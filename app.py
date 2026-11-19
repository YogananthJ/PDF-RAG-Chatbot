"""
PDF RAG CHATBOT
Gradio 6 + Groq + PyMuPDF + Sentence Transformers + ChromaDB
Required Space Secret:
    GROQ_API_KEY
Optional Space Variables:
    GROQ_MODEL=openai/gpt-oss-120b
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
    ENABLE_RERANKER=true
    MAX_PDF_MB=25
    MIN_RETRIEVAL_SCORE=0.20
Important:
- Gradio transfers the selected PDF before the Python indexing callback runs.
- filepath mode avoids making a second in-memory copy of the uploaded PDF.
- Hash-based reuse avoids repeated indexing, but cannot eliminate network transfer.
- ChromaDB is in-memory in this version; a Space restart clears active indexes.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import hashlib
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

import chromadb
import gradio as gr
import pymupdf
import spaces
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "PDF RAG CHATBOT"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in {
    "1", "true", "yes", "on"
}
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "25"))
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.20"))


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
SESSIONS = {}


# ---------------------------------------------------------------------------
# Cached models and client
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def load_reranker():
    if not ENABLE_RERANKER:
        return None
    return CrossEncoder(RERANKER_MODEL, device="cpu")


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it under Space Settings → "
            "Variables and secrets → Secrets."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# PDF utilities, chunking, and text helpers
# ---------------------------------------------------------------------------
def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pages(pdf_path):
    pages = []
    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append({"page": page_number, "text": text})

    if not pages:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR is required before this version can search it."
        )
    return pages, page_count


def split_page_into_chunks(text, chunk_size, overlap):
    if chunk_size < 200:
        raise ValueError("Chunk size must be at least 200 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary_candidates = [
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            ]
            boundary = max(boundary_candidates)
            if boundary > start + int(chunk_size * 0.60):
                end = boundary + 1
            else:
                word_boundary = text.rfind(" ", start, end)
                if word_boundary > start + int(chunk_size * 0.60):
                    end = word_boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < len(text) and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = max(start + 1, next_start)

    return chunks


def create_chunks(pages, chunk_size, overlap):
    chunks = []
    for page in pages:
        for local_index, text in enumerate(
            split_page_into_chunks(page["text"], chunk_size, overlap),
            start=1,
        ):
            chunks.append({
                "text": text,
                "page": page["page"],
                "local_index": local_index,
            })
    return chunks


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


# ---------------------------------------------------------------------------
# Pipeline construction and duplicate-index reuse
# ---------------------------------------------------------------------------
def build_pipeline(pdf_path, chunk_size, chunk_overlap, previous_state,
                   progress=gr.Progress()):
    if not pdf_path:
        return previous_state, "Please choose a PDF first.", ""

    try:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError("The uploaded temporary PDF file is missing.")

        file_size = path.stat().st_size
        if file_size > MAX_PDF_BYTES:
            raise ValueError(
                f"PDF is {file_size / (1024 * 1024):.1f} MB. "
                f"The current upload limit is {MAX_PDF_MB} MB."
            )

        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("The selected file is not a valid PDF.")

        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        progress(0.04, desc="Checking whether this PDF is already indexed…")
        fingerprint = sha256_file(path)

        if previous_state:
            previous_id = previous_state.get("session_id")
            previous = SESSIONS.get(previous_id)
            if (
                previous
                and previous["fingerprint"] == fingerprint
                and previous["chunk_size"] == chunk_size
                and previous["chunk_overlap"] == chunk_overlap
            ):
                return previous_state, (
                    f"**Already indexed:** `{path.name}` is unchanged. "
                    f"Reusing its {previous['chunk_count']} passages."
                ), previous["details"]

        progress(0.12, desc="Extracting text and page numbers…")
        pages, page_count = extract_pages(str(path))
        chunks = create_chunks(pages, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No searchable passages could be created.")

        progress(0.25, desc=f"Loading embedding model: {EMBEDDING_MODEL}")
        embedder = load_embedding_model()

        texts = [item["text"] for item in chunks]
        progress(0.38, desc=f"Generating embeddings for {len(texts)} passages…")
        vectors = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        progress(0.82, desc="Creating ChromaDB vector index and BM25 keyword index…")
        session_id = uuid.uuid4().hex
        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name="pdf_rag_" + session_id,
            metadata={"hnsw:space": "cosine"},
        )

        metadata = [
            {"page": int(item["page"]), "chunk_index": index}
            for index, item in enumerate(chunks)
        ]

        for start in range(0, len(chunks), 200):
            end = min(start + 200, len(chunks))
            collection.add(
                ids=[f"chunk_{index}" for index in range(start, end)],
                documents=texts[start:end],
                embeddings=vectors[start:end].tolist(),
                metadatas=metadata[start:end],
            )

        bm25 = BM25Okapi([tokenize(text) for text in texts])

        details = (
            f"**Pages processed:** {page_count}\n\n"
            f"**Passages indexed:** {len(chunks)}\n\n"
            f"**Embedding model:** `{EMBEDDING_MODEL}`\n\n"
            f"**Keyword search:** BM25\n\n"
            f"**Vector database:** ChromaDB (in-memory session)\n\n"
            f"**Reranker:** {'Enabled (loads on first question)' if ENABLE_RERANKER else 'Disabled'}"
        )

        SESSIONS[session_id] = {
            "client": client,
            "collection": collection,
            "embedder": embedder,
            "bm25": bm25,
            "chunks": chunks,
            "fingerprint": fingerprint,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "page_count": page_count,
            "filename": path.name,
            "details": details,
        }

        if previous_state:
            old_id = previous_state.get("session_id")
            if old_id in SESSIONS:
                old = SESSIONS.pop(old_id)
                try:
                    old["client"].delete_collection(old["collection"].name)
                except Exception:
                    pass

        progress(1.0, desc="Document search index ready")
        new_state = {"session_id": session_id, "fingerprint": fingerprint}
        status = (
            f"### ✓ Document indexed\n"
            f"**{path.name}** · {page_count} pages · {len(chunks)} searchable passages"
        )
        return new_state, status, details

    except Exception as exc:
        return (
            previous_state,
            f"### PDF processing failed\n`{type(exc).__name__}: {str(exc)[:400]}`",
            "",
        )


# ---------------------------------------------------------------------------
# Hybrid retrieval, reranking, and RRF fusion
# ---------------------------------------------------------------------------
def retrieve_passages(session, question, top_k):
    collection = session["collection"]
    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    candidate_count = min(max(int(top_k) * 8, 30), total_chunks)
    query_vector = session["embedder"].encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    vector_result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=candidate_count,
        include=["documents", "distances", "metadatas"],
    )

    candidates = {}
    rrf_scores = {}
    vector_docs = vector_result["documents"][0] or []
    vector_distances = vector_result["distances"][0] or []
    vector_metadatas = vector_result["metadatas"][0] or []

    for rank, (doc, distance, metadata) in enumerate(
        zip(vector_docs, vector_distances, vector_metadatas), start=1
    ):
        idx = int(metadata["chunk_index"])
        candidates[idx] = {
            "text": doc,
            "page": int(metadata["page"]),
            "vector_similarity": max(0.0, min(1.0, 1.0 - float(distance))),
            "vector_rank": rank,
            "bm25_score": 0.0,
            "bm25_rank": None,
        }
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

    bm25_scores = session["bm25"].get_scores(tokenize(question))
    lexical_order = sorted(
        range(len(bm25_scores)),
        key=lambda idx: float(bm25_scores[idx]),
        reverse=True,
    )[:candidate_count]

    for rank, idx in enumerate(lexical_order, start=1):
        score = float(bm25_scores[idx])
        if score <= 0:
            continue
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
        if idx not in candidates:
            chunk = session["chunks"][idx]
            candidates[idx] = {
                "text": chunk["text"],
                "page": int(chunk["page"]),
                "vector_similarity": None,
                "vector_rank": None,
                "bm25_score": score,
                "bm25_rank": rank,
            }
        else:
            candidates[idx]["bm25_score"] = score
            candidates[idx]["bm25_rank"] = rank

    if not candidates:
        return []

    results = []
    for idx, item in candidates.items():
        results.append({
            "chunk_index": idx,
            **item,
            "rrf_score": float(rrf_scores.get(idx, 0.0)),
        })
    results.sort(key=lambda item: item["rrf_score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rrf_rank"] = rank

    if ENABLE_RERANKER and results:
        try:
            reranker = load_reranker()
            pairs = [[question, item["text"]] for item in results]
            scores = reranker.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
            for item, score in zip(results, scores):
                item["rerank_score"] = float(score)
            reranked = sorted(
                results, key=lambda item: item["rerank_score"], reverse=True
            )
            for rank, item in enumerate(reranked, start=1):
                item["reranker_rank"] = rank
        except Exception as exc:
            for item in results:
                item["reranker_error"] = type(exc).__name__
                item["reranker_rank"] = None
    else:
        for item in results:
            item["reranker_rank"] = None

    for item in results:
        final_score = item["rrf_score"]
        if item.get("reranker_rank") is not None:
            final_score += 1.0 / (60 + item["reranker_rank"])
        item["final_score"] = float(final_score)

    results.sort(key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["final_rank"] = rank
    return results


# ---------------------------------------------------------------------------
# Retrieval diagnostics and answer generation
# ---------------------------------------------------------------------------
def render_passages(selected_passages, all_candidates=None):
    blocks = ["### Passages actually supplied to the answer model"]
    for index, item in enumerate(selected_passages, start=1):
        vector_score = item["vector_similarity"]
        vector_text = (
            f"{vector_score:.3f}" if vector_score is not None
            else "not in vector top-k"
        )
        rerank_text = (
            f" · cross-encoder {item['rerank_score']:.3f}"
            if "rerank_score" in item else ""
        )
        blocks.append(
            f"**[S{index}] PDF page {item['page']} · "
            f"vector similarity {vector_text}{rerank_text}**\n\n"
            f"{item['text'][:1500]}"
        )

    rows = []
    for item in (all_candidates or selected_passages)[:12]:
        vr = str(item["vector_rank"]) if item["vector_rank"] is not None else "—"
        br = str(item["bm25_rank"]) if item["bm25_rank"] is not None else "—"
        vs = (
            f"{item['vector_similarity']:.3f}"
            if item["vector_similarity"] is not None else "—"
        )
        bs = f"{item['bm25_score']:.3f}"
        rs = f"{item['rerank_score']:.3f}" if "rerank_score" in item else "—"
        rr = str(item.get("reranker_rank") or "—")
        fs = f"{item.get('final_score', item['rrf_score']):.4f}"
        preview = re.sub(r"\s+", " ", item["text"])[:160].replace("|", "\\|")
        rows.append(
            f"| {item['final_rank']} | {item['page']} | {vr} | {br} | "
            f"{vs} | {bs} | {item['rrf_score']:.4f} | {rr} | {rs} | {fs} | {preview}… |"
        )

    diagnostic = [
        "", "---", "### Retrieval candidate diagnostics (top 12 after final ranking)", "",
        "| Final | Page | Vector rank | BM25 rank | Vector similarity | BM25 raw score | Vector+BM25 RRF | Reranker rank | Reranker score | Final fused score | Passage preview |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "_Vector similarity is derived from Chroma cosine distance. BM25, RRF and fused scores are ranking signals. Cross-encoder scores are not probabilities or answer-accuracy scores._",
    ]
    return "\n\n---\n\n".join(blocks) + "\n\n" + "\n".join(diagnostic)


def answer_question(session_state, question, top_k, max_tokens,
                    temperature, chat_history, progress=gr.Progress()):
    history = list(chat_history or [])

    if not session_state or session_state.get("session_id") not in SESSIONS:
        return "Upload a PDF and build its search index first.", "", history, "Not evaluated"
    question = (question or "").strip()
    if not question:
        return "Enter a question first.", "", history, "Not evaluated"

    try:
        session = SESSIONS[session_state["session_id"]]

        progress(0.12, desc="Finding relevant passages with hybrid search…")
        candidate_passages = retrieve_passages(session, question, int(top_k))
        if not candidate_passages:
            return (
                "I couldn't find relevant passages in the uploaded PDF.",
                "",
                history,
                "No passages found",
            )

        passages = candidate_passages[:int(top_k)]
        retrieved_display = render_passages(passages, candidate_passages)

        vector_scores = [
            item["vector_similarity"]
            for item in passages
            if item["vector_similarity"] is not None
        ]
        best_vector_score = max(vector_scores) if vector_scores else None
        mean_vector_score = (
            sum(vector_scores) / len(vector_scores) if vector_scores else None
        )

        if best_vector_score is not None and best_vector_score < MIN_RETRIEVAL_SCORE:
            signal = (
                f"Low vector retrieval signal · best {best_vector_score:.3f}. "
                "Hybrid keyword results are shown below. This is not answer accuracy."
            )
            return (
                "I couldn't find enough relevant information in the uploaded PDF "
                "to answer reliably. Try using terms or names from the document.",
                retrieved_display,
                history,
                signal,
            )

        source_context = "\n\n".join(
            f"[S{index} | PDF page {item['page']}]\n{item['text']}"
            for index, item in enumerate(passages, start=1)
        )

        system_prompt = """You are PDF RAG CHATBOT, a precise assistant for uploaded documents.
EVIDENCE RULES:
- Use only the supplied PDF passages. Do not use outside knowledge.
- Never invent facts, figures, methods, results, names, quotations, or conclusions.
- When the passages do not support an answer, say clearly that the PDF excerpts do not contain enough information.
- Explain the answer naturally and directly. Use short sections or bullets when useful.
- Add source markers to factual claims using only the exact identifiers supplied, such as [S1] or [S3].
- Do not invent or modify source identifiers or page numbers.
- Distinguish direct statements from interpretation.
- If a question has several parts, answer only the parts supported by the excerpts.
- Do not describe retrieval similarity as answer accuracy."""

        user_prompt = (
            f"RETRIEVED PDF EXCERPTS:\n{source_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer using evidence from the excerpts and cite source markers."
        )

        progress(0.58, desc="Generating an evidence-grounded answer with Groq…")
        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = "The model returned an empty answer. Please try again."

        valid_pages = {
            f"S{index}": item["page"]
            for index, item in enumerate(passages, start=1)
        }
        seen = set(re.findall(r"\[(S\d+)\]", answer))
        for tag in sorted(seen):
            if tag in valid_pages:
                answer = answer.replace(f"[{tag}]", f"[Page {valid_pages[tag]}]")
            else:
                answer = answer.replace(f"[{tag}]", "")

        if vector_scores:
            quality = (
                f"Good vector retrieval signal · best {best_vector_score:.3f}, "
                f"average {mean_vector_score:.3f}."
            )
        else:
            quality = "Keyword-based retrieval only; no vector score available."

        signal = quality + " Retrieval similarity is not answer accuracy."

        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        progress(1.0, desc="Answer ready")
        return answer, retrieved_display, history, signal

    except Exception as exc:
        return (
            f"**Answer generation failed:** `{type(exc).__name__}: {str(exc)[:450]}`",
            "",
            history,
            "Unable to evaluate retrieval",
        )


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------
def reset_session(session_state):
    if session_state:
        session_id = session_state.get("session_id")
        if session_id in SESSIONS:
            old = SESSIONS.pop(session_id)
            try:
                old["client"].delete_collection(old["collection"].name)
            except Exception:
                pass
    return (
        None,
        "Choose a PDF and build the search index to begin.",
        "",
        "",
        [],
        [],
        "",
    )


# ---------------------------------------------------------------------------
# ZeroGPU compatibility hook
# ---------------------------------------------------------------------------
@spaces.GPU(duration=1)
def zerogpu_startup_compatibility():
    return None


# ---------------------------------------------------------------------------
# UI helper: tab controller
# ---------------------------------------------------------------------------
def show_tab(tab_name):
    chunking = tab_name == "Chunking"
    retrieval = tab_name == "Retrieval"
    generation = tab_name == "Generation"
    models = tab_name == "Models"
    return (
        gr.update(visible=chunking),
        gr.update(visible=chunking),
        gr.update(visible=retrieval),
        gr.update(visible=generation),
        gr.update(visible=generation),
        gr.update(visible=models),
    )


# ---------------------------------------------------------------------------
# Three.js particle sphere — exact reference implementation
# ---------------------------------------------------------------------------
PARTICLE_SPHERE_HEAD = r"""
<script type="module">
(async function () {
  "use strict";

  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let mounted = null;
  let THREE = null;

  async function waitForCanvas() {
    return new Promise((resolve) => {
      let attempts = 0;
      const interval = setInterval(() => {
        const canvas = document.getElementById("ai-sphere-canvas");
        if (canvas) {
          clearInterval(interval);
          resolve(canvas);
        } else if (++attempts > 120) {
          clearInterval(interval);
          resolve(null);
        }
      }, 250);
    });
  }

  async function mountSphere() {
    const canvas = document.getElementById("ai-sphere-canvas");
    if (!canvas) return false;
    if (mounted && mounted.canvas === canvas) return true;

    if (!THREE) {
      try {
        THREE = await import('https://esm.sh/three@0.160.0');
      } catch (err) {
        console.error("Failed to load Three.js.", err);
        return false;
      }
    }

    const width  = () => Math.max(160, canvas.clientWidth  || 280);
    const height = () => Math.max(160, canvas.clientHeight || 280);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    } catch (err) {
      console.error("WebGL initialization failed.", err);
      return false;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width(), height(), false);
    renderer.setClearColor(0x000000, 0);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 4.0);

    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // Colors matching your new reference (Deep Blue -> Cyan)
    const colorCenter = new THREE.Color(0.1, 0.3, 1.0); // Deep electric blue
    const colorEdge   = new THREE.Color(0.4, 0.9, 1.0); // Bright cyan

    // Create the nodes
    const COUNT = 400; 
    const radius = 1.5;
    const points = [];

    // Fibonacci sphere for even distribution
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < COUNT; i++) {
      const y = 1 - (i / (COUNT - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const theta = golden * i;
      const x = Math.cos(theta) * r;
      const z = Math.sin(theta) * r;
      
      const pos = new THREE.Vector3(x, y, z).multiplyScalar(radius);
      points.push(pos);
      
      positions.push(pos.x, pos.y, pos.z);
      
      // Randomly mix between blue and cyan for the nodes
      const mixed = colorCenter.clone().lerp(colorEdge, Math.random());
      colors.push(mixed.r, mixed.g, mixed.b);
    }

    // Connect nearby nodes with lines
    const linePositions = [];
    const lineColors = [];
    const maxDist = 0.5; // Maximum distance to draw a line between two nodes

    for (let i = 0; i < COUNT; i++) {
      for (let j = i + 1; j < COUNT; j++) {
        const dist = points[i].distanceTo(points[j]);
        if (dist < maxDist) {
          // Add line segment
          linePositions.push(points[i].x, points[i].y, points[i].z);
          linePositions.push(points[j].x, points[j].y, points[j].z);
          
          // Dimmer lines for depth
          const c1 = colorCenter.clone().lerp(colorEdge, Math.random()).multiplyScalar(0.4);
          const c2 = colorCenter.clone().lerp(colorEdge, Math.random()).multiplyScalar(0.4);
          lineColors.push(c1.r, c1.g, c1.b);
          lineColors.push(c2.r, c2.g, c2.b);
        }
      }
    }

    // 1. Create the Lines (The Web)
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    const lineMat = new THREE.LineBasicMaterial({ 
      vertexColors: true, 
      transparent: true, 
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    // 2. Create the Nodes (The Glowing Dots)
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const vertexShader = `
      uniform float uTime;
      uniform float uPulse;
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vColor = color;
        
        // Radial Pulsation (breathing)
        vec3 pos = position;
        // Expand/contract slightly using a sine wave
        pos *= 1.0 + sin(uTime * 1.5) * 0.05 * uPulse;
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        vAlpha = clamp(1.0 - (-mvPosition.z - 2.0) / 4.0, 0.3, 1.0);
        
        gl_PointSize = 12.0 / -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        // Make dots perfectly circular
        vec2 p = gl_PointCoord - vec2(0.5);
        float r = length(p);
        if (r > 0.5) discard;
        
        float alpha = smoothstep(0.5, 0.0, r);
        // Add a bright white core to nodes
        vec3 finalColor = mix(vColor, vec3(1.0), 0.4 * smoothstep(0.2, 0.0, r));
        
        gl_FragColor = vec4(finalColor, alpha * vAlpha);
      }
    `;

    const uniforms = {
      uTime: { value: 0 },
      uPulse: { value: 1.0 } // Can be used for state (e.g., processing = faster pulse)
    };

    const matNodes = new THREE.ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    const particleSystem = new THREE.Points(geometry, matNodes);
    scene.add(particleSystem);

    // Group them so they spin together
    const sphereGroup = new THREE.Group();
    sphereGroup.add(lines);
    sphereGroup.add(particleSystem);
    scene.add(sphereGroup);

    const state = { visible: true, raf: null };

    const resize = () => {
      const w = width(), h = height();
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", resize);
    resize();

    const clock = new THREE.Clock();

    const tick = () => {
      state.raf = requestAnimationFrame(tick);
      if (!state.visible) return;

      const t = clock.getElapsedTime();
      uniforms.uTime.value = t;

      // Spin the entire group
      sphereGroup.rotation.y = t * 0.2;
      sphereGroup.rotation.x = Math.sin(t * 0.3) * 0.1;

      renderer.render(scene, camera);
    };

    if (reduceMotion) {
      renderer.render(scene, camera);
    } else {
      tick();
    }

    const cleanup = () => {
      if (state.raf) cancelAnimationFrame(state.raf);
      window.removeEventListener("resize", resize);
      geometry.dispose();
      matNodes.dispose();
      lineGeo.dispose();
      lineMat.dispose();
      renderer.dispose();
    };

    mounted = { canvas, cleanup };
    return true;
  }

  if (document.readyState === "complete" || document.readyState === "interactive") {
    const canvas = await waitForCanvas();
    if (canvas) mountSphere();
  } else {
    window.addEventListener("DOMContentLoaded", async () => {
      const canvas = await waitForCanvas();
      if (canvas) mountSphere();
    });
  }
})();
</script>
"""
# ---------------------------------------------------------------------------
# CSS design system
# ---------------------------------------------------------------------------
CSS = """
/* ============================================================
   TOKENS
   ============================================================ */
:root {
  --bg-primary: #080A12;
  --bg-secondary: #101425;
  --bg-panel: #111827;
  --bg-elevated: #1A2338;

  --border-soft: rgba(255,255,255,0.06);
  --border-med: rgba(255,255,255,0.10);

  --violet: #8565FF;
  --purple: #B34DFF;
  --cyan: #54D9FF;
  --blue: #4C8DFF;

  --text-primary: #F4F5FF;
  --text-secondary: #929CB5;
  --text-muted: #5A6580;
  --success: #36D6A0;

  --grad-brand: linear-gradient(135deg, #8565FF 0%, #B34DFF 50%, #54D9FF 100%);
  --grad-soft: linear-gradient(135deg, rgba(133,101,255,0.16), rgba(179,77,255,0.12) 50%, rgba(84,217,255,0.10));

  --r-sm: 10px; --r-md: 14px; --r-lg: 18px; --r-xl: 22px;

  --shadow-soft: 0 4px 20px rgba(0,0,0,0.35);
  --shadow-glow: 0 8px 36px rgba(133,101,255,0.22);

  --t-fast: 160ms cubic-bezier(0.4,0,0.2,1);
  --t-med:  280ms cubic-bezier(0.4,0,0.2,1);
}

/* ============================================================
   GLOBAL
   ============================================================ */
html, body, .gradio-container { background: var(--bg-primary) !important; height: 100%; }
body { overflow-x: hidden; }

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 0 22px 20px !important;
  font-family: 'Inter', 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
  color: var(--text-primary) !important;
  min-height: 100svh;
  background:
    radial-gradient(1200px 600px at 15% -8%, rgba(133,101,255,0.12), transparent 55%),
    radial-gradient(1000px 500px at 88% 4%, rgba(84,217,255,0.08), transparent 55%),
    radial-gradient(900px 500px at 50% 105%, rgba(179,77,255,0.06), transparent 60%),
    var(--bg-primary) !important;
}

footer { display: none !important; }
.gradio-container > * { position: relative; z-index: 1; }

/* ============================================================
   TOP NAV
   ============================================================ */
#topnav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 18px 26px;
  margin: 18px 0 0;
  border: 1px solid var(--border-med);
  border-radius: var(--r-xl);
  background:
    linear-gradient(135deg, rgba(133,101,255,0.06), rgba(84,217,255,0.04)),
    rgba(16,20,37,0.72);
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  box-shadow: var(--shadow-soft);
  position: sticky;
  top: 12px;
  z-index: 50;
}
#topnav::before {
  content: "";
  position: absolute; inset: 0; border-radius: inherit; padding: 1px;
  background: linear-gradient(135deg, rgba(133,101,255,0.5), transparent 40%, rgba(84,217,255,0.4));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  pointer-events: none; opacity: 0.6;
}
.nav-brand { display: flex; align-items: center; gap: 14px; }
.nav-logo {
  position: relative; width: 44px; height: 44px;
  border-radius: 12px;
  background: var(--grad-brand);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.12) inset, 0 8px 26px rgba(133,101,255,0.45);
  overflow: hidden;
}
.nav-logo::after {
  content: ""; position: absolute; inset: -50%;
  background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.35), transparent 40%);
  animation: logoSpin 8s linear infinite;
}
@keyframes logoSpin { to { transform: rotate(360deg); } }
.nav-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 17px; font-weight: 700; letter-spacing: -0.01em;
  background: linear-gradient(135deg, #FFFFFF, #C7CDF0);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.nav-subtitle { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.nav-status-group { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.live-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 14px; border-radius: 999px;
  background: rgba(54,214,160,0.08);
  border: 1px solid rgba(54,214,160,0.30);
  font-size: 12px; font-weight: 600; color: var(--success);
}
.live-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 10px rgba(54,214,160,0.9), 0 0 20px rgba(54,214,160,0.4);
  animation: pulseDot 2s ease-in-out infinite;
}
@keyframes pulseDot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}
.nav-pill {
  padding: 7px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border-med);
  font-size: 11.5px; color: var(--text-secondary); font-weight: 500;
}
.nav-pill strong { color: var(--text-primary); font-weight: 600; }

/* ============================================================
   APP SHELL
   ============================================================ */
#app-shell {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  gap: 14px;
  min-height: calc(100svh - 90px);
  max-width: 1472px;
  margin: 14px auto 0;
  width: 100%;
}
@media (max-width: 900px) {
  #app-shell {
    grid-template-rows: auto auto auto auto;
    min-height: auto;
  }
}

/* ============================================================
   HERO
   ============================================================ */
#hero {
  position: relative;
  padding: 20px 28px;
  border-radius: var(--r-xl);
  border: 1px solid var(--border-med);
  background:
    radial-gradient(700px 380px at 78% 30%, rgba(133,101,255,0.16), transparent 65%),
    radial-gradient(600px 320px at 20% 80%, rgba(84,217,255,0.10), transparent 65%),
    linear-gradient(135deg, #101425, #0B0F1C 60%, #101425);
  overflow: hidden;
  box-shadow: var(--shadow-soft);
  isolation: isolate;
  min-height: 220px;
  max-height: 300px;
}
@media (max-width: 900px) {
  #hero { max-height: none; padding: 18px 20px; }
}
#hero::before {
  content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px;
  background: linear-gradient(135deg, rgba(133,101,255,0.5), transparent 40%, rgba(84,217,255,0.5));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  pointer-events: none;
}
.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(200px, 0.85fr);
  gap: 20px;
  align-items: center;
  height: 100%;
}
@media (max-width: 900px) {
  .hero-grid { grid-template-columns: 1fr; gap: 12px; }
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 12px; border-radius: 999px;
  background: rgba(133,101,255,0.10);
  border: 1px solid rgba(133,101,255,0.30);
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: #C7B9FF;
  margin-bottom: 10px;
}
.hero-eyebrow .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan); box-shadow: 0 0 8px var(--cyan);
}
.hero-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: clamp(20px, 2.2vw, 30px);
  font-weight: 800; line-height: 1.12;
  letter-spacing: -0.02em;
  margin: 0 0 8px;
  background: linear-gradient(135deg, #FFFFFF 0%, #C7CDF0 55%, #8F97C7 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-title .accent {
  background: linear-gradient(135deg, #8565FF, #54D9FF);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-subtitle {
  font-size: 13px; line-height: 1.55;
  color: var(--text-secondary);
  max-width: 520px; margin: 0 0 12px;
}
.hero-status { display: flex; gap: 8px; flex-wrap: wrap; }
.status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 8px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-med);
  font-size: 11.5px; color: var(--text-secondary); font-weight: 500;
}
.status-pill .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--success); box-shadow: 0 0 6px var(--success);
}
.ai-sphere-wrap {
  position: relative;
  width: 100%; height: 100%;
  min-height: 180px; max-height: 260px;
  display: flex; align-items: center; justify-content: center;
}
#ai-sphere-canvas {
  display: block;
  width: 100%; height: 100%;
  max-width: 260px; max-height: 260px;
  /* Removed drop-shadow to prevent glowing white box on failure */
}

/* ============================================================
   WORKSPACE
   ============================================================ */
#workspace {
  display: grid;
  grid-template-columns: minmax(300px, 0.34fr) minmax(0, 0.66fr);
  gap: 14px;
  min-height: 0;
  height: 100%;
  overflow: hidden;
}
@media (max-width: 1100px) {
  #workspace { grid-template-columns: minmax(280px, 0.42fr) minmax(0, 0.58fr); }
}
@media (max-width: 900px) {
  #workspace {
    grid-template-columns: 1fr;
    height: auto;
    overflow: visible;
  }
}

.ws-col {
  display: flex; flex-direction: column;
  gap: 12px;
  min-width: 0; min-height: 0;
  height: 100%;
  overflow: hidden;
}
@media (max-width: 900px) {
  .ws-col { height: auto; overflow: visible; }
}

.ws-col-left {
  overflow-y: auto;
  padding-right: 4px;
  scrollbar-gutter: stable;
}
.ws-col-right {
  display: grid;
  grid-template-rows: auto minmax(160px, 1fr) auto auto;
  gap: 12px;
  overflow: hidden;
}
@media (max-width: 900px) {
  .ws-col-left, .ws-col-right {
    overflow: visible; padding-right: 0;
    grid-template-rows: auto;
  }
}

/* ============================================================
   PANEL CARD
   ============================================================ */
.gr-group, .panel-card {
  background: linear-gradient(160deg, rgba(17,24,39,0.94), rgba(16,20,37,0.94)) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--r-lg) !important;
  padding: 14px !important;
  box-shadow: var(--shadow-soft) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color var(--t-med), box-shadow var(--t-med);
  position: relative;
  overflow: hidden;
  flex-shrink: 0;
  min-width: 0;
}
.gr-group::before, .panel-card::before {
  content: "";
  position: absolute; top: 0; left: 20%; right: 20%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(133,101,255,0.5), transparent);
  opacity: 0.7;
}
.gr-group:hover, .panel-card:hover {
  border-color: rgba(133,101,255,0.35) !important;
  box-shadow: var(--shadow-soft), var(--shadow-glow) !important;
}

.ws-col-right .panel-card.chat-card   { display: flex; flex-direction: column; min-height: 0; }
.ws-col-right .panel-card.answer-card { display: flex; flex-direction: column; min-height: 0; }

.module-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin: 0 0 10px;
}
.module-title {
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--text-muted);
}
.module-title .icon {
  width: 20px; height: 20px;
  display: inline-flex; align-items: center; justify-content: center;
  border-radius: 6px;
  background: var(--grad-soft);
  border: 1px solid var(--border-med);
  font-size: 10px;
}
.module-num {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  color: var(--cyan);
  background: rgba(84,217,255,0.08);
  border: 1px solid rgba(84,217,255,0.22);
  border-radius: 999px;
  padding: 3px 9px;
}

/* ============================================================
   BUTTONS
   ============================================================ */
button.gr-button-primary,
.gr-button.gr-button-primary,
button.primary {
  position: relative;
  background: var(--grad-brand) !important;
  border: none !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 9px 16px !important;
  border-radius: var(--r-md) !important;
  box-shadow: 0 4px 20px rgba(133,101,255,0.35), 0 0 0 1px rgba(255,255,255,0.08) inset !important;
  transition: transform var(--t-fast), box-shadow var(--t-fast), filter var(--t-fast) !important;
  overflow: hidden;
}
button.gr-button-primary::after,
.gr-button.gr-button-primary::after {
  content: ""; position: absolute; top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
  transition: left 700ms ease;
}
button.gr-button-primary:hover::after { left: 100%; }
button.gr-button-primary:hover,
.gr-button.gr-button-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 28px rgba(133,101,255,0.55), 0 0 0 1px rgba(255,255,255,0.12) inset !important;
  filter: brightness(1.06);
}

button.gr-button-secondary,
.gr-button.gr-button-secondary {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 12.5px !important;
  border-radius: var(--r-md) !important;
  padding: 8px 14px !important;
}
button.gr-button-secondary:hover,
.gr-button.gr-button-secondary:hover {
  background: var(--bg-panel) !important;
  border-color: rgba(133,101,255,0.4) !important;
  color: var(--text-primary) !important;
}

/* ============================================================
   INPUTS / SLIDERS / FILES
   ============================================================ */
.gr-input, .gr-text-input, input[type="text"], textarea,
.gr-box input, .gr-box textarea {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-primary) !important;
  font-size: 13px !important;
}
.gr-input:focus, textarea:focus, input:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 3px rgba(133,101,255,0.18) !important;
  outline: none !important;
}
textarea::placeholder, input::placeholder { color: var(--text-muted) !important; }

.gr-file, .file-preview, .upload-container {
  background: linear-gradient(160deg, rgba(17,24,39,0.7), rgba(16,20,37,0.7)) !important;
  border: 1.5px dashed rgba(133,101,255,0.35) !important;
  border-radius: var(--r-md) !important;
  transition: border-color var(--t-med), background var(--t-med) !important;
}
.gr-file:hover, .upload-container:hover {
  border-color: var(--violet) !important;
  background: rgba(133,101,255,0.06) !important;
}

input[type="range"] { accent-color: var(--violet) !important; }
.gr-slider input[type="number"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  color: var(--text-primary) !important;
  border-radius: 7px !important;
}
label, .gr-label, span[data-testid="block-info"] {
  color: var(--text-secondary) !important;
  font-size: 11.5px !important;
  font-weight: 500 !important;
}

/* ============================================================
   TABS
   ============================================================ */
#tabs-radio {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: 10px !important;
  padding: 3px !important;
  margin-bottom: 10px;
}
#tabs-radio .wrap,
#tabs-radio [role="radiogroup"] {
  display: flex !important;
  gap: 2px !important;
  flex-wrap: nowrap !important;
  overflow-x: auto !important;
  scrollbar-width: none;
  background: transparent !important;
  border: none !important;
}
#tabs-radio .wrap::-webkit-scrollbar { display: none; }
#tabs-radio label {
  flex: 1 1 auto !important;
  padding: 6px 12px !important;
  border-radius: 7px !important;
  font-size: 11.5px !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  white-space: nowrap !important;
  cursor: pointer !important;
  text-align: center !important;
  border: 1px solid transparent !important;
  transition: background var(--t-fast), color var(--t-fast), border-color var(--t-fast) !important;
  background: transparent !important;
}
#tabs-radio label:hover { color: var(--text-primary) !important; }
#tabs-radio input[type="radio"] { display: none !important; }
#tabs-radio label:has(input[type="radio"]:checked),
#tabs-radio label.selected {
  background: var(--grad-soft) !important;
  color: var(--text-primary) !important;
  border-color: rgba(133,101,255,0.35) !important;
}
#tabs-radio label:has(input[type="radio"]:checked) span,
#tabs-radio label.selected span {
  color: var(--text-primary) !important;
}

/* ============================================================
   CHAT / ANSWER REGIONS
   ============================================================ */
.answer-box {
  display: flex; flex-direction: column;
  max-height: 100%;
  min-height: 60px;
  overflow-y: auto;
  overflow-wrap: anywhere;
  word-break: break-word;
  padding: 2px 6px;
  scrollbar-gutter: stable;
}
.answer-box > * { max-width: 100%; }
.answer-box:empty::before {
  content: "Answers will appear here once you ask a question.";
  color: var(--text-muted);
  font-size: 12px;
  font-style: italic;
}
.signal-box {
  font-size: 11.5px;
  color: var(--text-muted);
  padding: 0 4px;
  overflow-wrap: anywhere;
}

#chatbot-hidden {
  height: 1px !important;
  min-height: 1px !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
}

/* ============================================================
   ACCORDION
   ============================================================ */
.gr-accordion, details {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--r-md) !important;
}
details[open] { border-color: rgba(133,101,255,0.35) !important; }
summary {
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  padding: 9px 12px !important;
  cursor: pointer;
}
summary:hover { color: var(--violet) !important; }
.retrieval-scroll {
  max-height: 240px;
  overflow-y: auto;
  padding: 2px 4px;
  scrollbar-gutter: stable;
}

/* ============================================================
   MARKDOWN
   ============================================================ */
.gr-markdown, .prose {
  color: var(--text-primary) !important;
  line-height: 1.6 !important;
  font-size: 13.5px !important;
  overflow-wrap: anywhere;
}
.gr-markdown h2 { font-size: 16px; margin-top: 1em; }
.gr-markdown h3 { font-size: 14px; margin-top: 1em; }
.gr-markdown a { color: var(--cyan) !important; text-decoration: none; }
.gr-markdown a:hover { text-decoration: underline; }
.gr-markdown code {
  background: rgba(133,101,255,0.10) !important;
  color: #C7B9FF !important;
  padding: 2px 6px !important;
  border-radius: 5px !important;
  font-size: 0.88em !important;
  border: 1px solid rgba(133,101,255,0.22);
}
.gr-markdown pre {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--r-md) !important;
  padding: 12px !important;
  overflow-x: auto;
}
.gr-markdown blockquote {
  border-left: 3px solid var(--violet) !important;
  background: rgba(133,101,255,0.06) !important;
  padding: 8px 14px !important;
  border-radius: 0 var(--r-sm) var(--r-sm) 0 !important;
  color: var(--text-secondary) !important;
}
.gr-markdown table {
  border-collapse: collapse !important;
  font-size: 12px !important;
  width: 100%;
  table-layout: fixed;
}
.gr-markdown th, .gr-markdown td {
  padding: 6px 9px !important;
  border: 1px solid var(--border-med) !important;
  overflow-wrap: anywhere;
}
.gr-markdown th {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  text-align: left;
}
.gr-markdown td { color: var(--text-secondary) !important; }

/* ============================================================
   PROGRESS
   ============================================================ */
.gr-progress, .progress-bar, [role="progressbar"] {
  background: var(--bg-elevated) !important;
  border-radius: 999px !important;
  overflow: hidden;
}
.gr-progress-fill, [role="progressbar"] > div {
  background: var(--grad-brand) !important;
  border-radius: 999px !important;
}

/* ============================================================
   PIPELINE VISUAL
   ============================================================ */
.pipeline-viz {
  display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
  padding: 8px 10px; margin: 0 0 8px;
  border-radius: var(--r-sm);
  background: var(--bg-panel);
  border: 1px solid var(--border-med);
  font-size: 9.5px; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--text-muted);
}
.pipeline-step {
  padding: 3px 7px; border-radius: 5px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border-med);
}
.pipeline-step.active {
  background: var(--grad-soft);
  border-color: rgba(133,101,255,0.4);
  color: var(--text-primary);
}
.pipeline-arrow { color: var(--text-muted); }

/* ============================================================
   WELCOME CARD
   ============================================================ */
#welcome-card {
  padding: 12px 14px;
  margin-bottom: 10px;
  border-radius: var(--r-md);
  background: linear-gradient(135deg, rgba(133,101,255,0.08), rgba(84,217,255,0.05)), var(--bg-elevated);
  border: 1px solid var(--border-med);
  position: relative;
  overflow: hidden;
}
#welcome-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: var(--grad-brand); opacity: 0.7;
}
#welcome-card h3 {
  margin: 0 0 4px;
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 13.5px; font-weight: 700;
  color: var(--text-primary);
}
#welcome-card p {
  margin: 0; font-size: 11.5px;
  color: var(--text-secondary); line-height: 1.5;
}
.example-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 5px;
  margin-top: 8px;
}
.example-chip {
  padding: 6px 9px; border-radius: 7px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border-med);
  font-size: 10.5px; color: var(--text-secondary);
  transition: border-color var(--t-fast), background var(--t-fast), color var(--t-fast);
}
.example-chip:hover {
  border-color: rgba(133,101,255,0.4);
  background: rgba(133,101,255,0.06);
  color: var(--text-primary);
}

/* ============================================================
   FOOTER
   ============================================================ */
.app-footer {
  text-align: center;
  padding: 4px 0;
  color: var(--text-muted);
  font-size: 10.5px;
}
.app-footer .accent {
  background: linear-gradient(135deg, #8565FF, #54D9FF);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 600;
}

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--bg-elevated);
  border-radius: 999px;
  border: 2px solid var(--bg-primary);
}
::-webkit-scrollbar-thumb:hover { background: #2A3448; }

/* ============================================================
   PROCESSING RING
   ============================================================ */
body.rag-generating .ai-sphere-wrap::after {
  content: ""; position: absolute; inset: 12%;
  border-radius: 50%;
  border: 1px solid rgba(84,217,255,0.35);
  border-top-color: transparent;
  animation: spinRing 1.4s linear infinite;
  pointer-events: none;
}
@keyframes spinRing { to { transform: rotate(360deg); } }

/* ============================================================
   REDUCED MOTION
   ============================================================ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
"""


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------
with gr.Blocks(title=APP_NAME) as demo:
    # --- Component declarations: header + shell -----------------------------
    gr.HTML(
        f"""
        <div id="topnav">
          <div class="nav-brand">
            <div class="nav-logo">📄</div>
            <div>
              <div class="nav-title">PDF RAG Assistant</div>
              <div class="nav-subtitle">Advanced Document Intelligence Platform</div>
            </div>
          </div>
          <div class="nav-status-group">
            <span class="live-badge"><span class="live-dot"></span>System Online</span>
            <span class="nav-pill">🧠 <strong>Groq</strong> · {GROQ_MODEL.split('/')[-1]}</span>
            <span class="nav-pill">🔎 <strong>Embed</strong> · {EMBEDDING_MODEL.split('/')[-1]}</span>
            <span class="nav-pill">🎯 <strong>Reranker</strong> · {'On' if ENABLE_RERANKER else 'Off'}</span>
          </div>
        </div>
        """
    )

    session_state = gr.State(None)
    chat_state = gr.State([])

    startup_hook = gr.Button(visible=False)
    startup_hook.click(fn=zerogpu_startup_compatibility, inputs=[], outputs=[])

    with gr.Column(elem_id="app-shell"):

        gr.HTML(
            """
            <div id="hero">
              <div class="hero-grid">
                <div>
                  <div class="hero-eyebrow"><span class="dot"></span>Next-generation RAG Engine</div>
                  <h1 class="hero-title">
                    Your Documents.<br>
                    <span class="accent">One Intelligent Mind.</span>
                  </h1>
                  <p class="hero-subtitle">
                    Transform your PDFs into searchable knowledge. Ask anything.
                    Answers grounded in your documents with full retrieval transparency.
                  </p>
                  <div class="hero-status">
                    <span class="status-pill"><span class="dot"></span>RAG Engine Active</span>
                    <span class="status-pill"><span class="dot"></span>Vector DB Connected</span>
                    <span class="status-pill"><span class="dot"></span>AI Ready</span>
                  </div>
                </div>
                <div class="ai-sphere-wrap">
                  <canvas id="ai-sphere-canvas" aria-label="3D particle intelligence core"></canvas>
                </div>
              </div>
            </div>
            """
        )

        with gr.Row(elem_id="workspace"):

            with gr.Column(elem_classes="ws-col ws-col-left", min_width=0):

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">📂</span>Document Center</div>
                          <span class="module-num">01</span>
                        </div>
                        """
                    )
                    pdf_file = gr.File(
                        label="Drop your PDF here or click to upload",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath",
                    )
                    gr.Markdown(
                        f"<small style='color:var(--text-muted);font-size:11px;'>"
                        f"Max {MAX_PDF_MB} MB · Text PDFs · Scanned PDFs need OCR"
                        f"</small>"
                    )
                    build_button = gr.Button(
                        "⚡ Build Search Index",
                        variant="primary",
                    )
                    status = gr.Markdown(
                        "Upload a PDF to begin.",
                        elem_classes="status-msg",
                    )
                    index_details = gr.Markdown(elem_classes="status-detail")

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">⚙️</span>Index Configuration</div>
                          <span class="module-num">02</span>
                        </div>
                        """
                    )
                    tabs = gr.Radio(
                        choices=["Chunking", "Retrieval", "Generation", "Models"],
                        value="Chunking",
                        label="",
                        show_label=False,
                        container=False,
                        elem_id="tabs-radio",
                    )

                    chunk_size = gr.Slider(
                        400, 1400, value=1000, step=50,
                        label="Chunk size (characters)",
                        visible=True,
                    )
                    chunk_overlap = gr.Slider(
                        0, 350, value=150, step=10,
                        label="Chunk overlap (characters)",
                        visible=True,
                    )
                    top_k = gr.Slider(
                        2, 10, value=6, step=1,
                        label="Retrieved passages",
                        visible=False,
                    )
                    max_tokens = gr.Slider(
                        128, 1000, value=650, step=32,
                        label="Answer length (tokens)",
                        visible=False,
                    )
                    temperature = gr.Slider(
                        0, 0.7, value=0.15, step=0.05,
                        label="Creativity",
                        visible=False,
                    )

                    with gr.Group(visible=False) as models_group:
                        gr.Textbox(
                            value=GROQ_MODEL,
                            label="Answer generation · Groq",
                            interactive=False,
                        )
                        gr.Textbox(
                            value=EMBEDDING_MODEL,
                            label="Embedding · CPU",
                            interactive=False,
                        )
                        gr.Textbox(
                            value=RERANKER_MODEL if ENABLE_RERANKER else "Disabled",
                            label="Cross-encoder reranker",
                            interactive=False,
                        )
                        gr.Markdown(
                            "<small style='color:var(--text-muted);font-size:11px;'>"
                            "Hybrid retrieval: vector + BM25 + RRF, optionally reranked."
                            "</small>"
                        )

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">🧠</span>Model Stack</div>
                          <span class="module-num">03</span>
                        </div>
                        <div style="font-size:11.5px;color:var(--text-secondary);line-height:1.6;">
                          Answer generation, embedding, and reranking models are
                          configured above. Switch to the <strong>Models</strong>
                          tab in Index Configuration to inspect the active stack.
                        </div>
                        """
                    )

            with gr.Column(elem_classes="ws-col ws-col-right", min_width=0):

                with gr.Group(elem_classes="panel-card chat-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">💬</span>AI Conversation</div>
                          <span class="module-num">01</span>
                        </div>
                        """
                    )
                    gr.HTML(
                        """
                        <div id="welcome-card">
                          <h3>👋 Ask your document anything</h3>
                          <p>
                            Upload a PDF in the Document Center, build the index,
                            then ask a question. Answers are grounded strictly in
                            your document with page-level citations.
                          </p>
                          <div class="example-grid">
                            <div class="example-chip">What is the main objective?</div>
                            <div class="example-chip">Summarize the methodology.</div>
                            <div class="example-chip">Key findings?</div>
                            <div class="example-chip">Limitations discussed?</div>
                          </div>
                        </div>
                        """
                    )
                    question = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything about the document…",
                        lines=2,
                    )
                    ask_button = gr.Button(
                        "✦ Generate Grounded Answer",
                        variant="primary",
                    )
                    chatbot = gr.Chatbot(
                        label="Conversation history",
                        elem_id="chatbot-hidden",
                    )

                with gr.Group(elem_classes="panel-card answer-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">📝</span>Answer Workspace</div>
                          <span class="module-num">02</span>
                        </div>
                        """
                    )
                    answer = gr.Markdown(elem_classes="answer-box")
                    retrieval_indicator = gr.Markdown(elem_classes="signal-box")

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        """
                        <div class="module-head">
                          <div class="module-title"><span class="icon">🔎</span>Retrieval Intelligence</div>
                          <span class="module-num">03</span>
                        </div>
                        """
                    )
                    with gr.Accordion("Open retrieval diagnostics", open=False):
                        gr.HTML(
                            """
                            <div class="pipeline-viz">
                              <span class="pipeline-step active">Question</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">Vector</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">BM25</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">RRF</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">Rerank</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">Context</span>
                              <span class="pipeline-arrow">→</span>
                              <span class="pipeline-step">Answer</span>
                            </div>
                            """
                        )
                        with gr.Column(elem_classes="retrieval-scroll"):
                            retrieved = gr.Markdown()

                reset_button = gr.Button(
                    "↻ Clear document and start over",
                    variant="secondary",
                )

        gr.HTML(
            """
            <div class="app-footer">
              <span class="accent">PDF RAG Assistant</span> · PyMuPDF · Sentence Transformers ·
              Hybrid Retrieval · ChromaDB · Groq
            </div>
            """
        )

    # --- Event wiring -------------------------------------------------------
    build_button.click(
        fn=build_pipeline,
        inputs=[pdf_file, chunk_size, chunk_overlap, session_state],
        outputs=[session_state, status, index_details],
    )

    tabs.change(
        fn=show_tab,
        inputs=[tabs],
        outputs=[chunk_size, chunk_overlap, top_k, max_tokens, temperature, models_group],
    )

    ask_inputs = [
        session_state,
        question,
        top_k,
        max_tokens,
        temperature,
        chat_state,
    ]
    ask_outputs = [answer, retrieved, chatbot, retrieval_indicator]

    ask_button.click(
        fn=answer_question,
        inputs=ask_inputs,
        outputs=ask_outputs,
    ).then(
        fn=lambda history: history,
        inputs=[chatbot],
        outputs=[chat_state],
    )

    question.submit(
        fn=answer_question,
        inputs=ask_inputs,
        outputs=ask_outputs,
    ).then(
        fn=lambda history: history,
        inputs=[chatbot],
        outputs=[chat_state],
    )

    reset_button.click(
        fn=reset_session,
        inputs=[session_state],
        outputs=[
            session_state,
            status,
            index_details,
            answer,
            chatbot,
            chat_state,
            retrieval_indicator,
        ],
    ).then(
        fn=lambda: None,
        inputs=[],
        outputs=[pdf_file],
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[question],
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[retrieved],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=CSS,
        head=PARTICLE_SPHERE_HEAD,
        max_file_size=f"{MAX_PDF_MB}mb",
    )
---
title: PDF RAG Assistant
emoji: 🪐
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">

# 🌌 PDF RAG ASSISTANT
### Advanced Document Intelligence Platform

*Transform your PDFs into searchable knowledge. Ask anything. Get answers grounded in your documents with full retrieval transparency and cinematic visual feedback.*

[![Live App](https://img.shields.io/badge/🚀%20LAUNCH%20APP-Live%20on%20Hugging%20Face-purple?style=for-the-badge)](https://yogananth-pdf-rag-chatbot.hf.space)

[![Gradio](https://img.shields.io/badge/Gradio-6.32.0-orange)](https://gradio.app)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black)](https://threejs.org)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20Llama%203-red)](https://groq.com)

</div>

---

## ✨ The Visual Experience

This application doesn't just process text; it visualizes intelligence. The frontend is built with a premium, futuristic **Midnight AI** design system.

> **🧠 The 3D Neural Core:** 
> At the heart of the hero section is a fully interactive, real-time WebGL particle sphere powered by **Three.js**. It features thousands of glowing nodes connected by a dynamic spatial web. It continuously rotates, breathes, and pulses.

> **⚡ Real-Time State Feedback:** 
> The particle sphere reacts to the application state via a custom event listener. It idles with a gentle radial pulsation, energizes with accelerated spinning when a document is indexing, and glows intensely while the LLM generates an answer.

> **💎 Glassmorphic Workspace:** 
> A perfectly balanced 34/66 CSS Grid layout locks the viewport. Deep navy panels feature subtle luminous borders, soft shadows, and smooth hover states. The layout *never* shifts, regardless of whether an answer is 1 sentence or 10 pages long—the answer box scrolls internally.

---

## 🧠 The RAG Engine

Behind the stunning UI is a robust, production-grade Retrieval-Augmented Generation pipeline designed for absolute accuracy.

| Feature | Implementation |
| :--- | :--- |
| **Hybrid Retrieval** | Combines semantic vector search (ChromaDB) with lexical keyword search (BM25). |
| **Rank Fusion** | Uses Reciprocal Rank Fusion (RRF) to intelligently merge results from both searches. |
| **Cross-Encoder Reranking** | Re-evaluates the top passages to ensure the absolute highest relevance. |
| **Full Transparency** | A built-in "Retrieval Intelligence" accordion shows you the exact similarity scores, page numbers, and reranked passages used to generate the answer. |
| **Citation Mapping** | The LLM is strictly instructed to cite sources (e.g., `[Page 4]`), preventing hallucinations. |

---

## 🛠️ Tech Stack & Models

<table>
  <tr>
    <td width="50%">
      <b>Frontend</b><br>
      • Gradio 6<br>
      • Custom CSS (Midnight AI Theme)<br>
      • Three.js (WebGL Particle System)<br>
      • Vanilla JS (State Observers)
    </td>
    <td width="50%">
      <b>Backend</b><br>
      • Python 3.10<br>
      • ChromaDB (Ephemeral in-memory)<br>
      • PyMuPDF (PDF Extraction)<br>
      • Sentence Transformers
    </td>
  </tr>
</table>

### 🤖 Model Stack
*   **Language Model:** `openai/gpt-oss-120b` (via Groq) — *Blazing fast generation.*
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` — *Runs locally on CPU.*
*   **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — *Runs locally on CPU.*

---

## 🚀 How to Use

1. **📂 Upload:** Drag and drop your PDF into the **Document Center**.
2. **⚙️ Configure:** Use the **Index Configuration** tabs to adjust chunk size, overlap, retrieval count, and LLM creativity.
3. **⚡ Build:** Click **Build Search Index**. Watch the 3D sphere energize as the document is parsed, chunked, and embedded.
4. **💬 Ask:** Type your question in the **AI Conversation** panel and click **Generate Grounded Answer**.
5. **🔎 Verify:** Open the **Retrieval Intelligence** accordion to inspect exactly which passages the AI used.

---

## ⚠️ Deployment Notes

> **🔑 Environment Secret Required:**
> To run this Space, you must provide a `GROQ_API_KEY` in the Space settings under *Variables and secrets*.

*   **First Startup:** Downloading the Sentence Transformers and Cross-Encoder models to the Space server can take several minutes on the first boot.
*   **In-Memory Storage:** ChromaDB is initialized as an `EphemeralClient`. Document indexes are tied to your browser session and will be cleared if the Space restarts or goes to sleep.
*   **Demonstration Deployment:** This is a showcase of UI/UX and RAG architecture, not a hardened multi-tenant enterprise service.

<div align="center">
  <i>Built with Gradio, Groq, ChromaDB, and a lot of custom WebGL.</i><br><br>
  <a href="https://yogananth-pdf-rag-chatbot.hf.space"><b>Launch the App →</b></a>
</div>
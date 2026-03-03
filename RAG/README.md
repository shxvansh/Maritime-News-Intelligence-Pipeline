# `RAG/` - Graph-Augmented Retrieval & Security Engine

This directory implements the interactive "Chat with your Data" phase of the pipeline. Standard RAG (Retrieval-Augmented Generation) often fails in domain-specific tasks because it relies solely on semantic similarity, missing exact keyword matches (like specific vessel names) and losing entity relationships.

This module solves that by fusing **Knowledge Graphs, Hybrid Search (Dense + Sparse), and in-Depth Guardrails**.

---

## File Roles & Responsibilities

### `qdrant_manager.py`
**Role:** Vector Database Abstraction
**Tasks:** Connects strictly to **Qdrant Cloud**. Handles the mathematical implementation of **Reciprocal Rank Fusion (RRF)**, which beautifully merges standard semantic search with exact keyword search limits.

### `ingester.py`
**Role:** Graph-Augmented Chunking
**Tasks:**
- Pulls completed intelligence articles from PostgreSQL.
- Splits content into sliding windows (3 sentences per chunk).
- **Core Innovation:** It pre-pends explicit Knowledge Graph entity tags obtained from the `MaritimeEvents` SQL table directly into the text chunk before embedding (e.g., `[Incident: Vessel Sanctions] [Vessel: MV BERTHA]`). This structurally tells the model what the paragraph represents *before* semantic embedding occurs.
- Computes both 1024-D Dense Vectors (via `BAAI/bge-large-en-v1.5`) and deterministically hashed Sparse Vectors (BM25 token distributions).

### `rag_chatbot.py`
**Role:** The Intelligence Synthesizer 

**Tasks:** Converts a user's natural language question into parallel dense/sparse search vectors. Filters the Qdrant database, retrieves the highly relevant enriched chunks, formats them into a strictly bounded prompt, and calls LLaMA-4 (Groq) to synthetically generate an answer grounded exclusively in the retrieved maritime events.

### `security.py`
**Role:**  Guardrails

**Tasks:**
- **Pre-Flight Filter:** Scans incoming user queries against 9 known prompt-injection and data exfiltration patterns.
- **Privacy Core (Clean-Room):** Evaluates retrieved context using **Microsoft Presidio** to detect and `<ANONYMIZED>` mask out Personally Identifiable Information (Emails, Phones, SSNs) before it is transmitted cross-network to the Groq LLM.
- **Output Guard (Hallucination Control):** A post-generation overlap analysis that calculates shared entity ratios. If the LLM's final answer contains proper nouns (entities) that did not exist in the retrieved context, the response is blocked as a hallucination.

### `evaluator.py`
**Role:** Live MLOps Scoring
**Tasks:** Acts as an impartial "LLM-as-a-Judge". Implements the industry-standard RAG Triad:
1. **Context Relevance:** Did the Qdrant hybrid search pull useful facts?
2. **Faithfulness:** Did the LLM fabricate anything in its answer?
3. **Answer Relevance:** Did the generated answer actually respond to the analyst's specific prompt?
It calculates these metrics on-the-fly to populate the reliability metrics card on the frontend Dashboard.

# Maritime News Intelligence Pipeline



An automated, intelligence pipeline that performs controlled web scraping of maritime news, NLP-driven entity extraction, LLM-assisted semantic enrichment, and Graph-Augmented Hybrid RAG. 

**Target Source:** MarineTraffic News API  
**Output:** Structured JSON intelligence, Knowledge Graphs, Theme Analytics, and an Analyst Dashboard

---

##  1. Architecture Overview

This project implements a five-layer synchronous batch processing architecture. It utilizes a combination of classical NLP (spaCy + GLiNER) and high-speed LLM inference (meta-llama/llama-4-scout-17b-16e-instruct via Groq LPUs) to transform unstructured narrative reporting into structured, queryable data.

> **Full Design Justifications:** See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a comprehensive breakdown of our design choices (Scraping Approach, NER Selection, Classification Tradeoffs, Storage Schema, and Graph-Augmented RAG strategy).

> **Performance & Metrics:** See [`docs/EVALUATION.md`](docs/EVALUATION.md) for pipeline latency, cost estimates, precision/recall analysis, and manual annotation samples. 

### Core Pipeline Flow (`pipeline/main.py`)
1. **Scraping Layer:** Natively paginates MarineTraffic's GraphQL API (bypassing HTML fragility and respecting anti-bot constraints).
2. **Preprocessing Layer:** Markdown sanitization, stopword removal, and deduplication hashing.
3. **NLP Layer:** Zero-shot incident classification (`facebook/bart-large-mnli`) and domain-specific Named Entity Recognition (`GLiNER`).
4. **LLM Extraction Layer:** Instructor-patched Groq client enforcing strict Pydantic JSON schemas to extract events, resolve coreferences, and generate semantic enrichments (Risk Levels, Geopolitical Flags).
5. **Storage Layer:** Writes structured `Article` and `MaritimeEvent` records to a Dockerized PostgreSQL instance.

---

##  2. System Capabilities & Features

### A. Graph-Augmented Hybrid RAG (`RAG/rag_chatbot.py`)
Standard RAG loses entity relationships. Our ingestion engine (`RAG/ingester.py`) queries PostgreSQL, extracts the complex `MaritimeEvents` related to the text, and dynamically injects Knowledge Graph tags (e.g., `[Incident: Vessel Sanctions] [Vessel: MV BERTHA]`) directly into the embedding chunk. 
We then utilize **Qdrant Cloud** to perform **Reciprocal Rank Fusion (RRF)** on:
- **Dense Vectors:** 1024-D semantic embeddings via `BAAI/bge-large-en-v1.5`
- **Sparse Vectors:** Deterministic MD5-hashed BM25 token frequencies for exact keyword matches.

### B. Security Layer (`RAG/security.py`)
1. **Pre-Flight Guardrail:** Checks for 9 variants of prompt injection and secret leaking attempts.
2. **Presidio PII Clean-Room:** Scrubs retrieved context of identifiable data (Emails, Phone numbers) before transmitting to the external LLM.
3. **Grounding Guard (Hallucination Control):** An overlap analysis engine that rejects the final output if the LLM hallucinates entities not found in the original context.

### C. Advanced Analytics & Generation Evaluation (`pipeline/analytics_engine.py` & `RAG/evaluator.py`)
- **Knowledge Graph:** Maps Vessel → Incident → Port relationships using `NetworkX` and exports them to interactive `PyVis` HTML components.
- **Topical Clustering:** Uses `BERTopic` (UMAP + HDBSCAN) to discover emerging macro-themes dynamically across the corpus.
- **RAG Evaluation (LLM-as-a-Judge):** Employs an automated evaluation triad scoring the pipeline on Context Relevance (100%), Faithfulness (95%), and Answer Relevance (95%). See [`RAG/evaluation_report.md`](RAG/evaluation_report.md) for full benchmark details.

---

##  3. Setup Instructions & Execution

### Prerequisites
- **Docker & Docker Compose** (Required for isolated PostgreSQL)
- **Python 3.11+**
- **Groq API Key** (Required for LLM Extraction)
- **Qdrant Cloud URL & Key** (Required for Hybrid Search)

### Quick Start Installation

1. **Clone the repository:**
```bash
git clone https://github.com/shxvansh/maritime-news-intelligence-pipeline.git
cd maritime-news-intelligence-pipeline
```

2. **Set up your environment variables:**
Rename the provided example file:
```bash
cp .env.example .env
```
*Edit `.env` and insert your `GROQ_API_KEY` and `QDRANT` credentials.*

3. **Run the Entire Stack with Docker:**
*Our `docker-compose.yml` orchestrates the PostgreSQL database, the core data pipeline, the FastAPI backend, and the Next.js frontend all at once. It mounts a persistent volume for the database and caches HuggingFace models to avoid re-downloads.*
```bash
docker compose up --build
```

Wait until you see the `maritime_api` service successfully preload models and the `maritime_frontend` service successfully compile.

### Accessing the System

- **Analyst Dashboard (Next.js):**
  Navigate to `http://localhost:3000` to access the main intelligence feed and analytics.

- **Backend API (FastAPI):**
  The API runs at `http://localhost:8000`. You can view the automated Swagger/OpenAPI docs at `http://localhost:8000/docs`.

From the Dashboard, you can:
- Trigger new Scraping/Extraction runs
- Ingest data into the Qdrant Cloud Vector DB
- View the enriched Intelligence Feed (with Executive Summaries & Risk Flags)
- Interrogate the RAG Chatbot
- Generate the Knowledge Graph & Topic Modeling Dashboards

---

##  4. Project Structure & Deliverables

```text
├── api/
│   └── main.py                 # FastAPI backend (Lifespan model loading)
├── RAG/
│   ├── ingester.py             # Prepares Graph-Augmented chunks for Qdrant
│   ├── qdrant_manager.py       # Hybrid Search interface
│   ├── rag_chatbot.py          # Dual search (Dense + Sparse) + LLM generation
│   ├── security.py             # Presidio limits + Hallucination checks
│   ├── evaluator.py            # Live LLM-as-a-judge (Context, Faithfulness, AR)
├── pipeline/
│   ├── main.py                 # The Synchronous Batch Pipeline Runner
│   ├── preprocessor.py         # Markdown cleaner, hashing, sentence splitting
│   ├── nlp_engine.py           # GLiNER NER & BART-large-mnli Zero-Shot
│   ├── llm_extractor.py        # Groq Client + Prompt Strategy + Coreference
│   ├── models.py               # Strict Pydantic JSON schemas
│   ├── database.py             # SQLAlchemy ORM (Articles & MaritimeEvents)
│   ├── graph_builder.py        # NetworkX entity relationship extraction
│   └── analytics_engine.py     # BERTopic analysis
├── docs/
│   ├── ARCHITECTURE.md         # Detailed design choices & prompt strategy
│   ├── EVALUATION.md           # Metrics, latency, cost & manual annotations
│   └── sample_outputs.json     # 10 Extracted Sample Events
├── frontend/                   # Next.js Analyst Dashboard Source
├── docker-compose.yml          # Persistent database environment
├── Dockerfile                  # Container instructions
└── requirements.txt
```

---

##  Model Selection Reasoning & Prompt Strategy

*An excerpt. See `docs/ARCHITECTURE.md` for full documentation.*

**Model Selection:**
*   **NER:** `GLiNER (Medium-v2.1)`. Chosen over fine-tuned spaCy models because zero-shot capability allows us to define arbitrary, highly specific maritime taxonomy classes (e.g., `"Vessel Name"`, `"Port"`, `"Incident Type"`, `"Cargo"`) at inference time without requiring any manually annotated training datasets.
*   **Classification:** `BART-large-mnli`. Chosen to operate locally as a zero-shot prior. It handles the speed and heavy lifting, providing a strong baseline classification hint (`"Collision"`, `"Sanctions"`) to the LLM.
*   **LLM Extraction:** `meta-llama/llama-4-scout-17b-instruct` via Groq. Chosen for its extreme inference speeds (800+ tokens/sec) and near-zero cost, making it feasible to run intensive structuring prompts across hundreds of articles.

**Prompt Strategy & Boundary Detection:**
The prompt enforces **three core heuristics**:
1. It injects a known list of vessel assets (from the MarineTraffic API metadata) into the context to enforce exact, normalized vessel naming.
2. It explicitly commands the resolution of coreferences/anaphoric references ("the ship", "the chemical carrier") back to their proper named entities, preventing duplicate ghost entities.
3. Event boundary detection is native: the prompt enforces multi-event JSON arrays via Pydantic mapping (`instructor`), allowing the LLM's vast context window to naturally separate different incidents mentioned within a single news report.

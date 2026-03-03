# `api/` - FastAPI Backend Service

This module serves as the production-ready REST API that powers the Next.js Analyst Dashboard. It is built using **FastAPI** to handle concurrent requests and efficiently manage heavy ML model lifetimes.

## Architecture

The API uses a decoupled, event-driven design to trigger processes in the `pipeline/`, `RAG/`, and `Scraper/` modules. It exposes local HTTP endpoints so the frontend dashboard remains completely stateless and lightweight.

---

## File Roles & Responsibilities

### `main.py`
**Role:** The centralized entry point, router, and lifespan controller.
**Responsibilities:**

1. **Lifespan Context Manager (Zero-Latency Inference):**
   - Pre-loads massive ML models (GLiNER, BART-MNLI, BGE-large sentence transformers) and external clients directly into server RAM upon startup.
   - *Why this matters:* Without preloading, every API request that needs to embed text would take 15-30 seconds to download and load a 1.3GB model into memory. By preloading via the `@asynccontextmanager` lifespan event, dashboard requests inference in milliseconds.

2. **Endpoints & Routing:**
   - **Data Acquisition:**
     - `POST /pipeline/scrape`: Triggers the MarineTraffic GraphQL web scraper.
     - `POST /pipeline/run`: Triggers the synchronous, batch NLP/LLM extraction pipeline across all freshly scraped articles.
   - **Database Access:**
     - `GET /db/articles`: Fetches the enriched, summarized `Article` rows from PostgreSQL.
     - `GET /db/events`: Fetches the detailed, normalized `MaritimeEvent` JSON rows from PostgreSQL.
   - **Advanced Analytics:**
     - `POST /analytics/topic-model`: Executes the BERTopic analysis script and updates the frontend HTML visualizations.
     - `POST /analytics/knowledge-graph`: Executes the NetworkX algorithms to rebuild the Entity Relationship graph.
   - **RAG Interrogation:**
     - `POST /rag/chat`: Receives natural language questions, invokes the secure Hybrid Search engine, and returns LLM-synthesized answers.
     - `POST /rag/evaluate`: Triggers the live LLM-as-a-judge (RAG Triad) evaluation to score the chatbot's generation in real-time.

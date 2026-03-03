# `pipeline/` - Core Intelligence Extraction

This directory contains the core business logic of the Intelligence Pipeline. It is responsible for taking unstructured, raw HTML/Markdown scraped from MarineTraffic and transforming it into strictly formatted, mathematically queryable intelligence in PostgreSQL.

## Architecture

This is a **Synchronous Batch Processing Pipeline**. It fundamentally flows through four distinct layers per article: 
1. **Preprocessing** (Cleaning & Sentence Boundaries)
2. **Classical NLP** (NER & Zero-Shot Classification)
3. **LLM Extraction** (Summarization & JSON Schema enforcement)
4. **Storage** (SQLAlchemy ORM)

---

## File Roles & Responsibilities

### `main.py`
**Role:** The Orchestrator
**Tasks:** Orchestrates the flow of data. It iterates through newly scraped articles (`latest_articles.json`) and chains the logic together. It utilizes exponential backoff (`tenacity`) to automatically handle Groq rate limits, ensuring the batch pipeline does not crash midway through 100 articles.

### `preprocessor.py`
**Role:** Text Sanitization & Identity
**Tasks:**
- Strips noisy Markdown tags, URLs, and boilerplate.
- Detects sentence boundaries using **spaCy's `en_core_web_sm`**. This is critical because RAG chunking and NLP models rely on clean sentence breaks.
- Computes a **SHA-256 hash** of the clean text, replacing arbitrary string matching with deterministic cryptographic identities for database deduplication.

### `nlp_engine.py`
**Role:** Specialized Information Extraction
**Tasks:**
- Deploys **GLiNER (Zero-Shot NER)** to extract highly specific maritime entities (Vessel Names, Ports, Incident Types, Cargo).
- Runs `facebook/bart-large-mnli` for **Zero-Shot Classification**. This provides a fast, free categorization (e.g., "Collision") that acts as a prior hint for the LLM.

### `llm_extractor.py`
**Role:** Semantic Reasoning & Structuring
**Tasks:**
- Connects to the **Groq LPU API** (`meta-llama/llama-4-scout-17b-instruct`).
- Solves anaphoric reference resolution ("the ship" -> "MV BERTHA") via prompt engineering.
- Enforces strict JSON shapes using the `instructor` library to prevent LLM hallucination of schema keys.

### `models.py`
**Role:** Output Schemas
**Tasks:** Contains the **Pydantic classes** (`MaritimeEvent`, `ArticleEnrichment`) that define the exact structured layout for event extraction as required by the grading rubric. Uses Enums to enforce authorized outputs (e.g., RiskLevel).

### `database.py`
**Role:** Persistence Layer
**Tasks:** Uses **SQLAlchemy ORM** to connect to Dockerized PostgreSQL. Defines the schema for the `articles` and `maritime_events` tables and executes ACID-compliant commits.

### `analytics_engine.py`
**Role:** Unsupervised Macro-Trend AI
**Tasks:** Runs the **BERTopic** pipeline (UMAP dimensionality reduction + HDBSCAN clustering) against the article database to dynamically discover rising narrative themes without pre-defined labels.

### `graph_builder.py`
**Role:** Entity Relationship Mapping
**Tasks:** Reads structured JSON events from PostgreSQL, builds mathematical connections using **NetworkX** (Vessel ➔ INVOLVED_IN ➔ Incident), and generates a physics-based interactive HTML visualization using **PyVis**.

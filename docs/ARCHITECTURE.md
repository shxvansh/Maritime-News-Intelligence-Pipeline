# System Architecture & Design Decisions

This document covers all design choices, tradeoffs, and engineering decisions made during the development of the Maritime News Intelligence Pipeline, as required by the technical assessment rubric.

---

## 1. Architecture Overview

The system follows a **modular layered architecture** with strict separation of concerns across five primary layers:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER                                 │
│                  Next.js Analyst Dashboard (frontend/)                   │
│         Ingestion Control │ Intel Feed │ Analytics │ RAG Chat            │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ HTTP (REST)
┌──────────────────────────────▼───────────────────────────────────────────┐
│                        API LAYER (FastAPI)                               │
│  Lifespan preloads: NLPEngine, LLMExtractor, SentenceTransformer,       │
│  SecurityManager into RAM for 0-latency cold starts                     │
│  Endpoints: /pipeline/*, /db/*, /analytics/*, /rag/*                    │
└─────┬──────────────┬──────────────┬──────────────┬───────────────────────┘
      │              │              │              │
┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────────────────┐
│  Scraper   │ │ NLP/LLM    │ │ Analytics  │ │ RAG Engine                 │
│  Layer     │ │ Pipeline   │ │ Layer      │ │ (Hybrid Search + Security) │
│            │ │            │ │            │ │                            │
│ GraphQL    │ │ spaCy      │ │ BERTopic   │ │ BGE-large (Dense 1024-D)   │
│ Client     │ │ GLiNER     │ │ UMAP+      │ │ + BM25 (Sparse)            │
│            │ │ BART-MNLI  │ │ HDBSCAN    │ │ + Qdrant RRF Fusion        │
│            │ │ Groq LLaMA │ │            │ │ + Presidio PII Masking     │
│            │ │ + Pydantic │ │            │ │ + Grounding Guard          │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └────────┬───────────────────┘
      │              │              │                  │
┌─────▼──────────────▼──────────────▼──────────────────▼───────────────────┐
│                         STORAGE LAYER                                    │
│  PostgreSQL (Docker)  │  Qdrant Cloud  │  JSON Files  │  HTML Outputs    │
│  Articles + Events    │  Dense+Sparse  │  Raw Scrape  │  KG + Topics     │
│  (SQLAlchemy ORM)     │  Vectors       │  Data        │  Dashboards      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Execution Model: Synchronous Batch

The core intelligence pipeline (`pipeline/main.py`) executes as a **synchronous, batch-oriented** process:

1. **Load** all raw articles from `latest_articles.json`
2. **Iterate** through each article sequentially: Preprocess → NER → Classification → LLM Extraction → DB Write
3. **Commit** results to PostgreSQL per-article

**Why Batch over Streaming?**
- The data source (MarineTraffic News) publishes articles on a daily/hourly cadence, not in real-time. A batch pipeline running on a cron schedule (e.g., every 6 hours) is the natural fit.
- LLM API calls to Groq have rate limits (30 requests/minute on free tier). A synchronous loop with exponential backoff (`tenacity`) handles this gracefully.
- For a production streaming upgrade, we would replace the loop with a **Celery task queue** backed by Redis, where each article is an independent task. The FastAPI endpoint `/pipeline/run` already serves as the trigger interface for this.




## 2. Data Acquisition Design (Scraper Layer)

### Approach: GraphQL API Reverse-Engineering vs. HTML Scraping

We chose to **reverse-engineer MarineTraffic's internal GraphQL API** rather than scraping their HTML pages. This was a deliberate design choice:

| Factor | HTML Scraping | GraphQL API (Our Approach) |
|---|---|---|
| **Reliability** | Breaks when CSS/HTML changes | Stable structured JSON responses |
| **Speed** | Slow (download full page + assets) | Fast (lightweight JSON payloads) |
| **Data Quality** | Requires complex parsing | Clean, typed fields from source |
| **Anti-Bot Risk** | High (Cloudflare, captchas) | Low (API endpoint is less monitored) |
| **Pagination** | Fragile (URL param guessing) | Native (`page`, `pageSize` variables) |

### robots.txt Compliance

Our approach aligns with `robots.txt` principles:
- We target only the publicly accessible news API endpoint used by their own frontend
- We implement **rate limiting** via pagination delays (one page request at a time, not parallel)
- We set a reasonable `User-Agent` header (not spoofing a browser identity for deceptive purposes)
- We fetch only publicly available content (no authentication bypass, no login scraping)
- The GraphQL endpoint is the same one their public website calls — we are not accessing private or restricted APIs

### Data Extraction

For each article, we extract:
| Field | Source |
|---|---|
| Title | `article.title` |
| Publication Date | `article.publishedAt` |
| Author | `article.author.data.attributes.name` |
| Tags/Category | `article.category.data.attributes.name` |
| Full Article Text | `article.content` (Markdown-formatted) |
| Source URL | Dynamically constructed from `category_id`, `slug`, and `year` |
| Associated Vessels | `article.assets.data` (vessel metadata from MarineTraffic AIS) |

---

## 3. Data Cleaning & Preprocessing Methodology

**File:** `pipeline/preprocessor.py`

The preprocessing pipeline applies four sequential transformations:

### Step 1: Markdown Sanitization (`clean_markdown()`)
MarineTraffic returns article content in Markdown format (not raw HTML). Our cleaner:
- Removes Markdown image tags `![alt](url)` (images are noise for NLP)
- Converts Markdown links `[text](url)` to plain text (preserves semantics, removes URL noise)
- Collapses excessive whitespace and newlines

### Step 2: Content Hashing (`compute_hash()`)
- Generates a **SHA-256 hash** of the cleaned text
- This hash serves as the **primary key** in PostgreSQL and the **deduplication identifier**
- Before processing any article, we query `SELECT * FROM articles WHERE id = <hash>` — if found, we skip
- This ensures re-running the pipeline on the same data is a no-op (idempotent processing)

### Step 3: Sentence Segmentation (`segment_sentences()`)
- Uses **spaCy's `en_core_web_sm`** sentence boundary detection
- Critical for: (a) RAG chunking (3-sentence chunks), (b) feeding cleaner inputs to NER models
- spaCy outperforms regex-based splitting on maritime text where abbreviations like "Ltd.", "M.V.", and "Capt." are frequent

### Step 4: Analytics Normalization (`normalize_for_analytics()`)
- Produces a **lemmatized, lowercase, stopword-removed** version of the text
- Used exclusively by the BERTopic clustering engine
- Lemmatization ensures "detained" and "detention" cluster together
- Stopword removal prevents common words from dominating topic distributions

---

## 4. NER Design Choices

**File:** `pipeline/nlp_engine.py`  
**Model:** `GLiNER (urchade/gliner_medium-v2.1)`

### Why GLiNER over spaCy NER or Fine-Tuned Models?

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **spaCy `en_core_web_sm` NER** | Fast, lightweight | Only recognizes generic entity types (PERSON, ORG, GPE). Cannot extract maritime-specific entities like "Vessel Name" or "Incident Type" without fine-tuning |  Rejected |
| **Fine-Tuned spaCy/BERT NER** | High precision for known entity types | Requires annotated maritime NER training data (expensive, time-consuming). Rigid — cannot adapt to new entity types without retraining |  Rejected |
| **GLiNER (Zero-Shot NER)** | Accepts arbitrary entity labels at inference time. No training data needed. Can extract "Vessel Name", "Cargo", "Incident Type" out of the box | Slightly slower than spaCy. Less precise than a fine-tuned model on edge cases |  Selected |

**Key Insight:** GLiNER allows us to define domain-specific entity labels (`["Vessel Name", "Port", "Organization", "Country", "Person", "Incident Type", "Date", "Cargo"]`) without any annotation or training. This makes it ideal for a rapid pipeline build targeting a specialized domain.

### Entity Labels Configuration
```python
self.ner_labels = [
    "Vessel Name",    # Ever Given, MSC FORTUNA
    "Port",           # Mumbai Port, Ningbo-Zhoushan  
    "Organization",   # Shipping Corp, IMO
    "Country",        # India, China
    "Person",         # Captain names
    "Incident Type",  # Collision, Fire, Sanctions
    "Date",           # 4 March 2025
    "Cargo"           # Crude Oil, LNG
]
```

---

## 5. Maritime Incident Classification

**Model:** `facebook/bart-large-mnli` (Zero-Shot) + `LLaMA-4` (Verification)

The classification of maritime incidents into structured categories is a core requirement of this intelligence pipeline. We evaluated four distinct architectural approaches before selecting a **Hybrid Ensemble** model.

### Trade-off Analysis

| Approach | Latency | Accuracy | Training Data | Rationale & Decisions |
| :--- | :--- | :--- | :--- | :--- |
| **Fine-tuned Classifier (BERT/RoBERTa)** | Very Low | High (on-domain) | High (~500+ labels) | **Rejected:** While extremely fast, it requires substantial manually annotated maritime training data which was not available for this task. It is also "rigid"—adding a new category like "Dark Fleet Activity" would require a full retraining cycle. |
| **Pure Zero-Shot (BART-MNLI)** | Medium | Medium-High | None | **Evaluated:** Excellent for cold-starts and handles diverse labels well. However, it can struggle with nuance where two categories overlap (e.g., a "Collision" that is also a "Military Activity"). |
| **Pure Prompt-based LLM** | High | Very High | None | **Rejected as Primary:** While contextually very smart, calling an LLM for *every* classification task is expensive and generates high latency. It also limits throughput due to API rate limits (Groq/OpenAI). |
| **Hybrid Ensemble (Our Choice)** | **Optimized** | **Highest** | **None** | **Selected:** We use the local BART-MNLI model to generate a "Probabilistic Prior" (a preliminary label). This label is then injected into the LLM's context. The LLM's task is shifted from *discovery* to *verification/correction*, which is a more robust reasoning pattern. |

### Implementation: The Hybrid Ensemble Strategy

Our system implements the **Hybrid Ensemble** approach as follows:

1.  **Stage 1: Local Inference (BART-MNLI):** Each article is processed by a local `facebook/bart-large-mnli` model. This model ranks the 9 required categories (`Collision`, `Piracy`, `Sanctions`, etc.) based on entailment scores. 
2.  **Stage 2: Prior Injection:** The top-scoring category is recorded as `nlp_classification` in the database but also passed forward as metadata.
3.  **Stage 3: LLM Verification:** During the event extraction phase, the LLM prompt is injected with this prior:  
    `"An initial automated pass suggested this incident is: '{category}'. Validate or correct this based on the full technical context."`
4.  **Final Resolution:** The LLM provides the final authoritative `incident_type` in the structured JSON output. If the LLM disagrees with the local model (e.g., identifying a subtle sanction evasion that looked like a standard port disruption), the LLM's higher-order reasoning prevails.

This design achieves **Production-Grade reliability** by combining the speed of local "Classical" NLP with the deep reasoning of Generative AI.

---

## 6. Event Extraction & Prompt Engineering Strategy

**File:** `pipeline/llm_extractor.py`  
**Model:** `meta-llama/llama-4-scout-17b-16e-instruct`

Our extraction layer is designed as a structured reasoning engine that transforms narrative text into strict, machine-usable intelligence. We address the three core challenges of maritime extraction—**Event Boundaries**, **Coreferences**, and **Normalization**—through a layered prompt engineering strategy.

### 1. How we Detect Event Boundaries
In a single news report, multiple distinct incidents (e.g., an initial collision followed by a separate rescue effort or an arrest) are often conflated. We solve this by:
- **Array-Centric Modelling:** Instead of asking for "the" event, we define a Pydantic `EventList` model representing a `List[MaritimeEvent]`.
- **Temporal/Geospatial Heuristics:** The prompt explicitly instructs the LLM to segment events based on the **"Who, Where, and When"**. A shift in date, location, or the primary vessels involved triggers the model to instantiate a new object in the event array.
- **Narrative Segmentation:** The LLM's vast 128k context window is leveraged to identify "transition sentences" in the article (e.g., *"In a separate incident..."*), which the model uses as natural boundary markers to stop one event object and start the next.

### 2. How we Resolve Coreferences
Maritime news heavily relies on anaphora (e.g., *"the Liberian carrier"*, *"the distressed tanker"*, *"it"*). To prevent "ghost entities" where "the ship" is extracted as a separate entity from "MV EVER GIVEN", we implement:
- **Pre-extraction Resolve Command:** The system prompt includes a high-priority instruction: *"Before populating the JSON fields, resolve all pronouns and generic references back to their specific named entities."*
- **Contextual Anchoring:** We utilize the **GLiNER NER** results from the previous pipeline stage as "Anchors". By presenting the LLM with the already-extracted entities as a reference list, it maps generic descriptions (like "the captain") to specific names found earlier.

### 3. How we Normalize Vessel Names
Vessel names are the primary keys of maritime intelligence, yet they are often abbreviated or misspelled in news reports. We achieve normalization using **External Asset Grounding**:
- **AIS Metadata Injection:** Our GraphQL scraper retrieves `assets` metadata (AIS-verified IMO names and callsigns) from MarineTraffic's backend.
- **Dynamic Canonical List:** This list is injected into every LLM prompt as a **"Canonical Vessel Context"**. 
- **The "Grounding Instruction":** The prompt enforces a strict rule: *"If the text refers to a vessel, you MUST map it to the most likely candidate in the Canonical Vessel list. Do not create new vessel variations if a canonical match is available."* This ensures that "The Bertha" is normalized to "MV BERTHA" based on real-world asset data, not LLM imagination.

### 4. Structured Output Enforcement (Instructor)
To guarantee zero-hallucination of the JSON schema itself, we use the `instructor` library to patch the Groq client. This enforces a **Pydantic-first extraction** loop:
1. The LLM attempts to populate the Pydantic `MaritimeEvent` model.
2. If the LLM returns an invalid date or an unknown Enum value (e.g., Risk Level: "Extreme" instead of "Critical"), the system automatically re-prompts the model with the validation error.
3. This guarantees that the final output is 100% compliant with the required storage schema.

---

## 7. Semantic Enrichment Prompt Strategy

**Requirement:** For each article, generate an Executive Summary, Risk Level, Impact Scope, Strategic Tags, and identify Geopolitical, Defense, and Sanction relevance.

We utilize a **multi-objective single-prompt strategy** to perform high-level intelligence synthesis. The prompt engineering is grounded in the following design principles:

### 1. Zero-Shot Analytical Priming
The model is primed with a "Expert Intelligence Analyst" persona. Instead of generic summarization, the prompt demands an **"Intelligence Brief"** format. 
- **Instruction:** *"Write a highly professional, 150-word intelligence brief. Focus on the 'So-What' (impact on global shipping) rather than just repeating the text."*

### 2. Standardized Scale Definition (Enum-Based)
To prevent the LLM from using subjective terms, we define strict categorical scales within the prompt:
- **Risk Level Scalar:** We provide clear definitions for each level (CRITICAL, HIGH, MEDIUM, LOW) based on real-world impact (e.g., "Critical = Loss of life or war zone").
- **Impact Scope Scalar:** Defined by the geographical footprint (Local, Regional, Global).
- **Reasoning:** By providing these definitions in the prompt, we normalize the "thinking" of the LLM across different news sources.

### 3. Indicator-Based Boolean Flags
For identifying **Geopolitical**, **Defense**, and **Sanctions** relevance, we use a "Definition + Example" pattern:
- **Geopolitical:** *"Does this involve state actors, territorial disputes, or government-level maritime policy?"*
- **Defense:** *"Are navies, coast guards, or military-contracted vessels explicitly mentioned?"*
- **Sanctions:** *"Does this involve OFAC, EU, UN sanctions, 'Dark Fleet' operations, or suspicious ship-to-ship transfers?"*

### 4. Strategic Tagging (Taxonomy Guidance)
To ensure tags are useful for enterprise search, we provide a **seed taxonomy** in the prompt (e.g., 'Supply Chain Shock', 'Piracy Trend') and instruct the model to generate 2-4 tags that characterize the *strategic nature* of the disruption, not just the entities involved.

---

## 8. Knowledge Graph Construction

**File:** `pipeline/graph_builder.py`  
**Library:** NetworkX + PyVis

### Graph Schema

| Node Type | Color | Examples |
|---|---|---|
| Vessel | Light Blue | MV BERTHA, MSC FORTUNA |
| Incident | Salmon/Red | Collision, Piracy, Sanctions |
| Organization | Light Green | IMO, Indian Navy, Maersk |
| Port | Orange | Ningbo-Zhoushan, Fujairah |
| Country | Orange | China, Malaysia, India |

### Edge (Relationship) Types

| Relationship | Example |
|---|---|
| `INVOLVED_IN` | Vessel → Incident |
| `LOCATED_AT` | Vessel → Port |
| `LOCATED_IN` | Port → Country |
| `ASSOCIATED_WITH` | Organization → Vessel |

### Implementation Details
- Nodes are **deduplicated** — if "Maersk" appears in 5 different events, it appears as a single node with 5 edges
- The graph is exports as an **interactive HTML file** using PyVis, with dark-theme styling consistent with the Next.js dashboard
- Physics-based layout with `repulsion()` tuning prevents node overlap while maintaining readable edge labels

---

## 9. Topic Modeling & Trend Analysis

**File:** `pipeline/analytics_engine.py`  
**Model:** BERTopic with fine-tuned sub-components

### Configuration for Small Datasets (20-100 articles)

Standard BERTopic parameters assume thousands of documents. We tuned the sub-models for our dataset size:

```python
umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=3, metric='euclidean', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
```

- `n_neighbors=5` allows UMAP to find structure in small datasets
- `min_cluster_size=3` permits small but meaningful clusters
- `ngram_range=(1,2)` captures bigrams like "oil spill" and "port disruption"

### Outputs Produced
1. **Top 5 Macro Themes** — Printed to terminal with keyword descriptors
2. **Frequency Visualization** — Interactive bar chart (Plotly) showing article counts per theme
3. **Temporal Distribution Chart** — Topics plotted over time to identify emerging trends

---

## 10. Storage Layer Design

### PostgreSQL Schema

Two normalized tables connected by a foreign key:

**`articles` Table (Primary Intelligence Records)**
| Column | Type | Purpose |
|---|---|---|
| `id` | VARCHAR (PK) | SHA-256 hash of content (dedup key) |
| `title` | VARCHAR | Article headline |
| `url` | VARCHAR | Source URL |
| `original_id` | VARCHAR (UNIQUE) | MarineTraffic article ID |
| `content` | TEXT | Full raw article text for RAG chunking |
| `nlp_classification` | VARCHAR | Zero-shot category label |
| `nlp_confidence` | FLOAT | Classifier confidence score |
| `processing_time` | FLOAT | Seconds taken to process this article |
| `named_entities` | JSONB | GLiNER NER results (flexible schema) |
| `executive_summary` | TEXT | LLM-generated 150-word summary |
| `risk_level` | VARCHAR | Low / Medium / High / Critical |
| `impact_scope` | VARCHAR | Local / Regional / Global |
| `strategic_relevance_tags` | JSONB | Array of strategic tags |
| `is_geopolitical` | BOOLEAN | Geopolitical relevance flag |
| `has_defense_implications` | BOOLEAN | Defense implications flag |
| `is_sanction_sensitive` | BOOLEAN | Sanctions sensitivity flag |
| `is_embedded` | BOOLEAN | RAG ingestion tracking flag |
| `created_at` | TIMESTAMP | Record creation time |

**`maritime_events` Table (Structured Event Records)**
| Column | Type | Purpose |
|---|---|---|
| `event_id` | UUID (PK) | System-generated unique ID |
| `article_hash` | VARCHAR (FK → articles.id) | Links event to parent article |
| `event_date` | VARCHAR | Flexible date format |
| `port` | VARCHAR | Event location port |
| `country` | VARCHAR | Event location country |
| `latitude` / `longitude` | FLOAT | Geocoordinates (nullable) |
| `vessels_involved` | JSONB | Array of vessel names |
| `organizations_involved` | JSONB | Array of organization names |
| `incident_type` | VARCHAR | Classification label |
| `casualties` | VARCHAR | Free-text casualties field |
| `cargo_type` | VARCHAR | Cargo involved |
| `summary` | TEXT | 1-2 sentence event summary |
| `confidence_score` | FLOAT | LLM extraction confidence |

**Design Decision: JSONB for arrays** — Rather than creating separate junction tables for vessel-event and org-event many-to-many relationships (which would be normalized but complex), we store arrays as JSONB. This simplifies queries for a pipeline that primarily writes and reads whole records, not individual relationships. PostgreSQL's JSONB supports indexing and querying inside arrays for future advanced use.

### Vector Database (Qdrant Cloud)

A single hybrid collection `maritime_news_v2` with:
- **Dense Vector:** 1024-dimensional BAAI/bge-large-en-v1.5 embeddings (Cosine distance)
- **Sparse Vector:** BM25-style keyword vectors using MD5 token hashing

Each vector point payload contains: `article_hash`, `title`, `text`, `graph_context`, `risk_level`, `classification`

---

## 11. RAG Architecture: Graph-Augmented Hybrid Search

### The Problem with Standard RAG
Standard RAG embeds text chunks and retrieves by semantic similarity. This loses:
1. **Entity relationships** — "Which vessels were involved in sanctions?" requires entity context, not just semantic similarity
2. **Keyword precision** — Dense embeddings alone may miss exact vessel names or port names that appear verbatim in the query

### Our Solution: Three Augmentations

#### Augmentation 1: Knowledge Graph Context Injection
Before embedding, we prepend structured entity metadata from the PostgreSQL `maritime_events` table to each text chunk:

```
Context: [Incident: Vessel Collision] [Vessel: MV EVER GIVEN] [Location: Suez Canal, Egypt]. 
The container ship ran aground on Tuesday after suffering a steering malfunction...
```

This means the embedding captures both the **narrative text** and the **structured entity relationships**, enabling queries like "What happened to MV EVER GIVEN?" to match on both the vessel name entity tag and the narrative description.

#### Augmentation 2: Hybrid Dense + Sparse Search
We perform two parallel searches in Qdrant:
- **Dense search** (BGE-large embeddings) — captures semantic meaning ("ships that were stopped" ≈ "vessel detention")
- **Sparse search** (BM25 keyword vectors) — captures exact term matches ("MV BERTHA" matches only chunks containing "MV BERTHA")

The results are merged using **Reciprocal Rank Fusion (RRF)**, which re-ranks results by combining their positions in both result lists.

#### Augmentation 3: Deterministic Sparse Hashing
We use `hashlib.md5` instead of Python's built-in `hash()` for sparse vector token indexing. Python's `hash()` is session-randomized (PYTHONHASHSEED), meaning sparse vectors produced during ingestion would be incomparable with query vectors produced in a different process. MD5 guarantees deterministic, cross-session consistency.

---

## 12. Security Architecture (Defense-Grade Guardrails)

**File:** `RAG/security.py`

### Layer 1: Prompt Injection Detection
Pre-flight regex scanning against 9 known injection patterns:
- "ignore all previous instructions"
- "system prompt", "developer mode", "jailbreak"
- "bypass security/restrictions/guardrails"
- "reveal the hidden prompt"

Also detects leaked secret patterns (AWS keys, Bearer tokens, API keys) to prevent data exfiltration via prompt.

### Layer 2: PII Sanitization (Microsoft Presidio)
Before sending retrieved context to the LLM, all text passes through Presidio:
- **Detected entities:** EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, CREDIT_CARD, SSN, IBAN, PERSON
- **Action:** Entities are replaced with `<ANONYMIZED>` placeholders
- **Defense-in-depth:** An additional regex pass catches API key patterns that Presidio might miss

### Layer 3: Output Grounding Check (Hallucination Guard)
After the LLM generates an answer, we perform a lightweight overlap analysis:
1. **Word overlap:** Extracts all 4+ character words from the answer and context. If overlap ratio < 20%, the answer is rejected
2. **Entity consistency:** Extracts capitalized proper nouns from both. If the answer introduces entities that don't exist in the context, it's flagged as hallucinated
3. **Rejection phrase pass-through:** Answers that correctly state "cannot be determined" are always allowed

### Rate Limiting
The `/rag/chat` endpoint is protected by `slowapi` at **10 requests per minute per IP**, preventing abuse of the Groq API quota.

---

## 13. API Design: FastAPI Lifespan Architecture

**File:** `api/main.py`

All ML models are loaded once during server startup using FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["nlp_engine"] = NLPEngine()              # GLiNER + BART-MNLI
    ml_models["llm_extractor"] = LLMExtractor()        # Groq client
    ml_models["sentence_transformer"] = SentenceTransformer("BAAI/bge-large-en-v1.5")  # 1.3GB
    ml_models["security_manager"] = SecurityManager()   # Presidio engines
    yield
    ml_models.clear()
```

**Why this matters:** Without lifespan preloading, every API request would need to load a 1.3GB embedding model, taking 15-30 seconds. With preloading, the models live in RAM and inference takes milliseconds.

---

## 14. Dockerization Strategy

### `Dockerfile` — Pipeline Image
- Base: `python:3.11-slim` (minimal attack surface)
- Pre-downloads all ML model weights (BART-MNLI, GLiNER) during image build
- This means container startup is **fast** — no 2GB download on first run
- Installs `libpq-dev` for PostgreSQL C-bindings

### `docker-compose.yml` — Full Stack
- **PostgreSQL service** with persistent volume (`postgres_data`) and health checks
- **Pipeline service** depends on `db` with `condition: service_healthy`, ensuring the database is ready before the pipeline starts
- Environment variables are injected from `.env` with sensible defaults
- JSON data files are mounted as volumes (not copied), so the evaluator can inspect outputs on their host machine

"""
Maritime News Intelligence Pipeline - FastAPI Backend
Wraps the entire pipeline (scraper, NLP, LLM, RAG, analytics) into REST endpoints.
"""

import os
import sys
import json
import hashlib
import uuid
import time
import matplotlib
matplotlib.use('Agg')  # Force thread-safe backend before any routes or modeling loads

# Ensure root project path is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from contextlib import asynccontextmanager

limiter = Limiter(key_func=get_remote_address)

# Global dictionary to hold our preloaded models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models during startup
    print("⏳ Preloading Machine Learning Models... (This ensures zero latency later)")
    
    from pipeline.nlp_engine import NLPEngine
    from pipeline.llm_extractor import LLMExtractor
    from sentence_transformers import SentenceTransformer
    
    ml_models["nlp_engine"] = NLPEngine()
    ml_models["llm_extractor"] = LLMExtractor()
    
    print("⏳ Loading RAG Vectors (BAAI/bge-large-en-v1.5)...")
    ml_models["sentence_transformer"] = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    print("⏳ Loading Security Manager (Presidio PII + Guardrails)...")
    from RAG.security import SecurityManager
    ml_models["security_manager"] = SecurityManager()
    
    print("✅ All models successfully preloaded into memory!")
    yield
    # Clean up the models on shutdown
    ml_models.clear()

app = FastAPI(
    title="Maritime Intelligence API",
    description="Backend service for the Maritime News Intelligence Pipeline.",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


class EvalRequest(BaseModel):
    question: str
    context: str
    answer: str


class PipelineRunRequest(BaseModel):
    batch_size: int = 20


# ---------------------------------------------------------------------------
# 1. SCRAPER ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/pipeline/scrape")
def run_scraper():
    """Triggers the GraphQL scraper and returns the count of fetched articles."""
    try:
        from Scraper.marinetraffic_scraper import MarineTrafficClient
        client = MarineTrafficClient()
        target_count = 100
        articles_per_page = 20
        total_articles = []

        def build_source_url(article) -> str:
            try:
                cat_data = article.get('category', {}).get('data', {})
                c_id = cat_data.get('id', 'unknown')
                c_name = cat_data.get('attributes', {}).get('name', 'general').lower().replace(' ', '-')
                pub = article.get('publishedAt', '')
                yr = pub[:4] if pub else '2026'
                return f"https://www.marinetraffic.com/en/maritime-news/{c_id}/{c_name}/{yr}/{article.get('id', '')}/{article.get('slug', '')}"
            except Exception:
                return f"https://www.marinetraffic.com/en/maritime-news/article/{article.get('slug', '')}"

        for page_num in range(1, (target_count // articles_per_page) + 1):
            result = client.get_latest_articles(
                category_id=None,
                page=page_num,
                page_size=articles_per_page,
                skip=0
            )
            page_articles = result["data"]["latestArticles"]
            if not page_articles:
                break
            for art in page_articles:
                art['source_url'] = build_source_url(art)
            total_articles.extend(page_articles)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "latest_articles.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(total_articles, f, indent=4, ensure_ascii=False)

        return {"status": "success", "articles_fetched": len(total_articles)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. FULL PIPELINE EXTRACTION ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/pipeline/run")
def run_extraction_pipeline(req: PipelineRunRequest):
    """Runs the full NLP + LLM extraction pipeline on the scraped articles."""
    from pipeline.preprocessor import Preprocessor
    from pipeline.database import SessionLocal, init_db, Article, MaritimeEvent

    init_db()
    preprocessor = Preprocessor()
    nlp_engine = ml_models["nlp_engine"]
    llm_extractor = ml_models["llm_extractor"]
    db = SessionLocal()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "latest_articles.json")

    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="latest_articles.json not found. Run scraper first.")

    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    batch = articles[:req.batch_size]
    results = []
    processed_count = 0

    for idx, article in enumerate(batch):
        raw_text = article.get("content", "")
        article_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        existing = db.query(Article).filter(Article.id == article_hash).first()
        if existing:
            continue

        processed_data = preprocessor.process_article(raw_text)
        clean_text = processed_data["llm_ready_text"]

        entities = nlp_engine.extract_entities(clean_text)
        classification = nlp_engine.classify_incident(clean_text[:1500])

        low_priority = ["Financial or Shipping Markets", "Regulatory Development"]
        incident_entities = [e for e in entities if e["label"] in ["Incident Type", "Cargo"]]

        if classification["label"] in low_priority and len(incident_entities) == 0:
            try:
                new_article = Article(
                    id=article_hash,
                    original_id=article.get("id"),
                    title=article.get("title", "Unknown"),
                    url=article.get("url", ""),
                    content=raw_text,
                    nlp_classification=classification["label"],
                    nlp_confidence=classification["score"],
                    processing_time=0,
                    named_entities={"entities": entities}
                )
                db.add(new_article)
                db.commit()
            except Exception:
                db.rollback()
            continue

        known_assets = article.get("assets", {}).get("data", [])
        start = time.time()
        extracted = llm_extractor.extract_events(
            text=clean_text[:1500],
            known_assets=known_assets,
            zero_shot_category=classification["label"]
        )
        elapsed = time.time() - start

        if extracted:
            enrichment = extracted.get("enrichment", {})
            try:
                new_article = Article(
                    id=article_hash,
                    original_id=article.get("id"),
                    title=article.get("title", "Unknown"),
                    url=article.get("source_url", ""),
                    content=raw_text,
                    nlp_classification=classification["label"],
                    nlp_confidence=classification["score"],
                    processing_time=elapsed,
                    named_entities={"entities": entities},
                    executive_summary=enrichment.get("executive_summary"),
                    risk_level=enrichment.get("risk_level"),
                    impact_scope=enrichment.get("impact_scope"),
                    strategic_relevance_tags=enrichment.get("strategic_relevance_tags", []),
                    is_geopolitical=enrichment.get("is_geopolitical", False),
                    has_defense_implications=enrichment.get("has_defense_implications", False),
                    is_sanction_sensitive=enrichment.get("is_sanction_sensitive", False)
                )
                db.add(new_article)

                for ev in extracted.get("events", []):
                    new_event = MaritimeEvent(
                        event_id=str(uuid.uuid4()),
                        article_hash=article_hash,
                        event_date=ev.get("event_date"),
                        port=ev.get("location", {}).get("port"),
                        country=ev.get("location", {}).get("country"),
                        latitude=ev.get("location", {}).get("lat"),
                        longitude=ev.get("location", {}).get("lon"),
                        vessels_involved=ev.get("vessels_involved", []),
                        organizations_involved=ev.get("organizations_involved", []),
                        incident_type=ev.get("incident_type"),
                        casualties=ev.get("casualties"),
                        cargo_type=ev.get("cargo_type"),
                        summary=ev.get("summary"),
                        confidence_score=ev.get("confidence_score")
                    )
                    db.add(new_event)

                db.commit()
                processed_count += 1
                results.append({
                    "title": article.get("title"),
                    "classification": classification["label"],
                    "risk_level": enrichment.get("risk_level"),
                    "events_extracted": len(extracted.get("events", []))
                })
            except Exception:
                db.rollback()

    db.close()
    return {"status": "success", "articles_processed": processed_count, "details": results}


# ---------------------------------------------------------------------------
# 3. DATABASE QUERY ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/db/articles")
def get_articles():
    """Returns all processed articles from the database."""
    from pipeline.database import SessionLocal, Article
    db = SessionLocal()
    articles = db.query(Article).order_by(Article.created_at.desc()).all()
    result = []
    for a in articles:
        result.append({
            "id": a.id,
            "title": a.title,
            "classification": a.nlp_classification,
            "confidence": a.nlp_confidence,
            "risk_level": a.risk_level,
            "impact_scope": a.impact_scope,
            "executive_summary": a.executive_summary,
            "strategic_tags": a.strategic_relevance_tags,
            "is_geopolitical": a.is_geopolitical,
            "has_defense_implications": a.has_defense_implications,
            "is_sanction_sensitive": a.is_sanction_sensitive,
            "named_entities": a.named_entities,
            "processing_time": a.processing_time,
            "url": a.url
        })
    db.close()
    return result


@app.get("/db/events")
def get_events():
    """Returns all extracted maritime events from the database."""
    from pipeline.database import SessionLocal, MaritimeEvent
    db = SessionLocal()
    events = db.query(MaritimeEvent).all()
    result = []
    for ev in events:
        result.append({
            "event_id": ev.event_id,
            "article_hash": ev.article_hash,
            "event_date": ev.event_date,
            "port": ev.port,
            "country": ev.country,
            "latitude": ev.latitude,
            "longitude": ev.longitude,
            "vessels_involved": ev.vessels_involved,
            "organizations_involved": ev.organizations_involved,
            "incident_type": ev.incident_type,
            "casualties": ev.casualties,
            "cargo_type": ev.cargo_type,
            "summary": ev.summary,
            "confidence_score": ev.confidence_score,
        })
    db.close()
    return result


# ---------------------------------------------------------------------------
# 4. ANALYTICS ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/analytics/knowledge-graph")
def build_knowledge_graph():
    """Triggers knowledge graph generation and returns the file path."""
    try:
        from pipeline.graph_builder import build_knowledge_graph
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "processed_test_results.json")
        build_knowledge_graph(json_path)
        return {"status": "success", "output": "knowledge_graph.html"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/topic-model")
def run_topic_model():
    """Triggers the BERTopic trend analysis and returns the top themes."""
    try:
        from pipeline.analytics_engine import run_topic_modeling
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "latest_articles.json")
        run_topic_modeling(json_path)
        return {"status": "success", "output": "theme_dashboard.html"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. SECURE RAG CHATBOT ENDPOINT (Graph-Augmented Hybrid Search + Guardrails)
# ---------------------------------------------------------------------------
@app.post("/rag/chat")
@limiter.limit("10/minute")
def rag_chat(req: ChatRequest, request: Request):
    """Answers a user question using Graph-Augmented Hybrid RAG with Defense-Grade Security."""
    import re
    from collections import Counter
    from RAG.qdrant_manager import QdrantManager
    from qdrant_client.http.models import SparseVector
    from groq import Groq

    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    model = ml_models["sentence_transformer"]
    security = ml_models["security_manager"]

    # --- SECURITY LAYER 1: Prompt Injection Guardrail ---
    if security.check_prompt_injection(req.question):
        print("🚨 Prompt Injection Detected! Rejecting query.")
        raise HTTPException(status_code=403, detail="Security violation: Malicious prompt pattern detected.")

    # Dense embedding with BGE instruct prefix
    instructed_query = BGE_QUERY_INSTRUCTION + req.question
    dense_vector = model.encode(instructed_query).tolist()

    # Sparse vector (BM25-like) for keyword matching
    tokens = re.findall(r'\b[a-z0-9]+\b', req.question.lower())
    token_counts = Counter(tokens)
    sparse_indices = [int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % 2_000_000 for t in token_counts]
    sparse_values = [float(c / len(tokens)) for c in token_counts.values()]
    sparse_vector = SparseVector(indices=sparse_indices, values=sparse_values)

    # Hybrid Search via Qdrant Cloud
    qdrant = QdrantManager()
    search_results = qdrant.hybrid_search(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=5
    )

    if not search_results:
        return {"answer": "No relevant information was found in the database.", "sources": []}

    context_blocks = []
    sources = []
    for idx, hit in enumerate(search_results):
        payload = hit.payload
        title = payload.get("title", "Unknown")
        text = payload.get("text", "")
        graph_context = payload.get("graph_context", "")
        risk_level = payload.get("risk_level", "Unknown")

        block = f"[Source #{idx+1}: {title} | Risk: {risk_level}]"
        if graph_context:
            block += f"\n{graph_context}"
        block += f"\n{text}"
        context_blocks.append(block)
        sources.append({
            "title": title,
            "score": round(hit.score, 3) if hit.score else 0,
            "risk_level": risk_level,
            "graph_context": graph_context
        })
    full_context = "\n\n---\n\n".join(context_blocks)

    # --- SECURITY LAYER 2: PII Data Sanitization ---
    full_context = security.sanitize_pii(full_context)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    prompt = f"""You are an expert Maritime Intelligence Analyst working for a defense organization.
Use ONLY the following extracted intelligence reports to answer the analyst's question.
Each source includes structured entity context (Vessels, Incidents, Locations) extracted
from a Knowledge Graph, followed by the relevant text passage.

If the answer cannot be determined from the provided context, state that clearly.
Do not speculate or introduce external information. Be precise and cite the source numbers.

INTELLIGENCE CONTEXT:
{full_context}

ANALYST QUESTION: {req.question}"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    # --- SECURITY LAYER 3: Output Grounding Check ---
    if not security.check_grounding(answer, full_context):
        print("🚨 Hallucination Detected! Rejecting response.")
        return {
            "answer": "⚠️ SECURITY ALERT: The generated intelligence report failed the confidence/grounding check. Returning no response to prevent hallucination.",
            "sources": sources,
            "context": full_context
        }

    return {"answer": answer, "sources": sources, "context": full_context}


# ---------------------------------------------------------------------------
# 5b. RAG EVALUATION ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/rag/evaluate")
def rag_evaluate(req: EvalRequest):
    """Evaluates a RAG response for context relevance, faithfulness, and answer relevance."""
    try:
        from RAG.evaluator import RAGEvaluator
        
        # Load the evaluator WITHOUT initializing duplicate instances of SentenceTransformer or Qdrant
        evaluator = RAGEvaluator(load_models=False)
        
        scores = evaluator.evaluate_triad(
            question=req.question,
            ground_truth="", # No ground truth during live inference
            context=req.context,
            answer=req.answer
        )
        return {"status": "success", "metrics": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5c. RAG INGESTION ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/rag/ingest")
def rag_ingest():
    """Triggers the Graph-Augmented Hybrid RAG ingestion from PostgreSQL to Qdrant Cloud."""
    try:
        from RAG.ingester import ingest_from_database
        ingest_from_database()
        return {"status": "success", "message": "RAG ingestion completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 6. PROCESSED DATA ENDPOINT (for dashboard pre-loaded view)
# ---------------------------------------------------------------------------
@app.get("/data/processed")
def get_processed_data():
    """Returns the processed_test_results.json for UI rendering when DB is unavailable."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "processed_test_results.json")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Processed data file not found.")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from contextlib import asynccontextmanager

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
    
    print("⏳ Loading RAG Vectors (all-MiniLM-L6-v2)...")
    ml_models["sentence_transformer"] = SentenceTransformer("all-MiniLM-L6-v2")
    
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
# 5. RAG CHATBOT ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/rag/chat")
def rag_chat(req: ChatRequest):
    """Answers a user question using the RAG pipeline (Qdrant + Groq)."""
    from RAG.qdrant_manager import QdrantManager
    from groq import Groq

    model = ml_models["sentence_transformer"]
    query_embedding = model.encode(req.question).tolist()

    qdrant = QdrantManager()
    search_results = qdrant.search(query_vector=query_embedding, limit=4)

    if not search_results:
        return {"answer": "No relevant information was found in the database.", "sources": []}

    context_blocks = []
    sources = []
    for hit in search_results:
        payload = hit.payload
        title = payload.get("title", "Unknown")
        text = payload.get("text", "")
        context_blocks.append(f"[Source: {title}]\n{text}")
        sources.append({"title": title, "score": round(hit.score, 3)})

    full_context = "\n\n---\n\n".join(context_blocks)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    prompt = f"""You are an expert Maritime Intelligence Analyst working for a defense organization.
Use ONLY the following extracted intelligence reports to answer the analyst's question.
If the answer cannot be determined from the provided context, state that clearly.
Do not speculate or introduce external information.

CONTEXT:
{full_context}

ANALYST QUESTION: {req.question}"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}


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

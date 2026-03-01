"""
RAG Ingester - Graph-Augmented Hybrid Ingestion Pipeline
Reads from PostgreSQL (not JSON files), augments chunks with Knowledge Graph
entities, embeds with BGE-large, and stores Dense + Sparse vectors in Qdrant Cloud.
"""

import os
import sys
import uuid
import re
from collections import Counter

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import Preprocessor
from pipeline.database import SessionLocal, Article, MaritimeEvent
from RAG.qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct, SparseVector


# ---------------------------------------------------------------------------
# Sparse Vector (BM25-like) Generator
# ---------------------------------------------------------------------------
def compute_sparse_vector(text: str) -> SparseVector:
    """
    Computes a simple BM25-inspired sparse vector from the text.
    Maps each unique token to an index (hash) and assigns its TF as the value.
    This allows Qdrant to perform exact keyword matching alongside dense search.
    """
    # Tokenize: lowercase, alphanumeric only
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    if not tokens:
        return SparseVector(indices=[], values=[])
    
    token_counts = Counter(tokens)
    
    # Use hash of each token as the sparse index (Qdrant expects integer indices)
    indices = []
    values = []
    for token, count in token_counts.items():
        # Map token to a stable integer index via hash (mod large prime for compactness)
        token_index = abs(hash(token)) % 2_000_000
        tf = count / len(tokens)  # Term Frequency normalization
        indices.append(token_index)
        values.append(float(tf))
    
    return SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------------
# Knowledge Graph Augmentation
# ---------------------------------------------------------------------------
def build_graph_context(events: list) -> str:
    """
    Takes the structured MaritimeEvent records linked to an article and
    constructs a Knowledge Graph context string to prepend to text chunks.
    
    Example output:
    "Context: [Incident: Vessel Collision] [Vessel: MV EVER GIVEN] [Location: Suez Canal, Egypt]"
    """
    if not events:
        return ""
    
    tags = []
    seen = set()  # Prevent duplicate tags across multiple events
    
    for event in events:
        if event.incident_type and event.incident_type not in seen:
            tags.append(f"[Incident: {event.incident_type}]")
            seen.add(event.incident_type)
        
        # Vessels
        vessels = event.vessels_involved or []
        for v in vessels:
            if v and v not in seen:
                tags.append(f"[Vessel: {v}]")
                seen.add(v)
        
        # Organizations
        orgs = event.organizations_involved or []
        for o in orgs:
            if o and o not in seen:
                tags.append(f"[Org: {o}]")
                seen.add(o)
        
        # Location
        location_parts = []
        if event.port:
            location_parts.append(event.port)
        if event.country:
            location_parts.append(event.country)
        loc_str = ", ".join(location_parts)
        if loc_str and loc_str not in seen:
            tags.append(f"[Location: {loc_str}]")
            seen.add(loc_str)
    
    if not tags:
        return ""
    
    return "Context: " + " ".join(tags) + ". "


# ---------------------------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------------------------
def ingest_from_database():
    """
    Production ingestion pipeline:
    1. Reads un-embedded articles from PostgreSQL
    2. Joins with MaritimeEvents for KG augmentation
    3. Chunks the text with graph context prepended
    4. Embeds with BGE-large (Dense 1024-D) + BM25 (Sparse)
    5. Upserts into Qdrant Cloud
    6. Marks articles as embedded in PostgreSQL
    """
    print("🚀 Initializing Graph-Augmented Hybrid RAG Ingestion...")
    print("=" * 60)
    
    # --- 1. Load the Embedding Model ---
    print("⏳ Loading embedding model (BAAI/bge-large-en-v1.5)...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print(f"   ✅ Model loaded. Vector dimension: {model.get_sentence_embedding_dimension()}")
    
    # --- 2. Connect to Qdrant Cloud ---
    qdrant = QdrantManager()
    
    # --- 3. Query PostgreSQL for un-embedded articles ---
    db = SessionLocal()
    
    try:
        articles = db.query(Article).filter(Article.is_embedded == False).all()
        
        if not articles:
            print("✅ All articles are already embedded. Nothing to ingest.")
            return
        
        print(f"📂 Found {len(articles)} un-embedded articles in PostgreSQL.")
        
        points_to_insert = []
        total_chunks = 0
        articles_processed = 0
        
        for article in articles:
            raw_text = article.content or ""
            title = article.title or "Unknown Title"
            article_hash = article.id
            
            if not raw_text or len(raw_text.strip()) < 50:
                # Mark as embedded even if empty (to avoid retrying forever)
                article.is_embedded = True
                continue
            
            # --- 4. Fetch linked MaritimeEvents for KG Augmentation ---
            events = db.query(MaritimeEvent).filter(
                MaritimeEvent.article_hash == article_hash
            ).all()
            
            graph_context = build_graph_context(events)
            
            # --- 5. Chunk the article text ---
            cleaned_data = Preprocessor.process_article(raw_text)
            sentences = cleaned_data.get("sentences", [])
            
            chunk_size = 3
            for i in range(0, len(sentences), chunk_size):
                chunk_text = " ".join(sentences[i:i + chunk_size])
                
                # Skip very short chunks
                if len(chunk_text.split()) < 5:
                    continue
                
                # --- 6. Prepend Graph Context to the chunk ---
                augmented_chunk = graph_context + chunk_text
                
                # --- 7. Generate Dense Embedding (BGE-large) ---
                # BGE models perform best with a task-specific prefix during encoding
                dense_embedding = model.encode(augmented_chunk).tolist()
                
                # --- 8. Generate Sparse Vector (BM25) ---
                sparse_vector = compute_sparse_vector(augmented_chunk)
                
                total_chunks += 1
                
                # --- 9. Build the Qdrant Point (Named Vectors) ---
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        QdrantManager.DENSE_VECTOR_NAME: dense_embedding,
                        QdrantManager.SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload={
                        "article_hash": article_hash,
                        "title": title,
                        "text": chunk_text,              # Store the raw text (without graph prefix)
                        "graph_context": graph_context,   # Store graph context separately for transparency
                        "risk_level": article.risk_level,
                        "classification": article.nlp_classification,
                    }
                )
                points_to_insert.append(point)
                
                # Batch insert every 200 points to prevent RAM spikes
                if len(points_to_insert) >= 200:
                    qdrant.insert_chunks(points_to_insert)
                    points_to_insert = []
            
            # --- 10. Mark article as embedded in PostgreSQL ---
            article.is_embedded = True
            articles_processed += 1
            
            if articles_processed % 10 == 0:
                print(f"   📊 Progress: {articles_processed}/{len(articles)} articles processed...")
        
        # Insert remaining chunks
        if points_to_insert:
            qdrant.insert_chunks(points_to_insert)
        
        # Commit the is_embedded flags to PostgreSQL
        db.commit()
        
        print("\n" + "=" * 60)
        print(f"✅ Ingestion Complete!")
        print(f"   Articles processed: {articles_processed}")
        print(f"   Total chunks embedded: {total_chunks}")
        print(f"   Embedding model: BAAI/bge-large-en-v1.5 (1024-D)")
        print(f"   Sparse index: BM25 keyword vectors")
        print(f"   Storage: Qdrant Cloud (Hybrid Collection)")
        print("=" * 60)
    
    except Exception as e:
        db.rollback()
        print(f"❌ Ingestion failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest_from_database()

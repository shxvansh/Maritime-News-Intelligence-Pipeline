import json
import uuid
import os
import sys

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import Preprocessor
from RAG.qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct

def ingest_articles(json_filepath):
    print("🚀 Initializing Vector Ingestion...")

    # Load up the local embedding model
    print("⏳ Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize the Qdrant connection
    qdrant = QdrantManager()

    if not os.path.exists(json_filepath):
        print(f"❌ Could not find {json_filepath}.")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"📂 Found {len(articles)} articles. Processing chunks...")
    
    points_to_insert = []
    total_chunks = 0
    
    for article in articles:
        raw_text = article.get("content", "")
        title = article.get("title", "Unknown Title")
        article_hash = article.get("hash", str(uuid.uuid4()))
        
        if not raw_text:
            continue
            
        # We reuse the Preprocessor's smart sentence segmenter
        # It cleans markdown and slices the article into precise sentences
        cleaned_data = Preprocessor.process_article(raw_text)
        sentences = cleaned_data.get("sentences", [])
        
        # We chunk them. For RAG, chunks of ~3-4 sentences provide good context
        chunk_size = 3
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i:i + chunk_size])
            
            # Skip overwhelmingly short bits
            if len(chunk_text.split()) < 5: 
                continue
                
            # Embed the specific chunk
            embedding = model.encode(chunk_text).tolist()
            total_chunks += 1
            
            # Construct a Qdrant 'Point' (Vector + Metadata)
            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant requires unique IDs for every chunk
                vector=embedding,
                payload={
                    "article_hash": article_hash,
                    "title": title,
                    "text": chunk_text
                }
            )
            points_to_insert.append(point)
            
            # Batch insert to prevent RAM explosion if list gets massive
            if len(points_to_insert) >= 500:
                qdrant.insert_chunks(points_to_insert)
                points_to_insert = []
                
    # Insert any remaining chunks
    if points_to_insert:
        qdrant.insert_chunks(points_to_insert)
        
    print(f"\n✅ Data ingested successfully! Embedded and stored {total_chunks} text chunks into Qdrant.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_json = os.path.join(base_dir, "latest_articles.json")
    
    ingest_articles(target_json)

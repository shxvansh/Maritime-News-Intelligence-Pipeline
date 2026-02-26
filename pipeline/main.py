import json
import os
import sys
import uuid
import time
import hashlib
from dotenv import load_dotenv

# Ensure the root directory is on the path so we can import 'pipeline'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import Preprocessor
from pipeline.nlp_engine import NLPEngine
from pipeline.llm_extractor import LLMExtractor
from pipeline.database import SessionLocal, init_db, Article, MaritimeEvent

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=dotenv_path)

def main():
    print("🚀 Initializing components...")
    init_db()  # Ensure tables exist
    preprocessor = Preprocessor()
    nlp_engine = NLPEngine()
    llm_extractor = LLMExtractor()
    db = SessionLocal()
    
    # Load raw data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latest_articles.json")
    
    if not os.path.exists(data_path):
        print(f"❌ Could not find {data_path}. Please ensure the scraper has generated this file.")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    # We will test on 20 articles to verify the pipeline logic and filters
    test_batch = articles[:20]
    print(f"\n📂 Found {len(articles)} articles. Processing a test batch of {len(test_batch)}...")
    
    total_processing_time = 0
    total_confidence_score = 0
    processed_count = 0
    
    for idx, article in enumerate(test_batch):
        print(f"\n--- Processing Article {idx+1}/{len(test_batch)}: {article.get('title')} ---")
        start_time = time.time()
        
        # 0. Deduplication Check via Database
        raw_text = article.get("content", "") # Changed from "text" to "content" to match original structure
        article_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        
        existing_article = db.query(Article).filter(Article.id == article_hash).first()
        if existing_article:
            print("⏭️ Skipping: This article has already been processed and is in the database.")
            continue
            
        # 1. Preprocessing (Cleaning & Normalizing)
        processed_data = preprocessor.process_article(raw_text) # Reverted to original Preprocessor usage
        clean_text = processed_data["llm_ready_text"] # Use the llm_ready_text from preprocessor
        print("✅ Preprocessing complete.")
        
        # 2. Local NLP Processing
        # NER
        entities = nlp_engine.extract_entities(clean_text)
        print(f"✅ NER complete: Found {len(entities)} entities.")
        
        # Classification (Zero-Shot)
        # LLM needs shorter text to avoid token limits, so we pass the first 1500 chars 
        llm_ready_text = clean_text[:1500] # Use the llm_ready_text from preprocessor, truncated
        classification = nlp_engine.classify_incident(llm_ready_text)
        print(f"✅ Classification complete: {classification['label']} (Confidence: {classification['score']:.2f})")
        
        # --- EARLY EXIT FILTER ---
        # Don't waste Groq TPM limits on purely financial/regulatory announcements if NO incidents/cargo were flagged
        low_priority_labels = ["Financial or Shipping Markets", "Regulatory Development"]
        incident_entities = [e for e in entities if e['label'] in ["Incident Type", "Cargo"]]
        
        if classification['label'] in low_priority_labels and len(incident_entities) == 0:
            print("⏭️ Filtering Out: Article marked as low priority with no physical incident entities.")
            
            # Save the early-exit to the DB so we don't re-process it tomorrow
            try:
                new_article = Article(
                    id=article_hash,
                    original_id=article.get("id"),
                    title=article.get("title", "Unknown Title"),
                    nlp_classification=classification['label'],
                    nlp_confidence=classification['score'],
                    processing_time=time.time() - start_time
                )
                db.add(new_article)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"❌ Database Insertion Failed for early exit: {e}")
            continue
            
        # 3. LLM Event Extraction
        known_assets = article.get("assets", {}).get("data", [])
        print("⏳ Passing to Groq LPU for structured extraction...")
        
        extracted_events = llm_extractor.extract_events(
            text=llm_ready_text, 
            known_assets=known_assets,
            zero_shot_category=classification['label']
        )
        
        elapsed = time.time() - start_time
        
        if extracted_events:
            print("✅ LLM Extraction successful!")
            try:
                # Extract the semantic enrichment from the Pydantic dictionary
                enrichment = extracted_events.get("enrichment", {})
                
                # Create the core Article record enriched with LLM intelligence
                new_article = Article(
                    id=article_hash,
                    original_id=article.get("id"),
                    title=article.get("title", "Unknown Title"),
                    nlp_classification=classification['label'],
                    nlp_confidence=classification['score'],
                    processing_time=elapsed,
                    executive_summary=enrichment.get("executive_summary"),
                    risk_level=enrichment.get("risk_level"),
                    impact_scope=enrichment.get("impact_scope"),
                    strategic_relevance_tags=enrichment.get("strategic_relevance_tags", []),
                    is_geopolitical=enrichment.get("is_geopolitical", False),
                    has_defense_implications=enrichment.get("has_defense_implications", False),
                    is_sanction_sensitive=enrichment.get("is_sanction_sensitive", False)
                )
                db.add(new_article)
                
                # Create the mapped MaritimeEvent records
                for ev in extracted_events.get("events", []):
                    new_event = MaritimeEvent(
                        event_id=str(uuid.uuid4()), # Generate a true unique ID, ignoring the LLM's mock ID
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
                total_processing_time += elapsed
                total_confidence_score += classification['score']
            except Exception as e:
                db.rollback()
                print(f"❌ Database Insertion Failed: {e}")
        else:
            print("❌ LLM Extraction failed.")

    db.close()

    if processed_count > 0:
        avg_time = total_processing_time / processed_count
        avg_conf = total_confidence_score / processed_count
        
        print("\n📊 --- Metrics for Executed Batch ---")
        print(f"⏱️  Average Processing Time: {avg_time:.2f} seconds per article")
        print(f"📈 Average NLP Confidence: {avg_conf:.2f}")

    print("\n🎉 Finished batch. Data stored securely in local PostgreSQL Database.")

if __name__ == "__main__":
    main()

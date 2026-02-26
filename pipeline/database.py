import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Float, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/maritime_news")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(String, primary_key=True, index=True) # Article Hash 
    title = Column(String, nullable=False)
    original_id = Column(String, unique=True, index=True) 
    nlp_classification = Column(String)
    nlp_confidence = Column(Float)
    processing_time = Column(Float)
    
    # Enrichment fields
    executive_summary = Column(String, nullable=True)
    risk_level = Column(String, nullable=True)
    impact_scope = Column(String, nullable=True)
    strategic_relevance_tags = Column(JSON, default=list) # Store list of tags as JSON
    is_geopolitical = Column(Boolean, default=False)
    has_defense_implications = Column(Boolean, default=False)
    is_sanction_sensitive = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class MaritimeEvent(Base):
    __tablename__ = "maritime_events"
    
    event_id = Column(String, primary_key=True, index=True) # EVT-2024-02-24-001
    article_hash = Column(String, ForeignKey("articles.id"))
    event_date = Column(String) # Stored as string to support YYYY-MM-DD or imprecise dates
    
    # Nested Location Data
    port = Column(String, nullable=True)
    country = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Arrays/Lists are stored as JSON for flexibility in Postgres
    vessels_involved = Column(JSON, default=list)
    organizations_involved = Column(JSON, default=list)
    
    incident_type = Column(String)
    casualties = Column(String)
    cargo_type = Column(String, nullable=True)
    summary = Column(String)
    confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Creates the tables if they do not exist."""
    print("Initializing Database Schemas...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database Scaffolding Complete!")

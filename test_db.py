from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipeline.database import Article
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/maritime_news")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()
articles = db.query(Article).all()
print(f"Total articles in DB: {len(articles)}")
for a in articles[:5]:
    print(f"ID: {a.id}, Orig_ID: {a.original_id}, URL: {a.url}")
db.close()

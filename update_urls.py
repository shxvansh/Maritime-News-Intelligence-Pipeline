import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipeline.database import Article

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/maritime_news")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()
with open("latest_articles.json", "r") as f:
    latest = json.load(f)

url_map = {}
for art in latest:
    url_map[str(art.get("id"))] = art.get("source_url", "")

articles = db.query(Article).all()
updated = 0
for a in articles:
    orig_id = str(a.original_id)
    if orig_id in url_map and url_map[orig_id]:
        a.url = url_map[orig_id]
        updated += 1

db.commit()
print(f"Updated {updated} article URLs in DB")
db.close()

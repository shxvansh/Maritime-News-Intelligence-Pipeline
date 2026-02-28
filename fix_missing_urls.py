import json
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipeline.database import Article

load_dotenv()

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

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()
with open("latest_articles.json", "r") as f:
    latest = json.load(f)

url_map = {}
for art in latest:
    url_map[str(art.get("id"))] = build_source_url(art)

articles = db.query(Article).all()
updated = 0
for a in articles:
    orig_id = str(a.original_id)
    if not a.url and orig_id in url_map:
        a.url = url_map[orig_id]
        updated += 1
    elif not a.url and orig_id == 'None':
        # the db might have orig_id saved as string 'None' or actual None
        pass

db.commit()
print(f"Updated {updated} missing article URLs in DB")
for a in articles[:5]:
    print(f"ID: {a.id}, Orig_ID: {a.original_id}, URL: {a.url}")
db.close()

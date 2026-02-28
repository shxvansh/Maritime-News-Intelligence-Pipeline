import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load .env to capture GROQ_API_KEY
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

def ask_question(question: str):
    """Answers a user's question using RAG by fetching from Qdrant and prompting Groq."""
    print(f"\n🤔 Question: {question}")
    print("⏳ Searching database for context...")
    
    # 1. Embed the Question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(question).tolist()
    
    # 2. Retrieve Similar Chunks from Qdrant
    qdrant = QdrantManager()
    search_results = qdrant.search(query_vector=query_embedding, limit=4)
    
    if not search_results:
        print("❌ Could not find any relevant information in the database.")
        return
        
    print(f"✅ Found {len(search_results)} highly relevant excerpts!")
    
    # Compile the fetched context into a single string for the LLM
    context_blocks = []
    for idx, hit in enumerate(search_results):
        score = hit.score
        payload = hit.payload
        article_title = payload.get("title", "Unknown")
        article_text = payload.get("text", "")
        context_blocks.append(f"[Article: {article_title}]\n{article_text}")
        
    full_context = "\n\n---\n\n".join(context_blocks)
    
    # 3. Prompt the LLM using Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable is missing.")
        return
        
    print("⏳ Asking the LLM to analyze and synthesize the answer...")
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert Maritime Intelligence Analyst.
    Use ONLY the following extracted text blocks to answer the user's question. 
    If you cannot find the answer explicitly within the text blocks provided below, 
    say "I cannot answer this based on the provided context." Do not hallucinate external facts.

    CONTEXT BLOCKS:
    {full_context}

    USER QUESTION: {question}
    """
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, # We want highly deterministic factual answers
    )
    
    answer = response.choices[0].message.content
    
    print("\n\n" + "="*50)
    print("🤖 MARITIME AI RESPONSE:")
    print("="*50)
    print(answer)
    print("\nSource Contexts Used:")
    for hit in search_results:
        print(f"  - {hit.payload.get('title')} (Relevance Score: {hit.score:.2f})")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask questions about maritime news using RAG.')
    parser.add_argument('query', type=str, nargs='?', default="Which vessels were recently arrested or detained in China?",
                        help='The question you want to ask the RAG chatbot.')
    
    args = parser.parse_args()
    ask_question(args.query)

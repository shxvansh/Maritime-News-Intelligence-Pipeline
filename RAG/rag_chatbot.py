"""
RAG Chatbot - Graph-Augmented Hybrid Retrieval + LLM Generation
Uses BGE-large with instruct prompting, Hybrid Search (Dense + Sparse),
and Groq LLaMA-4 for precise, grounded maritime intelligence answers.
"""

import os
import sys
import re
import argparse
from collections import Counter
from dotenv import load_dotenv

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import SparseVector
from groq import Groq

# Load .env to capture GROQ_API_KEY
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ---------------------------------------------------------------------------
# BGE Instruct Prefix (Critical for retrieval performance)
# ---------------------------------------------------------------------------
# BGE models are explicitly trained with task-specific prefixes.
# For queries (retrieval), prepend this instruction to shape the vector.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def compute_sparse_vector(text: str) -> SparseVector:
    """
    Computes a BM25-inspired sparse vector from the query text.
    Must use the same logic as ingester.py so the vectors are comparable.
    """
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    if not tokens:
        return SparseVector(indices=[], values=[])
    
    token_counts = Counter(tokens)
    
    indices = []
    values = []
    for token, count in token_counts.items():
        token_index = abs(hash(token)) % 2_000_000
        tf = count / len(tokens)
        indices.append(token_index)
        values.append(float(tf))
    
    return SparseVector(indices=indices, values=values)


def ask_question(question: str, model: SentenceTransformer = None):
    """
    Answers a user's question using Graph-Augmented Hybrid RAG.
    
    Pipeline:
    1. Embed the query with BGE-large (instruct prefix) -> Dense Vector
    2. Tokenize the query -> Sparse Vector (BM25)
    3. Hybrid Search in Qdrant (Dense + Sparse fused via RRF)
    4. Compile retrieved chunks + graph context for the LLM
    5. Generate a grounded answer via Groq (LLaMA-4)
    """
    print(f"\n🤔 Question: {question}")
    print("⏳ Running Hybrid Search (Dense + Sparse)...")
    
    # --- 1. Load or reuse the embedding model ---
    if model is None:
        print("⏳ Loading embedding model (BAAI/bge-large-en-v1.5)...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    # --- 2. Embed the Query (with BGE instruct prefix) ---
    instructed_query = BGE_QUERY_INSTRUCTION + question
    dense_query_vector = model.encode(instructed_query).tolist()
    
    # --- 3. Generate Sparse Vector for the Query ---
    sparse_query_vector = compute_sparse_vector(question)
    
    # --- 4. Hybrid Search in Qdrant Cloud ---
    qdrant = QdrantManager()
    search_results = qdrant.hybrid_search(
        dense_vector=dense_query_vector,
        sparse_vector=sparse_query_vector,
        limit=5
    )
    
    if not search_results:
        print("❌ Could not find any relevant information in the database.")
        return {"answer": "No relevant information was found.", "sources": []}
    
    print(f"✅ Found {len(search_results)} relevant chunks via Hybrid Search!")
    
    # --- 5. Compile context for the LLM ---
    context_blocks = []
    sources = []
    for idx, hit in enumerate(search_results):
        payload = hit.payload
        article_title = payload.get("title", "Unknown")
        article_text = payload.get("text", "")
        graph_context = payload.get("graph_context", "")
        risk_level = payload.get("risk_level", "Unknown")
        
        # Build a rich context block including graph metadata
        block = f"[Source #{idx+1}: {article_title} | Risk: {risk_level}]"
        if graph_context:
            block += f"\n{graph_context}"
        block += f"\n{article_text}"
        
        context_blocks.append(block)
        sources.append({
            "title": article_title,
            "score": round(hit.score, 3) if hit.score else 0,
            "risk_level": risk_level,
            "graph_context": graph_context
        })
    
    full_context = "\n\n---\n\n".join(context_blocks)
    
    # --- 6. Generate the Answer via Groq (LLaMA-4) ---
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable is missing.")
        return {"answer": "API key not configured.", "sources": []}
    
    print("⏳ Asking LLaMA-4 to synthesize the intelligence report...")
    client = Groq(api_key=api_key)
    
    prompt = f"""You are an expert Maritime Intelligence Analyst working for a defense organization.
Use ONLY the following extracted intelligence reports to answer the analyst's question.
Each source includes structured entity context (Vessels, Incidents, Locations) extracted
from a Knowledge Graph, followed by the relevant text passage.

If the answer cannot be determined from the provided context, state that clearly.
Do not speculate or introduce external information. Be precise and cite the source numbers.

INTELLIGENCE CONTEXT:
{full_context}

ANALYST QUESTION: {question}"""
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    answer = response.choices[0].message.content
    
    print("\n\n" + "=" * 60)
    print("🤖 MARITIME AI RESPONSE:")
    print("=" * 60)
    print(answer)
    print("\n📎 Sources Retrieved (Hybrid Search):")
    for src in sources:
        print(f"  - {src['title']} (Score: {src['score']}, Risk: {src['risk_level']})")
        if src['graph_context']:
            print(f"    Graph: {src['graph_context']}")
    print("=" * 60)
    
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Ask questions about maritime news using Graph-Augmented Hybrid RAG.'
    )
    parser.add_argument(
        'query', type=str, nargs='?',
        default="Which vessels were recently arrested or detained in China?",
        help='The question you want to ask the RAG chatbot.'
    )
    
    args = parser.parse_args()
    ask_question(args.query)

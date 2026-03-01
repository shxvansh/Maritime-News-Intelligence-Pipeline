"""
RAG Evaluator - LLM-as-a-Judge Evaluation Framework
Evaluates the RAG pipeline using the "RAG Triad" metrics:
1. Context Relevance
2. Faithfulness
3. Answer Relevance
"""

import os
import json
import re
import sys
from groq import Groq
from dotenv import load_dotenv

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from RAG.qdrant_manager import QdrantManager
from RAG.rag_chatbot import ask_question, compute_sparse_vector, BGE_QUERY_INSTRUCTION

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


class RAGEvaluator:
    def __init__(self, load_models=True):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=self.api_key)
        
        # Use a high-quality model as the impartial judge
        self.judge_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        self.model = None
        self.qdrant = None
        if load_models:
            print("⏳ Loading embedding model for evaluation fetcher...")
            # Since evaluation loads the model, we pass it to query functions directly to avoid reloading
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            self.qdrant = QdrantManager()

    def get_context_and_answer(self, question: str):
        """Fetch the full text context and ask the chatbot to generate an answer."""
        
        # 1. We ask the chatbot natively to get its answer and source metadata
        # Supress stdout completely if you want a cleaner console, but we'll leave it 
        # for a cool visual trace here.
        print("\n--- Sending request to Intelligence System ---")
        result = ask_question(question, self.model)
        answer = result["answer"]
        
        # 2. Extract full context exactly as the chatbot did internally 
        # (Since the chatbot only returns source titles, not the big text blocks)
        instructed_query = BGE_QUERY_INSTRUCTION + question
        dense_query_vector = self.model.encode(instructed_query).tolist()
        sparse_query_vector = compute_sparse_vector(question)
        
        search_results = self.qdrant.hybrid_search(
            dense_vector=dense_query_vector,
            sparse_vector=sparse_query_vector,
            limit=5
        )
        
        context_blocks = []
        for idx, hit in enumerate(search_results):
            payload = hit.payload
            title = payload.get("title", "Unknown")
            text = payload.get("text", "")
            graph_context = payload.get("graph_context", "")
            risk_level = payload.get("risk_level", "Unknown")
            
            block = f"[Source #{idx+1}: {title} | Risk: {risk_level}]\n"
            if graph_context:
                block += f"{graph_context}\n"
            block += f"{text}"
            context_blocks.append(block)
            
        full_context = "\n\n---\n\n".join(context_blocks)
        
        return full_context, answer

    def parse_score(self, llm_response: str) -> float:
        """Parses a 0.0 to 1.0 score from the LLM's raw string response."""
        # Find SCORE: X.X pattern
        match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', llm_response)
        if match:
            return float(match.group(1))
        # Fallback heuristic
        numbers = re.findall(r'0\.[0-9]+|1\.0|0\.0', llm_response)
        if numbers:
            return float(numbers[-1])
        return 0.0

    def evaluate_triad(self, question: str, ground_truth: str, context: str, answer: str):
        """Calculates Context Relevance, Faithfulness, and Answer Relevance."""
        metrics = {}
        
        # 1. Context Relevance
        prompt_cr = f"""You are an objective judge evaluating a RAG (Retrieval-Augmented Generation) system.
TASK: Evaluate 'Context Relevance'. Does the RETRIEVED CONTEXT contain the facts necessary to answer the user's QUESTION?

QUESTION: {question}
RETRIEVED CONTEXT: 
{context}

Provide a brief reasoning, then end with strict formatting "SCORE: X.X" where X.X is a float between 0.0 and 1.0 (e.g. 1.0 for perfect, 0.5 for partial, 0.0 for irrelevant)."""
        
        res_cr = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt_cr}],
            temperature=0.0
        ).choices[0].message.content
        metrics["Context Relevance"] = self.parse_score(res_cr)

        # 2. Faithfulness
        prompt_fa = f"""You are an objective judge evaluating a RAG system.
TASK: Evaluate 'Faithfulness'. Can the GENERATED ANSWER be factually verified STRICTLY using the RETRIEVED CONTEXT? 
Note: If the answer correctly states that it cannot answer due to lack of context, score it 1.0.

RETRIEVED CONTEXT: 
{context}
GENERATED ANSWER: 
{answer}

Provide a brief reasoning, then end with strict formatting "SCORE: X.X" where X.X is a float between 0.0 and 1.0."""
        
        res_fa = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt_fa}],
            temperature=0.0
        ).choices[0].message.content
        metrics["Faithfulness"] = self.parse_score(res_fa)

        # 3. Answer Relevance
        if ground_truth:
            prompt_ar = f"""You are an objective judge evaluating a RAG system.
TASK: Evaluate 'Answer Relevance'. Based on the explicitly required GROUND TRUTH, does the GENERATED ANSWER correctly address the user's QUESTION without generating irrelevant tangents?

QUESTION: {question}
GROUND TRUTH: {ground_truth}
GENERATED ANSWER: {answer}

Provide a brief reasoning, then end with strict formatting "SCORE: X.X" where X.X is a float between 0.0 and 1.0."""
        else:
            prompt_ar = f"""You are an objective judge evaluating a RAG system.
TASK: Evaluate 'Answer Relevance'. Does the GENERATED ANSWER correctly and directly address the user's QUESTION without generating irrelevant tangents?

QUESTION: {question}
GENERATED ANSWER: {answer}

Provide a brief reasoning, then end with strict formatting "SCORE: X.X" where X.X is a float between 0.0 and 1.0."""

        
        res_ar = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt_ar}],
            temperature=0.0
        ).choices[0].message.content
        metrics["Answer Relevance"] = self.parse_score(res_ar)

        return metrics

    def run_evaluation(self, dataset_path: str):
        print(f"\n🚀 Initializing RAG Evaluation using dataset: {os.path.basename(dataset_path)}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        results = []
        for i, item in enumerate(eval_data):
            q = item['question']
            gt = item['ground_truth']
            
            print(f"\n" + "-"*60)
            print(f"📝 Evaluating Case {i+1}/{len(eval_data)}")
            print(f"Q: {q}")
            
            context, answer = self.get_context_and_answer(q)
            
            print(f"🧠 Running LLM-as-a-Judge grading...")
            scores = self.evaluate_triad(q, gt, context, answer)
            
            print(f"📊 Scores: CR={scores['Context Relevance']} | FA={scores['Faithfulness']} | AR={scores['Answer Relevance']}")
            
            results.append({
                "question": q,
                "scores": scores
            })

        print("\n\n" + "="*50)
        print("🏆 FINAL EVALUATION REPORT")
        print("="*50)
        
        avg_cr = sum([r['scores']['Context Relevance'] for r in results]) / len(results)
        avg_fa = sum([r['scores']['Faithfulness'] for r in results]) / len(results)
        avg_ar = sum([r['scores']['Answer Relevance'] for r in results]) / len(results)
        overall = (avg_cr + avg_fa + avg_ar) / 3

        print(f"Overall RAG Score:    {overall:.2%}")
        print(f"- Context Relevance:  {avg_cr:.2%}")
        print(f"- Faithfulness:       {avg_fa:.2%}")
        print(f"- Answer Relevance:   {avg_ar:.2%}")
        print("="*50)
        
        # Write markdown report
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# RAG Evaluation Report\n\n")
            f.write("## Overall Metrics\n")
            f.write(f"- **Overall RAG Score:** {overall:.2%}\n")
            f.write(f"- **Context Relevance (Retrieval):** {avg_cr:.2%}\n")
            f.write(f"- **Faithfulness (Hallucination Control):** {avg_fa:.2%}\n")
            f.write(f"- **Answer Relevance (Generation Accuracy):** {avg_ar:.2%}\n\n")
            f.write("## Detailed Test Cases\n")
            for i, r in enumerate(results):
                f.write(f"### Q{i+1}: {r['question']}\n")
                s = r['scores']
                f.write(f"- **Context Relevance:** {s['Context Relevance']} \n")
                f.write(f"- **Faithfulness:** {s['Faithfulness']} \n")
                f.write(f"- **Answer Relevance:** {s['Answer Relevance']} \n\n")

        print(f"📁 Detailed report saved to: {report_path}\n")


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.json")
    evaluator = RAGEvaluator()
    evaluator.run_evaluation(dataset)

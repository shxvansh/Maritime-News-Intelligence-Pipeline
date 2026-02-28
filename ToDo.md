# Project To-Do List & Prioritization

This document outlines the remaining tasks for the Maritime News Intelligence Pipeline, ordered by priority to ensure a logical flow from foundational understanding to production deployment and final documentation.

## Phase 1: Codebase Comprehension & Verification
*Before building further, solidify the understanding of the current architecture and ensure all initial requirements are met.*

1. **Review and Understand the Existing Codebase:** Deep dive into all existing scripts, modules, and data flows to gain a comprehensive understanding of the current system architecture.
2. **Analyze the Knowledge Graph:** Study the `graph_builder.py` implementation, understand how NetworkX builds nodes/edges, and review the generated `knowledge_graph.png`.
3. **Conduct Resource Evaluation:** Document and evaluate every technology and resource used (e.g., GraphQL, GLiNER, LLaMA, Qdrant, BERTopic). For each, note how it works, outline alternatives, and evaluate the advantages and disadvantages of our selection.
4. **Cross-Check Requirements (Email Review):** Read the original project specifications/email to verify that all core technical requirements have been fulfilled.

## Phase 2: Core Feature Refinement & Security
*Improve the quality, safety, and depth of the intelligence extraction.*

5. **Enhance Topic Modeling & Trend Analysis:** Re-evaluate and refine the BERTopic implementation (`analytics_engine.py`) to ensure the extracted macro-themes are as accurate and insightful as possible.
6. **Implement LLM Hallucination Awareness:** Audit prompts and extraction pipelines to ensure strict factual adherence parameters are in place.
7. **Build Hallucination Feedback Loops:** Implement automated guardrails and feedback mechanisms for all LLM outputs to detect, flag, and mitigate hallucinations.
8. **Integrate Security Features & Guardrails:** Add robust security checks and prompt-injection guardrails specifically surrounding the RAG architecture and LLM parsing logic.

## Phase 3: Backend Architecture & Production Readiness
*Transform the standalone scripts into a robust, deployable service.*

9. **FastAPI Integration:** Wrap the entire backend (scrapers, preprocessors, LLM extraction, RAG, and analytics) into a cohesive, high-performance `FastAPI` application.
10. **Design for Scalability:** Structure the architecture for true production readiness (e.g., defining asynchronous queues, connection pooling, and error handling).
11. **Performance Analysis:** Conduct the performance analysis requested in the project email, benchmarking latency, throughput, and token costs.

## Phase 4: Frontend Integration & Visualization
*Create a unified interface to interact with the intelligence pipeline.*

12. **Dashboard Architecture Design:** Figure out the optimal architecture to connect the FastAPI backend, the Knowledge Graph, the Topic Modeling visualizations, and the RAG Chatbot into a single, unified interface.
13. **Build the Dashboard:** Develop the actual user-facing dashboard (e.g., using Streamlit, React, or pure HTML/JS) to visualize all maritime intelligence.

## Phase 5: Documentation & Stretch Goals
*Finalize the project with professional-grade documentation and advanced features.*

14. **Granular Codebook Documentation:** Create a dedicated `.md` file for *every* code section/module, explicitly explaining its logic, inputs, and outputs.
15. **Advanced Feature Implementation:** If time permits, implement remaining advanced requirements or stretch goals from the "Senior Section" of the project brief.
16. **Granular Version Control:** Carefully review and stage all modified files individually (`git add <file>`), ensuring each commit features explicit, properly formatted commit messages. Do **not** use `git add .` or blind commits.

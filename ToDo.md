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

## Phase 3: Backend Architecture & Production Readiness
*Transform the standalone scripts into a robust, deployable service.*

10. **Design for Scalability:** Structure the architecture for true production readiness (e.g., defining asynchronous queues, connection pooling, and error handling).
11. **Performance Analysis:** Conduct the performance analysis requested in the project email, benchmarking latency, throughput, and token costs.

## Phase 5: Documentation & Stretch Goals
*Finalize the project with professional-grade documentation and advanced features.*

14. **Granular Codebook Documentation:** Create a dedicated `.md` file for *every* code section/module, explicitly explaining its logic, inputs, and outputs.

16. **Granular Version Control:** Carefully review and stage all modified files individually (`git add <file>`), ensuring each commit features explicit, properly formatted commit messages. Do **not** use `git add .` or blind commits.

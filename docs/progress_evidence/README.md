# Development Progress Evidence

This folder documents the development, testing, debugging, and improvement of KnowledgeVault.

## Phase 1 — Document Ingestion and Chunking

The first prototype supported PDF upload, page-level text extraction, metadata preservation, and overlapping text chunks.

Evidence:

- `01_phase1_document_chunking.png`

## Phase 2 — Embeddings and Retrieval

The next version generated local embeddings using `nomic-embed-text`, stored chunks in ChromaDB, and displayed retrieved document context.

Evidence:

- `02_phase2_semantic_retrieval.png`

## Initial Evaluation

The initial evaluation achieved approximately 50% because some relevant chunks were ranked below unrelated content and the first local language model did not consistently follow the retrieved evidence.

Evidence:

- `03_initial_evaluation_50_percent.png`

## Retrieval Improvements

The system was improved using:

- Smaller document chunks
- Chunk overlap
- Cosine-distance configuration
- Lexical keyword matching
- Semantic and lexical hybrid retrieval
- Candidate reranking
- Evidence filtering
- A stronger local language model

Evidence:

- `04_hybrid_retrieval_improvement.png`

## Final Evaluation

The final automated evaluation passed all included test cases.

```text
Passed: 10/10
Pass rate: 100.0%
# AI Usage Disclosure

## Purpose

AI tools were used during the development of KnowledgeVault to support ideation, debugging, code review, interface planning, testing, and documentation.

AI tools were not used as a replacement for understanding the project. The implementation was tested, reviewed, and adjusted during development.

## AI Tools Used

### ChatGPT

ChatGPT was used for:

- Breaking the assignment into development phases
- Planning the RAG architecture
- Suggesting the initial folder structure
- Explaining virtual environments and dependency isolation
- Drafting Python implementation examples
- Debugging Python, LangChain, ChromaDB, and Ollama issues
- Improving document chunking and retrieval
- Designing a hybrid semantic and lexical retrieval strategy
- Drafting evaluation scripts
- Reviewing failed evaluation results
- Improving grounding and fallback behaviour
- Preparing README documentation
- Preparing this AI usage disclosure

### Ollama Models

The following local models are part of the application itself:

- `nomic-embed-text` for document and query embeddings
- `qwen2.5:3b` for grounded answer generation

These models run locally through Ollama.

## Human Decisions and Contributions

The following decisions and actions were completed and verified manually:

- Selected the project topic and scope
- Chose LangChain, ChromaDB, Streamlit, and Ollama
- Installed and configured Python
- Created and activated the project virtual environment
- Installed project dependencies
- Installed Ollama models
- Created the project files and folders
- Ran the Streamlit application
- Uploaded and processed test documents
- Tested document extraction and chunking
- Tested semantic retrieval
- Reviewed source references
- Ran the automated evaluation pipeline
- Identified incorrect retrieval results
- Compared evaluation outputs
- Rebuilt the Chroma database after configuration changes
- Verified unsupported-question fallback behaviour
- Confirmed the final 10/10 automated evaluation result
- Reviewed the final implementation and documentation

## How AI-Generated Suggestions Were Verified

AI-generated code and suggestions were not accepted without testing.

The following verification process was used:

1. Run each implementation phase separately.
2. Test document upload and extraction.
3. Inspect generated document chunks.
4. Verify filename and page metadata.
5. Generate embeddings through Ollama.
6. Store and retrieve chunks through ChromaDB.
7. Ask supported and unsupported questions.
8. Review generated answers against the original PDF.
9. Confirm visible source references.
10. Run automated evaluation questions.
11. Investigate failed questions instead of manually changing results.
12. Improve retrieval and prompting.
13. Rebuild the knowledge base.
14. Run the evaluation again.
15. Preserve the final evaluation report.

## Problems Identified During Development

### Python environment conflicts

Multiple Python versions were installed on the development computer. A clean Python installation and project-specific virtual environment were created to prevent dependency conflicts.

### Ollama PATH issue

The Ollama command was temporarily unavailable inside the VS Code terminal because the Windows PATH had changed. The PATH was repaired and Ollama was verified using:

```powershell
ollama --version
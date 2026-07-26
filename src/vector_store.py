import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "knowledge_vault"

EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "during",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}


TOKEN_ALIASES = {
    "tools": "tool",
    "suggested": "suggest",
    "suggesting": "suggest",
    "documents": "document",
    "deliverables": "deliverable",
    "keys": "key",
    "passwords": "password",
    "tokens": "token",
    "credentials": "credential",
    "used": "use",
    "using": "use",
    "committed": "commit",
    "committing": "commit",
    "assigned": "assign",
    "minimum": "least",
    "mandatory": "requirement",
    "required": "requirement",
    "requires": "requirement",
}


def get_embedding_model() -> OllamaEmbeddings:
    """Return the local Ollama embedding model."""

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )


def clear_vector_store() -> None:
    """
    Clear the Chroma collection safely.

    The database directory is not deleted because Windows may lock
    Chroma files while the application is running.
    """

    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIRECTORY),
    )

    try:
        vector_store.delete_collection()

    except Exception as error:
        error_message = str(error).lower()

        collection_missing = (
            "does not exist" in error_message
            or "not found" in error_message
        )

        if not collection_missing:
            raise RuntimeError(
                f"Could not clear the Chroma collection: {error}"
            ) from error


def create_vector_store(
    documents: list[Document],
    reset_database: bool = True,
) -> Chroma:
    """Generate embeddings and store document chunks in Chroma."""

    if not documents:
        raise ValueError("No document chunks were provided.")

    if reset_database:
        clear_vector_store()
    else:
        CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store() -> Chroma:
    """Load the existing Chroma vector database."""

    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIRECTORY),
    )


def normalize_token(token: str) -> str:
    """Normalize common word variations for keyword retrieval."""

    token = token.lower()

    if token in TOKEN_ALIASES:
        return TOKEN_ALIASES[token]

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("s") and len(token) > 4:
        return token[:-1]

    return token


def tokenize_text(text: str) -> set[str]:
    """Convert text into normalized retrieval terms."""

    raw_tokens = re.findall(
        r"[a-zA-Z0-9%]+",
        text.lower(),
    )

    normalized_tokens = {
        normalize_token(token)
        for token in raw_tokens
        if len(token) > 1 and token not in STOP_WORDS
    }

    return normalized_tokens


def calculate_lexical_score(
    query: str,
    document_text: str,
) -> float:
    """Calculate direct term overlap between a query and chunk."""

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return 0.0

    document_tokens = tokenize_text(document_text)

    matches = query_tokens.intersection(document_tokens)

    return len(matches) / len(query_tokens)


def get_all_documents(
    vector_store: Chroma,
) -> list[Document]:
    """Read all stored chunks for lexical retrieval."""

    stored_data = vector_store.get(
        include=["documents", "metadatas"],
    )

    documents = stored_data.get("documents", [])
    metadatas = stored_data.get("metadatas", [])

    return [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
        for text, metadata in zip(documents, metadatas)
        if text
    ]


def document_key(document: Document) -> tuple:
    """Create a stable identifier for a document chunk."""

    return (
        document.metadata.get("source"),
        document.metadata.get("page_number"),
        document.metadata.get("chunk_id"),
    )


def search_documents(
    query: str,
    number_of_results: int = 3,
) -> list[tuple[Document, float]]:
    """
    Perform hybrid semantic and lexical retrieval.

    Chroma provides semantic similarity while keyword overlap
    improves exact questions involving names, percentages,
    tools, API keys, and repository formats.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError("The search query cannot be empty.")

    vector_store = load_vector_store()

    candidate_count = max(number_of_results * 4, 12)

    semantic_results = vector_store.similarity_search_with_score(
        query=clean_query,
        k=candidate_count,
    )

    all_documents = get_all_documents(vector_store)

    merged_candidates: dict[tuple, dict] = {}

    for document, distance in semantic_results:
        # Chroma is configured to use cosine distance.
        semantic_score = max(
            0.0,
            min(1.0, 1.0 - float(distance)),
        )

        key = document_key(document)

        merged_candidates[key] = {
            "document": document,
            "semantic_score": semantic_score,
        }

    # Include every chunk for exact keyword comparison.
    for document in all_documents:
        key = document_key(document)

        if key not in merged_candidates:
            merged_candidates[key] = {
                "document": document,
                "semantic_score": 0.0,
            }

    reranked_results = []

    for candidate in merged_candidates.values():
        document = candidate["document"]
        semantic_score = candidate["semantic_score"]

        lexical_score = calculate_lexical_score(
            query=clean_query,
            document_text=document.page_content,
        )
        
        query_terms = tokenize_text(clean_query)
        document_terms = tokenize_text(document.page_content)
        
        document_text_lower = document.page_content.lower()

        requirement_intent = bool(
            query_terms.intersection(
                {
                    "minimum",
                    "least",
                    "requirement",
                    "maximum",
                    "mandatory",
                }
            )
        )

        quantitative_bonus = 0.0

        if requirement_intent:
            if re.search(
                r"\b(at least|minimum|maximum|no fewer than|no more than)\b",
                document_text_lower,
            ):
                quantitative_bonus += 0.20

            if re.search(r"\b\d+\b", document_text_lower):
                quantitative_bonus += 0.12

            if any(
                term in document_text_lower
                for term in (
                    "mandatory",
                    "required",
                    "requirements",
                    "must contain",
                    "should contain",
                )
            ):
                quantitative_bonus += 0.08

        important_matches = (
            query_terms
            .intersection(document_terms)
            .intersection(
                {
                    "least",
                    "requirement",
                    "maximum",
                    "percentage",
                }
            )
        )

        intent_bonus = min(
            len(important_matches) * 0.08,
            0.16,
        )

        combined_score = min(
            1.0,
            semantic_score * 0.55
            + lexical_score * 0.45
            + intent_bonus
            + quantitative_bonus,
        )

        document.metadata["semantic_score"] = semantic_score
        document.metadata["lexical_score"] = lexical_score
        document.metadata["combined_score"] = combined_score

        reranked_results.append(
            (document, combined_score)
        )

    reranked_results.sort(
        key=lambda result: (
            result[1],
            result[0].metadata.get("lexical_score", 0),
            result[0].metadata.get("semantic_score", 0),
        ),
        reverse=True,
    )

    return reranked_results[:number_of_results]
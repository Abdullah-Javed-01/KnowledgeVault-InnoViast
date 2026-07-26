from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save a Streamlit uploaded PDF inside data/uploads."""

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(uploaded_file.name).name
    file_path = UPLOAD_DIRECTORY / safe_filename

    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def load_pdf(file_path: Path) -> list[Document]:
    """Extract text and metadata from every page of a PDF."""

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are currently supported.")

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    
    if not pages:
        raise ValueError(
            "The PDF contains no readable pages."
        )

    has_extractable_text = any(
        page.page_content.strip()
        for page in pages
    )

    if not has_extractable_text:
        raise ValueError(
            "No selectable text was found. "
            "This PDF may be scanned or image-based, "
            "and OCR is not currently supported."
        )

    for page in pages:
        original_page_number = page.metadata.get("page", 0)

        page.metadata["source"] = file_path.name
        page.metadata["page_number"] = int(original_page_number) + 1

    return pages


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split document pages into smaller overlapping text chunks."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["character_count"] = len(chunk.page_content)

    return chunks


def process_uploaded_files(
    uploaded_files: list[Any],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> tuple[list[Path], list[Document], list[Document]]:
    """Save, load, and split multiple uploaded PDF documents."""

    saved_files: list[Path] = []
    all_pages: list[Document] = []

    for uploaded_file in uploaded_files:
        saved_file = save_uploaded_file(uploaded_file)
        pages = load_pdf(saved_file)

        saved_files.append(saved_file)
        all_pages.extend(pages)

    chunks = split_documents(
        documents=all_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return saved_files, all_pages, chunks

def clear_uploaded_files() -> None:
    """Delete locally saved uploads while preserving .gitkeep."""

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for file_path in UPLOAD_DIRECTORY.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink()
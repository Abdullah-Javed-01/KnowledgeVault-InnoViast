import html
import re
import streamlit as st

from src.document_processor import (
    clear_uploaded_files,
    process_uploaded_files,
)
from src.rag_chain import answer_question
from src.vector_store import clear_vector_store, create_vector_store


st.set_page_config(
    page_title="KnowledgeVault",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    /* Main page */
    .stApp {
        background:
            radial-gradient(circle at 85% 10%, rgba(124, 58, 237, 0.12), transparent 28%),
            radial-gradient(circle at 15% 90%, rgba(14, 165, 233, 0.08), transparent 25%),
            #0B1120;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Header */
    .kv-header {
        padding: 1.7rem 1.9rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.16),
                rgba(15, 23, 42, 0.92)
            );
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
    }

    .kv-title {
        margin: 0;
        color: #F8FAFC;
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.04em;
    }

    .kv-subtitle {
        margin: 0.45rem 0 0;
        color: #94A3B8;
        font-size: 1rem;
        line-height: 1.6;
    }

    .kv-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .kv-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.38rem 0.7rem;
        border: 1px solid rgba(167, 139, 250, 0.23);
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.11);
        color: #C4B5FD;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        min-height: 116px;
        padding: 1.15rem 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 17px;
        background: rgba(17, 24, 39, 0.72);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.13);
    }

    .metric-label {
        margin-bottom: 0.55rem;
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 750;
        line-height: 1;
    }

    .metric-note {
        margin-top: 0.55rem;
        color: #64748B;
        font-size: 0.76rem;
    }

    /* Empty state */
    .empty-state {
        padding: 3.8rem 2rem;
        border: 1px dashed rgba(148, 163, 184, 0.27);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.55);
        text-align: center;
    }

    .empty-icon {
        margin-bottom: 0.8rem;
        font-size: 3.2rem;
    }

    .empty-title {
        margin: 0;
        color: #F8FAFC;
        font-size: 1.35rem;
        font-weight: 700;
    }

    .empty-text {
        max-width: 560px;
        margin: 0.65rem auto 0;
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* Source cards */
    .source-card {
        margin: 0.75rem 0;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.78);
    }

    .source-title {
        color: #E2E8F0;
        font-size: 0.91rem;
        font-weight: 700;
    }

    .source-meta {
        margin: 0.35rem 0 0.7rem;
        color: #8B5CF6;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .source-content {
        color: #CBD5E1;
        font-size: 0.86rem;
        line-height: 1.58;
        white-space: pre-wrap;
    }

    /* General Streamlit components */
    div[data-testid="stFileUploader"] {
        padding: 0.65rem;
        border: 1px dashed rgba(167, 139, 250, 0.35);
        border-radius: 14px;
        background: rgba(124, 58, 237, 0.05);
    }

    div[data-testid="stChatMessage"] {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.64);
    }

    div[data-testid="stChatInput"] {
        border-radius: 16px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.44);
    }

    div.stButton > button {
        min-height: 2.8rem;
        border-radius: 12px;
        font-weight: 650;
    }

    div.stButton > button[kind="primary"] {
        border: none;
        background: linear-gradient(135deg, #7C3AED, #6D28D9);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
    }

    .sidebar-heading {
        margin-bottom: 0.2rem;
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 750;
    }

    .sidebar-description {
        margin-bottom: 1rem;
        color: #94A3B8;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .document-row {
        margin: 0.45rem 0;
        padding: 0.7rem 0.8rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        background: rgba(30, 41, 59, 0.55);
        color: #CBD5E1;
        font-size: 0.8rem;
    }
</style>
"""


def initialize_session_state() -> None:
    """Initialize values used across Streamlit reruns."""

    defaults = {
        "processed_files": [],
        "document_pages": [],
        "document_chunks": [],
        "vector_store_ready": False,
        "messages": [],
        "uploader_key": 0,
        "clear_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Display the application header."""

    st.markdown(
        """
        <div class="kv-header">
            <h1 class="kv-title">📚 KnowledgeVault</h1>
            <p class="kv-subtitle">
                Ask questions about your PDF documents using a private,
                locally powered Retrieval-Augmented Generation assistant.
            </p>
            <div class="kv-badges">
                <span class="kv-badge">🔒 Local and private</span>
                <span class="kv-badge">🔎 Hybrid retrieval</span>
                <span class="kv-badge">📎 Source references</span>
                <span class="kv-badge">🧠 Grounded answers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str | int,
    note: str,
) -> None:
    """Render a custom status metric."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Display the interface shown before a knowledge base is created."""

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🗂️</div>
            <h2 class="empty-title">Your knowledge base is empty</h2>
            <p class="empty-text">
                Upload one or more PDF documents from the sidebar and
                select <strong>Process Documents</strong>. KnowledgeVault
                will extract the text, create embeddings, and prepare the
                documents for grounded question answering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_sources(sources: list[dict]) -> None:
    """Display document evidence below an assistant answer."""

    if not sources:
        return

    with st.expander(
        f"📎 View supporting sources ({len(sources)})"
    ):
        for position, source in enumerate(sources, start=1):
            filename = html.escape(
                str(source.get("source", "Unknown document"))
            )

            page_number = html.escape(
                str(source.get("page_number", "Unknown"))
            )

            chunk_id = html.escape(
                str(source.get("chunk_id", "Unknown"))
            )

            score = float(
                source.get("relevance_score", 0)
            )

            raw_content = str(
                source.get("content", "")
            ).strip()

            clean_content = re.sub(
                r"\s+",
                " ",
                raw_content,
            )

            if len(clean_content) > 900:
                clean_content = (
                    clean_content[:900]
                    .rsplit(" ", 1)[0]
                    + "..."
                )

            content = html.escape(clean_content)

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">
                        {position}. {filename}
                    </div>
                    <div class="source-meta">
                        Page {page_number}
                        &nbsp;•&nbsp;
                        Chunk {chunk_id}
                        &nbsp;•&nbsp;
                        Score {score:.3f}
                    </div>
                    <div class="source-content">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def display_chat_history() -> None:
    """Render stored chat messages."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_sources(
                    message.get("sources", [])
                )


def display_document_chunks() -> None:
    """Display a compact preview of processed chunks."""

    chunks = st.session_state.document_chunks

    if not chunks:
        return

    with st.expander(
        f"🧩 Inspect processed chunks ({len(chunks)})"
    ):
        preview_count = min(5, len(chunks))

        for index, chunk in enumerate(
            chunks[:preview_count],
            start=1,
        ):
            source = chunk.metadata.get(
                "source",
                "Unknown document",
            )

            page_number = chunk.metadata.get(
                "page_number",
                "Unknown",
            )

            chunk_id = chunk.metadata.get(
                "chunk_id",
                index,
            )

            st.markdown(
                f"**Chunk {chunk_id} · {source} · Page {page_number}**"
            )

            st.caption(
                f"{len(chunk.page_content)} characters"
            )

            st.write(chunk.page_content)

            if index < preview_count:
                st.divider()


def clear_chat() -> None:
    """Remove the current conversation."""

    st.session_state.messages = []
    st.rerun()


def clear_knowledge_base() -> None:
    """Clear vectors, documents, uploaded files, and chat history."""

    try:
        clear_vector_store()
        clear_uploaded_files()

    except Exception as error:
        st.session_state.clear_error = str(error)
        return

    st.session_state.processed_files = []
    st.session_state.document_pages = []
    st.session_state.document_chunks = []
    st.session_state.vector_store_ready = False
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.session_state.clear_error = ""

    st.rerun()


initialize_session_state()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-heading">Knowledge Base</div>
        <div class="sidebar-description">
            Upload PDF files and prepare them for semantic search.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} document(s) selected"
        )

        for uploaded_file in uploaded_files:
            safe_name = html.escape(uploaded_file.name)
            file_size_kb = uploaded_file.size / 1024

            st.markdown(
                f"""
                <div class="document-row">
                    📄 {safe_name}<br>
                    <span style="color:#64748B">
                        {file_size_kb:.1f} KB
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    process_documents = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    st.divider()

    with st.expander("⚙️ RAG Settings"):
        chunk_size = st.slider(
            "Chunk size",
            min_value=300,
            max_value=1500,
            value=500,
            step=100,
            help="Maximum approximate characters per document chunk.",
        )

        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=25,
            help="Text repeated between neighbouring chunks.",
        )

        number_of_results = 3

        st.caption("Retrieved evidence chunks: `3`")

    with st.expander("🧠 System Details"):
        st.caption("RAG framework")
        st.code("LangChain", language=None)

        st.caption("Vector database")
        st.code("ChromaDB", language=None)

        st.caption("Embedding model")
        st.code("nomic-embed-text", language=None)

        st.caption("Generation model")
        st.code("qwen2.5:3b", language=None)
        
        st.caption("Evidence chunks")
        st.code("3", language=None)

        st.caption("Runtime")
        st.code("Ollama · Local", language=None)

    st.divider()

    button_column_1, button_column_2 = st.columns(2)

    with button_column_1:
        st.button(
            "Clear Chat",
            use_container_width=True,
            on_click=clear_chat,
            disabled=not st.session_state.messages,
        )

    with button_column_2:
        st.button(
            "Clear All",
            use_container_width=True,
            on_click=clear_knowledge_base,
            disabled=not st.session_state.vector_store_ready,
        )

if st.session_state.clear_error:
    st.error(
        "The knowledge base could not be cleared: "
        f"{st.session_state.clear_error}"
    )

if process_documents:
    if chunk_overlap >= chunk_size:
        st.sidebar.error(
            "Chunk overlap must be smaller than chunk size."
        )

    else:
        try:
            with st.spinner(
                "Extracting text, generating embeddings, "
                "and building the knowledge base..."
            ):
                saved_files, pages, chunks = (
                    process_uploaded_files(
                        uploaded_files=uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )

                create_vector_store(
                    documents=chunks,
                    reset_database=True,
                )

                st.session_state.processed_files = saved_files
                st.session_state.document_pages = pages
                st.session_state.document_chunks = chunks
                st.session_state.vector_store_ready = True
                st.session_state.messages = []

            st.toast(
                "Knowledge base created successfully.",
                icon="✅",
            )

        except Exception as error:
            st.session_state.vector_store_ready = False
            st.error(
                f"Document processing failed: {error}"
            )


render_header()

if st.session_state.vector_store_ready:
    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = (
        st.columns(4)
    )

    with metric_column_1:
        render_metric_card(
            "Documents",
            len(st.session_state.processed_files),
            "Indexed PDF files",
        )

    with metric_column_2:
        render_metric_card(
            "Pages",
            len(st.session_state.document_pages),
            "Extracted pages",
        )

    with metric_column_3:
        render_metric_card(
            "Chunks",
            len(st.session_state.document_chunks),
            "Stored in ChromaDB",
        )

    with metric_column_4:
        render_metric_card(
            "Status",
            "Ready",
            "Knowledge base active",
        )

    st.markdown("")

    display_document_chunks()

    st.markdown("## Chat with your documents")

    st.caption(
        "KnowledgeVault answers only from retrieved document evidence. "
        "Expand the sources below an answer to verify it."
    )

    if not st.session_state.messages:
        st.info(
            "Your knowledge base is ready. Ask a question below."
        )

    display_chat_history()

    user_question = st.chat_input(
        "Ask a question about your documents"
    )

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching the knowledge base..."
                ):
                    result = answer_question(
                        question=user_question,
                        number_of_results=number_of_results,
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                display_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "is_fallback": result["is_fallback"],
                    }
                )

            except Exception as error:
                error_message = (
                    "KnowledgeVault encountered an error while "
                    f"processing the question: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "is_fallback": True,
                    }
                )

else:
    render_empty_state()

    st.markdown("### How it works")

    step_column_1, step_column_2, step_column_3 = st.columns(3)

    with step_column_1:
        render_metric_card(
            "01 · Upload",
            "PDFs",
            "Select one or more documents",
        )

    with step_column_2:
        render_metric_card(
            "02 · Process",
            "Index",
            "Create chunks and embeddings",
        )

    with step_column_3:
        render_metric_card(
            "03 · Ask",
            "Chat",
            "Receive grounded answers",
        )
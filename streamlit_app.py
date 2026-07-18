import streamlit as st

from rag.config import get_settings
from rag.embeddings import create_embedding
from rag.routers.llm_call_router import LLMCallRouter
from rag.services.document_registry import build_registry
from rag.services.ingestion_service import IngestionService
from rag.services.prompt_builder import build_grounded_prompt
from rag.services.retrieval_service import RetrievalService
from rag.vector_stores import available_backends, create_vector_store


st.set_page_config(page_title="RAG Playground", layout="wide")


@st.cache_resource(show_spinner=False)
def get_vector_store(backend: str):
    settings = get_settings().model_copy(update={"vector_db_backend": backend})
    return create_vector_store(settings)


@st.cache_resource(show_spinner=False)
def get_registry(mongo_uri: str):
    # Keyed on the URI so caching is per-connection; None when Mongo is disabled.
    return build_registry(get_settings())


st.title("RAG Playground")
st.caption("Pick a vector DB + embedding model and compare ingestion/retrieval without touching .env.")

settings = get_settings()

with st.sidebar:
    st.header("Configuration")
    backend = st.selectbox(
        "Vector DB backend",
        available_backends(),
        help="Each backend has its own storage, so re-ingest after switching to compare.",
    )
    collection_name = st.text_input("Collection name", value=settings.collection_name)
    chunk_size = st.number_input("Chunk size", min_value=50, value=settings.chunk_size)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=settings.chunk_overlap)
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=settings.retrieval_top_k)

try:
    embedding = create_embedding(settings)
except Exception as exc:  # e.g. missing OPENROUTER_API_KEY
    st.error(f"Could not initialize the '{settings.embedding_backend}' embedding: {exc}")
    st.stop()

with st.sidebar:
    st.caption(f"Embedding model: `{embedding.model_id}` · dim {embedding.dimension}")

vector_store = get_vector_store(backend)
registry = get_registry(settings.mongo_uri)

if registry is None:
    st.sidebar.info("MongoDB registry disabled (set MONGO_URI in .env to enable dedup + tracking).")

ingestion_service = IngestionService(
    vector_store=vector_store,
    embedding=embedding,
    collection_name=collection_name,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    registry=registry,
)
retrieval_service = RetrievalService(
    vector_store=vector_store,
    embedding=embedding,
    collection_name=collection_name,
    top_k=top_k,
)


def show_ingestion_result(result) -> None:
    if result.skipped:
        st.info(
            f"Skipped — this document is already ingested in '{result.collection_name}' "
            f"with embedding model `{result.embedding_model}` ({result.total_chunks} chunks)."
        )
    else:
        st.success(
            f"Ingested {result.total_chunks} chunks into '{result.collection_name}' "
            f"using `{result.embedding_model}`."
        )
    st.json(result.model_dump())


ingest_tab, query_tab, registry_tab = st.tabs(["Ingestion", "Query", "Documents"])

with ingest_tab:
    st.subheader("Ingest raw text")
    text_source_id = st.text_input("Source ID", value="doc-1", key="text_source_id")
    raw_text = st.text_area("Raw text", height=200)
    if st.button("Ingest text", disabled=not raw_text.strip()):
        with st.spinner(f"Ingesting into {backend} via {embedding.model_id}..."):
            result = ingestion_service.ingest_text(source_id=text_source_id, raw_text=raw_text)
        show_ingestion_result(result)

    st.divider()

    st.subheader("Ingest a file")
    uploaded_file = st.file_uploader("Upload .pdf, .docx, or .txt", type=["pdf", "docx", "txt"])
    file_source_id = st.text_input("Source ID (optional, defaults to filename)", key="file_source_id")
    if st.button("Ingest file", disabled=uploaded_file is None):
        with st.spinner(f"Parsing and ingesting into {backend} via {embedding.model_id}..."):
            try:
                result = ingestion_service.ingest_file(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    source_id=file_source_id or None,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                show_ingestion_result(result)

with query_tab:
    st.subheader("Ask a question")
    st.caption(f"Query will be embedded with `{embedding.model_id}` — must match the docs' model.")
    question = st.text_input("Question")
    generate_answer = st.checkbox(
        "Generate answer with local LLM",
        value=True,
        help="Uncheck to see retrieval only, skipping the local model (downloads on first run).",
    )
    if st.button("Ask", disabled=not question.strip()):
        with st.spinner(f"Searching {backend}..."):
            retrieved_chunks = retrieval_service.retrieve(question=question)

        if not retrieved_chunks:
            st.warning("No chunks retrieved. Did you ingest into this backend + collection with this embedding model?")
        else:
            st.markdown("### Retrieved chunks")
            for chunk in retrieved_chunks:
                with st.expander(f"{chunk.chunk_id} · score {chunk.score:.4f}"):
                    st.write(chunk.text)
                    st.caption(f"source: {chunk.source_id} · metadata: {chunk.metadata}")

        if generate_answer:
            prompt = build_grounded_prompt(question=question, retrieved_chunks=retrieved_chunks)
            with st.spinner("Generating answer..."):
                answer, provider_used = LLMCallRouter(settings=settings).generate(prompt=prompt)
            st.markdown("### Answer")
            st.write(answer)
            st.caption(f"provider: {provider_used}")

with registry_tab:
    st.subheader("Ingested documents (MongoDB registry)")
    if registry is None:
        st.info("Set MONGO_URI in .env and run `docker compose up -d` to track ingested documents.")
    else:
        documents = registry.list_documents()
        if not documents:
            st.write("No documents ingested yet.")
        else:
            st.caption(f"{len(documents)} document(s) tracked. Dedup key: doc_hash + embedding_model + collection.")
            st.dataframe(documents, use_container_width=True)

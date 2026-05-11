
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.embedder import EmbeddingGenerator
from src.retriever import FAISSRetriever
from src.chatbot import RAGChain
from config import DATA_PROC_DIR, TOP_K, SIMILARITY_THRESHOLD


st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 RAG Document Chatbot")
st.caption("Ask questions about your uploaded PDF documents.")

# ── Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=15, value=TOP_K)
    threshold = st.slider(
        "Similarity threshold", min_value=0.0, max_value=1.0,
        value=SIMILARITY_THRESHOLD, step=0.05,
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. PDFs are chunked & embedded\n"
        "2. Your question is matched to the most relevant chunks\n"
        "3. GPT-4o answers using only those chunks\n"
        "4. Low-confidence chunks are filtered out to reduce hallucinations"
    )

# ── Load chain 
@st.cache_resource(show_spinner="Loading vector index…")
def load_chain(top_k: int, threshold: float) -> RAGChain:
    embedder  = EmbeddingGenerator()
    retriever = FAISSRetriever(
        embedder=embedder,
        top_k=top_k,
        similarity_threshold=threshold,
    )
    retriever.load(DATA_PROC_DIR)
    return RAGChain(retriever=retriever)


try:
    chain = load_chain(top_k, threshold)
except FileNotFoundError:
    st.error(
        "No FAISS index found. Run `python scripts/ingest.py` first, "
        "then restart the app."
    )
    st.stop()

# ── Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for line in msg["sources"]:
                    st.markdown(f"📄 **{line}**")

# ── Input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = chain.ask(prompt)

        st.markdown(response.answer)
        sources_md = ""
        
        source_lines = []
        if response.source_documents:
            seen = set()
            with st.expander("📎 Sources"):
                for doc in response.source_documents:
                    src  = doc.metadata.get("source", "Unknown").split("\\")[-1].split("/")[-1]
                    page = doc.metadata.get("page", "?")
                    score = doc.metadata.get("similarity_score", "?")
                    key  = (src, page)
                    if key not in seen:
                        seen.add(key)
                        line = f"{src} — page {page} · score {score}"
                        st.markdown(f"📄 **{line}**")
                        source_lines.append(line)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response.answer,
        "sources": source_lines,
    })
"""
RAG Chat Application — Streamlit UI
Run with: streamlit run app.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline
from rag.generator import GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background: #0d0d0d; color: #e8e8e8; }
.stApp { background: #0d0d0d; }

.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #00ff88;
    letter-spacing: -1px;
}
.header-sub {
    font-size: 0.85rem;
    color: #666;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 4px 0;
}
.stat-label { font-size: 0.7rem; color: #555; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; }
.stat-value { font-size: 1rem; color: #00ff88; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

.source-card {
    background: #111;
    border-left: 3px solid #00ff88;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #aaa;
}
.score-badge {
    display: inline-block;
    background: #00ff8822;
    color: #00ff88;
    border: 1px solid #00ff8844;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "pipeline": None,
        "messages": [],
        "indexed": False,
        "index_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-title">⚙ CONFIG</div>', unsafe_allow_html=True)
    st.markdown("---")

    # API key
    st.markdown("**LLM Provider**")
    provider = st.selectbox("Provider", ["groq", "openai", "ollama"], index=0)

    api_key = ""
    if provider in ("groq", "openai"):
        env_key = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        api_key = st.text_input(
            f"{env_key}",
            value=os.getenv(env_key, ""),
            type="password",
            placeholder="Paste your key here",
        )

    # Model selection
    model_options = {
        "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
    }
    model = st.selectbox("Model", model_options[provider])

    st.markdown("---")

    # Retrieval settings
    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    score_threshold = st.slider("Min similarity score", 0.0, 1.0, 0.2, step=0.05)
    use_mmr = st.toggle("Use MMR (diverse retrieval)", value=False)

    st.markdown("---")

    # Generation settings
    st.markdown("**Generation Settings**")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=256)
    stream_mode = st.toggle("Stream responses", value=True)

    st.markdown("---")

    # Document upload
    st.markdown("**Index Documents**")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
    reset_store = st.checkbox("Reset vector store before indexing", value=False)

    if st.button("🚀 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    # Save uploads to temp dir
                    tmpdir = tempfile.mkdtemp()
                    for f in uploaded_files:
                        dest = Path(tmpdir) / f.name
                        dest.write_bytes(f.read())

                    # Build pipeline
                    gen_config = GenerationConfig(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    backend_kwargs = {}
                    if api_key:
                        backend_kwargs["api_key"] = api_key

                    pipeline = RAGPipeline(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        provider=provider,
                        llm_config=gen_config,
                        **backend_kwargs,
                    )

                    n_chunks = pipeline.index([tmpdir], reset=reset_store)
                    st.session_state.pipeline = pipeline
                    st.session_state.indexed = True
                    st.session_state.index_stats = {
                        **pipeline.stats,
                        "files": len(uploaded_files),
                        "chunks": n_chunks,
                    }
                    st.success(f"✅ Indexed {n_chunks} chunks from {len(uploaded_files)} file(s)")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.exception("Indexing error")

    # Stats panel
    if st.session_state.indexed and st.session_state.index_stats:
        s = st.session_state.index_stats
        st.markdown("---")
        st.markdown("**Index Stats**")
        for label, val in [
            ("Files indexed", s.get("files", "-")),
            ("Chunks stored", s.get("chunks", "-")),
            ("Embedding model", s.get("embedding_model", "-")),
            ("LLM", f"{s.get('llm_provider','-')}/{s.get('llm_model','-')}"),
        ]:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">{label}</div>'
                f'<div class="stat-value">{val}</div></div>',
                unsafe_allow_html=True,
            )


# ── Main area ──────────────────────────────────────────────────────────────
col_title, col_clear = st.columns([6, 1])
with col_title:
    st.markdown('<div class="header-title">🔍 RAG CHAT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">Retrieval-Augmented Generation · '
        'Ask questions grounded in your documents</div>',
        unsafe_allow_html=True,
    )
with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) used", expanded=False):
                for src in msg["sources"]:
                    meta = src.get("metadata", {})
                    source_name = Path(meta.get("source", "unknown")).name
                    page = meta.get("page", "")
                    page_str = f" · page {int(page)+1}" if page != "" else ""
                    score = src.get("similarity_score", 0)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>{src["rank"]}.</b> {source_name}{page_str}'
                        f'<span class="score-badge">sim: {score:.3f}</span>'
                        f'<br><span style="color:#555">{src["content"][:200]}...</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Input
if not st.session_state.indexed:
    st.info("👆 Upload documents and click **Index Documents** in the sidebar to get started.")
else:
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            pipeline: RAGPipeline = st.session_state.pipeline
            # Update config from current sidebar values
            pipeline.generator.config.temperature = temperature
            pipeline.generator.config.max_tokens = max_tokens
            pipeline.generator.config.model = model

            sources = []
            try:
                if stream_mode:
                    # Retrieve first, then stream
                    if use_mmr:
                        retrieved = pipeline.retriever.retrieve_with_mmr(
                            prompt, top_k=top_k, score_threshold=score_threshold
                        )
                    else:
                        retrieved = pipeline.retriever.retrieve(
                            prompt, top_k=top_k, score_threshold=score_threshold
                        )
                    sources = retrieved

                    if not retrieved:
                        answer = "I couldn't find relevant information in the indexed documents."
                        st.markdown(answer)
                    else:
                        answer_chunks = []
                        placeholder = st.empty()
                        for token in pipeline.generator.stream_generate(prompt, retrieved):
                            answer_chunks.append(token)
                            placeholder.markdown("".join(answer_chunks) + "▌")
                        answer = "".join(answer_chunks)
                        placeholder.markdown(answer)
                else:
                    with st.spinner("Thinking..."):
                        response = pipeline.query(
                            prompt,
                            top_k=top_k,
                            score_threshold=score_threshold,
                            use_mmr=use_mmr,
                        )
                    answer = response.answer
                    sources = response.sources
                    st.markdown(answer)
                    if response.total_tokens:
                        st.caption(f"Tokens used: {response.total_tokens}")

                # Show sources inline
                if sources:
                    with st.expander(f"📎 {len(sources)} source(s) used", expanded=False):
                        for src in sources:
                            meta = src.get("metadata", {})
                            source_name = Path(meta.get("source", "unknown")).name
                            page = meta.get("page", "")
                            page_str = f" · page {int(page)+1}" if page != "" else ""
                            score = src.get("similarity_score", 0)
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["rank"]}.</b> {source_name}{page_str}'
                                f'<span class="score-badge">sim: {score:.3f}</span>'
                                f'<br><span style="color:#555">{src["content"][:200]}...</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.error(answer)
                logger.exception("Query error")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
"""
RAG Pipeline — Main Orchestrator
Ties together: Load → Chunk → Embed → Store → Retrieve → Generate
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from rag.loader import DocumentLoader
from rag.chunker import DocumentChunker
from rag.embedder import EmbeddingManager
from rag.vectorstore import VectorStore
from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator, GenerationConfig, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Typical usage:
        pipeline = RAGPipeline(provider="groq")
        pipeline.index(["./data/pdf", "./data/text_files"])
        response = pipeline.query("What is attention mechanism?")
        print(response.answer)
        print(response.format_sources())
    """

    def __init__(
        self,
        # Embedding
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        # Chunking
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        # Vector store
        collection_name: str = "rag_documents",
        persist_dir: str = "./data/vector_store",
        # LLM
        provider: str = "groq",
        llm_config: Optional[GenerationConfig] = None,
        **generator_kwargs,
    ):
        logger.info("Initializing RAG Pipeline...")

        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingManager(model_name=embedding_model, device=device)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedder,
        )
        self.generator = RAGGenerator(
            provider=provider,
            config=llm_config,
            **generator_kwargs,
        )

        logger.info("RAG Pipeline ready.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        paths: List[str],
        reset: bool = False,
        batch_size: int = 64,
    ) -> int:
        """
        Load, chunk, embed, and store documents from a list of file/directory paths.

        Args:
            paths:      List of file or directory paths to index.
            reset:      If True, wipe the vector store before indexing.
            batch_size: Embedding batch size.

        Returns:
            Number of chunks indexed.
        """
        if reset:
            logger.warning("Resetting vector store before indexing.")
            self.vector_store.delete_collection()

        # 1. Load
        logger.info(f"Loading documents from: {paths}")
        documents = self.loader.load_from_paths(paths)
        if not documents:
            logger.warning("No documents loaded. Check paths and file types.")
            return 0

        # 2. Chunk
        chunks = self.chunker.split(documents)

        # 3. Embed
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed(texts, batch_size=batch_size, show_progress=True)

        # 4. Store
        self.vector_store.add_documents(chunks, embeddings)

        logger.info(f"Indexing complete. {len(chunks)} chunks stored.")
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.2,
        use_mmr: bool = False,
        metadata_filter: Optional[Dict] = None,
    ) -> RAGResponse:
        """
        Full RAG query: retrieve relevant chunks → generate grounded answer.

        Args:
            question:        Natural language question.
            top_k:           Number of chunks to retrieve.
            score_threshold: Minimum similarity score for a chunk to be used.
            use_mmr:         If True, use MMR for diverse retrieval.
            metadata_filter: Optional metadata filter (e.g. {"source": "report.pdf"}).

        Returns:
            RAGResponse with answer, sources, and token info.
        """
        if self.vector_store.count == 0:
            raise RuntimeError("Vector store is empty. Call pipeline.index() first.")

        # Retrieve
        if use_mmr:
            retrieved = self.retriever.retrieve_with_mmr(
                question, top_k=top_k, score_threshold=score_threshold
            )
        else:
            retrieved = self.retriever.retrieve(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )

        if not retrieved:
            logger.warning("No relevant documents found above threshold.")
            return RAGResponse(
                answer="I couldn't find relevant information in the indexed documents to answer this question.",
                sources=[],
            )

        # Generate
        return self.generator.generate(question, retrieved)

    def stream_query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> Iterator[str]:
        """
        Streaming version of query(). Yields answer tokens as they arrive.
        Retrieval is still synchronous; only generation is streamed.
        """
        if self.vector_store.count == 0:
            raise RuntimeError("Vector store is empty. Call pipeline.index() first.")

        retrieved = self.retriever.retrieve(
            question, top_k=top_k, score_threshold=score_threshold
        )

        if not retrieved:
            yield "I couldn't find relevant information in the indexed documents."
            return

        yield from self.generator.stream_generate(question, retrieved)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Quick summary of pipeline state."""
        return {
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dimension,
            "llm_provider": self.generator.provider,
            "llm_model": self.generator.config.model,
            "docs_in_store": self.vector_store.count,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }
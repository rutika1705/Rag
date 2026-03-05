"""
Vector Store Module
Persistent ChromaDB-backed store for document embeddings.
"""

import logging
import uuid #creates unique IDs
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for storing and querying document embeddings.

    Design decisions:
    - PersistentClient: survives process restarts — no re-indexing needed.
    - get_or_create_collection: idempotent initialization.
    - Metadata flattening: ChromaDB only accepts str/int/float in metadata.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._init_store()

    def _init_store(self) -> None:
        import os
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine distance
            )
            logger.info(
                f"VectorStore ready. Collection: '{self.collection_name}' | "
                f"Docs: {self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise

    @property
    def count(self) -> int:
        """Number of documents currently stored."""
        return self.collection.count() if self.collection else 0

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only accepts scalar metadata values."""
        flat = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            else:
                flat[k] = str(v)
        return flat

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add documents + pre-computed embeddings to the store.

        Args:
            documents:  LangChain Document objects.
            embeddings: np.ndarray of shape (N, dim), L2-normalized.
            batch_size: Insert in batches to avoid memory spikes.

        Returns:
            List of assigned document IDs.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )

        all_ids = []
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_embs = embeddings[start : start + batch_size]

            ids, texts, metas, emb_list = [], [], [], []
            for i, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                doc_id = f"doc_{uuid.uuid4().hex[:10]}_{start + i}"
                meta = self._flatten_metadata({
                    **doc.metadata,
                    "doc_index": start + i,
                    "content_length": len(doc.page_content),
                })
                ids.append(doc_id)
                texts.append(doc.page_content)
                metas.append(meta)
                emb_list.append(emb.tolist())

            self.collection.add(
                ids=ids,
                embeddings=emb_list,
                documents=texts,
                metadatas=metas,
            )
            all_ids.extend(ids)

        logger.info(f"Added {len(all_ids)} documents. Total in store: {self.count}")
        return all_ids

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most similar documents.

        Args:
            query_embedding: Shape (dim,), L2-normalized.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts with keys: id, content, metadata, distance, similarity_score, rank.
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self.count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for rank, (doc_id, doc, meta, dist) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ),
                start=1,
            ):
                # cosine distance in [0,2]; convert to similarity in [0,1]
                similarity = 1 - (dist / 2)
                retrieved.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity_score": round(similarity, 4),
                    "rank": rank,
                })

        return retrieved

    def delete_collection(self) -> None:
        """Drop and recreate the collection (full reset)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' has been reset.")
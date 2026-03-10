"""
Vector Store Module
Persistent ChromaDB-backed store for document embeddings.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
   

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
                metadata={"hnsw:space": "cosine"},
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
        """ChromaDB only accepts scalar metadata values. bool checked before int to preserve type."""
        flat = {}
        for k, v in metadata.items():
            if isinstance(v, bool):
                flat[k] = v
            elif isinstance(v, (str, int, float)):
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
      
    
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )

        # Validate embedding dimensionality against existing collection
        if self.count > 0:
            try:
                existing = self.collection.get(limit=1, include=["embeddings"])
                if existing["embeddings"] is not None and len(existing["embeddings"]) > 0:
                    expected_dim = len(existing["embeddings"][0])
                    if embeddings.shape[1] != expected_dim:
                        raise ValueError(
                            f"Embedding dim mismatch: got {embeddings.shape[1]}, "
                            f"expected {expected_dim}."
                        )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Could not validate embedding dimensions: {e}")

        all_ids = []
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_embs = embeddings[start : start + batch_size]

            ids, texts, metas, emb_list = [], [], [], []
            for i, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                # Deterministic ID based on content hash to prevent duplicates
                content_hash = uuid.uuid5(
                    uuid.NAMESPACE_X500,
                    doc.page_content + str(doc.metadata.get("source", "")),
                ).hex[:16]
                doc_id = f"doc_{content_hash}"

                meta = self._flatten_metadata({
                    **doc.metadata,
                    "doc_index": start + i,
                    "content_length": len(doc.page_content),
                })
                ids.append(doc_id)
                texts.append(doc.page_content)
                metas.append(meta)
                emb_list.append(emb.tolist())

            try:
                # upsert instead of add: prevents duplicate entries on re-indexing
                self.collection.upsert(
                    ids=ids,
                    embeddings=emb_list,
                    documents=texts,
                    metadatas=metas,
                )
                all_ids.extend(ids)
            except Exception as e:
                logger.error(
                    f"Batch insertion failed at offset {start}–{start + len(batch_docs) - 1}: {e}. "
                    f"Successfully committed {len(all_ids)} documents before this batch."
                )
                raise

        logger.info(f"Upserted {len(all_ids)} documents. Total in store: {self.count}")
        return all_ids

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        
        # Guard: ChromaDB raises if n_results=0
        if self.count == 0:
            logger.warning("Query called on empty collection.")
            return []

        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self.count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB query failed. "
                f"where filter={where!r} may be invalid. Original error: {e}"
            ) from e

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

    def reset_collection(self) -> None:
       
        self.client.delete_collection(self.collection_name)
        try:
            new_collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            self.collection = None
            logger.error(
                f"Failed to recreate collection '{self.collection_name}' after deletion. "
                f"Store is now in a broken state. Call _init_store() to recover. Error: {e}"
            )
            raise
        self.collection = new_collection
        logger.warning(f"Collection '{self.collection_name}' has been reset.")
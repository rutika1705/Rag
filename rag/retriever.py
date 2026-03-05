"""
Retriever Module
Semantic search over the vector store with MMR and score filtering.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from rag.embedder import EmbeddingManager
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Semantic retriever combining query embedding + vector store search.

    Supports:
    - Standard top-K retrieval
    - Score threshold filtering
    - Maximal Marginal Relevance (MMR) for diversity
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Standard top-K semantic retrieval.

        Args:
            query:           Natural language query string.
            top_k:           Maximum number of results to return.
            score_threshold: Minimum cosine similarity score (0.0–1.0).
            metadata_filter: Optional ChromaDB 'where' clause for filtering by metadata.

        Returns:
            List of result dicts sorted by similarity (descending).
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Retrieving | query='{query}' | top_k={top_k} | threshold={score_threshold}")

        query_emb = self.embedding_manager.embed_query(query)
        results = self.vector_store.query(
            query_embedding=query_emb,
            top_k=top_k,
            where=metadata_filter,
        )

        # Apply score threshold
        filtered = [r for r in results if r["similarity_score"] >= score_threshold]
        logger.info(f"Retrieved {len(filtered)}/{len(results)} results (above threshold)")
        return filtered

    def retrieve_with_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance retrieval — balances relevance vs diversity.

        Useful when top results are near-duplicate chunks from the same document.

        Args:
            fetch_k:      Candidate pool size (fetch more, then re-rank).
            lambda_mult:  0.0 = max diversity, 1.0 = max relevance. 0.5 is balanced.

        Returns:
            Diverse top-K results.
        """
        logger.info(f"MMR Retrieval | query='{query}' | top_k={top_k} | lambda={lambda_mult}")

        query_emb = self.embedding_manager.embed_query(query)

        # Fetch a larger pool first
        candidates = self.vector_store.query(
            query_embedding=query_emb,
            top_k=min(fetch_k, self.vector_store.count),
        )
        candidates = [c for c in candidates if c["similarity_score"] >= score_threshold]

        if len(candidates) <= top_k:
            return candidates

        # Re-embed candidate texts and run MMR
        candidate_texts = [c["content"] for c in candidates]
        candidate_embs = self.embedding_manager.embed(candidate_texts)

        selected_indices = self._mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            top_k=top_k,
            lambda_mult=lambda_mult,
        )

        mmr_results = [candidates[i] for i in selected_indices]
        # Re-assign ranks after MMR reordering
        for i, r in enumerate(mmr_results, start=1):
            r["rank"] = i

        return mmr_results

    @staticmethod
    def _mmr(
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        top_k: int,
        lambda_mult: float,
    ) -> List[int]:
        """
        Core MMR algorithm.
        Iteratively selects the candidate that maximizes:
            lambda * sim(query, doc) - (1 - lambda) * max_sim(doc, selected)
        """
        selected: List[int] = []
        remaining = list(range(len(candidate_embs)))

        query_sims = candidate_embs @ query_emb  # cosine sim (normalized)

        while len(selected) < top_k and remaining:
            if not selected:
                # First pick: most relevant to query
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                selected_embs = candidate_embs[selected]

                def mmr_score(i: int) -> float:
                    relevance = lambda_mult * query_sims[i]
                    redundancy = (1 - lambda_mult) * np.max(selected_embs @ candidate_embs[i])
                    return relevance - redundancy

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return selected
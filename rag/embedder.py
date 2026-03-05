"""
Embedding Module
Converts text into dense vector representations using SentenceTransformers.
"""

import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages a SentenceTransformer model for generating text embeddings.

    Embedding model options (trade-off: quality vs speed):
      - all-MiniLM-L6-v2      → 384-dim  | fast, good for general use
      - all-mpnet-base-v2     → 768-dim  | slower, higher quality
      - BAAI/bge-small-en-v1.5 → 384-dim | optimized for retrieval tasks
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace model identifier.
            device:     'cpu', 'cuda', or 'mps' (Apple Silicon).
        """
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise
        # ← BUG FIX: dimension/embed/embed_query were indented inside _load_model
        #   in the original code, making them unreachable nested functions.
        #   They are now correctly defined at class level below.

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        if not self.model:
            raise RuntimeError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Args:
            texts:         A single string or list of strings.
            batch_size:    Number of texts to encode in each batch.
            show_progress: Show tqdm progress bar for large batches.

        Returns:
            np.ndarray of shape (N, embedding_dim).
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Embedding {len(texts)} text(s)...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize → cosine sim == dot product
        )
        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.embed([query])[0]
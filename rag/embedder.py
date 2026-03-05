"""
1.load_model:to load_model with model_name
2.dimension:model not load then return dimension
3.embed:covert text into vectors(embeddings)
4.embed_query :turn query into embeddings by calling embed function
"""
"""
line_no:85
a = [3, 4]    # "I love cats"
b = [6, 8]    # "I absolutely love cats"
# dot product
a · b = (3×6) + (4×8) = 18 + 32 = 50

# lengths
|a| = √(9+16)  = 5
|b| = √(36+64) = 10

# cosine similarity
cos_sim = 50 / (5 × 10)
        = 50 / 50
        = 1.0   identical meaning!"""
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
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer = None #hint that it will hold sentencetransformer
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


    @property # access it without parentheses()
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        if not self.model:
            raise RuntimeError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()  #number like 384 or 768

    def embed(
        self,
        texts: Union[str, List[str]],# either one string or a list of strings
        batch_size: int = 64,# process 64 texts at a time (no memory overload) and default(64)
        show_progress: bool = False,
    ) -> np.ndarray:
       
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
            #with normalization alll vectors same length
            normalize_embeddings=True,  # L2-normalize → cosine sim == dot product
        )
        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.embed([query])[0] #(1, 384)  → row_no(no_need), 384 numbers

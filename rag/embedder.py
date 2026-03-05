"""
Embedding Module
Converts text into dense vector representations using SentenceTransformers.
"""
import logging
from typing import List,Union,Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger=logging.getLogger(__name__)
class EmbeddingManager:
    def __init__(self,model_name:str="all-MiniLM-L6-v2",device:str="cpu"):
        self.model_name=model_name
        self.device=device
        self.model:Optional[SentenceTransformer] = None
        self._load_model()
    def _load_model(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model=SentenceTransformer(self.model_name,device=self.device)
            logger.info(
                f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"

            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}' : {e}"
                         )
            raise
    @property
    def dimension(self) ->int:
        if not self.model:
            raise RuntimeError("model not loaded.")
        return self.model.get_sentence_embedding_dimension()
    def embed(
        self,
        texts: Union[str,List[str]],
        batch_size: int=64,
        show_progress:bool=False,
        ) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model not loaded")
        if isinstance(texts,str):
            texts=[texts]

        logger.debug(f"Embedding {len(texts)} text(s)...")
        embeddings=self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            )
        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    def embed_query(self,query:str) -> np.ndarray:
        return self.embed([query])[0]


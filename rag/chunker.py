"""
Splits documents into semantically meaningful chunks for retrieval.
defining splitter f
def split:calling _splitter assigning no to every chunk according which file belongs to file1,chunk3
"""

import logging
from typing import List
#Document — a container holding page_content & metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """. The splitter tries \n\n (paragraph break) first, then \n (newline), 
    then .  (sentence end), then   (word boundary), and finally ""
      (character-by-character as a last resort)."""
   
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000, # max limit 850 also fine if its clean boundry
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        """
        Args:
            chunk_size:    Max characters per chunk. 
            chunk_overlap:to preserve context of by adding 1-2 sentence from previous chunk
            separators:    Ordered list of split boundaries (tries each in order).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        #_splitter :private attribute
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len, # tells to measure chunk size in characters
            separators=self.separators,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into chunks.

        Returns:
            List of Document chunks, each carrying the original metadata
            plus chunk-level fields: chunk_index, total_chunks, original_source.
        """
        if not documents:
            logger.warning("split() called with empty document list.")
            return []

        chunks = self._splitter.split_documents(documents)

        
        # no of chunks produced per source file stord in this dict.
        source_counter: dict = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown") #src = "fileA.pdf"
            """If src exists in the dict → return its current value
            If src doesn't exist → return 0 (the default)"""
            source_counter[src] = source_counter.get(src, 0) + 1 #source_counter = {"fileA.pdf": 1}
            chunk.metadata["chunk_index"] = source_counter[src] #chunk[0].metadata = {"source": "fileA.pdf", "chunk_index": 1}

        logger.info(
            f"Split {len(documents)} document(s) → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
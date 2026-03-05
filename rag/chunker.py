"""
Text Chunking Module
Splits documents into semantically meaningful chunks for retrieval.
"""
import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger=logging.getLogger(__name__)
class DocumentChunker:
    """
    splits documents into overlapping chunks using recursive
    character-based splitting with configurable strategy"""
    default_separators=["\n\n","\n","."," ",""]

    def __init__(
            self,
            chunk_size: int=1000,
            chunk_overlap: int=200,
            separators: List[str]=None,
            ):
       self.chunk_size=chunk_size
       self.chunk_overlap=chunk_overlap
       self.separators=separators or self.default_separators

       self._splitter=RecursiveCharacterTextSplitter(
           chunk_size=self.chunk_size,
           chunk_overlap=self.chunk_overlap,
           length_function=len,
           separators=self.separators,
           )
    def split(self,documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("split() called with empty document list.")
            return []
        chunks=self._splitter.split_documents(documents)
        
        source_counter: dict={}
        for chunk in chunks:
            src=chunk.metadata.get("source","unknown")
            source_counter[src]=source_counter.get(src,0)+1
            chunk.metadata["chunk_index"]=source_counter[src]
        logger.info(
            f"Split {len(documents)} document(s) -> {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        
        )
        return chunks
    def split_text(self,text:str,metadata:dict=None) -> List[Document]:
        doc=Document(page_content=text,metadata=metadata or {})
        return self.split([doc])
    
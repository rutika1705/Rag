"""
Document Loader Module
Handles loading of TXT, PDF, and directory-based documents.
1.load_file :if single file path 
2.load_directory:if folder path we have
3.load_from paths: if we have folder and files together in list
"""

import logging
#Path=it has helper method(.exists(), .suffix, .is_dir())
from pathlib import Path 
"""type hints =what kind of data to expect.
List[str] = a list of strings, like ["hello", "world"]
Optional[str] = either a string or None"""
from typing import List, Optional
from langchain_core.documents import Document
"""TextLoader → reads .txt files
PyMuPDFLoader → reads .pdf files 
DirectoryLoader → scans a whole folder & uses diff loaders for each file"""
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyMuPDFLoader,
)
#Creates a logger specifically for *this file*
logger = logging.getLogger(__name__)


class DocumentLoader:
    # everything capital(SUPPORTED_EXTENSIONS) means this value won't change" 

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file (txt or pdf)."""
        #Wraps the plain string "documents/report.pdf" into path for methods .exists()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}")

        logger.info(f"Loading file: {file_path}")
        if path.suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8") #str(path) converts Path object back to a plain string
        else:
            loader = PyMuPDFLoader(str(path))
        
        #Actually reads the file and returns a list of Documen
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document(s) from {file_path}")
        return docs

    def load_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,#this can be either list of string or none
    ) -> List[Document]:
        """Load all supported files from a directory."""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        """loader.load_directory("docs/", extensions=[".pdf"])
           # extensions = [".pdf"]

            allowed = [".pdf"] or [".txt", ".pdf"]"""

        allowed = extensions or list(self.SUPPORTED_EXTENSIONS)
        #all_docs will contain list of doc
        all_docs: List[Document] = []

        for ext in allowed:
            ext_glob = f"**/*{ext}"  #"**/*.txt" 
            if ext == ".txt":
                loader = DirectoryLoader(
                    str(path),
                    glob=ext_glob,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}, #kwargs(keyword arguments):asses extra options
                    show_progress=False,# don't show a loading bar
                    silent_errors=True,#if one file fails, skip
                )
            elif ext == ".pdf":
                loader = DirectoryLoader(
                    str(path),
                    glob=ext_glob,
                    loader_cls=PyMuPDFLoader,
                    show_progress=False,
                    silent_errors=True,
                )
            else:
                continue

            try:
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"  [{ext}] Loaded {len(docs)} documents")
            except Exception as e:
                logger.warning(f"  [{ext}] Error loading: {e}")

        logger.info(f"Total documents loaded from '{dir_path}': {len(all_docs)}")
        return all_docs

    def load_from_paths(self, paths: List[str]) -> List[Document]:
        """paths is list which contain paths of file & path of folders"""
        all_docs: List[Document] = []
        for p in paths: 
            path = Path(p)
            if path.is_dir():
                all_docs.extend(self.load_directory(p))
            elif path.is_file():
                all_docs.extend(self.load_file(p))
            else:
                logger.warning(f"Path does not exist, skipping: {p}")
        return all_docs
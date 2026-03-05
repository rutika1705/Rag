"""
Document Loader Module
Handles loading of TXT, PDF, and directory-based documents.
"""

import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyMuPDFLoader,
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Unified document loader supporting .txt and .pdf files,
    both individually and from directories.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file (txt or pdf)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}")

        logger.info(f"Loading file: {file_path}")
        if path.suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            loader = PyMuPDFLoader(str(path))

        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document(s) from {file_path}")
        return docs

    def load_directory(
        self,
        dir_path: str,
        glob: str = "**/*.*",
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load all supported files from a directory."""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        allowed = extensions or list(self.SUPPORTED_EXTENSIONS)
        all_docs: List[Document] = []

        for ext in allowed:
            ext_glob = f"**/*{ext}"
            if ext == ".txt":
                loader = DirectoryLoader(
                    str(path),
                    glob=ext_glob,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    show_progress=False,
                    silent_errors=True,
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
        """Load documents from a mixed list of file/directory paths."""
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
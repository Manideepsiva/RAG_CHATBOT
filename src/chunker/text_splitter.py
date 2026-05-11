from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class PDFChunker:
    """
    Load one or more PDFs and split them into overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Characters shared between consecutive chunks.
        Overlap preserves context across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    
    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        """Load a single PDF; return a list of per-page Documents."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        logger.info("  → %d pages loaded", len(pages))
        return pages

    def load_directory(self, directory: str | Path) -> List[Document]:
        """Load all PDFs from *directory*; return combined page Documents."""
        directory = Path(directory)
        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in: {directory}")

        all_pages: List[Document] = []
        for f in pdf_files:
            all_pages.extend(self.load_pdf(f))

        logger.info(
            "Loaded %d pages from %d PDFs", len(all_pages), len(pdf_files)
        )
        return all_pages

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split Documents into overlapping chunks.

        Each chunk inherits the metadata (source filename, page number)
        of its parent page so citations remain traceable.
        """
        chunks = self._splitter.split_documents(documents)

        for c in chunks:
            c.page_content = c.page_content.strip()

       
        chunks = [c for c in chunks if len(c.page_content) > 20]

        logger.info(
            "Split %d pages → %d chunks (size=%d, overlap=%d)",
            len(documents), len(chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def load_and_split(self, source: str | Path) -> List[Document]:
        """
        Convenience: load PDF(s) from *source* and split into chunks.

        *source* can be a single .pdf file or a directory of PDFs.
        """
        source = Path(source)
        docs   = self.load_directory(source) if source.is_dir() else self.load_pdf(source)
        return self.split(docs)
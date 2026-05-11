
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.chunker import PDFChunker


class TestPDFChunker:
    def setup_method(self):
        self.chunker = PDFChunker(chunk_size=200, chunk_overlap=20)

    def _make_docs(self, n=3):
        return [
            Document(
                page_content="Word " * 80,            
                metadata={"source": f"doc_{i}.pdf", "page": i},
            )
            for i in range(n)
        ]

    def test_split_produces_chunks(self):
        docs   = self._make_docs(2)
        chunks = self.chunker.split(docs)
        assert len(chunks) > len(docs), "Expected more chunks than pages"

    def test_chunks_respect_max_size(self):
        chunks = self.chunker.split(self._make_docs(1))
        for c in chunks:
            assert len(c.page_content) <= self.chunker.chunk_size + 50  
    def test_metadata_preserved(self):
        docs   = self._make_docs(1)
        chunks = self.chunker.split(docs)
        for c in chunks:
            assert "source" in c.metadata
            assert "page" in c.metadata

    def test_empty_chunks_removed(self):
        docs   = [Document(page_content="   ", metadata={})]
        chunks = self.chunker.split(docs)
        assert len(chunks) == 0

    def test_missing_pdf_raises(self):
        with pytest.raises(FileNotFoundError):
            self.chunker.load_pdf("/nonexistent/path/file.pdf")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No PDF files"):
            self.chunker.load_directory(tmp_path)

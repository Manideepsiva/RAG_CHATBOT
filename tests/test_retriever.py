
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.retriever import FAISSRetriever


def _mock_embedder():
    e = MagicMock()
    e.langchain_embeddings = MagicMock()
    return e


def _make_chunks(n=5):
    return [
        Document(
            page_content=f"Chunk content number {i}. " * 10,
            metadata={"source": "test.pdf", "page": i},
        )
        for i in range(n)
    ]


class TestFAISSRetriever:
    def test_build_requires_chunks(self):
        r = FAISSRetriever(embedder=_mock_embedder())
        with pytest.raises(ValueError, match="empty"):
            r.build_index([])

    def test_retrieve_before_load_raises(self):
        r = FAISSRetriever(embedder=_mock_embedder())
        with pytest.raises(RuntimeError, match="no index loaded"):
            r.retrieve("test query")

    def test_threshold_filters_low_scores(self):
        r = FAISSRetriever(
            embedder=_mock_embedder(),
            top_k=3,
            similarity_threshold=0.9,
        )
       
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="chunk A", metadata={}), 10.0),  
            (Document(page_content="chunk B", metadata={}), 0.1),   
        ]
        r._store = mock_store

        results = r.retrieve("test")
        assert len(results) == 1
        assert results[0].page_content == "chunk B"

    def test_similarity_score_in_metadata(self):
        r = FAISSRetriever(
            embedder=_mock_embedder(),
            similarity_threshold=0.0,  
        )
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="x", metadata={}), 1.0),
        ]
        r._store = mock_store
        results = r.retrieve("q")
        assert "similarity_score" in results[0].metadata


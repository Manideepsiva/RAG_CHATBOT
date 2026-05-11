

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.chatbot import RAGChain, RAGResponse


def _mock_retriever(chunks=None):
    r = MagicMock()
    r.retrieve.return_value = chunks if chunks is not None else []
    return r


class TestRAGChain:
    def _chain(self, chunks=None):
        with patch("src.chatbot.rag_chain.ChatOpenAI") as MockLLM:
            mock_llm        = MagicMock()
            mock_llm.invoke = MagicMock(return_value=MagicMock(content="Test answer"))
            MockLLM.return_value = mock_llm
            chain            = RAGChain(retriever=_mock_retriever(chunks))
            chain._llm       = mock_llm
        return chain

    def test_empty_question_returns_prompt(self):
        chain = self._chain()
        resp  = chain.ask("   ")
        assert "Please enter" in resp.answer

    def test_no_context_returns_safe_response(self):
        chain = self._chain(chunks=[])
        resp  = chain.ask("What is the policy?")
        assert "don't have enough information" in resp.answer
        assert resp.source_documents == []

    def test_answer_with_context(self):
        chunks = [
            Document(
                page_content="The refund policy allows 30-day returns.",
                metadata={"source": "policy.pdf", "page": 1, "similarity_score": 0.92},
            )
        ]
        chain = self._chain(chunks=chunks)
        resp  = chain.ask("What is the refund policy?")
        assert resp.answer == "Test answer"
        assert len(resp.source_documents) == 1

    def test_format_sources_deduplicates(self):
        doc = Document(
            page_content="text",
            metadata={"source": "a.pdf", "page": 1, "similarity_score": 0.8},
        )
        resp = RAGResponse(question="q", answer="a", source_documents=[doc, doc])
        lines = resp.format_sources().strip().split("\n")
        assert len(lines) == 1   # deduped

    def test_rag_response_str(self):
        resp = RAGResponse(question="Q?", answer="A.", source_documents=[])
        assert "Q?" in str(resp) and "A." in str(resp)
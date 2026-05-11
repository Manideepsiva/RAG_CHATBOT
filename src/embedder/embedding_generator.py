
from __future__ import annotations

import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates dense vector embeddings using Hugging Face models.

    Parameters
    ----------
    model : str
        Default:
        'sentence-transformers/all-MiniLM-L6-v2'
            → fast, lightweight, 384-dim embeddings
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ) -> None:
        self.model = model

        self._embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=kwargs.get("model_kwargs", {}),
            encode_kwargs=kwargs.get("encode_kwargs", {}),
        )

        logger.info("EmbeddingGenerator ready — model: %s", model)

    

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Underlying LangChain embeddings object.
        Required by FAISS.from_documents() and similar helpers.
        """
        return self._embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of raw strings; returns one vector per string."""
        logger.debug("Embedding %d texts...", len(texts))
        return self._embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self._embeddings.embed_query(query)
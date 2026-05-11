

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from config import TOP_K, SIMILARITY_THRESHOLD, FAISS_INDEX_NAME
from src.embedder import EmbeddingGenerator

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    Wraps a FAISS vector store using cosine similarity.

    Parameters
    ----------
    embedder : EmbeddingGenerator
    top_k : int
        Candidate chunks fetched per query before threshold filtering.
    similarity_threshold : float
        Minimum cosine similarity (0-1) to keep a chunk.
        0.5+ = good match, 0.7+ = strong match.
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.embedder             = embedder
        self.top_k                = top_k
        self.similarity_threshold = similarity_threshold
        self._store: FAISS | None = None

   

    def build_index(self, chunks: List[Document]) -> None:
        """Build a FAISS index using cosine similarity."""
        if not chunks:
            raise ValueError("Cannot build an index")

        logger.info("Building FAISS index from %d chunks…", len(chunks))
        self._store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedder.langchain_embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        logger.info("Index built — %d vectors", self._store.index.ntotal)

    

    def save(self, index_dir: str | Path) -> Path:
        """Save FAISS index + docstore to disk."""
        self._require_store("save")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        save_path = index_dir / FAISS_INDEX_NAME
        self._store.save_local(str(save_path))
        logger.info("FAISS index saved → %s", save_path)
        return save_path

    def load(self, index_dir: str | Path) -> None:
        """Load a previously saved FAISS index from disk."""
        index_path = Path(index_dir) / FAISS_INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"No saved FAISS index at: {index_path}\n"
                "Run `python scripts/ingest.py` first."
            )

        logger.info("Loading FAISS index from %s…", index_path)
        self._store = FAISS.load_local(
            str(index_path),
            embeddings=self.embedder.langchain_embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded — %d vectors", self._store.index.ntotal)



    def retrieve(self, query: str) -> List[Document]:
        """
        Return top-k chunks that pass the cosine similarity threshold.

        Cosine similarity is already 0-1 so no conversion needed.
        Higher score = better match.
        """
        self._require_store("retrieve")

        results: List[Tuple[Document, float]] = (
            self._store.similarity_search_with_score(query, k=self.top_k)
        )

        filtered: List[Document] = []
        for doc, score in results:
            # With COSINE strategy, score is already cosine similarity (0-1)
            # No conversion needed unlike L2
            cosine_score = float(score)
            logger.debug(
                "Chunk score %.4f | text: %s…",
                cosine_score, doc.page_content[:60],
            )
            if cosine_score >= self.similarity_threshold:
                doc.metadata["similarity_score"] = round(cosine_score, 4)
                filtered.append(doc)
            else:
                logger.debug(
                    "Dropped chunk (score %.4f < threshold %.4f)",
                    cosine_score, self.similarity_threshold,
                )

        logger.info(
            "Query '%s…' → %d/%d chunks passed threshold %.2f",
            query[:50], len(filtered), len(results), self.similarity_threshold,
        )
        return filtered

    def as_langchain_retriever(self):
        """LangChain-compatible retriever for use in chains."""
        self._require_store("as_langchain_retriever")
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

    

    def _require_store(self, op: str) -> None:
        if self._store is None:
            raise RuntimeError(
                f"Cannot call '{op}': no index loaded. "
                "Call build_index() or load() first."
            )
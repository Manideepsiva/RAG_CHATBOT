"""
src
---
RAG Document Chatbot — source package.

Sub-packages
------------
chunker   : PDF loading and text splitting
embedder  : OpenAI embedding generation
retriever : FAISS vector store build / query
chatbot   : LangChain RAG chain and CLI
"""

from src.chunker   import PDFChunker
from src.embedder  import EmbeddingGenerator
from src.retriever import FAISSRetriever
from src.chatbot   import RAGChain, RAGResponse

__all__ = [
    "PDFChunker",
    "EmbeddingGenerator",
    "FAISSRetriever",
    "RAGChain",
    "RAGResponse",
]

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from config import (
    LLM_MODEL,
    TEMPERATURE,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
)
from src.retriever import FAISSRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response returned by RAGChain.ask()."""

    question: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)

    def format_sources(self) -> str:
        if not self.source_documents:
            return "No sources retrieved."

        seen, lines = set(), []

        for doc in self.source_documents:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("similarity_score", "?")

            key = (src, page)

            if key not in seen:
                seen.add(key)
                lines.append(f"  • {src}  (page {page}, score {score})")

        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"Q: {self.question}\n\n"
            f"A: {self.answer}\n\n"
            f"Sources:\n{self.format_sources()}"
        )


class RAGChain:
    """
    Orchestrates retrieval + generation for the RAG chatbot.
    """

    _NO_CONTEXT = (
        "I don't have enough information in the provided documents "
        "to answer this question."
    )

    def __init__(
        self,
        retriever: FAISSRetriever,
        llm_model: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
    ) -> None:

        self.retriever = retriever

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )

        self._llm = ChatGroq(
            model=llm_model,
            temperature=temperature,
            api_key=groq_api_key,
        )

        logger.info(
            "RAGChain ready — model: %s, temp: %s",
            llm_model,
            temperature,
        )

    

    def ask(self, question: str) -> RAGResponse:
        """Answer *question* using the full RAG pipeline."""

        question = question.strip()

        if not question:
            return RAGResponse(
                question=question,
                answer="Please enter a question.",
            )

       
        chunks = self.retriever.retrieve(question)

        
        if not chunks:
            logger.info("No chunks above threshold — skipping LLM call.")

            return RAGResponse(
                question=question,
                answer=self._NO_CONTEXT,
            )

       
        context = "\n\n---\n\n".join(
            f"[Chunk {i} | {doc.metadata.get('source', '?')} "
            f"p.{doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content}"
            for i, doc in enumerate(chunks, 1)
        )

        
        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            question=question,
        )

  
        response = self._llm.invoke(prompt)

        return RAGResponse(
            question=question,
            answer=response.content.strip(),
            source_documents=chunks,
        )



    def chat(self) -> None:
        """Start an interactive command-line chat session."""

        print("\n" + "═" * 60)
        print("  RAG Document Chatbot  —  type 'exit' to quit")
        print("═" * 60 + "\n")

        while True:
            try:
                question = input("You: ").strip()

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                sys.exit(0)

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            resp = self.ask(question)

            print(f"\nAssistant: {resp.answer}\n")

            if resp.source_documents:
                print(f"Sources:\n{resp.format_sources()}\n")

            print("─" * 60)
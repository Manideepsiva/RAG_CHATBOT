"""
scripts/chat.py
---------------
CLI entrypoint for the RAG chatbot.

Usage
-----
    python scripts/chat.py                          # interactive loop
    python scripts/chat.py -q "What is the policy?" # single question
    python scripts/chat.py --threshold 0.6 --top_k 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import EmbeddingGenerator
from src.retriever import FAISSRetriever
from src.chatbot import RAGChain
from config import DATA_PROC_DIR

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Document Chatbot")
    p.add_argument("--index_dir", type=Path, default=DATA_PROC_DIR)
    p.add_argument("--question", "-q", type=str, default=None,
                   help="Single question (non-interactive mode)")
    p.add_argument("--top_k",    type=int,   default=None)
    p.add_argument("--threshold",type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    embedder = EmbeddingGenerator()
    kw = {k: v for k, v in
          [("top_k", args.top_k), ("similarity_threshold", args.threshold)]
          if v is not None}
    retriever = FAISSRetriever(embedder=embedder, **kw)
    retriever.load(args.index_dir)

    chain = RAGChain(retriever=retriever)

    if args.question:
        print(chain.ask(args.question))
    else:
        chain.chat()


if __name__ == "__main__":
    main()
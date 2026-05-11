"""
scripts/ingest.py
-----------------
CLI: load PDFs → chunk → embed → build FAISS index → save to disk.

Usage
-----
    python scripts/ingest.py
    python scripts/ingest.py --pdf_dir data/raw/ --index_dir data/processed/
    python scripts/ingest.py --pdf_dir data/raw/report.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import PDFChunker
from src.embedder import EmbeddingGenerator
from src.retriever import FAISSRetriever
from config import DATA_RAW_DIR, DATA_PROC_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs → FAISS index")
    p.add_argument("--pdf_dir",      type=Path, default=DATA_RAW_DIR,
                   help="PDF file or directory (default: data/raw/)")
    p.add_argument("--index_dir",    type=Path, default=DATA_PROC_DIR,
                   help="Where to save the index (default: data/processed/)")
    p.add_argument("--chunk_size",   type=int,  default=None)
    p.add_argument("--chunk_overlap",type=int,  default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    # 1. Chunk
    kw      = {k: v for k, v in
               [("chunk_size", args.chunk_size), ("chunk_overlap", args.chunk_overlap)]
               if v is not None}
    chunks  = PDFChunker(**kw).load_and_split(args.pdf_dir)
    logger.info("Total chunks: %d", len(chunks))

    # 2. Embed + index
    embedder  = EmbeddingGenerator()
    retriever = FAISSRetriever(embedder=embedder)
    retriever.build_index(chunks)

    # 3. Save
    saved = retriever.save(args.index_dir)
    logger.info("Done in %.1fs — index saved to %s",
                time.perf_counter() - t0, saved)


if __name__ == "__main__":
    main()

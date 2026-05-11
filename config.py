

import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


BASE_DIR      = Path(__file__).resolve().parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"


OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.0"))


CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


TOP_K                = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


FAISS_INDEX_NAME = "faiss_index"  


SYSTEM_PROMPT = """You are a helpful assistant that answers questions \
strictly based on the provided document context.

Rules:
- Only use information from the CONTEXT below.
- If the answer is not in the context, say:
  "I don't have enough information to answer this question."
- Cite the source document and page number when possible.
- Be concise and accurate.
"""

RAG_PROMPT_TEMPLATE = """
{system_prompt}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://rag_user:rag_password@localhost:5432/rag_db",
    )

    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM = 384  # must match VECTOR(384) in 01_schema.sql

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 250))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 40))

    TOP_K = int(os.getenv("TOP_K", 5))


config = Config()

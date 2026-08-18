"""
Local embedding model wrapper.

Using sentence-transformers means embeddings are generated locally,
free, and reproducibly -- no API key or network call required for
this step. Only the final answer-generation step needs an API key.
"""
from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.config import config


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]

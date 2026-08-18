"""
Simple word-based chunker with overlap.

A production system might chunk by sentence boundaries or use a proper
tokenizer, but a word-count sliding window is transparent, dependency-light,
and good enough to demonstrate the pipeline end to end.
"""
from __future__ import annotations

from src.config import config


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE_TOKENS
    overlap = overlap or config.CHUNK_OVERLAP_TOKENS

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap  # slide forward, keeping some overlap

    return chunks

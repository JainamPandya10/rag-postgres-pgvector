"""
Ingestion pipeline:

  file on disk -> extract text -> chunk -> embed -> INSERT into Postgres

Run:
    python -m src.ingest --path data/sample_docs
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from tqdm import tqdm
from pypdf import PdfReader

from src.db import get_connection
from src.chunking import chunk_text
from src.embeddings import embed_texts


def extract_text(filepath: Path) -> str:
    if filepath.suffix.lower() == ".pdf":
        reader = PdfReader(str(filepath))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return filepath.read_text(encoding="utf-8", errors="ignore")


def insert_document(conn, title: str, source_path: str, author: str | None, tags: list[str]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (title, source_path, author, published_date, tags)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (title, source_path, author, date.today(), tags),
        )
        return cur.fetchone()[0]


def insert_chunks(conn, document_id: int, chunks: list[str], embeddings: list[list[float]]):
    with conn.cursor() as cur:
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding;
                """,
                (document_id, idx, chunk, len(chunk.split()), emb),
            )


def ingest_path(path: str, author: str | None = None, tags: list[str] | None = None):
    tags = tags or []
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("**/*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in {".txt", ".md", ".pdf"}]

    if not files:
        print(f"No .txt/.md/.pdf files found under {path}")
        return

    with get_connection() as conn:
        for f in tqdm(files, desc="Ingesting documents"):
            text = extract_text(f)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = embed_texts(chunks)

            doc_id = insert_document(conn, title=f.stem, source_path=str(f), author=author, tags=tags)
            insert_chunks(conn, doc_id, chunks, embeddings)

    print(f"Done. Ingested {len(files)} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument("--author", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    args = parser.parse_args()

    ingest_path(args.path, author=args.author, tags=args.tags)

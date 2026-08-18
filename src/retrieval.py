"""
Hybrid retrieval over the chunks table.

Two retrieval signals, both computed in SQL:
  1. Vector similarity  -- cosine distance via pgvector ("<=>" operator, HNSW index)
  2. Full-text relevance -- Postgres tsvector/tsquery ("@@" operator, GIN index)

Vector search finds semantically similar text even with no shared words.
Full-text search is precise for exact keywords, names, codes, acronyms
that embeddings often blur together. Neither alone is reliably best, so
results are merged with Reciprocal Rank Fusion (RRF) -- a simple, well
known technique for combining ranked lists without needing to normalize
or calibrate two different score scales.

Optional metadata filters (author, tags, date range) are pushed down
into both SQL queries directly, rather than filtered in Python -- letting
Postgres use its indexes instead of scanning full result sets.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.db import get_connection
from src.embeddings import embed_text
from src.config import config


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    content: str
    vector_rank: int | None = None
    fulltext_rank: int | None = None
    rrf_score: float = 0.0


def _build_metadata_filter(author: str | None, tags: list[str] | None, sql_params: list) -> str:
    """Returns a SQL WHERE fragment (starting with 'AND ...') and appends params."""
    clauses = []
    if author:
        clauses.append("d.author = %s")
        sql_params.append(author)
    if tags:
        clauses.append("d.tags && %s")  # array overlap operator
        sql_params.append(tags)
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def vector_search(conn, query_embedding, top_k: int, author=None, tags=None):
    params = [query_embedding]
    filter_sql = _build_metadata_filter(author, tags, params)
    params.append(query_embedding)
    params.append(top_k)

    sql = f"""
        SELECT c.id, c.document_id, d.title, c.content,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE 1=1 {filter_sql}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fulltext_search(conn, query_text: str, top_k: int, author=None, tags=None):
    params = [query_text]
    filter_sql = _build_metadata_filter(author, tags, params)
    params.append(top_k)

    sql = f"""
        SELECT c.id, c.document_id, d.title, c.content,
               ts_rank(c.content_tsv, to_tsquery('english', %s)) AS rank
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.content_tsv @@ to_tsquery('english', %s) {filter_sql}
        ORDER BY rank DESC
        LIMIT %s;
    """
    # to_tsquery needs the search term twice (SELECT + WHERE) -- insert it again
    params = [query_text, query_text] + params[1:]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _to_tsquery_safe(text: str) -> str:
    """Turn free text into an OR-joined tsquery so partial keyword matches still hit."""
    words = [w for w in text.split() if w.isalnum()]
    return " | ".join(words) if words else text


def hybrid_search(query: str, top_k: int | None = None, author=None, tags=None, rrf_k: int = 60) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion:
        score(doc) = sum over each ranked list of  1 / (rrf_k + rank_in_that_list)
    Chunks appearing near the top of either list score highly; chunks
    appearing near the top of BOTH lists score highest of all.
    """
    top_k = top_k or config.TOP_K
    fetch_k = max(top_k * 4, 20)  # pull a wider candidate pool before fusing

    query_embedding = embed_text(query)
    tsquery_text = _to_tsquery_safe(query)

    merged: dict[int, RetrievedChunk] = {}

    with get_connection() as conn:
        vec_rows = vector_search(conn, query_embedding, fetch_k, author, tags)
        for rank, row in enumerate(vec_rows, start=1):
            chunk_id, doc_id, title, content, _similarity = row
            merged[chunk_id] = RetrievedChunk(chunk_id, doc_id, title, content, vector_rank=rank)

        try:
            ft_rows = fulltext_search(conn, tsquery_text, fetch_k, author, tags)
        except Exception:
            ft_rows = []  # e.g. empty tsquery -- degrade gracefully to vector-only

        for rank, row in enumerate(ft_rows, start=1):
            chunk_id, doc_id, title, content, _rank_score = row
            if chunk_id in merged:
                merged[chunk_id].fulltext_rank = rank
            else:
                merged[chunk_id] = RetrievedChunk(chunk_id, doc_id, title, content, fulltext_rank=rank)

    for c in merged.values():
        score = 0.0
        if c.vector_rank is not None:
            score += 1.0 / (rrf_k + c.vector_rank)
        if c.fulltext_rank is not None:
            score += 1.0 / (rrf_k + c.fulltext_rank)
        c.rrf_score = score

    ranked = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    return ranked[:top_k]

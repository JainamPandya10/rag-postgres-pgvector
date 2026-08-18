-- ============================================================
-- RAG schema: relational metadata + vector embeddings + full text
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- Documents (the "source" table) ----------
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    source_path     TEXT,
    author          TEXT,
    published_date  DATE,
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- Chunks (the retrieval unit) ----------
-- Embedding dimension = 384 because we use the
-- sentence-transformers/all-MiniLM-L6-v2 model.
-- Change this if you swap embedding models.
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INT NOT NULL,
    content          TEXT NOT NULL,
    token_count      INT,
    embedding        VECTOR(384),
    content_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- ---------- Indexes ----------
-- Vector similarity index (HNSW = fast approximate nearest neighbor search)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search index (classic keyword search, GIN on tsvector)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON chunks USING GIN (content_tsv);

-- Foreign key lookup index
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- Metadata filter indexes (common WHERE clauses: by tag, by date)
CREATE INDEX IF NOT EXISTS idx_documents_tags
    ON documents USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_documents_published_date
    ON documents (published_date);

-- ---------- A view that's convenient for debugging / demos ----------
CREATE OR REPLACE VIEW chunk_details AS
SELECT
    c.id            AS chunk_id,
    c.document_id,
    d.title         AS document_title,
    d.author,
    d.tags,
    c.chunk_index,
    c.content,
    c.created_at
FROM chunks c
JOIN documents d ON d.id = c.document_id;

-- =====================================================================
-- Migration: Upgrade document_embeddings to Gemini Embeddings (768-dim)
-- Run this in your Supabase SQL Editor (https://app.supabase.com/project/_/sql)
-- =====================================================================

-- 1. Drop existing match_documents function that depends on VECTOR(384)
DROP FUNCTION IF EXISTS public.match_documents(VECTOR(384), INT, TEXT);
DROP FUNCTION IF EXISTS public.match_documents(VECTOR, INT, TEXT);

-- 2. Clear old 384-dimensional vectors so they are not mixed with new embeddings
TRUNCATE TABLE public.document_embeddings;

-- 3. Drop old index
DROP INDEX IF EXISTS public.idx_document_embeddings_embedding;

-- 4. Alter column dimension to 768 (standard output dimension of gemini-embedding-001)
ALTER TABLE public.document_embeddings 
    ALTER COLUMN embedding TYPE VECTOR(768);

-- 5. Recreate HNSW cosine distance index for 768 dimensions
CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding 
ON public.document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 6. Create new match_documents function for 768-dimensional query embeddings
CREATE OR REPLACE FUNCTION public.match_documents (
    query_embedding VECTOR(768),
    match_count INT DEFAULT 5,
    filter_repository TEXT DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    repository TEXT,
    document_type TEXT,
    source_id TEXT,
    developer TEXT,
    text TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.repository,
        d.document_type,
        d.source_id,
        d.developer,
        d.text,
        d.metadata,
        ROUND((1 - (d.embedding <=> query_embedding))::numeric, 4)::float AS similarity
    FROM public.document_embeddings d
    WHERE (filter_repository IS NULL OR d.repository = filter_repository)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

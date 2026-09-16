-- =====================================================================
-- Migration: Upgrade document_embeddings from 384-dim to 768-dim
-- Run this in your Supabase SQL Editor (https://app.supabase.com/project/_/sql)
-- =====================================================================

-- 1. Drop existing match_documents function (which references vector(384))
DROP FUNCTION IF EXISTS public.match_documents(vector(384), integer, text);
DROP FUNCTION IF EXISTS public.match_documents(vector, integer, text);
DROP FUNCTION IF EXISTS public.match_documents(vector(768), integer, text);

-- 2. Drop the HNSW vector index first so the column type can be altered
DROP INDEX IF EXISTS public.idx_document_embeddings_embedding;

-- 3. Truncate existing 384-dimensional vectors
TRUNCATE TABLE public.document_embeddings;

-- 4. Alter the embedding column type to VECTOR(768)
ALTER TABLE public.document_embeddings 
    ALTER COLUMN embedding TYPE vector(768);

-- 5. Recreate HNSW cosine distance index for 768 dimensions
CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding 
ON public.document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 6. Create match_documents function accepting query_embedding VECTOR(768)
CREATE OR REPLACE FUNCTION public.match_documents (
    query_embedding vector(768),
    match_count int DEFAULT 5,
    filter_repository text DEFAULT NULL
)
RETURNS TABLE (
    id text,
    repository text,
    document_type text,
    source_id text,
    developer text,
    text text,
    metadata jsonb,
    similarity float
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

-- =====================================================================
-- Migration: Prepare document_embeddings for BAAI/bge-base-en-v1.5 (768-dim)
-- Run this in your Supabase SQL Editor (https://app.supabase.com/project/_/sql)
-- =====================================================================

-- 1. Truncate existing vectors so old model embeddings are not mixed with BGE vectors
TRUNCATE TABLE public.document_embeddings;

-- 2. Verify or ensure the embedding column is VECTOR(768)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'document_embeddings' 
          AND column_name = 'embedding'
    ) THEN
        ALTER TABLE public.document_embeddings ADD COLUMN embedding VECTOR(768);
    ELSE
        ALTER TABLE public.document_embeddings ALTER COLUMN embedding TYPE VECTOR(768);
    END IF;
END $$;

-- 3. Recreate HNSW cosine distance index for 768 dimensions
DROP INDEX IF EXISTS public.idx_document_embeddings_embedding;
CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding 
ON public.document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 4. Create or update match_documents function for 768-dimensional query embeddings
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

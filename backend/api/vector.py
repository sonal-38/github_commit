"""
FastAPI router for Vector Database (Supabase PostgreSQL + pgvector) and Semantic Search endpoints.

Endpoints:
- POST /vector/repositories/{owner}/{repo}/index:
    Loads relational GitHub data from Supabase, builds semantic documents,
    generates embeddings, and upserts them into Supabase PostgreSQL (pgvector).
- GET /vector/search:
    Performs semantic vector search against Supabase pgvector using query embedding.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

from vector.indexer import VectorIndexer
from vector.embeddings import EmbeddingError
from database.supabase_client import (
    SupabaseConfigurationError,
    SupabaseDatabaseError,
)

router = APIRouter(
    prefix="/vector",
    tags=["Vector Database & Semantic Search"],
)


class IndexStats(BaseModel):
    commits: int = Field(0, description="Number of commit documents indexed")
    commit_files: int = Field(0, description="Number of commit file documents indexed")
    pull_requests: int = Field(0, description="Number of pull request documents indexed")
    reviews: int = Field(0, description="Number of code review documents indexed")
    review_comments: int = Field(0, description="Number of review comment documents indexed")
    issues: int = Field(0, description="Number of issue documents indexed")
    issue_comments: int = Field(0, description="Number of issue comment documents indexed")
    changed_files: int = Field(0, description="Number of changed file documents indexed")


class VectorIndexResponse(BaseModel):
    repository: str
    indexed: IndexStats
    total_vectors: int
    new_vectors: Optional[int] = None
    message: Optional[str] = None


class SearchResultItem(BaseModel):
    score: float = Field(..., description="Cosine similarity score (0 to 1)")
    document_type: str = Field(..., description="Category (commit, pull_request, issue, etc.)")
    repository: str
    developer: str
    source_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


@router.post(
    "/repositories/{owner}/{repo}/index",
    response_model=VectorIndexResponse,
    summary="Index repository records from Supabase into pgvector",
    description=(
        "Reads normalized repository records from Supabase, transforms them into domain-rich "
        "semantic documents, generates vector embeddings, and idempotently upserts them into "
        "Supabase PostgreSQL using the pgvector extension."
    ),
)
def index_repository_vectors(
    owner: str = Path(..., description="Repository owner login (e.g. sonal-38)"),
    repo: str = Path(..., description="Repository name (e.g. smart-payment-platform)"),
    incremental: bool = Query(
        True,
        description="When True, only embed and upsert new documents not already indexed. Set to False for full re-index.",
    ),
):
    try:
        indexer = VectorIndexer()
        result = indexer.index_repository(owner=owner, repo=repo, incremental=incremental)
        return result
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during vector indexing: {str(e)}",
        )


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic similarity search across indexed repository documents",
    description=(
        "Converts the search query into a dense vector embedding and queries Supabase PostgreSQL "
        "using pgvector cosine similarity retrieval."
    ),
)
def search_vectors(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    repository: Optional[str] = Query(
        None,
        description="Optional repository filter (e.g. sonal-38/smart-payment-platform)",
    ),
    limit: int = Query(
        5,
        ge=1,
        le=50,
        description="Maximum number of top matching documents to return",
    ),
):
    try:
        indexer = VectorIndexer()
        result = indexer.search(query=q, repository=repository, limit=limit)
        return result
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during semantic search: {str(e)}",
        )

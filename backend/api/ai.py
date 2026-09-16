"""
FastAPI router for AI & RAG (Retrieval-Augmented Generation) endpoints.

Endpoints:
- POST /ai/ask:
    Takes a question and optional repository name. Automatically parses chronological
    constraints and performs exact GitHub event date filtering against Supabase PostgreSQL,
    or executes pgvector semantic search, and invokes Gemini to synthesize a grounded answer.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai.rag_service import RAGService
from ai.gemini_service import GeminiConfigurationError, GeminiAPIError
from database.supabase_client import SupabaseConfigurationError, SupabaseDatabaseError
from vector.embeddings import EmbeddingError

router = APIRouter(
    prefix="/ai",
    tags=["AI & RAG Assistant"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="Engineering question about the codebase or repository history")
    repository: Optional[str] = Field(None, description="Repository identifier (e.g. 'owner/repo')")
    top_k: Optional[int] = Field(5, ge=1, le=25, description="Number of evidence documents to retrieve (1-25)")


class SourceItem(BaseModel):
    document_type: str = Field(..., description="Type of evidence document (commit, pull_request, issue, etc.)")
    source_id: str = Field(..., description="Unique source identifier (commit SHA, PR #, issue #, file path)")
    developer: Optional[str] = Field(None, description="Author or developer responsible")
    date: Optional[str] = Field(None, description="Original GitHub event timestamp")


class FilterMetadata(BaseModel):
    date_start: Optional[str] = Field(None, description="Start date timestamp filter (ISO 8601 UTC)")
    date_end: Optional[str] = Field(None, description="End date timestamp filter (ISO 8601 UTC)")
    target_entity: Optional[str] = Field(None, description="Identified GitHub object table (commits, pull_requests, etc.)")
    date_field: Optional[str] = Field(None, description="Exact GitHub event timestamp column used")
    is_transition: Optional[bool] = Field(None, description="Whether chronological sequence reasoning is enabled")
    relative_keyword: Optional[str] = Field(None, description="Relative time constraint (e.g. 'recent', 'latest')")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    filters: Optional[FilterMetadata] = Field(None, description="Chronological filters applied to query if detected")


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the repository using Gemini + date-aware pgvector RAG",
    description=(
        "Executes a date-aware grounded RAG query: automatically detects chronological boundaries "
        "(e.g., 'after August 1, 2026', 'August 2026', 'transition'), performs exact Supabase event "
        "timestamp filtering or pgvector semantic search, orders evidence chronologically when needed, "
        "and prompts Gemini to answer strictly based on repository facts, citing verifiable sources."
    ),
)
def ask_question(request: AskRequest) -> AskResponse:
    clean_q = request.question.strip()
    if not clean_q:
        raise HTTPException(status_code=400, detail="The 'question' field cannot be empty.")

    try:
        rag = RAGService(default_top_k=request.top_k or 5)
        response_dict = rag.answer_question(
            question=clean_q,
            repository=request.repository,
            top_k=request.top_k,
        )
        return AskResponse(**response_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except GeminiConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except GeminiAPIError as e:
        raise HTTPException(status_code=502, detail=f"Gemini AI Service Error: {str(e)}")
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal RAG pipeline error: {str(e)}")

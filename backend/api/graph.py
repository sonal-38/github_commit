"""
FastAPI router for Neo4j Knowledge Graph endpoints.

Provides:
- POST /graph/repositories/{owner}/{repo}/build: Idempotently builds the Neo4j graph from Supabase data.
- GET /graph/repositories/{owner}/{repo}/summary: Returns verification counts of nodes and relationships from Neo4j.
"""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from graph.builder import KnowledgeGraphBuilder
from database.neo4j_client import (
    Neo4jConfigurationError,
    Neo4jConnectionError,
    Neo4jExecutionError,
)
from database.supabase_client import (
    SupabaseConfigurationError,
    SupabaseDatabaseError,
)

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


class NodeCounts(BaseModel):
    repositories: int = Field(..., description="Number of Repository nodes")
    developers: int = Field(..., description="Number of Developer nodes")
    commits: int = Field(..., description="Number of Commit nodes")
    pull_requests: int = Field(..., description="Number of PullRequest nodes")
    reviews: int = Field(..., description="Number of Review nodes")
    review_comments: int = Field(..., description="Number of ReviewComment nodes")
    issues: int = Field(..., description="Number of Issue nodes")
    issue_comments: int = Field(..., description="Number of IssueComment nodes")
    files: int = Field(..., description="Number of File nodes")
    commit_files: int = Field(0, description="Number of CommitFile nodes")


class GraphBuildResponse(BaseModel):
    repository: str
    full_name: str
    graph_build_status: str = "success"
    nodes_created_or_updated: NodeCounts
    relationships_created_or_updated: int


class GraphSummaryResponse(BaseModel):
    repository: str
    full_name: str
    status: str
    nodes: NodeCounts
    relationships_count: int


@router.post(
    "/repositories/{owner}/{repo}/build",
    response_model=GraphBuildResponse,
    summary="Build Neo4j Knowledge Graph from Supabase data",
    description=(
        "Reads normalized GitHub entities from Supabase PostgreSQL and constructs an "
        "idempotent Neo4j Knowledge Graph connecting repositories, developers, commits, "
        "pull requests, reviews, comments, issues, and changed files."
    ),
)
def build_repository_graph(
    owner: str = Path(..., description="GitHub repository owner (e.g. 'sonal-38')"),
    repo: str = Path(..., description="GitHub repository name (e.g. 'smart-payment-platform')"),
):
    try:
        builder = KnowledgeGraphBuilder()
        result = builder.build_repository_graph(owner=owner, repo=repo)
        return result
    except Neo4jConfigurationError as cfg_err:
        raise HTTPException(status_code=400, detail=str(cfg_err))
    except Neo4jConnectionError as conn_err:
        raise HTTPException(status_code=conn_err.status_code, detail=conn_err.message)
    except Neo4jExecutionError as exec_err:
        raise HTTPException(status_code=exec_err.status_code, detail=exec_err.message)
    except SupabaseConfigurationError as s_cfg_err:
        raise HTTPException(status_code=400, detail=str(s_cfg_err))
    except SupabaseDatabaseError as s_db_err:
        detail_msg = s_db_err.message
        if s_db_err.details and s_db_err.details != s_db_err.message:
            detail_msg = f"{s_db_err.message} [Details: {s_db_err.details}]"
        raise HTTPException(status_code=s_db_err.status_code, detail=detail_msg)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while building the knowledge graph: {str(e)}",
        )


@router.get(
    "/repositories/{owner}/{repo}/summary",
    response_model=GraphSummaryResponse,
    summary="Get Neo4j Knowledge Graph summary and node counts",
    description="Inspects Neo4j Aura directly to verify node counts and relationship count for the repository.",
)
def get_graph_summary(
    owner: str = Path(..., description="GitHub repository owner"),
    repo: str = Path(..., description="GitHub repository name"),
):
    try:
        builder = KnowledgeGraphBuilder()
        return builder.get_repository_summary(owner=owner, repo=repo)
    except Neo4jConfigurationError as cfg_err:
        raise HTTPException(status_code=400, detail=str(cfg_err))
    except Neo4jConnectionError as conn_err:
        raise HTTPException(status_code=conn_err.status_code, detail=conn_err.message)
    except Neo4jExecutionError as exec_err:
        raise HTTPException(status_code=exec_err.status_code, detail=exec_err.message)
    except SupabaseDatabaseError as s_db_err:
        raise HTTPException(status_code=s_db_err.status_code, detail=s_db_err.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred retrieving graph summary: {str(e)}",
        )

"""
FastAPI router for Knowledge Discovery and HDBSCAN Clustering.

Endpoints:
- POST /knowledge/repositories/{owner}/{repo}/cluster:
    Constructs knowledge documents from Supabase GitHub evidence,
    generates vectors via the existing BGE embedding model, and
    discovers semantic clusters using HDBSCAN.
- GET /knowledge/repositories/{owner}/{repo}/documents:
    Previews constructed knowledge documents and type distributions
    prior to clustering.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

from knowledge.clustering import (
    load_knowledge_documents,
    run_knowledge_clustering,
)
from knowledge.interpretation import (
    run_knowledge_interpretation,
)
from database.supabase_client import (
    SupabaseConfigurationError,
    SupabaseDatabaseError,
)
from database.neo4j_client import (
    Neo4jConfigurationError,
    Neo4jConnectionError,
    Neo4jExecutionError,
)
from graph.builder import KnowledgeGraphBuilder
from vector.embeddings import EmbeddingError

router = APIRouter(
    prefix="/knowledge",
    tags=["Knowledge Discovery & Clustering"],
)


class DocumentSummary(BaseModel):
    id: str
    document_type: str
    source_id: str
    preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClusterGroup(BaseModel):
    cluster_id: int = Field(..., description="Cluster identifier (>=0 for clusters, -1 for noise)")
    document_count: int = Field(..., description="Number of documents in this cluster")
    document_type_counts: Dict[str, int] = Field(default_factory=dict, description="Counts by document type")
    documents: List[DocumentSummary] = Field(default_factory=list, description="Document previews in cluster")


class ClusterResponse(BaseModel):
    status: str = Field(..., description="'completed', 'insufficient_data', or 'no_data'")
    repository: str
    document_count: int
    cluster_count: int = 0
    noise_count: int = 0
    noise_documents: Optional[int] = None
    minimum_required: Optional[int] = None
    message: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    cluster_summary: Dict[str, Any] = Field(default_factory=dict)
    clusters: List[ClusterGroup] = Field(default_factory=list)


class ClusterInterpretation(BaseModel):
    cluster_id: int = Field(..., description="HDBSCAN cluster identifier (>= 0)")
    candidate_label: str = Field(..., description="Evidence-derived candidate knowledge area name")
    document_count: int = Field(..., description="Number of documents in this cluster")
    representative_documents: List[str] = Field(
        default_factory=list,
        description="Top representative document IDs with highest membership probability",
    )
    representative_terms: List[str] = Field(
        default_factory=list,
        description="Dominant domain terms extracted from commits, PRs, and file paths",
    )
    important_files: List[str] = Field(
        default_factory=list,
        description="Ranked repeated/important file paths associated with this cluster",
    )
    all_document_ids: List[str] = Field(
        default_factory=list,
        description="All document IDs in this cluster for complete traceability",
    )
    evidence_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary of document types, PR authors, and commit authors",
    )


class InterpretationResponse(BaseModel):
    status: str = Field(..., description="'completed', 'insufficient_data', or 'no_data'")
    repository: str
    clusters: List[ClusterInterpretation] = Field(default_factory=list)
    noise_documents: int = Field(0, description="Count of noise documents (cluster_id = -1)")
    noise_document_ids: List[str] = Field(default_factory=list, description="IDs of noise documents")
    total_documents: Optional[int] = Field(None, description="Total evidence documents analyzed")


class KnowledgeAreaGraphResponse(BaseModel):
    repository: str = Field(..., description="Repository full name (owner/repo)")
    status: str = Field(..., description="Status of the knowledge area graph operation")
    knowledge_areas: int = Field(..., description="Count of KnowledgeArea nodes in the repository")
    evidence_relationships: int = Field(..., description="Count of EVIDENCED_BY relationships created or existing")
    developers_reached_through_evidence: int = Field(
        ..., description="Count of distinct developers associated through evidence traceability"
    )


@router.post(
    "/repositories/{owner}/{repo}/cluster",
    response_model=ClusterResponse,
    summary="Discover knowledge clusters from repository evidence via HDBSCAN",
    description=(
        "Extracts GitHub evidence from Supabase, formats standardized Knowledge Documents "
        "(Commit, CommitFile, PR, PRFile), generates embeddings using BGE, and applies "
        "HDBSCAN clustering to discover semantic engineering groupings."
    ),
)
def cluster_repository_knowledge(
    owner: str = Path(..., description="Repository owner (e.g. sonal-38)"),
    repo: str = Path(..., description="Repository name (e.g. smart-payment-platform)"),
    min_cluster_size: int = Query(5, ge=2, description="Minimum size of clusters for HDBSCAN"),
    min_samples: int = Query(3, ge=1, description="HDBSCAN min_samples parameter"),
    metric: str = Query("euclidean", description="Distance metric (e.g. 'euclidean')"),
    cluster_selection_method: str = Query("eom", description="'eom' or 'leaf'"),
):
    print(
        f"\n>>> [API REQUEST] POST /knowledge/repositories/{owner}/{repo}/cluster "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='{metric}', method='{cluster_selection_method}')",
        flush=True,
    )
    try:
        result = run_knowledge_clustering(
            owner=owner,
            repo=repo,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        return result
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during knowledge clustering: {str(e)}",
        )


@router.get(
    "/repositories/{owner}/{repo}/documents",
    summary="Inspect constructed knowledge documents for a repository",
    description="Loads evidence from Supabase and returns the constructed KnowledgeDocument list.",
)
def inspect_knowledge_documents(
    owner: str = Path(..., description="Repository owner"),
    repo: str = Path(..., description="Repository name"),
):
    print(f"\n>>> [API REQUEST] GET /knowledge/repositories/{owner}/{repo}/documents", flush=True)
    try:
        docs = load_knowledge_documents(owner=owner, repo=repo)
        type_counts = {}
        for d in docs:
            type_counts[d.document_type] = type_counts.get(d.document_type, 0) + 1

        return {
            "repository": f"{owner}/{repo}",
            "total_documents": len(docs),
            "document_type_counts": type_counts,
            "documents": [
                {
                    "id": d.id,
                    "type": d.document_type,
                    "document_type": d.document_type,
                    "source_id": d.source_id,
                    "timestamp": d.timestamp,
                    "contained_commit_ids": d.contained_commit_ids,
                    "text_length": len(d.text),
                    "metadata": d.metadata,
                }
                for d in docs
            ],
        }
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/repositories/{owner}/{repo}/interpret",
    response_model=InterpretationResponse,
    summary="Interpret HDBSCAN clusters into candidate knowledge areas",
    description=(
        "Consumes Step 10 semantic clustering outputs and deterministically interprets each "
        "cluster using commit messages, PR titles/bodies, contained commits, and important file paths. "
        "Preserves PR vs commit author distinction and complete traceability."
    ),
)
def interpret_repository_knowledge(
    owner: str = Path(..., description="Repository owner (e.g. sonal-38)"),
    repo: str = Path(..., description="Repository name (e.g. smart_payment_platform)"),
    min_cluster_size: int = Query(5, ge=2, description="Minimum size of clusters for HDBSCAN"),
    min_samples: int = Query(3, ge=1, description="HDBSCAN min_samples parameter"),
    metric: str = Query("euclidean", description="Distance metric (e.g. 'euclidean')"),
    cluster_selection_method: str = Query("eom", description="'eom' or 'leaf'"),
):
    print(
        f"\n>>> [API REQUEST] POST /knowledge/repositories/{owner}/{repo}/interpret "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})",
        flush=True,
    )
    try:
        result = run_knowledge_interpretation(
            owner=owner,
            repo=repo,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        return result
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during knowledge interpretation: {str(e)}",
        )


@router.post(
    "/repositories/{owner}/{repo}/graph",
    response_model=KnowledgeAreaGraphResponse,
    summary="Connect interpreted Knowledge Areas to the existing Neo4j graph",
    description=(
        "Consumes Step 11 interpreted Knowledge Areas and links them into the existing Neo4j "
        "graph via (:KnowledgeArea)-[:EVIDENCED_BY]->(:Commit|:PullRequest|:CommitFile). "
        "Preserves existing graph structure and provides multi-developer traceability without direct KNOWS edges."
    ),
)
def connect_knowledge_areas_to_graph(
    owner: str = Path(..., description="Repository owner (e.g. sonal-38)"),
    repo: str = Path(..., description="Repository name (e.g. smart-payment-platform)"),
    min_cluster_size: int = Query(5, ge=2, description="Minimum size of clusters for HDBSCAN"),
    min_samples: int = Query(3, ge=1, description="HDBSCAN min_samples parameter"),
    metric: str = Query("euclidean", description="Distance metric (e.g. 'euclidean')"),
    cluster_selection_method: str = Query("eom", description="'eom' or 'leaf'"),
):
    print(
        f"\n>>> [API REQUEST] POST /knowledge/repositories/{owner}/{repo}/graph "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})",
        flush=True,
    )
    try:
        builder = KnowledgeGraphBuilder()
        result = builder.build_knowledge_area_graph(
            owner=owner,
            repo=repo,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        return result
    except SupabaseConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except SupabaseDatabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Neo4jConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Neo4jConnectionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Neo4jExecutionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while attaching knowledge areas to graph: {str(e)}",
        )



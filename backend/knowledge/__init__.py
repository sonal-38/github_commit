"""Knowledge extraction, clustering, and interpretation module."""
from knowledge.clustering import (
    KnowledgeDocument,
    load_knowledge_documents,
    build_embeddings,
    cluster_embeddings,
    run_knowledge_clustering,
)
from knowledge.interpretation import (
    extract_file_paths,
    extract_representative_terms,
    select_representative_documents,
    generate_candidate_label,
    extract_evidence_summary,
    interpret_clusters,
    run_knowledge_interpretation,
)

__all__ = [
    "KnowledgeDocument",
    "load_knowledge_documents",
    "build_embeddings",
    "cluster_embeddings",
    "run_knowledge_clustering",
    "extract_file_paths",
    "extract_representative_terms",
    "select_representative_documents",
    "generate_candidate_label",
    "extract_evidence_summary",
    "interpret_clusters",
    "run_knowledge_interpretation",
]

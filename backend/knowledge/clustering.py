"""
Knowledge Document Construction and HDBSCAN Clustering Pipeline.

Pipeline:
    Supabase GitHub data
            ↓
    Knowledge documents (Commit, CommitFile, PR, PRFile)
            ↓
    Existing BGE embedding system (BAAI/bge-base-en-v1.5)
            ↓
    Embedding vectors (768-dim normalized)
            ↓
    HDBSCAN clustering (sklearn.cluster.HDBSCAN)
            ↓
    Cluster labels (0, 1, 2, ..., -1 for noise)
            ↓
    Cluster summary/report

This module is strictly for semantic clustering discovery.
It does NOT create KnowledgeArea nodes, does NOT score developer expertise,
and does NOT modify Supabase tables, Neo4j, or RAG.
"""
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from database.supabase_client import SupabaseClient, SupabaseDatabaseError, get_supabase_client
from vector.embeddings import EmbeddingService, EmbeddingError

logger = logging.getLogger(__name__)


class KnowledgeDocument:
    """
    Internal representation of an engineering knowledge document derived from GitHub evidence.
    Keeps source identity, contained commits, and original event timestamps fully intact.
    """
    def __init__(
        self,
        id: str,
        document_type: str = "",
        repository: str = "",
        source_id: str = "",
        text: str = "",
        timestamp: Optional[str] = None,
        contained_commit_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        type: Optional[str] = None,
        cluster_id: Optional[int] = None,
        cluster_probability: Optional[float] = None,
    ):
        self.id = id
        self.document_type = type or document_type
        self.repository = repository
        self.source_id = source_id
        self.text = text
        self.timestamp = timestamp
        self.contained_commit_ids = contained_commit_ids or []
        self.metadata = metadata or {}
        self.cluster_id = cluster_id
        self.cluster_probability = cluster_probability

    @property
    def type(self) -> str:
        return self.document_type

    @type.setter
    def type(self, val: str):
        self.document_type = val

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.document_type,
            "document_type": self.document_type,
            "repository": self.repository,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.contained_commit_ids:
            d["contained_commit_ids"] = self.contained_commit_ids
        if self.cluster_id is not None:
            d["cluster_id"] = self.cluster_id
        if self.cluster_probability is not None:
            d["cluster_probability"] = self.cluster_probability
        return d


def format_commit_document(
    commit: Dict[str, Any],
    repo_name: str,
    dev_map: Dict[int, Dict[str, Any]],
    commit_files: Optional[List[Dict[str, Any]]] = None,
) -> Optional[KnowledgeDocument]:
    """Constructs a knowledge document from a Git commit record."""
    sha = (commit.get("sha") or "").strip()
    if not sha:
        return None

    dev_id = commit.get("developer_id")
    dev = dev_map.get(dev_id, {}) if dev_id else {}
    author_login = dev.get("login") or dev.get("name") or commit.get("author_login") or "Unknown"
    message = (commit.get("message") or "").strip()
    committed_at = (commit.get("committed_at") or commit.get("date") or "").strip()

    text_parts = [
        f"Repository: {repo_name}",
        "",
        "Commit:",
        f"{message if message else 'No commit message provided.'}",
        "",
        "Author:",
        f"{author_login}",
        "",
        "Commit SHA:",
        f"{sha}",
    ]
    if committed_at:
        text_parts.extend(["", "Committed at:", committed_at])

    if commit_files:
        filenames = [cf.get("filename") for cf in commit_files if cf.get("filename")]
        if filenames:
            text_parts.extend(["", "Changed Files:"] + [f"- {fn}" for fn in filenames[:15]])
            if len(filenames) > 15:
                text_parts.append(f"... and {len(filenames) - 15} more files")

    text = "\n".join(text_parts).strip()
    doc_id = f"commit:{repo_name}:{sha}"

    commit_meta = {
        "commit_sha": sha,
        "canonical_id": f"commit:{sha}",
        "author_login": author_login,
        "committed_at": committed_at,
        "html_url": commit.get("html_url") or "",
    }
    if message:
        commit_meta["message"] = message
    if commit_files:
        commit_meta["changed_files"] = [cf.get("filename") for cf in commit_files if cf.get("filename")]

    return KnowledgeDocument(
        id=doc_id,
        document_type="commit",
        repository=repo_name,
        source_id=sha,
        text=text,
        timestamp=committed_at if committed_at else None,
        metadata=commit_meta,
    )


def format_commit_file_document(
    cf: Dict[str, Any],
    repo_name: str,
    commit_info: Optional[Dict[str, Any]] = None,
    dev_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Optional[KnowledgeDocument]:
    """Constructs a knowledge document for a file changed in a specific commit."""
    filename = (cf.get("filename") or "").strip()
    commit_sha = (cf.get("commit_sha") or "").strip()
    if not commit_sha and commit_info:
        commit_sha = (commit_info.get("sha") or "").strip()

    if not filename or not commit_sha:
        return None

    commit_info = commit_info or {}
    dev_map = dev_map or {}

    dev_name = "Unknown"
    dev_id = commit_info.get("developer_id")
    if dev_id and dev_id in dev_map:
        dev_obj = dev_map[dev_id]
        dev_name = dev_obj.get("login") or dev_obj.get("name") or dev_name
    elif commit_info.get("author_login"):
        dev_name = commit_info.get("author_login")
    elif cf.get("developer"):
        dev_name = cf.get("developer")

    commit_message = (commit_info.get("message") or "").strip()
    committed_at = (commit_info.get("committed_at") or commit_info.get("date") or "").strip()

    status = cf.get("status") or "modified"
    additions = cf.get("additions") if cf.get("additions") is not None else 0
    deletions = cf.get("deletions") if cf.get("deletions") is not None else 0
    changes = cf.get("changes") if cf.get("changes") is not None else (additions + deletions)
    patch = (cf.get("patch") or "").strip()
    if len(patch) > 1500:
        patch = patch[:1500] + "\n... [diff truncated]"

    text_parts = [
        f"Repository:\n{repo_name}",
        "",
        f"Commit:\n{commit_sha}",
    ]
    if commit_message:
        text_parts.extend(["", f"Commit Message:\n{commit_message}"])
    text_parts.extend([
        "",
        f"Changed file:\n{filename}",
        "",
        f"Status:\n{status}",
        "",
        f"Changes:\n+{additions} additions, -{deletions} deletions ({changes} changes)",
    ])
    if patch:
        text_parts.extend(["", f"Patch:\n{patch}"])

    text = "\n".join(text_parts).strip()
    doc_id = f"commit_file:{repo_name}:{commit_sha}:{filename}"
    source_id = f"{commit_sha}:{filename}"

    return KnowledgeDocument(
        id=doc_id,
        document_type="commit_file",
        repository=repo_name,
        source_id=source_id,
        text=text,
        timestamp=committed_at if committed_at else None,
        metadata={
            "commit_sha": commit_sha,
            "filename": filename,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "changes": changes,
            "author_login": dev_name,
            "committed_at": committed_at,
            "blob_url": cf.get("blob_url") or "",
        },
    )


def format_pull_request_document(
    pr: Dict[str, Any],
    repo_name: str,
    dev_map: Dict[int, Dict[str, Any]],
    pr_files: Optional[List[Dict[str, Any]]] = None,
    contained_commits: Optional[List[Dict[str, Any]]] = None,
) -> Optional[KnowledgeDocument]:
    """
    Constructs a knowledge document from a Pull Request record.
    Includes PR-level metadata and actual contained commit summaries
    while keeping PR author and commit authors strictly separated.
    """
    pr_number = pr.get("github_pr_number") or pr.get("number")
    if pr_number is None:
        return None

    dev_id = pr.get("developer_id")
    dev = dev_map.get(dev_id, {}) if dev_id else {}
    author_login = dev.get("login") or dev.get("name") or pr.get("author_login") or "Unknown"

    title = (pr.get("title") or "").strip()
    body = (pr.get("body") or "").strip()
    state = pr.get("state") or "unknown"
    created_at = (pr.get("created_at") or "").strip()
    updated_at = (pr.get("updated_at") or "").strip()
    closed_at = (pr.get("closed_at") or "").strip()
    merged_at = (pr.get("merged_at") or "").strip()
    html_url = (pr.get("html_url") or "").strip()

    text_parts = [
        f"Pull Request:\n#{pr_number}",
        "",
        f"Repository:\n{repo_name}",
        "",
        f"Title:\n{title}",
        "",
        f"Description:\n{body if body else '(No description provided)'}",
        "",
        f"Author:\n{author_login}",
        "",
        f"State:\n{state}",
    ]
    if created_at:
        text_parts.extend(["", f"Created:\n{created_at}"])
    if merged_at:
        text_parts.extend(["", f"Merged at:\n{merged_at}"])
    elif closed_at:
        text_parts.extend(["", f"Closed at:\n{closed_at}"])

    contained_commit_ids: List[str] = []
    if contained_commits:
        text_parts.extend(["", "Contained commits:"])
        for c in contained_commits[:25]:
            c_sha = (c.get("sha") or "").strip()
            if not c_sha:
                continue
            c_author = c.get("author_login") or c.get("author_name") or c.get("author") or "Unknown"
            c_msg = (c.get("message") or "").split("\n")[0].strip() or "No commit message provided"
            c_date = (c.get("date") or c.get("committed_at") or "").strip()
            short_sha = c_sha[:8]
            date_str = f" ({c_date})" if c_date else ""
            text_parts.append(f"- {short_sha} — {c_author} — {c_msg}{date_str}")
            contained_commit_ids.append(f"commit:{c_sha}")

        if len(contained_commits) > 25:
            text_parts.append(f"... and {len(contained_commits) - 25} more commits")

    if pr_files:
        file_names = [f.get("filename") for f in pr_files if f.get("filename")]
        if file_names:
            text_parts.extend(["", "PR changed files:"] + [f"- {fn}" for fn in file_names[:15]])
            if len(file_names) > 15:
                text_parts.append(f"... and {len(file_names) - 15} more files")

    text = "\n".join(text_parts).strip()
    doc_id = f"pr:{repo_name}:{pr_number}"

    return KnowledgeDocument(
        id=doc_id,
        document_type="pull_request",
        repository=repo_name,
        source_id=str(pr_number),
        text=text,
        timestamp=created_at if created_at else None,
        contained_commit_ids=contained_commit_ids,
        metadata={
            "pr_number": pr_number,
            "title": title,
            "body": body,
            "author_login": author_login,
            "state": state,
            "created_at": created_at,
            "updated_at": updated_at,
            "closed_at": closed_at,
            "merged_at": merged_at,
            "html_url": html_url,
            "contained_commit_ids": contained_commit_ids,
            "contained_commits": contained_commits or [],
            "changed_files": file_names if pr_files else [],
            "commit_count": len(contained_commits) if contained_commits else 0,
        },
    )


def format_pr_file_document(
    cf: Dict[str, Any],
    pr_number: int,
    repo_name: str,
    timestamp: Optional[str] = None,
) -> Optional[KnowledgeDocument]:
    """Constructs a knowledge document for a file changed in a pull request."""
    filename = (cf.get("filename") or "").strip()
    if not filename:
        return None

    status = cf.get("status") or "modified"
    additions = cf.get("additions") if cf.get("additions") is not None else 0
    deletions = cf.get("deletions") if cf.get("deletions") is not None else 0
    changes = cf.get("changes") if cf.get("changes") is not None else (additions + deletions)
    patch = (cf.get("patch") or "").strip()
    if len(patch) > 1500:
        patch = patch[:1500] + "\n... [diff truncated]"

    text_parts = [
        f"Repository:\n{repo_name}",
        "",
        f"Pull Request:\n#{pr_number}",
        "",
        f"Changed File:\n{filename}",
        "",
        f"Change Status:\n{status}",
        "",
        f"Stats:\n+{additions} / -{deletions} ({changes} changes)",
    ]
    if timestamp:
        text_parts.extend(["", f"Timestamp:\n{timestamp}"])
    if patch:
        text_parts.extend(["", f"Patch:\n{patch}"])

    text = "\n".join(text_parts).strip()
    doc_id = f"pr_file:{repo_name}:{pr_number}:{filename}"
    source_id = f"{pr_number}:{filename}"

    return KnowledgeDocument(
        id=doc_id,
        document_type="pr_file",
        repository=repo_name,
        source_id=source_id,
        text=text,
        timestamp=timestamp,
        metadata={
            "pr_number": pr_number,
            "filename": filename,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "changes": changes,
            "blob_url": cf.get("blob_url") or "",
        },
    )


def load_knowledge_documents(
    owner: str,
    repo: str,
    supabase_client: Optional[SupabaseClient] = None,
    github_client: Optional[Any] = None,
    pr_commits_map: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> List[KnowledgeDocument]:
    """
    Extracts GitHub evidence from Supabase and transforms records into
    normalized KnowledgeDocument objects.
    Preserves actual PR -> Commit relationships using the official GitHub commits endpoint,
    maintains strictly separate PR author and Commit author identities,
    and preserves original GitHub event timestamps.
    """
    sb = supabase_client or get_supabase_client()
    owner_clean = owner.strip()
    repo_clean = repo.strip()
    full_name = f"{owner_clean}/{repo_clean}"

    # 1. Look up repository record
    repos = sb.select("repositories", {"full_name": f"eq.{full_name}"})
    if not repos:
        repos = sb.select(
            "repositories",
            {"name": f"eq.{repo_clean}", "owner_login": f"eq.{owner_clean}"},
        )
    if not repos:
        alt_repo = repo_clean.replace("_", "-") if "_" in repo_clean else repo_clean.replace("-", "_")
        alt_owner = owner_clean.replace("_", "-") if "_" in owner_clean else owner_clean.replace("-", "_")
        repos = sb.select(
            "repositories",
            {"name": f"eq.{alt_repo}", "owner_login": f"eq.{alt_owner}"},
        )

    if not repos:
        raise SupabaseDatabaseError(
            f"Repository '{full_name}' not found in Supabase. Ingest repository first.",
            status_code=404,
        )

    repo_record = repos[0]
    repo_id = repo_record.get("id")
    resolved_full_name = repo_record.get("full_name") or full_name
    resolved_repo_name = repo_record.get("name") or repo_clean

    # 2. Extract records from Supabase tables
    commits = sb.select("commits", {"repository_id": f"eq.{repo_id}"})
    prs = sb.select("pull_requests", {"repository_id": f"eq.{repo_id}"})
    developers_list = sb.select("developers")
    dev_map: Dict[int, Dict[str, Any]] = {d["id"]: d for d in developers_list if "id" in d}

    # Fetch commit files
    commit_files = sb.select("commit_files", {"repository": f"eq.{resolved_full_name}"})
    if not commit_files and resolved_full_name != full_name:
        commit_files = sb.select("commit_files", {"repository": f"eq.{full_name}"})
    if not commit_files:
        commit_files = sb.select("commit_files", {"repository": f"eq.{resolved_repo_name}"})

    # Group commit files by sha
    commit_files_by_sha: Dict[str, List[Dict[str, Any]]] = {}
    for cf in commit_files:
        sha = cf.get("commit_sha")
        if sha:
            commit_files_by_sha.setdefault(sha, []).append(cf)

    # Fetch PR changed files
    pr_id_map: Dict[int, Dict[str, Any]] = {p["id"]: p for p in prs if "id" in p}
    changed_files = []
    if pr_id_map:
        for pid in pr_id_map.keys():
            pr_cfiles = sb.select("changed_files", {"pull_request_id": f"eq.{pid}"})
            changed_files.extend(pr_cfiles)

    pr_files_map: Dict[int, List[Dict[str, Any]]] = {}
    for cf in changed_files:
        pr_files_map.setdefault(cf.get("pull_request_id"), []).append(cf)

    # Fetch PR Commits using official GitHub endpoint GET /repos/{owner}/{repo}/pulls/{pull_number}/commits
    pr_commits_cache: Dict[int, List[Dict[str, Any]]] = {}
    if pr_commits_map:
        pr_commits_cache.update(pr_commits_map)

    # If github_client was provided or can be instantiated, fetch commits for PRs not already in cache
    gh_client = github_client
    if gh_client is None:
        try:
            from github.client import GitHubClient
            gh_client = GitHubClient()
        except Exception:
            gh_client = None

    for p in prs:
        pr_num = p.get("github_pr_number") or p.get("number")
        if pr_num and pr_num not in pr_commits_cache and gh_client is not None:
            try:
                pr_commits = gh_client.get_pull_request_commits(owner_clean, repo_clean, pr_num)
                pr_commits_cache[pr_num] = pr_commits
            except Exception as e:
                logger.warning(
                    f"Could not fetch commits for PR #{pr_num} from GitHub API ({full_name}): {e}"
                )
                pr_commits_cache[pr_num] = []

    # 3. Construct documents with strict deduplication
    documents: List[KnowledgeDocument] = []
    seen_doc_ids = set()

    def add_doc(doc: Optional[KnowledgeDocument]):
        if doc and doc.id not in seen_doc_ids and doc.text.strip():
            seen_doc_ids.add(doc.id)
            documents.append(doc)

    # A. Commits
    for c in commits:
        sha = c.get("sha")
        c_files = commit_files_by_sha.get(sha, [])
        doc = format_commit_document(c, resolved_full_name, dev_map, c_files)
        add_doc(doc)

    # B. CommitFiles
    commit_map: Dict[str, Dict[str, Any]] = {c["sha"]: c for c in commits if "sha" in c}
    for cf in commit_files:
        parent_commit = commit_map.get(cf.get("commit_sha"), {})
        doc = format_commit_file_document(cf, resolved_full_name, parent_commit, dev_map)
        add_doc(doc)

    # C. Pull Requests (with contained commit references)
    for p in prs:
        p_files = pr_files_map.get(p.get("id"), [])
        pr_num = p.get("github_pr_number") or p.get("number")
        p_commits = pr_commits_cache.get(pr_num, []) if pr_num else []
        doc = format_pull_request_document(
            p, resolved_full_name, dev_map, p_files, contained_commits=p_commits
        )
        add_doc(doc)

    # D. PR Files
    for cf in changed_files:
        pr_obj = pr_id_map.get(cf.get("pull_request_id"), {})
        pr_num = pr_obj.get("github_pr_number", 0)
        pr_ts = pr_obj.get("created_at")
        doc = format_pr_file_document(cf, pr_num, resolved_full_name, timestamp=pr_ts)
        add_doc(doc)

    return documents


def build_embeddings(
    documents: List[KnowledgeDocument],
    embedding_service: Optional[EmbeddingService] = None,
) -> np.ndarray:
    """
    Generates 768-dimensional normalized dense vectors using the existing BGE embedding service.
    Validates vector count, dimensions, and ensures no NaN/inf values.
    """
    if not documents:
        return np.empty((0, 768), dtype=np.float32)

    service = embedding_service or EmbeddingService()
    texts = [doc.text for doc in documents]

    raw_vectors = service.embed_documents(texts, batch_size=32)

    if len(raw_vectors) != len(documents):
        raise ValueError(
            f"Embedding count mismatch: expected {len(documents)} vectors, got {len(raw_vectors)}"
        )

    embeddings = np.array(raw_vectors, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embeddings.shape}")

    expected_dim = service.dimension
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
        )

    if np.isnan(embeddings).any():
        raise ValueError("Invalid embedding vectors: detected NaN values")

    if np.isinf(embeddings).any():
        raise ValueError("Invalid embedding vectors: detected Infinite values")

    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Executes HDBSCAN clustering over embedding vectors.
    Returns integer cluster labels for each vector (0, 1, 2, ..., -1 for noise),
    and optionally membership probabilities if return_probabilities is True.
    """
    if len(embeddings) < min_cluster_size:
        raise ValueError(
            f"Insufficient samples for clustering: have {len(embeddings)}, "
            f"min_cluster_size={min_cluster_size}"
        )

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        from hdbscan import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(embeddings)
    labels = clusterer.labels_
    probabilities = getattr(clusterer, "probabilities_", None)

    if return_probabilities:
        return labels, probabilities
    return labels


def build_cluster_summary(
    repository: str,
    documents: List[KnowledgeDocument],
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregates clustering results into structured statistics, document mappings,
    and readable document previews. Preserves noise (-1) as a distinct group.
    Maintains document-to-cluster mappings with probabilities and PR contained commits.
    """
    if len(documents) != len(labels):
        raise ValueError("Document count and label count must match")

    clusters_map: Dict[int, List[Tuple[KnowledgeDocument, Optional[float]]]] = {}
    prob_list = list(probabilities) if probabilities is not None else [None] * len(labels)

    for doc, label, prob in zip(documents, labels, prob_list):
        lbl = int(label)
        p_val = round(float(prob), 4) if prob is not None else None
        doc.cluster_id = lbl
        doc.cluster_probability = p_val
        clusters_map.setdefault(lbl, []).append((doc, p_val))

    # Sort regular clusters by size descending (cluster 0, 1, 2, ...)
    regular_labels = sorted(
        [lbl for lbl in clusters_map.keys() if lbl != -1],
        key=lambda x: len(clusters_map[x]),
        reverse=True,
    )

    clusters_list = []
    cluster_summary_dict: Dict[str, Any] = {}

    for lbl in regular_labels:
        items = clusters_map[lbl]
        type_counts: Dict[str, int] = {}
        doc_previews = []
        for d, p_val in items:
            type_counts[d.document_type] = type_counts.get(d.document_type, 0) + 1
            doc_info = {
                "id": d.id,
                "document_type": d.document_type,
                "source_id": d.source_id,
                "preview": d.text.split("\n\n")[0] if "\n\n" in d.text else d.text[:120],
                "metadata": d.metadata,
            }
            if p_val is not None:
                doc_info["cluster_probability"] = p_val
            if d.contained_commit_ids:
                doc_info["contained_commit_ids"] = d.contained_commit_ids
            doc_previews.append(doc_info)

        cluster_summary_dict[str(lbl)] = {
            "document_count": len(items),
            "document_type_counts": type_counts,
        }

        clusters_list.append({
            "cluster_id": lbl,
            "document_count": len(items),
            "document_type_counts": type_counts,
            "documents": doc_previews,
        })

    # Add noise cluster (-1) if present
    noise_items = clusters_map.get(-1, [])
    noise_count = len(noise_items)
    if noise_items:
        type_counts = {}
        doc_previews = []
        for d, p_val in noise_items:
            type_counts[d.document_type] = type_counts.get(d.document_type, 0) + 1
            doc_info = {
                "id": d.id,
                "document_type": d.document_type,
                "source_id": d.source_id,
                "preview": d.text.split("\n\n")[0] if "\n\n" in d.text else d.text[:120],
                "metadata": d.metadata,
            }
            if p_val is not None:
                doc_info["cluster_probability"] = p_val
            if d.contained_commit_ids:
                doc_info["contained_commit_ids"] = d.contained_commit_ids
            doc_previews.append(doc_info)

        cluster_summary_dict["-1"] = {
            "document_count": noise_count,
            "document_type_counts": type_counts,
        }

        clusters_list.append({
            "cluster_id": -1,
            "document_count": noise_count,
            "document_type_counts": type_counts,
            "documents": doc_previews,
        })

    # Flatten document-to-cluster mappings
    doc_mappings = []
    for doc, label, prob in zip(documents, labels, prob_list):
        p_val = round(float(prob), 4) if prob is not None else None
        item = {
            "id": doc.id,
            "type": doc.document_type,
            "document_type": doc.document_type,
            "source_id": doc.source_id,
            "timestamp": doc.timestamp,
            "cluster_id": int(label),
        }
        if p_val is not None:
            item["cluster_probability"] = p_val
        if doc.contained_commit_ids:
            item["contained_commit_ids"] = doc.contained_commit_ids
        doc_mappings.append(item)

    return {
        "status": "completed",
        "repository": repository,
        "document_count": len(documents),
        "cluster_count": len(regular_labels),
        "noise_count": noise_count,
        "noise_documents": noise_count,
        "parameters": parameters or {},
        "documents": doc_mappings,
        "cluster_summary": cluster_summary_dict,
        "clusters": clusters_list,
    }


def run_knowledge_clustering(
    owner: str,
    repo: str,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    supabase_client: Optional[SupabaseClient] = None,
    github_client: Optional[Any] = None,
    pr_commits_map: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> Dict[str, Any]:
    """
    Full pipeline execution for knowledge document construction and HDBSCAN clustering.
    Handles empty data and small datasets gracefully.
    """
    owner_clean = owner.strip()
    repo_clean = repo.strip()
    full_name = f"{owner_clean}/{repo_clean}"

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
    }

    # 1. Load knowledge documents from Supabase + GitHub PR commits
    documents = load_knowledge_documents(
        owner=owner_clean,
        repo=repo_clean,
        supabase_client=supabase_client,
        github_client=github_client,
        pr_commits_map=pr_commits_map,
    )

    # 2. Handle empty data
    if not documents:
        return {
            "status": "no_data",
            "repository": full_name,
            "document_count": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "noise_documents": 0,
            "parameters": params,
            "documents": [],
            "cluster_summary": {},
            "clusters": [],
        }

    # 3. Handle small datasets
    if len(documents) < min_cluster_size:
        doc_mappings = [
            {
                "id": d.id,
                "type": d.document_type,
                "document_type": d.document_type,
                "source_id": d.source_id,
                "timestamp": d.timestamp,
                "cluster_id": -1,
                "cluster_probability": 0.0,
                **( {"contained_commit_ids": d.contained_commit_ids} if d.contained_commit_ids else {} )
            }
            for d in documents
        ]
        return {
            "status": "insufficient_data",
            "repository": full_name,
            "document_count": len(documents),
            "minimum_required": min_cluster_size,
            "message": f"Not enough documents for HDBSCAN clustering (found {len(documents)}, minimum required is {min_cluster_size}).",
            "cluster_count": 0,
            "noise_count": len(documents),
            "noise_documents": len(documents),
            "parameters": params,
            "documents": doc_mappings,
            "cluster_summary": {
                "-1": {
                    "document_count": len(documents)
                }
            },
            "clusters": [],
        }

    # 4. Generate vectors using existing BGE embeddings
    embeddings = build_embeddings(documents, embedding_service=embedding_service)

    # 5. Execute HDBSCAN clustering with membership probabilities
    clustering_result = cluster_embeddings(
        embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        return_probabilities=True,
    )
    if isinstance(clustering_result, tuple):
        labels, probabilities = clustering_result
    else:
        labels, probabilities = clustering_result, None

    # 6. Build and return summary report
    return build_cluster_summary(
        repository=full_name,
        documents=documents,
        labels=labels,
        probabilities=probabilities,
        parameters=params,
    )

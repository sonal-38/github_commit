"""
Vector Indexer and Search pipeline.

Extracts normalized records from Supabase (without calling GitHub),
constructs semantic documents, embeds them using SentenceTransformers,
and upserts deterministic vectors to Supabase PostgreSQL (pgvector).
Also provides semantic similarity retrieval with optional repository filtering.
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from database.supabase_client import SupabaseClient, SupabaseDatabaseError
from vector.embeddings import EmbeddingService
from vector.document_builder import DocumentBuilder, DocumentItem

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Coordinates data extraction from Supabase, document building,
    embedding generation, and idempotent vector storage in Supabase pgvector.
    """

    def __init__(
        self,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.supabase = supabase_client or SupabaseClient()
        self.embeddings = embedding_service or EmbeddingService()

    def index_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Loads all relational records for the repository from Supabase,
        converts them to semantic documents, embeds them, and upserts to Supabase pgvector.
        Idempotent: Re-running this replaces/updates existing records on primary key conflict.
        """
        if hasattr(self.supabase, "verify_connectivity"):
            self.supabase.verify_connectivity()

        owner_clean = owner.strip()
        repo_clean = repo.strip()
        full_name = f"{owner_clean}/{repo_clean}"

        # 1. Fetch repository record from Supabase (tolerant of hyphens/underscores)
        repos = self.supabase.select("repositories", {"full_name": f"eq.{full_name}"})
        if not repos:
            repos = self.supabase.select(
                "repositories",
                {"name": f"eq.{repo_clean}", "owner_login": f"eq.{owner_clean}"}
            )
        if not repos:
            # Try alternate hyphen/underscore version
            alt_repo = repo_clean.replace("_", "-") if "_" in repo_clean else repo_clean.replace("-", "_")
            alt_owner = owner_clean.replace("_", "-") if "_" in owner_clean else owner_clean.replace("-", "_")
            repos = self.supabase.select(
                "repositories",
                {"name": f"eq.{alt_repo}", "owner_login": f"eq.{alt_owner}"}
            )

        if not repos:
            raise SupabaseDatabaseError(
                f"Repository '{full_name}' was not found in Supabase. "
                "Please run ingestion first via POST /github/repositories/{owner}/{repo}/ingest",
                status_code=404
            )

        repo_record = repos[0]
        repo_id = repo_record.get("id")
        resolved_full_name = repo_record.get("full_name") or full_name
        resolved_repo_name = repo_record.get("name") or repo_clean

        # 2. Fetch all relational data from Supabase
        commits = self.supabase.select("commits", {"repository_id": f"eq.{repo_id}"})
        prs = self.supabase.select("pull_requests", {"repository_id": f"eq.{repo_id}"})
        issues = self.supabase.select("issues", {"repository_id": f"eq.{repo_id}"})

        # Fetch developers map
        developers_list = self.supabase.select("developers")
        dev_map: Dict[int, Dict[str, Any]] = {d["id"]: d for d in developers_list if "id" in d}

        # Fetch PR-dependent children
        pr_id_map: Dict[int, Dict[str, Any]] = {p["id"]: p for p in prs if "id" in p}
        pr_ids = list(pr_id_map.keys())

        reviews = []
        review_comments = []
        changed_files = []

        if pr_ids:
            for pid in pr_ids:
                pr_reviews = self.supabase.select("reviews", {"pull_request_id": f"eq.{pid}"})
                reviews.extend(pr_reviews)
                pr_comms = self.supabase.select("review_comments", {"pull_request_id": f"eq.{pid}"})
                review_comments.extend(pr_comms)
                pr_cfiles = self.supabase.select("changed_files", {"pull_request_id": f"eq.{pid}"})
                changed_files.extend(pr_cfiles)

        # Fetch Issue-dependent children
        issue_id_map: Dict[int, Dict[str, Any]] = {i["id"]: i for i in issues if "id" in i}
        issue_ids = list(issue_id_map.keys())
        issue_comments = []

        if issue_ids:
            for iid in issue_ids:
                ics = self.supabase.select("issue_comments", {"issue_id": f"eq.{iid}"})
                issue_comments.extend(ics)

        # 3. Build documents across categories
        documents: List[DocumentItem] = []
        stats = {
            "commits": 0,
            "pull_requests": 0,
            "reviews": 0,
            "review_comments": 0,
            "issues": 0,
            "issue_comments": 0,
            "changed_files": 0,
        }

        # Build commit documents
        for c in commits:
            doc = DocumentBuilder.build_commit_doc(c, resolved_full_name, dev_map)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["commits"] += 1

        # Map files and reviews to PRs for richer PR documents
        pr_files_map: Dict[int, List[Dict[str, Any]]] = {}
        for cf in changed_files:
            pr_files_map.setdefault(cf.get("pull_request_id"), []).append(cf)

        pr_reviews_map: Dict[int, List[Dict[str, Any]]] = {}
        for rv in reviews:
            pr_reviews_map.setdefault(rv.get("pull_request_id"), []).append(rv)

        # Build pull request documents
        for p in prs:
            p_id = p.get("id")
            p_files = pr_files_map.get(p_id, [])
            p_revs = pr_reviews_map.get(p_id, [])
            doc = DocumentBuilder.build_pr_doc(p, resolved_full_name, dev_map, p_files, p_revs)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["pull_requests"] += 1

        # Build review documents
        for rv in reviews:
            pr_obj = pr_id_map.get(rv.get("pull_request_id"), {})
            pr_num = pr_obj.get("github_pr_number", 0)
            doc = DocumentBuilder.build_review_doc(rv, pr_num, resolved_full_name, dev_map)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["reviews"] += 1

        # Build review comment documents
        for rc in review_comments:
            pr_obj = pr_id_map.get(rc.get("pull_request_id"), {})
            pr_num = pr_obj.get("github_pr_number", 0)
            doc = DocumentBuilder.build_review_comment_doc(rc, pr_num, resolved_full_name, dev_map)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["review_comments"] += 1

        # Map comments to issues for summary
        issue_comms_map: Dict[int, List[Dict[str, Any]]] = {}
        for ic in issue_comments:
            issue_comms_map.setdefault(ic.get("issue_id"), []).append(ic)

        # Build issue documents
        for i in issues:
            i_id = i.get("id")
            i_comms = issue_comms_map.get(i_id, [])
            doc = DocumentBuilder.build_issue_doc(i, resolved_full_name, dev_map, i_comms)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["issues"] += 1

        # Build issue comment documents
        for ic in issue_comments:
            i_obj = issue_id_map.get(ic.get("issue_id"), {})
            i_num = i_obj.get("github_issue_number", 0)
            doc = DocumentBuilder.build_issue_comment_doc(ic, i_num, resolved_full_name, dev_map)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["issue_comments"] += 1

        # Build changed files documents
        for cf in changed_files:
            pr_obj = pr_id_map.get(cf.get("pull_request_id"), {})
            pr_num = pr_obj.get("github_pr_number", 0)
            doc = DocumentBuilder.build_changed_file_doc(cf, pr_num, resolved_full_name)
            if doc and doc.text.strip():
                documents.append(doc)
                stats["changed_files"] += 1

        if not documents:
            return {
                "repository": resolved_full_name,
                "indexed": stats,
                "total_vectors": 0,
                "message": "No indexable documents found for repository.",
            }

        # 4. Generate embeddings in batches using Gemini RETRIEVAL_DOCUMENT
        texts_to_embed = [doc.text for doc in documents]
        vectors = self.embeddings.embed_batch(
            texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=20,
        )

        # 5. Prepare records for Supabase pgvector table (document_embeddings)
        # Uses deterministic stable_key as the primary key 'id' to guarantee idempotency.
        now_iso = datetime.now(timezone.utc).isoformat()
        records = []
        for doc, vec in zip(documents, vectors):
            doc_meta = dict(doc.metadata or {})
            doc_meta["embedding_model"] = "gemini-embedding-001"
            records.append({
                "id": doc.stable_key,
                "repository": doc.repository,
                "document_type": doc.document_type,
                "source_id": doc.source_id,
                "developer": doc.developer,
                "text": doc.text,
                "embedding": vec,
                "metadata": doc_meta,
                "updated_at": now_iso,
            })

        # 6. Upsert in batches into Supabase PostgreSQL (pgvector)
        # on_conflict="id" ensures duplicate prevention: updates existing rows if re-indexed.
        chunk_size = 50
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            self.supabase.upsert(
                table="document_embeddings",
                data=chunk,
                on_conflict="id",
            )

        total_vectors = len(records)

        return {
            "repository": resolved_full_name,
            "indexed": stats,
            "total_vectors": total_vectors,
        }

    def search(
        self,
        query: str,
        repository: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Embeds the search query and retrieves the most semantically relevant documents
        from Supabase PostgreSQL using the pgvector cosine similarity function match_documents.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return {"query": query, "results": []}

        # 1. Generate query embedding using Gemini RETRIEVAL_QUERY (768 dimensions)
        query_vector = self.embeddings.embed_text(clean_query, task_type="RETRIEVAL_QUERY")

        # 2. Execute pgvector cosine similarity search via Supabase RPC
        rpc_params: Dict[str, Any] = {
            "query_embedding": query_vector,
            "match_count": limit,
        }
        if repository and repository.strip():
            rpc_params["filter_repository"] = repository.strip()

        try:
            matches = self.supabase.rpc("match_documents", rpc_params)
        except SupabaseDatabaseError as e:
            logger.warning("Supabase match_documents RPC error: %s", str(e))
            # If match_documents function doesn't exist or table is empty, return empty results
            # so RAG can fall back gracefully to repository records or return helpful guidance
            return {"query": clean_query, "results": []}
        except Exception as e:
            logger.warning("Unexpected error during semantic vector search: %s", str(e))
            return {"query": clean_query, "results": []}

        # 3. Format results to preserve API schema
        results = []
        for row in matches:
            results.append({
                "score": float(row.get("similarity", 0.0)),
                "document_type": row.get("document_type", ""),
                "repository": row.get("repository", ""),
                "developer": row.get("developer") or "Unknown",
                "source_id": str(row.get("source_id", "")),
                "text": row.get("text", ""),
                "metadata": row.get("metadata") or {},
            })

        return {
            "query": clean_query,
            "results": results,
        }

"""
RAG Service for AI Digital Shadow.

Coordinates the end-to-end Retrieval-Augmented Generation pipeline:
1. User question input.
2. Chronological & date constraint detection (DateQueryDetector).
3. If date constraints exist:
   - Queries Supabase structured tables using original GitHub event timestamps
     (e.g., commits.committed_at, pull_requests.merged_at/created_at, issues.created_at).
   - Resolves eligible records within the exact chronological range.
   - Orders evidence chronologically (oldest -> newest) for timeline and transition queries.
   - Uses pgvector semantic reranking or filtering where appropriate.
4. If no date constraint exists:
   - Uses pgvector semantic cosine similarity search.
5. Builds clear, structured evidence context with explicit event timestamps and sources.
6. Prompts Gemini to produce a strictly grounded answer with zero external hallucinations.
7. Returns grounded answer, verified source citations, and date filters applied.
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union

from vector.indexer import VectorIndexer
from vector.document_builder import DocumentBuilder
from ai.gemini_service import GeminiService, GeminiConfigurationError, GeminiAPIError
from ai.date_filter import DateQueryDetector, DateConstraint
from database.supabase_client import SupabaseClient, SupabaseDatabaseError

logger = logging.getLogger(__name__)


class RAGService:
    """
    Coordinates date-aware structured retrieval from Supabase, pgvector semantic search,
    and Gemini grounded answer generation.
    """

    def __init__(
        self,
        vector_indexer: Optional[VectorIndexer] = None,
        gemini_service: Optional[GeminiService] = None,
        supabase_client: Optional[SupabaseClient] = None,
        default_top_k: int = 5,
        min_similarity_threshold: float = 0.25,
    ):
        self.indexer = vector_indexer or VectorIndexer()
        self.gemini = gemini_service or GeminiService()
        self.supabase = supabase_client or SupabaseClient()
        self.default_top_k = default_top_k
        self.min_similarity_threshold = min_similarity_threshold

    def answer_question(
        self,
        question: str,
        repository: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the date-aware RAG pipeline:
        - Detects date/time constraints (e.g., "after August 1, 2026", "in August 2026", "transition").
        - For date queries: queries Supabase structured tables directly using original GitHub timestamps.
        - For semantic queries: queries pgvector.
        - Structures context chronologically where appropriate.
        - Calls Gemini for grounded answer generation with sources.
        """
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        k = top_k if top_k is not None and top_k > 0 else self.default_top_k
        if k > 25:
            k = 25  # Safe ceiling

        repo_clean = repository.strip() if repository and repository.strip() else None

        # 1. Detect date constraints from question
        date_constraint = DateQueryDetector.detect(clean_question)

        relevant_docs: List[Dict[str, Any]] = []
        filter_metadata: Optional[Dict[str, Any]] = None

        if date_constraint and (date_constraint.start_iso or date_constraint.end_iso or date_constraint.relative_keyword):
            filter_metadata = date_constraint.to_dict()
            relevant_docs = self._retrieve_by_exact_date(
                constraint=date_constraint,
                repository=repo_clean,
                query=clean_question,
                limit=k,
            )
        elif date_constraint and date_constraint.is_transition:
            # Transition query without explicit date bounds: retrieve semantic docs and sort chronologically
            filter_metadata = date_constraint.to_dict()
            relevant_docs = self._retrieve_and_sort_transition_docs(
                query=clean_question,
                repository=repo_clean,
                limit=k,
            )
        else:
            # Standard semantic vector search via pgvector
            search_result = self.indexer.search(
                query=clean_question,
                repository=repo_clean,
                limit=k,
            )
            raw_results: List[Dict[str, Any]] = search_result.get("results", [])
            relevant_docs = [
                doc for doc in raw_results
                if doc.get("score", 0.0) >= self.min_similarity_threshold
            ]

        # 2. No-Evidence Handling: Return controlled response if no relevant documents match
        if not relevant_docs:
            resp: Dict[str, Any] = {
                "question": clean_question,
                "answer": "I could not find enough repository evidence to answer this question.",
                "sources": [],
            }
            if filter_metadata:
                resp["filters"] = filter_metadata
            return resp

        # 3. Build Structured Context for Gemini
        context_str = self._build_context(relevant_docs, is_transition=bool(date_constraint and date_constraint.is_transition))

        # 4. Generate Grounded Answer using Gemini
        answer = self.gemini.generate_grounded_answer(
            question=clean_question,
            context=context_str,
        )

        # 5. Build clean source citations for response
        sources = self._extract_sources(relevant_docs)

        result: Dict[str, Any] = {
            "question": clean_question,
            "answer": answer,
            "sources": sources,
        }
        if filter_metadata:
            result["filters"] = filter_metadata

        return result

    def _retrieve_by_exact_date(
        self,
        constraint: DateConstraint,
        repository: Optional[str],
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Queries Supabase structured tables using original GitHub event timestamp columns:
        - commits.committed_at
        - pull_requests.created_at / merged_at / closed_at
        - issues.created_at / closed_at
        - reviews.submitted_at
        - review_comments.created_at
        - issue_comments.created_at
        """
        # Resolve repository_id if repository specified
        repo_id: Optional[int] = None
        repo_full_name: str = repository or ""

        if repository:
            repo_records = self._safe_select("repositories", [("full_name", f"eq.{repository}")])
            if not repo_records and "/" in repository:
                owner, rname = repository.split("/", 1)
                repo_records = self._safe_select("repositories", [
                    ("name", f"eq.{rname}"),
                    ("owner_login", f"eq.{owner}")
                ])
            if repo_records:
                repo_id = repo_records[0].get("id")
                repo_full_name = repo_records[0].get("full_name") or repository

        # Fetch developers lookup map
        developers_list = self._safe_select("developers", None)
        dev_map: Dict[int, Dict[str, Any]] = {d["id"]: d for d in developers_list if "id" in d}

        entities_to_query = [constraint.target_entity] if constraint.target_entity else ["commits", "pull_requests", "issues"]
        all_matches: List[Dict[str, Any]] = []

        for entity in entities_to_query:
            table_name = entity
            date_col = constraint.date_field or self._default_date_column_for(table_name)
            if not date_col:
                continue

            query_param_list: List[tuple] = []
            if repo_id is not None:
                query_param_list.append(("repository_id", f"eq.{repo_id}"))

            # Apply PostgREST comparison operators
            if constraint.start_iso and constraint.end_iso:
                # Query lower bound and upper bound
                query_param_list.append((date_col, f"gte.{constraint.start_iso}"))
                query_param_list.append((date_col, f"lt.{constraint.end_iso}"))
            elif constraint.start_iso:
                query_param_list.append((date_col, f"gte.{constraint.start_iso}"))
            elif constraint.end_iso:
                query_param_list.append((date_col, f"lte.{constraint.end_iso}"))

            # Fetch records from Supabase structured table
            records = self._safe_select(table_name, query_param_list)

            # Python-side boundary check to ensure UTC precision
            filtered_records = []
            for rec in records:
                rec_date = rec.get(date_col)
                if not rec_date:
                    continue
                if constraint.start_iso and rec_date < constraint.start_iso:
                    continue
                if constraint.end_iso and rec_date >= constraint.end_iso:
                    continue
                filtered_records.append(rec)

            # Sort chronologically by original GitHub event date
            filtered_records.sort(
                key=lambda r: r.get(date_col) or "",
                reverse=bool(constraint.relative_keyword == "recent")
            )

            # Convert to DocumentItem representation
            for rec in filtered_records[:limit]:
                doc_item = self._convert_record_to_doc(
                    entity=table_name,
                    record=rec,
                    repo_name=repo_full_name,
                    dev_map=dev_map,
                    date_field=date_col,
                )
                if doc_item:
                    all_matches.append(doc_item)

        # Sort all retrieved documents chronologically (oldest -> newest) for consistent timeline presentation
        all_matches.sort(key=lambda d: d.get("metadata", {}).get("github_event_date") or "")
        return all_matches[:limit]

    def _retrieve_and_sort_transition_docs(
        self,
        query: str,
        repository: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        For evolution/transition queries, retrieves top semantic documents and
        orders them strictly chronologically using original GitHub event timestamps.
        """
        search_result = self.indexer.search(
            query=query,
            repository=repository,
            limit=limit,
        )
        raw_results = search_result.get("results", [])
        docs = [d for d in raw_results if d.get("score", 0.0) >= self.min_similarity_threshold]

        # Order chronologically based on metadata timestamp
        def extract_ts(doc: Dict[str, Any]) -> str:
            meta = doc.get("metadata") or {}
            return (
                meta.get("committed_at")
                or meta.get("created_at")
                or meta.get("submitted_at")
                or ""
            )

        docs.sort(key=extract_ts)
        return docs

    def _convert_record_to_doc(
        self,
        entity: str,
        record: Dict[str, Any],
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
        date_field: str,
    ) -> Optional[Dict[str, Any]]:
        """Transforms a Supabase relational row into a structured evidence item."""
        event_date = record.get(date_field) or ""

        if entity == "commits":
            doc = DocumentBuilder.build_commit_document(record, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "commit",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        elif entity == "pull_requests":
            doc = DocumentBuilder.build_pr_doc(record, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "pull_request",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        elif entity == "issues":
            doc = DocumentBuilder.build_issue_doc(record, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "issue",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        elif entity == "reviews":
            pr_id = record.get("pull_request_id", 0)
            doc = DocumentBuilder.build_review_doc(record, pr_id, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "review",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        elif entity == "review_comments":
            pr_id = record.get("pull_request_id", 0)
            doc = DocumentBuilder.build_review_comment_doc(record, pr_id, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "review_comment",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        elif entity == "issue_comments":
            issue_id = record.get("issue_id", 0)
            doc = DocumentBuilder.build_issue_comment_doc(record, issue_id, repo_name, dev_map)
            if doc:
                meta = doc.metadata
                meta["github_event_date"] = event_date
                return {
                    "score": 1.0,
                    "document_type": "issue_comment",
                    "repository": repo_name,
                    "developer": doc.developer,
                    "source_id": doc.source_id,
                    "text": doc.text,
                    "metadata": meta,
                }
        return None

    def _default_date_column_for(self, table: str) -> Optional[str]:
        mapping = {
            "commits": "committed_at",
            "pull_requests": "created_at",
            "issues": "created_at",
            "reviews": "submitted_at",
            "review_comments": "created_at",
            "issue_comments": "created_at",
        }
        return mapping.get(table)

    def _safe_select(
        self,
        table: str,
        query_params: Optional[Union[Dict[str, str], List[tuple]]]
    ) -> List[Dict[str, Any]]:
        """Executes a Supabase SELECT query catching errors gracefully."""
        try:
            return self.supabase.select(table, query_params)
        except Exception as e:
            logger.warning("Supabase SELECT failed for table '%s': %s", table, str(e))
            return []

    def _build_context(self, documents: List[Dict[str, Any]], is_transition: bool = False) -> str:
        """
        Constructs a structured, human-readable evidence context from retrieved documents.
        Clearly separates each evidence item with metadata and exact event dates.
        """
        evidence_blocks = []

        for idx, doc in enumerate(documents, start=1):
            doc_type = doc.get("document_type", "document").upper()
            source_id = doc.get("source_id", "unknown")
            developer = doc.get("developer", "Unknown")
            repo = doc.get("repository", "")
            meta = doc.get("metadata", {})
            text_body = doc.get("text", "").strip()

            event_date = (
                meta.get("github_event_date")
                or meta.get("committed_at")
                or meta.get("created_at")
                or meta.get("submitted_at")
                or "Unknown Date"
            )

            block_lines = [
                f"[Evidence #{idx}]",
                f"Type: {doc_type}",
                f"Source Identifier: {source_id}",
                f"Developer / Author: {developer}",
                f"Original GitHub Event Date: {event_date}",
            ]
            if repo:
                block_lines.append(f"Repository: {repo}")

            if "title" in meta:
                block_lines.append(f"Title: {meta['title']}")
            if "state" in meta:
                block_lines.append(f"State: {meta['state']}")
            if "pull_number" in meta:
                block_lines.append(f"Pull Request #: {meta['pull_number']}")
            if "issue_number" in meta:
                block_lines.append(f"Issue #: {meta['issue_number']}")

            block_lines.append("Content / Details:")
            block_lines.append(text_body)

            evidence_blocks.append("\n".join(block_lines))

        prefix = ""
        if is_transition:
            prefix = "NOTE: The following evidence is arranged in CHRONOLOGICAL ORDER (oldest to newest):\n\n"

        return prefix + "\n\n----------------------------------------\n\n".join(evidence_blocks)

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts verified source identifiers and metadata from the retrieved documents.
        Deduplicates sources based on document_type and source_id.
        """
        seen_keys = set()
        sources = []

        for doc in documents:
            doc_type = doc.get("document_type", "")
            source_id = str(doc.get("source_id", ""))
            key = f"{doc_type}:{source_id}"

            if key not in seen_keys:
                seen_keys.add(key)
                sources.append({
                    "document_type": doc_type,
                    "source_id": source_id,
                    "developer": doc.get("developer") or "Unknown",
                })

        return sources

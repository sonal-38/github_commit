"""
Document builder for converting structured Supabase records into semantic text documents.

Produces human-readable, domain-rich context documents for:
- Commits (message, developer, repo, date)
- Pull Requests (title, body, author, reviews, changed files summary)
- Reviews (body, state, reviewer, PR)
- Review Comments (diff hunk, file path, comment text, line)
- Issues (title, body, creator, state, comments summary)
- Issue Comments (discussion text, commenter, issue)
- Changed Files (filename, status, additions/deletions, patch preview)

Generates deterministic UUIDs for Qdrant point upserts to ensure idempotency.
"""
import uuid
from typing import Any, Dict, List, Optional


class DocumentItem:
    """Represents a prepared document ready for embedding and Qdrant ingestion."""
    def __init__(
        self,
        point_id: str,
        stable_key: str,
        document_type: str,
        repository: str,
        developer: str,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.point_id = point_id          # Valid UUID string for Qdrant
        self.stable_key = stable_key      # Human-readable deterministic string (e.g. commit:repo:sha)
        self.document_type = document_type
        self.repository = repository
        self.developer = developer
        self.source_id = source_id
        self.text = text
        self.metadata = metadata or {}

    def to_payload(self) -> Dict[str, Any]:
        """Returns the full metadata payload to store with the vector in Qdrant."""
        payload = {
            "document_type": self.document_type,
            "repository": self.repository,
            "developer": self.developer,
            "source_id": self.source_id,
            "stable_key": self.stable_key,
            "text": self.text,
        }
        payload.update(self.metadata)
        return payload


class DocumentBuilder:
    """
    Transforms relational records from Supabase into semantically rich document items.
    """

    @staticmethod
    def _create_deterministic_id(key: str) -> str:
        """
        Converts a deterministic string key into an RFC 4122 compliant UUIDv5.
        Qdrant strictly requires point IDs to be unsigned 64-bit integers or UUIDs.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    @classmethod
    def build_commit_doc(
        cls,
        commit: Dict[str, Any],
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from a Git commit record."""
        sha = commit.get("sha", "")
        if not sha:
            return None

        dev_id = commit.get("developer_id")
        dev = dev_map.get(dev_id, {}) if dev_id else {}
        dev_name = dev.get("login") or dev.get("name") or "Unknown"
        message = (commit.get("message") or "").strip()
        date = commit.get("committed_at") or ""
        short_sha = sha[:8]

        text_parts = [
            f"Repository: {repo_name}",
            f"Developer: {dev_name}",
            f"Date: {date}" if date else "",
            "",
            "Commit:",
            f"SHA: {sha}",
            f"Message: {message if message else 'No commit message provided.'}",
        ]
        text = "\n".join(p for p in text_parts if p is not None).strip()

        stable_key = f"commit:{repo_name}:{sha}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="commit",
            repository=repo_name,
            developer=dev_name,
            source_id=sha,
            text=text,
            metadata={
                "commit_sha": sha,
                "short_sha": short_sha,
                "committed_at": date,
                "html_url": commit.get("html_url", ""),
            },
        )

    @classmethod
    def build_pr_doc(
        cls,
        pr: Dict[str, Any],
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
        files: Optional[List[Dict[str, Any]]] = None,
        reviews: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from a Pull Request record."""
        pr_number = pr.get("github_pr_number")
        if pr_number is None:
            return None

        dev_id = pr.get("developer_id")
        dev = dev_map.get(dev_id, {}) if dev_id else {}
        author = dev.get("login") or dev.get("name") or "Unknown"
        title = (pr.get("title") or "").strip()
        body = (pr.get("body") or "").strip()
        state = pr.get("state") or "unknown"
        created_at = pr.get("created_at") or ""

        # Summarize changed files if available
        files_summary = ""
        if files:
            file_names = [f.get("filename", "") for f in files if f.get("filename")]
            if file_names:
                files_summary = ", ".join(file_names[:10])
                if len(file_names) > 10:
                    files_summary += f" and {len(file_names) - 10} more files"

        # Summarize reviews if available
        review_summary = ""
        if reviews:
            review_states = [r.get("state", "") for r in reviews if r.get("state")]
            if review_states:
                review_summary = f"{len(reviews)} reviews ({', '.join(set(review_states))})"

        text_parts = [
            f"Repository: {repo_name}",
            "",
            "Pull Request:",
            f"Number: #{pr_number}",
            f"Title: {title}",
            f"Author: {author}",
            f"State: {state}",
            f"Created At: {created_at}" if created_at else "",
            f"Description:\n{body}" if body else "Description:\n(No description provided)",
        ]
        if files_summary:
            text_parts.append(f"Changed Files: {files_summary}")
        if review_summary:
            text_parts.append(f"Reviews: {review_summary}")

        text = "\n".join(text_parts).strip()
        stable_key = f"pr:{repo_name}:{pr_number}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="pull_request",
            repository=repo_name,
            developer=author,
            source_id=str(pr_number),
            text=text,
            metadata={
                "pr_number": pr_number,
                "title": title,
                "state": state,
                "created_at": created_at,
                "html_url": pr.get("html_url", ""),
            },
        )

    @classmethod
    def build_review_doc(
        cls,
        review: Dict[str, Any],
        pr_number: int,
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from a PR Review record."""
        review_id = review.get("github_review_id") or review.get("id")
        if not review_id:
            return None

        reviewer_id = review.get("reviewer_id")
        reviewer = dev_map.get(reviewer_id, {}) if reviewer_id else {}
        reviewer_name = reviewer.get("login") or reviewer.get("name") or "Unknown"
        body = (review.get("body") or "").strip()
        state = (review.get("state") or "COMMENTED").strip()
        submitted_at = review.get("submitted_at") or ""

        text_parts = [
            f"Repository: {repo_name}",
            f"Pull Request: #{pr_number}",
            "",
            "Code Review:",
            f"Reviewer: {reviewer_name}",
            f"State: {state}",
            f"Date: {submitted_at}" if submitted_at else "",
            f"Review Feedback:\n{body if body else f'Status marked as {state}'}",
        ]
        text = "\n".join(p for p in text_parts if p).strip()

        stable_key = f"review:{repo_name}:{review_id}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="review",
            repository=repo_name,
            developer=reviewer_name,
            source_id=str(review_id),
            text=text,
            metadata={
                "review_id": review_id,
                "pr_number": pr_number,
                "state": state,
                "submitted_at": submitted_at,
            },
        )

    @classmethod
    def build_review_comment_doc(
        cls,
        comment: Dict[str, Any],
        pr_number: int,
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from an inline PR diff review comment."""
        comment_id = comment.get("github_comment_id") or comment.get("id")
        if not comment_id:
            return None

        dev_id = comment.get("developer_id")
        dev = dev_map.get(dev_id, {}) if dev_id else {}
        author = dev.get("login") or dev.get("name") or "Unknown"
        body = (comment.get("body") or "").strip()
        path = comment.get("path") or ""
        diff_hunk = (comment.get("diff_hunk") or "").strip()
        created_at = comment.get("created_at") or ""

        text_parts = [
            f"Repository: {repo_name}",
            f"Pull Request: #{pr_number}",
            f"File: {path}" if path else "",
            "",
            "Inline Code Review Comment:",
            f"Author: {author}",
            f"Date: {created_at}" if created_at else "",
            f"Diff Context:\n{diff_hunk}" if diff_hunk else "",
            f"Comment:\n{body if body else 'No comment text'}",
        ]
        text = "\n".join(p for p in text_parts if p).strip()

        stable_key = f"review_comment:{repo_name}:{comment_id}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="review_comment",
            repository=repo_name,
            developer=author,
            source_id=str(comment_id),
            text=text,
            metadata={
                "comment_id": comment_id,
                "pr_number": pr_number,
                "file_path": path,
                "created_at": created_at,
            },
        )

    @classmethod
    def build_issue_doc(
        cls,
        issue: Dict[str, Any],
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from an Issue record."""
        issue_number = issue.get("github_issue_number")
        if issue_number is None:
            return None

        dev_id = issue.get("developer_id")
        dev = dev_map.get(dev_id, {}) if dev_id else {}
        creator = dev.get("login") or dev.get("name") or "Unknown"
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        state = issue.get("state") or "unknown"
        created_at = issue.get("created_at") or ""

        # Summarize comments if available
        comments_preview = ""
        if comments:
            comments_preview = f"Comments count: {len(comments)}"

        text_parts = [
            f"Repository: {repo_name}",
            "",
            "Issue:",
            f"Number: #{issue_number}",
            f"Title: {title}",
            f"Created by: {creator}",
            f"State: {state}",
            f"Created At: {created_at}" if created_at else "",
            f"Description:\n{body}" if body else "Description:\n(No description provided)",
        ]
        if comments_preview:
            text_parts.append(comments_preview)

        text = "\n".join(text_parts).strip()
        stable_key = f"issue:{repo_name}:{issue_number}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="issue",
            repository=repo_name,
            developer=creator,
            source_id=str(issue_number),
            text=text,
            metadata={
                "issue_number": issue_number,
                "title": title,
                "state": state,
                "created_at": created_at,
                "html_url": issue.get("html_url", ""),
            },
        )

    @classmethod
    def build_issue_comment_doc(
        cls,
        comment: Dict[str, Any],
        issue_number: int,
        repo_name: str,
        dev_map: Dict[int, Dict[str, Any]],
    ) -> Optional[DocumentItem]:
        """Builds a semantic document from an Issue Comment record."""
        comment_id = comment.get("github_comment_id") or comment.get("id")
        if not comment_id:
            return None

        dev_id = comment.get("developer_id")
        dev = dev_map.get(dev_id, {}) if dev_id else {}
        author = dev.get("login") or dev.get("name") or "Unknown"
        body = (comment.get("body") or "").strip()
        created_at = comment.get("created_at") or ""

        text_parts = [
            f"Repository: {repo_name}",
            f"Issue: #{issue_number}",
            "",
            "Issue Discussion Comment:",
            f"Author: {author}",
            f"Date: {created_at}" if created_at else "",
            f"Comment:\n{body if body else 'No comment text'}",
        ]
        text = "\n".join(p for p in text_parts if p).strip()

        stable_key = f"issue_comment:{repo_name}:{comment_id}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="issue_comment",
            repository=repo_name,
            developer=author,
            source_id=str(comment_id),
            text=text,
            metadata={
                "comment_id": comment_id,
                "issue_number": issue_number,
                "created_at": created_at,
            },
        )

    @classmethod
    def build_changed_file_doc(
        cls,
        cf: Dict[str, Any],
        pr_number: int,
        repo_name: str,
    ) -> Optional[DocumentItem]:
        """Builds a semantic document for changed files in a pull request."""
        filename = cf.get("filename")
        if not filename:
            return None

        status = cf.get("status") or "modified"
        additions = cf.get("additions", 0)
        deletions = cf.get("deletions", 0)
        changes = cf.get("changes", 0)
        patch = (cf.get("patch") or "").strip()

        # Limit patch size to 1500 chars to avoid vector noise
        if len(patch) > 1500:
            patch = patch[:1500] + "\n... [diff truncated]"

        text_parts = [
            f"Repository: {repo_name}",
            f"Pull Request: #{pr_number}",
            "",
            f"Changed File: {filename}",
            f"Change Status: {status}",
            f"Stats: +{additions} / -{deletions} ({changes} changes)",
        ]
        if patch:
            text_parts.append(f"Diff Patch Preview:\n{patch}")

        text = "\n".join(text_parts).strip()
        stable_key = f"file:{repo_name}:{pr_number}:{filename}"
        point_id = cls._create_deterministic_id(stable_key)

        return DocumentItem(
            point_id=point_id,
            stable_key=stable_key,
            document_type="changed_file",
            repository=repo_name,
            developer="Unknown",
            source_id=filename,
            text=text,
            metadata={
                "filename": filename,
                "pr_number": pr_number,
                "status": status,
                "additions": additions,
                "deletions": deletions,
            },
        )

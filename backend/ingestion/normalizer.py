"""
Normalized entity models and transformation functions for GitHub data.

Converts diverse GitHub API payload representations into consistent,
relational, and type-safe internal models ready for future storage layers.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. Normalized Pydantic Entity Models
# =====================================================================

class NormalizedRepository(BaseModel):
    """Normalized repository entity."""
    repository_id: Optional[int] = None
    name: str
    full_name: str
    owner_login: str
    html_url: Optional[str] = None


class NormalizedCommit(BaseModel):
    """Normalized commit entity preserving developer and repository relationships."""
    sha: str
    repository: str
    message: str = ""
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    author_login: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None


class NormalizedPullRequest(BaseModel):
    """Normalized pull request entity preserving author and repository relationships."""
    number: int
    repository: str
    title: str
    body: Optional[str] = None
    state: str
    author_login: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    merged_at: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    url: Optional[str] = None


class NormalizedReview(BaseModel):
    """Normalized review entity linking reviewer to PR and repository."""
    id: int
    repository: str
    pull_request_number: int
    reviewer_login: Optional[str] = None
    body: Optional[str] = None
    state: str
    submitted_at: Optional[str] = None
    commit_id: Optional[str] = None
    url: Optional[str] = None


class NormalizedReviewComment(BaseModel):
    """Normalized inline diff review comment linking commenter to PR and code location."""
    id: int
    repository: str
    pull_request_number: int
    commenter_login: Optional[str] = None
    body: Optional[str] = None
    path: Optional[str] = None
    line: Optional[int] = None
    diff_hunk: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    commit_id: Optional[str] = None
    url: Optional[str] = None


class NormalizedIssue(BaseModel):
    """Normalized issue entity linking author to repository (excluding pull requests)."""
    number: int
    repository: str
    title: str
    body: Optional[str] = None
    state: str
    author_login: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    url: Optional[str] = None


class NormalizedIssueComment(BaseModel):
    """Normalized issue discussion comment linking commenter to issue and repository."""
    id: int
    repository: str
    issue_number: int
    commenter_login: Optional[str] = None
    body: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    url: Optional[str] = None


class NormalizedChangedFile(BaseModel):
    """Normalized changed file entity linking file diff to pull request and repository."""
    repository: str
    pull_request_number: int
    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    sha: Optional[str] = None
    patch: Optional[str] = None
    blob_url: Optional[str] = None


# =====================================================================
# 2. Ingestion Response Summaries
# =====================================================================

class IngestionCounts(BaseModel):
    """Entity counts produced during an ingestion run."""
    commits: int = 0
    pull_requests: int = 0
    reviews: int = 0
    review_comments: int = 0
    issues: int = 0
    issue_comments: int = 0
    changed_files: int = 0


class IngestionSummaryResponse(BaseModel):
    """Summary response returned by the ingestion endpoint."""
    repository: str
    status: str = "completed"
    counts: IngestionCounts


# =====================================================================
# 3. Normalization Helper Functions
# =====================================================================

def normalize_repository(raw_repo: Dict[str, Any], default_owner: str, default_repo: str) -> NormalizedRepository:
    """Normalize raw repository data into NormalizedRepository."""
    owner_login = raw_repo.get("owner_login") or default_owner
    repo_name = raw_repo.get("name") or default_repo
    full_name = raw_repo.get("full_name") or f"{owner_login}/{repo_name}"
    html_url = raw_repo.get("html_url") or raw_repo.get("url") or f"https://github.com/{full_name}"

    return NormalizedRepository(
        repository_id=raw_repo.get("repository_id") or raw_repo.get("id"),
        name=repo_name,
        full_name=full_name,
        owner_login=owner_login,
        html_url=html_url,
    )


def normalize_commit(raw_commit: Dict[str, Any], repo_full_name: str) -> NormalizedCommit:
    """Normalize commit data into NormalizedCommit."""
    return NormalizedCommit(
        sha=raw_commit.get("sha", ""),
        repository=repo_full_name,
        message=raw_commit.get("message", ""),
        author_name=raw_commit.get("author_name"),
        author_email=raw_commit.get("author_email"),
        author_login=raw_commit.get("author_login"),
        date=raw_commit.get("date"),
        url=raw_commit.get("url"),
    )


def normalize_pull_request(raw_pr: Dict[str, Any], repo_full_name: str) -> NormalizedPullRequest:
    """Normalize pull request data into NormalizedPullRequest."""
    # Standardize author_login from either author_login or user_login
    author_login = raw_pr.get("author_login") or raw_pr.get("user_login")

    return NormalizedPullRequest(
        number=raw_pr.get("number", 0),
        repository=repo_full_name,
        title=raw_pr.get("title", ""),
        body=raw_pr.get("body"),
        state=raw_pr.get("state", "open"),
        author_login=author_login,
        created_at=raw_pr.get("created_at"),
        updated_at=raw_pr.get("updated_at"),
        closed_at=raw_pr.get("closed_at"),
        merged_at=raw_pr.get("merged_at"),
        merge_commit_sha=raw_pr.get("merge_commit_sha"),
        url=raw_pr.get("html_url") or raw_pr.get("url"),
    )


def normalize_review(raw_review: Dict[str, Any], repo_full_name: str, pr_number: int) -> NormalizedReview:
    """Normalize pull request review data into NormalizedReview."""
    reviewer_login = raw_review.get("reviewer_login") or raw_review.get("user_login")

    return NormalizedReview(
        id=raw_review.get("id", 0),
        repository=repo_full_name,
        pull_request_number=pr_number,
        reviewer_login=reviewer_login,
        body=raw_review.get("body"),
        state=raw_review.get("state", "COMMENTED"),
        submitted_at=raw_review.get("submitted_at"),
        commit_id=raw_review.get("commit_id"),
        url=raw_review.get("html_url") or raw_review.get("url"),
    )


def normalize_review_comment(raw_comment: Dict[str, Any], repo_full_name: str, pr_number: int) -> NormalizedReviewComment:
    """Normalize inline code review comment data into NormalizedReviewComment."""
    commenter_login = raw_comment.get("commenter_login") or raw_comment.get("user_login")

    return NormalizedReviewComment(
        id=raw_comment.get("id", 0),
        repository=repo_full_name,
        pull_request_number=pr_number,
        commenter_login=commenter_login,
        body=raw_comment.get("body"),
        path=raw_comment.get("path"),
        line=raw_comment.get("line"),
        diff_hunk=raw_comment.get("diff_hunk"),
        created_at=raw_comment.get("created_at"),
        updated_at=raw_comment.get("updated_at"),
        commit_id=raw_comment.get("commit_id"),
        url=raw_comment.get("html_url") or raw_comment.get("url"),
    )


def normalize_issue(raw_issue: Dict[str, Any], repo_full_name: str) -> NormalizedIssue:
    """Normalize issue data into NormalizedIssue."""
    author_login = raw_issue.get("author_login") or raw_issue.get("user_login")

    return NormalizedIssue(
        number=raw_issue.get("number", 0),
        repository=repo_full_name,
        title=raw_issue.get("title", ""),
        body=raw_issue.get("body"),
        state=raw_issue.get("state", "open"),
        author_login=author_login,
        created_at=raw_issue.get("created_at"),
        updated_at=raw_issue.get("updated_at"),
        closed_at=raw_issue.get("closed_at"),
        url=raw_issue.get("html_url") or raw_issue.get("url"),
    )


def normalize_issue_comment(raw_comment: Dict[str, Any], repo_full_name: str, issue_number: int) -> NormalizedIssueComment:
    """Normalize issue discussion comment data into NormalizedIssueComment."""
    commenter_login = raw_comment.get("commenter_login") or raw_comment.get("user_login")

    return NormalizedIssueComment(
        id=raw_comment.get("id", 0),
        repository=repo_full_name,
        issue_number=issue_number,
        commenter_login=commenter_login,
        body=raw_comment.get("body"),
        created_at=raw_comment.get("created_at"),
        updated_at=raw_comment.get("updated_at"),
        url=raw_comment.get("html_url") or raw_comment.get("url"),
    )


def normalize_changed_file(raw_file: Dict[str, Any], repo_full_name: str, pr_number: int) -> NormalizedChangedFile:
    """Normalize changed file data into NormalizedChangedFile."""
    return NormalizedChangedFile(
        repository=repo_full_name,
        pull_request_number=pr_number,
        filename=raw_file.get("filename", ""),
        status=raw_file.get("status", "modified"),
        additions=raw_file.get("additions", 0) or 0,
        deletions=raw_file.get("deletions", 0) or 0,
        changes=raw_file.get("changes", 0) or 0,
        sha=raw_file.get("sha"),
        patch=raw_file.get("patch"),
        blob_url=raw_file.get("blob_url"),
    )

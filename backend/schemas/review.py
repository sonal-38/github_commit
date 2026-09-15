from typing import List, Optional
from pydantic import BaseModel


class PullRequestReviewItem(BaseModel):
    """Clean representation of a pull request review."""
    id: int
    user_login: Optional[str] = None
    body: Optional[str] = None
    state: str
    submitted_at: Optional[str] = None
    commit_id: Optional[str] = None
    html_url: Optional[str] = None


class PullRequestReviewListResponse(BaseModel):
    """List response for pull request reviews."""
    repository: str
    pull_number: int
    total_reviews: int
    reviews: List[PullRequestReviewItem]


class PullRequestReviewCommentItem(BaseModel):
    """Clean representation of a pull request review comment (code comment/diff discussion)."""
    id: int
    user_login: Optional[str] = None
    body: Optional[str] = None
    path: Optional[str] = None
    line: Optional[int] = None
    diff_hunk: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    commit_id: Optional[str] = None
    html_url: Optional[str] = None


class PullRequestReviewCommentListResponse(BaseModel):
    """List response for pull request review comments."""
    repository: str
    pull_number: int
    total_comments: int
    comments: List[PullRequestReviewCommentItem]

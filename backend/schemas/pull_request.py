from typing import List, Optional
from pydantic import BaseModel


class PullRequestItem(BaseModel):
    """Clean representation of a single GitHub pull request."""
    number: int
    title: str
    body: Optional[str] = None
    state: str
    user_login: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    merged_at: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    html_url: Optional[str] = None


class PullRequestListResponse(BaseModel):
    """Complete pull requests response for a repository."""
    repository: str
    total_pull_requests: int
    pull_requests: List[PullRequestItem]

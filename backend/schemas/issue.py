from typing import List, Optional
from pydantic import BaseModel


class IssueItem(BaseModel):
    """Clean representation of a GitHub issue (excluding pull requests)."""
    number: int
    title: str
    body: Optional[str] = None
    state: str
    user_login: Optional[str] = None
    labels: List[str] = []
    assignees: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    html_url: Optional[str] = None


class IssueListResponse(BaseModel):
    """Complete issues response for a repository."""
    repository: str
    total_issues: int
    issues: List[IssueItem]


class IssueCommentItem(BaseModel):
    """Clean representation of a GitHub issue comment."""
    id: int
    user_login: Optional[str] = None
    body: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    html_url: Optional[str] = None


class IssueCommentListResponse(BaseModel):
    """Complete comments response for an issue."""
    repository: str
    issue_number: int
    total_comments: int
    comments: List[IssueCommentItem]

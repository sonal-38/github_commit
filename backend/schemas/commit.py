from typing import List, Optional
from pydantic import BaseModel


class CommitItem(BaseModel):
    """Clean representation of a single GitHub commit."""
    sha: str
    message: str
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    author_login: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None


class CommitHistoryResponse(BaseModel):
    """Complete commit history response for a repository."""
    repository: str
    total_commits: int
    commits: List[CommitItem]

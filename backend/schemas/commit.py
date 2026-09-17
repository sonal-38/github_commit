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


class CommitFileItem(BaseModel):
    """Clean representation of a file changed in a commit."""
    filename: str
    status: Optional[str] = "modified"
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: Optional[str] = None
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None


class CommitDetailResponse(BaseModel):
    """Detailed commit response including list of changed files."""
    sha: str
    repository: str
    message: str = ""
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    total_files: int = 0
    files: List[CommitFileItem] = []


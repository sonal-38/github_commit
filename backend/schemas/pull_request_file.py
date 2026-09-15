from typing import List, Optional
from pydantic import BaseModel


class PullRequestFileItem(BaseModel):
    """Clean representation of a changed file in a pull request."""
    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None
    contents_url: Optional[str] = None
    sha: Optional[str] = None
    patch: Optional[str] = None


class PullRequestFileListResponse(BaseModel):
    """Complete changed files response for a pull request."""
    repository: str
    pull_request_number: int
    total_files: int
    files: List[PullRequestFileItem]

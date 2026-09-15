from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from github.client import GitHubClient, GitHubClientError
from schemas.commit import CommitHistoryResponse

router = APIRouter(
    prefix="/github",
    tags=["GitHub"]
)


@router.get("/repositories", response_model=List[Dict[str, Any]])
def get_repositories():
    """
    Retrieve accessible repositories for the configured GitHub account.

    Returns:
        List of repository objects containing name, owner, private, and url.
    """
    client = GitHubClient()
    try:
        repositories = client.get_repositories()
        return repositories
    except GitHubClientError as err:
        # Re-raise with the specific status code and safe message
        raise HTTPException(
            status_code=err.status_code,
            detail=err.message
        )
    except Exception:
        # Catch unexpected errors without leaking tokens or internal details
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while communicating with GitHub"
        )


@router.get("/repositories/{owner}/{repo}/commits", response_model=CommitHistoryResponse)
def get_repository_commits(owner: str, repo: str):
    """
    Retrieve all commits for a specified repository with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')

    Returns:
        CommitHistoryResponse containing repository name, total_commits, and the list of clean commits.
    """
    client = GitHubClient()
    try:
        commits = client.get_repository_commits(owner=owner, repo=repo)
        return CommitHistoryResponse(
            repository=repo,
            total_commits=len(commits),
            commits=commits,
        )
    except GitHubClientError as err:
        raise HTTPException(
            status_code=err.status_code,
            detail=err.message
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while communicating with GitHub"
        )

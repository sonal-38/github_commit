from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from github.client import GitHubClient, GitHubClientError

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

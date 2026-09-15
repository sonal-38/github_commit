from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from github.client import GitHubClient, GitHubClientError
from schemas.commit import CommitHistoryResponse
from schemas.pull_request import PullRequestListResponse
from schemas.review import (
    PullRequestReviewListResponse,
    PullRequestReviewCommentListResponse,
)
from schemas.issue import (
    IssueListResponse,
    IssueCommentListResponse,
)
from schemas.pull_request_file import PullRequestFileListResponse
from ingestion.normalizer import IngestionSummaryResponse
from ingestion.service import IngestionService
from database.supabase_client import SupabaseConfigurationError, SupabaseDatabaseError

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


@router.get("/repositories/{owner}/{repo}/pull-requests", response_model=PullRequestListResponse)
def get_repository_pull_requests(owner: str, repo: str):
    """
    Retrieve all pull requests (open, closed, merged) for a specified repository with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')

    Returns:
        PullRequestListResponse containing repository name, total_pull_requests, and the list of clean pull requests.
    """
    client = GitHubClient()
    try:
        pull_requests = client.get_repository_pull_requests(owner=owner, repo=repo)
        return PullRequestListResponse(
            repository=repo,
            total_pull_requests=len(pull_requests),
            pull_requests=pull_requests,
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


@router.get("/repositories/{owner}/{repo}/pull-requests/{pull_number}/reviews", response_model=PullRequestReviewListResponse)
def get_pull_request_reviews(owner: str, repo: str, pull_number: int):
    """
    Retrieve all reviews for a specific pull request with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')
        pull_number: Pull request number (e.g., 1)

    Returns:
        PullRequestReviewListResponse containing repository, pull_number, total_reviews, and clean reviews.
    """
    client = GitHubClient()
    try:
        reviews = client.get_pull_request_reviews(owner=owner, repo=repo, pull_number=pull_number)
        return PullRequestReviewListResponse(
            repository=repo,
            pull_number=pull_number,
            total_reviews=len(reviews),
            reviews=reviews,
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


@router.get("/repositories/{owner}/{repo}/pull-requests/{pull_number}/comments", response_model=PullRequestReviewCommentListResponse)
def get_pull_request_comments(owner: str, repo: str, pull_number: int):
    """
    Retrieve all review comments (inline code diff comments) for a specific pull request with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')
        pull_number: Pull request number (e.g., 1)

    Returns:
        PullRequestReviewCommentListResponse containing repository, pull_number, total_comments, and clean comments.
    """
    client = GitHubClient()
    try:
        comments = client.get_pull_request_comments(owner=owner, repo=repo, pull_number=pull_number)
        return PullRequestReviewCommentListResponse(
            repository=repo,
            pull_number=pull_number,
            total_comments=len(comments),
            comments=comments,
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


@router.get("/repositories/{owner}/{repo}/issues", response_model=IssueListResponse)
def get_repository_issues(owner: str, repo: str):
    """
    Retrieve all issues for a specified repository with complete pagination.
    Note: Pull requests are filtered out and excluded from this response.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')

    Returns:
        IssueListResponse containing repository, total_issues, and clean issues.
    """
    client = GitHubClient()
    try:
        issues = client.get_repository_issues(owner=owner, repo=repo)
        return IssueListResponse(
            repository=repo,
            total_issues=len(issues),
            issues=issues,
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


@router.get("/repositories/{owner}/{repo}/issues/{issue_number}/comments", response_model=IssueCommentListResponse)
def get_issue_comments(owner: str, repo: str, issue_number: int):
    """
    Retrieve all comments for a specific issue with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')
        issue_number: Issue number (e.g., 1)

    Returns:
        IssueCommentListResponse containing repository, issue_number, total_comments, and clean comments.
    """
    client = GitHubClient()
    try:
        comments = client.get_issue_comments(owner=owner, repo=repo, issue_number=issue_number)
        return IssueCommentListResponse(
            repository=repo,
            issue_number=issue_number,
            total_comments=len(comments),
            comments=comments,
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


@router.get("/repositories/{owner}/{repo}/pull-requests/{pull_number}/files", response_model=PullRequestFileListResponse)
def get_pull_request_files(owner: str, repo: str, pull_number: int):
    """
    Retrieve all changed files for a specific pull request with complete pagination.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')
        pull_number: Pull request number (e.g., 1)

    Returns:
        PullRequestFileListResponse containing repository, pull_request_number, total_files, and clean changed files.
    """
    client = GitHubClient()
    try:
        files = client.get_pull_request_files(owner=owner, repo=repo, pull_number=pull_number)
        return PullRequestFileListResponse(
            repository=repo,
            pull_request_number=pull_number,
            total_files=len(files),
            files=files,
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


@router.post("/repositories/{owner}/{repo}/ingest", response_model=IngestionSummaryResponse)
def ingest_repository(owner: str, repo: str):
    """
    Ingest, normalize, and store all engineering data for a specified repository in Supabase PostgreSQL.
    Fetches repositories, commits, PRs, reviews, comments, issues, issue comments,
    and changed files, transforming them into a consistent internal normalized representation
    and persisting them into Supabase relational tables with foreign-key relationships.

    Args:
        owner: GitHub username or organization (e.g., 'sonal-38')
        repo: Repository name (e.g., 'smart-payment-platform')

    Returns:
        IngestionSummaryResponse containing repository name, status, storage provider, and entity counts.
    """
    try:
        service = IngestionService()
        summary = service.ingest_repository(owner=owner, repo=repo)
        return summary
    except SupabaseConfigurationError as cfg_err:
        raise HTTPException(
            status_code=400,
            detail=str(cfg_err)
        )
    except SupabaseDatabaseError as db_err:
        raise HTTPException(
            status_code=db_err.status_code,
            detail=db_err.message
        )
    except GitHubClientError as err:
        raise HTTPException(
            status_code=err.status_code,
            detail=err.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during repository ingestion and database storage"
        )





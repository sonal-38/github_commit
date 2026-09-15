"""
Ingestion service coordinating GitHub data collection and normalization.

Pulls data across all Step 1–6 GitHub endpoints, applies the normalization
pipeline, preserves relational links, and produces structured in-memory records.
"""
from typing import Optional
from github.client import GitHubClient, GitHubClientError
from ingestion.normalizer import (
    NormalizedRepository,
    NormalizedCommit,
    NormalizedPullRequest,
    NormalizedReview,
    NormalizedReviewComment,
    NormalizedIssue,
    NormalizedIssueComment,
    NormalizedChangedFile,
    IngestionCounts,
    IngestionSummaryResponse,
    normalize_repository,
    normalize_commit,
    normalize_pull_request,
    normalize_review,
    normalize_review_comment,
    normalize_issue,
    normalize_issue_comment,
    normalize_changed_file,
)


class IngestionService:
    """Service to ingest and normalize GitHub repository data."""

    def __init__(self, client: Optional[GitHubClient] = None):
        self.client = client or GitHubClient()

    def ingest_repository(self, owner: str, repo: str) -> IngestionSummaryResponse:
        """
        Execute full data ingestion and normalization for a given repository.

        Coordinates:
        1. Repository metadata fetching & normalization
        2. Commit history collection & normalization
        3. Pull Request collection & normalization
        4. PR Reviews, Review Comments, and Changed Files collection per PR
        5. Issues collection & normalization (PRs excluded)
        6. Issue Comments collection per Issue

        Returns:
            IngestionSummaryResponse with accurate entity counts.
        """
        repo_full_name = f"{owner}/{repo}"

        # 1. Fetch Repository Metadata
        raw_repo_data = {}
        if hasattr(self.client, "get_repository"):
            try:
                raw_repo_data = self.client.get_repository(owner, repo)
            except GitHubClientError:
                # If single repo fetch fails or isn't available, fallback to basic details
                raw_repo_data = {"name": repo, "owner_login": owner, "full_name": repo_full_name}
        else:
            raw_repo_data = {"name": repo, "owner_login": owner, "full_name": repo_full_name}

        _normalized_repo = normalize_repository(raw_repo_data, default_owner=owner, default_repo=repo)

        # 2. Fetch & Normalize Commits
        raw_commits = self.client.get_repository_commits(owner=owner, repo=repo)
        normalized_commits = [
            normalize_commit(commit, repo_full_name)
            for commit in raw_commits
        ]

        # 3. Fetch & Normalize Pull Requests
        raw_prs = self.client.get_repository_pull_requests(owner=owner, repo=repo)
        normalized_prs = [
            normalize_pull_request(pr, repo_full_name)
            for pr in raw_prs
        ]

        # 4. Fetch & Normalize PR Sub-resources (Reviews, Review Comments, Changed Files)
        normalized_reviews = []
        normalized_review_comments = []
        normalized_changed_files = []

        for pr in normalized_prs:
            pr_num = pr.number

            # Fetch Reviews for this PR
            raw_reviews = self.client.get_pull_request_reviews(owner=owner, repo=repo, pull_number=pr_num)
            for review in raw_reviews:
                normalized_reviews.append(normalize_review(review, repo_full_name, pr_num))

            # Fetch Review Comments (diff discussions) for this PR
            raw_review_comments = self.client.get_pull_request_comments(owner=owner, repo=repo, pull_number=pr_num)
            for comment in raw_review_comments:
                normalized_review_comments.append(normalize_review_comment(comment, repo_full_name, pr_num))

            # Fetch Changed Files for this PR
            raw_files = self.client.get_pull_request_files(owner=owner, repo=repo, pull_number=pr_num)
            for file_item in raw_files:
                normalized_changed_files.append(normalize_changed_file(file_item, repo_full_name, pr_num))

        # 5. Fetch & Normalize Issues (Excludes PRs)
        raw_issues = self.client.get_repository_issues(owner=owner, repo=repo)
        normalized_issues = [
            normalize_issue(issue, repo_full_name)
            for issue in raw_issues
        ]

        # 6. Fetch & Normalize Issue Comments
        normalized_issue_comments = []
        for issue in normalized_issues:
            issue_num = issue.number
            raw_issue_comments = self.client.get_issue_comments(owner=owner, repo=repo, issue_number=issue_num)
            for comment in raw_issue_comments:
                normalized_issue_comments.append(normalize_issue_comment(comment, repo_full_name, issue_num))

        # 7. Construct Summary Response with Counts
        counts = IngestionCounts(
            commits=len(normalized_commits),
            pull_requests=len(normalized_prs),
            reviews=len(normalized_reviews),
            review_comments=len(normalized_review_comments),
            issues=len(normalized_issues),
            issue_comments=len(normalized_issue_comments),
            changed_files=len(normalized_changed_files),
        )

        return IngestionSummaryResponse(
            repository=repo_full_name,
            status="completed",
            counts=counts,
        )

import os
from typing import Any, Dict, List
import requests


class GitHubClientError(Exception):
    """Custom exception raised when GitHub operations fail."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GitHubClient:
    """Simple client to interact with GitHub REST API."""

    BASE_URL = "https://api.github.com"

    def __init__(self):
        # Read the token from environment variables
        self.token = os.getenv("GITHUB_TOKEN")

    def _get_headers(self) -> Dict[str, str]:
        """Validate and return authentication headers for GitHub API."""
        if not self.token or self.token.strip() == "" or self.token.strip() == "your_token_here":
            raise GitHubClientError("GitHub token is not configured", status_code=500)

        return {
            "Authorization": f"Bearer {self.token.strip()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "AI-Digital-Shadow-Backend",
        }

    def get_repositories(self) -> List[Dict[str, Any]]:
        """
        Fetch repositories accessible to the authenticated user.

        Returns a simplified list of repositories with:
        - name: repository name
        - owner: login of repository owner
        - private: boolean indicating if repository is private
        - url: html URL to view repository in browser
        """
        headers = self._get_headers()
        url = f"{self.BASE_URL}/user/repos"
        params = {
            "affiliation": "owner,collaborator,organization_member",
            "sort": "updated",
            "per_page": 100,
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(
                f"Failed to connect to GitHub API: {str(e)}",
                status_code=503,
            )

        # Handle specific HTTP status codes
        if response.status_code == 401:
            raise GitHubClientError(
                "GitHub token is invalid or expired",
                status_code=401,
            )
        elif response.status_code != 200:
            raise GitHubClientError(
                f"GitHub API returned error status {response.status_code}",
                status_code=response.status_code,
            )

        # Parse JSON data
        try:
            repos_data = response.json()
        except ValueError:
            raise GitHubClientError(
                "Received invalid JSON response from GitHub API",
                status_code=502,
            )

        # Format and extract only the required fields
        parsed_repositories: List[Dict[str, Any]] = []
        for repo in repos_data:
            owner_data = repo.get("owner") or {}
            parsed_repositories.append({
                "name": repo.get("name"),
                "owner": owner_data.get("login"),
                "private": repo.get("private", False),
                "url": repo.get("html_url"),
            })

        return parsed_repositories

    def get_repository_commits(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Fetch all commits for a given repository with pagination.
        Processes pages of 100 commits until no more are returned.
        """
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits"
        all_commits: List[Dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            params = {
                "per_page": per_page,
                "page": page,
            }

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=15,
                )
            except requests.exceptions.RequestException as e:
                raise GitHubClientError(
                    f"Failed to connect to GitHub API: {str(e)}",
                    status_code=503,
                )

            # Handle specific HTTP status codes
            if response.status_code == 404:
                raise GitHubClientError("GitHub repository not found", status_code=404)
            elif response.status_code == 401:
                raise GitHubClientError("GitHub token is invalid or expired", status_code=401)
            elif response.status_code == 403:
                raise GitHubClientError("GitHub API rate limit exceeded or access forbidden", status_code=403)
            elif response.status_code != 200:
                raise GitHubClientError(
                    f"GitHub API returned error status {response.status_code}",
                    status_code=response.status_code,
                )

            try:
                commits_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            # If response is not a list or is empty, we reached the end
            if not isinstance(commits_data, list) or len(commits_data) == 0:
                break

            for item in commits_data:
                commit_info = item.get("commit") or {}
                git_author = commit_info.get("author") or {}
                git_committer = commit_info.get("committer") or {}
                gh_author = item.get("author") or {}

                # Safely resolve author fields with sensible defaults
                author_name = git_author.get("name") or git_committer.get("name")
                author_email = git_author.get("email") or git_committer.get("email")
                author_login = gh_author.get("login")
                date = git_author.get("date") or git_committer.get("date")

                all_commits.append({
                    "sha": item.get("sha", ""),
                    "message": commit_info.get("message", ""),
                    "author_name": author_name,
                    "author_email": author_email,
                    "author_login": author_login,
                    "date": date,
                    "url": item.get("html_url"),
                })

            # Safe stopping condition: fewer items than per_page means we are on the final page
            if len(commits_data) < per_page:
                break

            page += 1

        return all_commits

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

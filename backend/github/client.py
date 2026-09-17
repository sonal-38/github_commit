import os
from typing import Any, Dict, List, Optional, Tuple
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

    def __init__(self, token: Optional[str] = None):
        # Read the token from argument or environment variables
        self.token = token if token is not None else os.getenv("GITHUB_TOKEN")

    @staticmethod
    def _clean_owner_repo(owner: str, repo: str) -> tuple[str, str]:
        """
        Normalize and sanitize owner and repo strings.
        Handles cases where users input 'owner/repo' in either parameter,
        or include leading/trailing whitespace and slashes.
        """
        owner_clean = (owner or "").strip(" \t\n\r/")
        repo_clean = (repo or "").strip(" \t\n\r/")

        if "/" in repo_clean:
            parts = [p.strip() for p in repo_clean.split("/") if p.strip()]
            if len(parts) >= 2:
                owner_clean = parts[0]
                repo_clean = parts[1]
            elif len(parts) == 1:
                repo_clean = parts[0]
        elif "/" in owner_clean:
            parts = [p.strip() for p in owner_clean.split("/") if p.strip()]
            if len(parts) >= 2:
                owner_clean = parts[0]
                repo_clean = parts[1]
            elif len(parts) == 1:
                owner_clean = parts[0]

        return owner_clean, repo_clean

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
        - full_name: owner/name format
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
                "full_name": repo.get("full_name") or f"{owner_data.get('login')}/{repo.get('name')}",
                "private": repo.get("private", False),
                "url": repo.get("html_url"),
            })

        return parsed_repositories

    def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetch repository details for a specific repository.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        try:
            response = requests.get(url, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(f"Failed to connect to GitHub API: {str(e)}", status_code=503)

        if response.status_code == 404:
            raise GitHubClientError(
                f"GitHub repository '{owner}/{repo}' not found. Please verify the owner ('{owner}') and repository name ('{repo}'). If the repository is private, ensure your GITHUB_TOKEN has the 'repo' scope.",
                status_code=404,
            )
        elif response.status_code == 401:
            raise GitHubClientError("GitHub token is invalid or expired", status_code=401)
        elif response.status_code == 403:
            raise GitHubClientError("GitHub API rate limit exceeded or access forbidden", status_code=403)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API returned error status {response.status_code}", status_code=response.status_code)

        try:
            repo_data = response.json()
        except ValueError:
            raise GitHubClientError("Received invalid JSON response from GitHub API", status_code=502)

        owner_data = repo_data.get("owner") or {}
        return {
            "repository_id": repo_data.get("id"),
            "name": repo_data.get("name"),
            "full_name": repo_data.get("full_name"),
            "owner_login": owner_data.get("login"),
            "html_url": repo_data.get("html_url"),
        }

    def get_repository_commits(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Fetch all commits for a given repository with pagination.
        Processes pages of 100 commits until no more are returned.
        Handles empty repositories gracefully.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
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

            # 409 Conflict indicates an empty repository (0 commits pushed)
            if response.status_code == 409:
                return []

            # Handle specific HTTP status codes
            if response.status_code == 404:
                # Check if GitHub reported that the repository is empty
                try:
                    err_json = response.json()
                    msg = err_json.get("message", "")
                    if "empty" in msg.lower():
                        return []
                except Exception:
                    pass
                raise GitHubClientError(
                    f"GitHub repository '{owner}/{repo}' not found. Please verify the owner ('{owner}') and repository name ('{repo}'). If the repository is private, ensure your GITHUB_TOKEN has the 'repo' scope.",
                    status_code=404,
                )
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

    def get_repository_pull_requests(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Fetch all pull requests (open, closed, merged) for a repository with pagination.
        Processes pages of 100 pull requests until no more are returned.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls"
        all_prs: List[Dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            params = {
                "state": "all",
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
                raise GitHubClientError(
                    f"GitHub repository '{owner}/{repo}' not found. Please verify the owner ('{owner}') and repository name ('{repo}').",
                    status_code=404,
                )
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
                prs_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            # If response is not a list or is empty, we reached the end
            if not isinstance(prs_data, list) or len(prs_data) == 0:
                break

            for item in prs_data:
                user_info = item.get("user") or {}

                all_prs.append({
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "body": item.get("body"),
                    "state": item.get("state", "unknown"),
                    "user_login": user_info.get("login"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "closed_at": item.get("closed_at"),
                    "merged_at": item.get("merged_at"),
                    "merge_commit_sha": item.get("merge_commit_sha"),
                    "html_url": item.get("html_url"),
                })

            # Safe stopping condition: fewer items than per_page means we are on the final page
            if len(prs_data) < per_page:
                break

            page += 1

        return all_prs

    def get_pull_request_reviews(self, owner: str, repo: str, pull_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all reviews for a specific pull request with pagination.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}/reviews"
        all_reviews: List[Dict[str, Any]] = []
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Pull request #{pull_number} or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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
                reviews_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(reviews_data, list) or len(reviews_data) == 0:
                break

            for item in reviews_data:
                user_info = item.get("user") or {}
                all_reviews.append({
                    "id": item.get("id"),
                    "user_login": user_info.get("login"),
                    "body": item.get("body"),
                    "state": item.get("state", "COMMENTED"),
                    "submitted_at": item.get("submitted_at"),
                    "commit_id": item.get("commit_id"),
                    "html_url": item.get("html_url"),
                })

            if len(reviews_data) < per_page:
                break

            page += 1

        return all_reviews

    def get_pull_request_comments(self, owner: str, repo: str, pull_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all review comments (inline/diff comments) for a specific pull request with pagination.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
        all_comments: List[Dict[str, Any]] = []
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Pull request #{pull_number} or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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
                comments_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(comments_data, list) or len(comments_data) == 0:
                break

            for item in comments_data:
                user_info = item.get("user") or {}
                all_comments.append({
                    "id": item.get("id"),
                    "user_login": user_info.get("login"),
                    "body": item.get("body"),
                    "path": item.get("path"),
                    "line": item.get("line"),
                    "diff_hunk": item.get("diff_hunk"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "commit_id": item.get("commit_id"),
                    "html_url": item.get("html_url"),
                })

            if len(comments_data) < per_page:
                break

            page += 1

        return all_comments

    def get_repository_issues(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Fetch all issues for a repository (excluding pull requests) with pagination.
        Processes pages of 100 issues until no more are returned.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
        all_issues: List[Dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            params = {
                "state": "all",
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"GitHub repository '{owner}/{repo}' not found. Please verify the owner ('{owner}') and repository name ('{repo}').",
                    status_code=404,
                )
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
                issues_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(issues_data, list) or len(issues_data) == 0:
                break

            for item in issues_data:
                # GitHub's issues endpoint includes pull requests; exclude them
                if "pull_request" in item:
                    continue

                user_info = item.get("user") or {}
                raw_labels = item.get("labels") or []
                raw_assignees = item.get("assignees") or []

                labels = [
                    lbl.get("name")
                    for lbl in raw_labels
                    if isinstance(lbl, dict) and lbl.get("name")
                ]
                assignees = [
                    assignee.get("login")
                    for assignee in raw_assignees
                    if isinstance(assignee, dict) and assignee.get("login")
                ]

                all_issues.append({
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "body": item.get("body"),
                    "state": item.get("state", "open"),
                    "user_login": user_info.get("login"),
                    "labels": labels,
                    "assignees": assignees,
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "closed_at": item.get("closed_at"),
                    "html_url": item.get("html_url"),
                })

            if len(issues_data) < per_page:
                break

            page += 1

        return all_issues

    def get_issue_comments(self, owner: str, repo: str, issue_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all comments for a specific issue with pagination.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        all_comments: List[Dict[str, Any]] = []
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Issue #{issue_number} or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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
                comments_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(comments_data, list) or len(comments_data) == 0:
                break

            for item in comments_data:
                user_info = item.get("user") or {}
                all_comments.append({
                    "id": item.get("id"),
                    "user_login": user_info.get("login"),
                    "body": item.get("body"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "html_url": item.get("html_url"),
                })

            if len(comments_data) < per_page:
                break

            page += 1

        return all_comments

    def get_pull_request_files(self, owner: str, repo: str, pull_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all changed files for a specific pull request with pagination.
        Processes pages of 100 files until no more are returned.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}/files"
        all_files: List[Dict[str, Any]] = []
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Pull request #{pull_number} or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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
                files_data = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(files_data, list) or len(files_data) == 0:
                break

            for item in files_data:
                all_files.append({
                    "filename": item.get("filename", ""),
                    "status": item.get("status", "modified"),
                    "additions": item.get("additions", 0),
                    "deletions": item.get("deletions", 0),
                    "changes": item.get("changes", 0),
                    "blob_url": item.get("blob_url"),
                    "raw_url": item.get("raw_url"),
                    "contents_url": item.get("contents_url"),
                    "sha": item.get("sha"),
                    "patch": item.get("patch"),
                })

            if len(files_data) < per_page:
                break

            page += 1

        return all_files

    def get_pull_request_commits(self, owner: str, repo: str, pull_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all commits belonging to a specific pull request with pagination.
        Endpoint: GET /repos/{owner}/{repo}/pulls/{pull_number}/commits
        Processes pages of 100 commits until no more are returned.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}/commits"
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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Pull request #{pull_number} or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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

            if not isinstance(commits_data, list) or len(commits_data) == 0:
                break

            for item in commits_data:
                commit_info = item.get("commit") or {}
                git_author = commit_info.get("author") or {}
                git_committer = commit_info.get("committer") or {}
                gh_author = item.get("author") or {}

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

            if len(commits_data) < per_page:
                break

            page += 1

        return all_commits

    def get_commit_details(self, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        """
        Fetch details for a specific commit, including its changed files with pagination.
        Handles commits with many files by fetching subsequent pages of files.
        """
        owner, repo = self._clean_owner_repo(owner, repo)
        headers = self._get_headers()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits/{sha}"
        all_files: List[Dict[str, Any]] = []
        page = 1
        per_page = 100
        commit_info: Dict[str, Any] = {
            "sha": sha,
            "message": "",
            "author_name": None,
            "author_email": None,
            "date": None,
            "url": None,
        }

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

            if response.status_code == 404:
                raise GitHubClientError(
                    f"Commit '{sha}' or repository '{owner}/{repo}' not found",
                    status_code=404,
                )
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
                commit_payload = response.json()
            except ValueError:
                raise GitHubClientError(
                    "Received invalid JSON response from GitHub API",
                    status_code=502,
                )

            if not isinstance(commit_payload, dict):
                break

            if page == 1:
                raw_commit = commit_payload.get("commit") or {}
                raw_author = raw_commit.get("author") or {}
                commit_info = {
                    "sha": commit_payload.get("sha") or sha,
                    "message": raw_commit.get("message", ""),
                    "author_name": raw_author.get("name"),
                    "author_email": raw_author.get("email"),
                    "date": raw_author.get("date"),
                    "url": commit_payload.get("html_url"),
                }

            files_data = commit_payload.get("files") or []
            if not isinstance(files_data, list) or len(files_data) == 0:
                break

            for item in files_data:
                all_files.append({
                    "filename": item.get("filename", ""),
                    "status": item.get("status", "modified"),
                    "additions": item.get("additions", 0) or 0,
                    "deletions": item.get("deletions", 0) or 0,
                    "changes": item.get("changes", 0) or 0,
                    "blob_url": item.get("blob_url"),
                    "raw_url": item.get("raw_url"),
                    "contents_url": item.get("contents_url"),
                    "sha": item.get("sha"),
                    "patch": item.get("patch"),
                })

            if len(files_data) < per_page:
                break

            page += 1

        commit_info["files"] = all_files
        return commit_info

    def get_commit_files(self, owner: str, repo: str, sha: str) -> List[Dict[str, Any]]:
        """
        Convenience method to retrieve only the list of changed files for a specific commit.
        """
        details = self.get_commit_details(owner=owner, repo=repo, sha=sha)
        return details.get("files", [])



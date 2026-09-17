"""
Unit and integration tests for commit files tracking.

Verifies:
1. GitHub commit file extraction and pagination logic.
2. Normalization of commit files into NormalizedCommitFile.
3. Database storage and unique constraint handling for (repository, commit_sha, filename).
4. Full ingestion flow associating files with their parent commits.
"""
import unittest
from unittest.mock import MagicMock, patch

from github.client import GitHubClient, GitHubClientError
from ingestion.normalizer import (
    NormalizedCommit,
    NormalizedCommitFile,
    NormalizedRepository,
    normalize_commit_file,
    IngestionCounts,
)
from database.storage import SupabaseStorageService
from ingestion.service import IngestionService


class TestCommitFileNormalization(unittest.TestCase):
    """Test normalization of commit files."""

    def test_normalize_commit_file_complete(self):
        raw_file = {
            "filename": "services/payment.py",
            "status": "modified",
            "additions": 45,
            "deletions": 12,
            "changes": 57,
            "patch": "@@ -10,4 +10,6 @@ ...",
            "blob_url": "https://github.com/org/repo/blob/abc123/services/payment.py",
            "raw_url": "https://github.com/org/repo/raw/abc123/services/payment.py",
            "sha": "file_blob_sha_123",
        }
        normalized = normalize_commit_file(raw_file, "org/repo", "abc123456")

        self.assertEqual(normalized.repository, "org/repo")
        self.assertEqual(normalized.commit_sha, "abc123456")
        self.assertEqual(normalized.filename, "services/payment.py")
        self.assertEqual(normalized.status, "modified")
        self.assertEqual(normalized.additions, 45)
        self.assertEqual(normalized.deletions, 12)
        self.assertEqual(normalized.changes, 57)
        self.assertEqual(normalized.patch, "@@ -10,4 +10,6 @@ ...")
        self.assertEqual(normalized.blob_url, "https://github.com/org/repo/blob/abc123/services/payment.py")
        self.assertEqual(normalized.raw_url, "https://github.com/org/repo/raw/abc123/services/payment.py")

    def test_normalize_commit_file_missing_fields(self):
        raw_file = {
            "filename": "README.md",
        }
        normalized = normalize_commit_file(raw_file, "org/repo", "def789")

        self.assertEqual(normalized.repository, "org/repo")
        self.assertEqual(normalized.commit_sha, "def789")
        self.assertEqual(normalized.filename, "README.md")
        self.assertEqual(normalized.status, "modified")
        self.assertEqual(normalized.additions, 0)
        self.assertEqual(normalized.deletions, 0)
        self.assertEqual(normalized.changes, 0)
        self.assertIsNone(normalized.patch)
        self.assertIsNone(normalized.blob_url)
        self.assertIsNone(normalized.raw_url)


class TestGitHubClientCommitDetails(unittest.TestCase):
    """Test GitHub client get_commit_details with pagination."""

    @patch("github.client.requests.get")
    def test_get_commit_details_single_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sha": "a1b2c3d4",
            "commit": {
                "message": "fix: update payment handler",
                "author": {
                    "name": "Jane Developer",
                    "email": "jane@example.com",
                    "date": "2026-03-01T10:00:00Z",
                },
            },
            "html_url": "https://github.com/owner/repo/commit/a1b2c3d4",
            "files": [
                {
                    "filename": "src/checkout.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 2,
                    "changes": 12,
                    "patch": "@@ -1,2 +1,3 @@",
                    "blob_url": "https://github.com/owner/repo/blob/a1b2c3d4/src/checkout.py",
                    "raw_url": "https://github.com/owner/repo/raw/a1b2c3d4/src/checkout.py",
                }
            ],
        }
        mock_get.return_value = mock_response

        client = GitHubClient(token="mock_token")
        details = client.get_commit_details("owner", "repo", "a1b2c3d4")

        self.assertEqual(details["sha"], "a1b2c3d4")
        self.assertEqual(details["message"], "fix: update payment handler")
        self.assertEqual(details["author_name"], "Jane Developer")
        self.assertEqual(len(details["files"]), 1)
        self.assertEqual(details["files"][0]["filename"], "src/checkout.py")
        self.assertEqual(details["files"][0]["additions"], 10)

    @patch("github.client.requests.get")
    def test_get_commit_details_multi_page_pagination(self, mock_get):
        # Page 1: 100 files
        page1_files = [
            {"filename": f"file_{i}.py", "status": "added", "additions": 5, "deletions": 0, "changes": 5}
            for i in range(100)
        ]
        # Page 2: 25 files
        page2_files = [
            {"filename": f"file_{100 + i}.py", "status": "added", "additions": 5, "deletions": 0, "changes": 5}
            for i in range(25)
        ]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {
            "sha": "big_commit_sha",
            "commit": {"message": "Large refactor", "author": {}},
            "files": page1_files,
        }

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {
            "sha": "big_commit_sha",
            "commit": {"message": "Large refactor", "author": {}},
            "files": page2_files,
        }

        mock_get.side_effect = [resp1, resp2]

        client = GitHubClient(token="mock_token")
        details = client.get_commit_details("owner", "repo", "big_commit_sha")

        self.assertEqual(details["sha"], "big_commit_sha")
        self.assertEqual(len(details["files"]), 125)
        self.assertEqual(details["files"][0]["filename"], "file_0.py")
        self.assertEqual(details["files"][124]["filename"], "file_124.py")
        self.assertEqual(mock_get.call_count, 2)

    @patch("github.client.requests.get")
    def test_get_commit_details_no_files(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sha": "empty_commit",
            "commit": {"message": "Initial empty commit", "author": {}},
            "files": [],
        }
        mock_get.return_value = mock_response

        client = GitHubClient(token="mock_token")
        details = client.get_commit_details("owner", "repo", "empty_commit")

        self.assertEqual(len(details["files"]), 0)


class TestStorageAndDuplicateHandling(unittest.TestCase):
    """Test SupabaseStorageService handling of commit files and upsert idempotency."""

    def test_store_commit_files_upsert_payload_and_uniqueness(self):
        mock_supabase = MagicMock()

        def mock_upsert(table, data, on_conflict=None):
            if table == "repositories":
                return [{"id": 1, **(data if isinstance(data, dict) else data[0])}]
            if isinstance(data, list):
                return [{"id": idx + 1, **item} for idx, item in enumerate(data)]
            return [{"id": 1, **data}]

        mock_supabase.upsert.side_effect = mock_upsert

        storage = SupabaseStorageService(client=mock_supabase)

        repo = NormalizedRepository(
            repository_id=1,
            name="test-repo",
            full_name="org/test-repo",
            owner_login="org",
            html_url="https://github.com/org/test-repo",
        )
        commit = NormalizedCommit(
            sha="commit_sha_1",
            repository="org/test-repo",
            message="initial",
        )
        commit_files = [
            NormalizedCommitFile(
                repository="org/test-repo",
                commit_sha="commit_sha_1",
                filename="main.py",
                status="added",
                additions=20,
                deletions=0,
                changes=20,
            ),
            NormalizedCommitFile(
                repository="org/test-repo",
                commit_sha="commit_sha_1",
                filename="utils.py",
                status="added",
                additions=10,
                deletions=0,
                changes=10,
            ),
        ]

        # First store
        counts = storage.store_all(
            repository=repo,
            commits=[commit],
            pull_requests=[],
            reviews=[],
            review_comments=[],
            issues=[],
            issue_comments=[],
            changed_files=[],
            commit_files=commit_files,
        )

        self.assertEqual(counts["commit_files"], 2)

        # Verify on_conflict key is strictly (repository, commit_sha, filename)
        calls = [c for c in mock_supabase.upsert.call_args_list if c[0][0] == "commit_files"]
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertEqual(kwargs.get("on_conflict"), "repository,commit_sha,filename")

        payload = calls[0][0][1]
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["repository"], "org/test-repo")
        self.assertEqual(payload[0]["commit_sha"], "commit_sha_1")
        self.assertEqual(payload[0]["filename"], "main.py")

        # Second store (repeated ingestion simulation)
        # Updates existing rows on conflict rather than throwing duplicate errors
        counts_second = storage.store_all(
            repository=repo,
            commits=[commit],
            pull_requests=[],
            reviews=[],
            review_comments=[],
            issues=[],
            issue_comments=[],
            changed_files=[],
            commit_files=commit_files,
        )
        self.assertEqual(counts_second["commit_files"], 2)


class TestIngestionServiceFlow(unittest.TestCase):
    """Test full IngestionService coordinating commit details fetching and commit_files storage."""

    def test_ingest_repository_coordinates_commit_files(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()

        # Setup mock client returns
        mock_client.get_repository.return_value = {
            "id": 100,
            "name": "payment-api",
            "owner_login": "org",
            "full_name": "org/payment-api",
        }
        mock_client.get_repository_commits.return_value = [
            {"sha": "c1", "message": "feat: add stripe", "author_login": "alice", "date": "2026-03-01T00:00:00Z"},
            {"sha": "c2", "message": "fix: bug in webhook", "author_login": "bob", "date": "2026-03-02T00:00:00Z"},
        ]
        mock_client.get_commit_details.side_effect = lambda owner, repo, sha: {
            "sha": sha,
            "files": [
                {"filename": f"src/{sha}_file.py", "status": "modified", "additions": 15, "deletions": 5, "changes": 20}
            ] if sha == "c1" else [
                {"filename": "src/webhook.py", "status": "modified", "additions": 3, "deletions": 1, "changes": 4},
                {"filename": "tests/test_webhook.py", "status": "added", "additions": 20, "deletions": 0, "changes": 20},
            ]
        }
        mock_client.get_repository_pull_requests.return_value = []
        mock_client.get_repository_issues.return_value = []

        mock_storage.store_all.return_value = {
            "repositories": 1,
            "developers": 2,
            "commits": 2,
            "pull_requests": 0,
            "reviews": 0,
            "review_comments": 0,
            "issues": 0,
            "issue_comments": 0,
            "changed_files": 0,
            "commit_files": 3,
        }

        service = IngestionService(client=mock_client, storage_service=mock_storage)
        summary = service.ingest_repository(owner="org", repo="payment-api")

        self.assertEqual(summary.status, "completed")
        self.assertEqual(summary.counts.commits, 2)
        self.assertEqual(summary.counts.commit_files, 3)

        # Verify store_all received the commit_files
        self.assertTrue(mock_storage.store_all.called)
        _, kwargs = mock_storage.store_all.call_args
        commit_files_arg = kwargs.get("commit_files")
        self.assertIsNotNone(commit_files_arg)
        self.assertEqual(len(commit_files_arg), 3)

        # Check association: files match their parent commit
        c1_files = [f for f in commit_files_arg if f.commit_sha == "c1"]
        c2_files = [f for f in commit_files_arg if f.commit_sha == "c2"]
        self.assertEqual(len(c1_files), 1)
        self.assertEqual(c1_files[0].filename, "src/c1_file.py")
        self.assertEqual(len(c2_files), 2)
        self.assertEqual(c2_files[0].filename, "src/webhook.py")
        self.assertEqual(c2_files[1].filename, "tests/test_webhook.py")


if __name__ == "__main__":
    unittest.main()

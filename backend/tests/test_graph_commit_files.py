"""
Unit and integration tests for Commit -> CHANGED -> CommitFile Knowledge Graph relationships.

Verifies:
1. CommitFile constraint creation in init_schema.
2. Building (:Commit)-[:CHANGED]->(:CommitFile) relationships idempotently.
3. Node identifier format: repository + ":" + commit_sha + ":" + filename.
4. Handling multiple files changed by a single commit without duplicate commits.
5. Preserving (:Commit)-[:AUTHORED_BY]->(:Developer) relationships.
6. Separation of (:PullRequest)-[:CHANGED]->(:File) from CommitFiles.
7. Verification of repository summary counts including commit_files.
"""
import unittest
from unittest.mock import MagicMock, patch

from graph.builder import KnowledgeGraphBuilder


class TestGraphCommitFiles(unittest.TestCase):
    """Test suite for Commit -> CommitFile graph modeling."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.mock_neo4j = MagicMock()

    @patch("graph.builder.get_supabase_client")
    def test_init_schema_includes_commit_file_constraint(self, mock_get_sb):
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j)

        builder.init_schema()

        # Check all executed queries for CommitFile constraint
        executed_queries = [call[0][0] for call in self.mock_neo4j.execute_query.call_args_list]
        commit_file_constraints = [
            q for q in executed_queries
            if "CommitFile" in q and "IS UNIQUE" in q
        ]
        self.assertTrue(
            len(commit_file_constraints) > 0,
            "Expected uniqueness constraint for CommitFile in init_schema"
        )

    @patch("graph.builder.get_supabase_client")
    def test_build_repository_graph_creates_commit_files_and_relationships(self, mock_get_sb):
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j)

        # Mock Supabase data
        self.mock_supabase.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "name": "smart-payment-platform", "full_name": "sonal-38/smart-payment-platform", "owner_login": "sonal-38"}],
            "developers": [{"id": 10, "github_login": "alice", "name": "Alice Dev", "email": "alice@example.com"}],
            "commits": [
                {
                    "id": 101,
                    "sha": "a1b2c3d4",
                    "repository_id": 1,
                    "developer_id": 10,
                    "message": "feat: add payment retry logic and webhook",
                    "committed_at": "2026-03-01T12:00:00Z",
                    "html_url": "https://github.com/sonal-38/smart-payment-platform/commit/a1b2c3d4",
                }
            ],
            "commit_files": [
                {
                    "repository": "sonal-38/smart-payment-platform",
                    "commit_sha": "a1b2c3d4",
                    "filename": "services/payment.py",
                    "status": "modified",
                    "additions": 45,
                    "deletions": 10,
                    "changes": 55,
                    "patch": "@@ -1,5 +1,10 @@ ...",
                    "blob_url": "https://github.com/sonal-38/smart-payment-platform/blob/a1b2c3d4/services/payment.py",
                },
                {
                    "repository": "sonal-38/smart-payment-platform",
                    "commit_sha": "a1b2c3d4",
                    "filename": "services/webhook.py",
                    "status": "added",
                    "additions": 30,
                    "deletions": 0,
                    "changes": 30,
                    "patch": "@@ -0,0 +1,30 @@ ...",
                    "blob_url": "https://github.com/sonal-38/smart-payment-platform/blob/a1b2c3d4/services/webhook.py",
                },
            ],
            "pull_requests": [],
            "issues": [],
        }.get(table, [])

        result = builder.build_repository_graph("sonal-38", "smart-payment-platform")

        self.assertEqual(result["graph_build_status"], "success")
        self.assertEqual(result["nodes_created_or_updated"]["commits"], 1)
        self.assertEqual(result["nodes_created_or_updated"]["commit_files"], 2)

        # Inspect all Cypher queries executed
        query_calls = self.mock_neo4j.execute_query.call_args_list

        # 1. Verify Commit node MERGE
        commit_calls = [
            c for c in query_calls
            if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("sha") == "a1b2c3d4" and "MERGE (c:Commit" in c[0][0]
        ]
        self.assertEqual(len(commit_calls), 1, "Expected exactly 1 Commit node MERGE call")

        # 2. Verify CommitFile node MERGE calls
        commit_file_calls = [
            c for c in query_calls
            if len(c[0]) > 1 and isinstance(c[0][1], dict) and "MERGE (cf:CommitFile" in c[0][0]
        ]
        self.assertEqual(len(commit_file_calls), 2, "Expected 2 CommitFile node MERGE calls")

        # Verify identifier format: repository:sha:filename
        cf_ids = {c[0][1]["id"] for c in commit_file_calls}
        expected_ids = {
            "sonal-38/smart-payment-platform:a1b2c3d4:services/payment.py",
            "sonal-38/smart-payment-platform:a1b2c3d4:services/webhook.py",
        }
        self.assertEqual(cf_ids, expected_ids)

        # Verify properties passed to Cypher
        payment_call = [c for c in commit_file_calls if c[0][1]["filename"] == "services/payment.py"][0]
        params = payment_call[0][1]
        self.assertEqual(params["commit_sha"], "a1b2c3d4")
        self.assertEqual(params["status"], "modified")
        self.assertEqual(params["additions"], 45)
        self.assertEqual(params["deletions"], 10)
        self.assertEqual(params["changes"], 55)
        self.assertIn("MERGE (c)-[:CHANGED]->(cf)", payment_call[0][0])

        # 3. Verify Author relationship preserved
        authored_calls = [
            c for c in query_calls
            if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("dev_login") == "alice" and "AUTHORED_BY" in c[0][0]
        ]
        self.assertEqual(len(authored_calls), 1)

    @patch("graph.builder.get_supabase_client")
    def test_single_commit_multiple_files_does_not_duplicate_commit_nodes(self, mock_get_sb):
        """Verify that 1 commit with 5 files only merges 1 commit and merges 5 commit files."""
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j)

        five_files = [
            {
                "repository": "org/repo",
                "commit_sha": "c100",
                "filename": f"file_{i}.py",
                "status": "modified",
                "additions": i,
                "deletions": 0,
                "changes": i,
            }
            for i in range(5)
        ]

        self.mock_supabase.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "name": "repo", "full_name": "org/repo", "owner_login": "org"}],
            "developers": [],
            "commits": [{"id": 1, "sha": "c100", "repository_id": 1}],
            "commit_files": five_files,
            "pull_requests": [],
            "issues": [],
        }.get(table, [])

        result = builder.build_repository_graph("org", "repo")
        self.assertEqual(result["nodes_created_or_updated"]["commits"], 1)
        self.assertEqual(result["nodes_created_or_updated"]["commit_files"], 5)

        # Count commit MERGE calls
        commit_calls = [
            c for c in self.mock_neo4j.execute_query.call_args_list
            if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("sha") == "c100" and "MERGE (c:Commit" in c[0][0]
        ]
        self.assertEqual(len(commit_calls), 1)

    @patch("graph.builder.get_supabase_client")
    def test_idempotent_rebuild_uses_merge(self, mock_get_sb):
        """Verifies that all queries use MERGE to guarantee idempotency on rebuild."""
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j)

        self.mock_supabase.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "name": "repo", "full_name": "org/repo", "owner_login": "org"}],
            "developers": [{"id": 1, "github_login": "dev1"}],
            "commits": [{"id": 1, "sha": "c100", "repository_id": 1, "developer_id": 1}],
            "commit_files": [{"repository": "org/repo", "commit_sha": "c100", "filename": "app.py"}],
            "pull_requests": [],
            "issues": [],
        }.get(table, [])

        builder.build_repository_graph("org", "repo")

        # Verify that CREATE is never used without MERGE
        for c in self.mock_neo4j.execute_query.call_args_list:
            cypher = c[0][0]
            if "CREATE CONSTRAINT" in cypher:
                continue
            self.assertIn("MERGE", cypher, f"Query must use MERGE for idempotency: {cypher}")

    @patch("graph.builder.get_supabase_client")
    def test_repository_summary_includes_commit_files_count(self, mock_get_sb):
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j)

        def mock_query(cypher, params=None):
            if "MATCH (r:Repository)" in cypher and "RETURN r.id" in cypher:
                return [{"id": "org/repo", "name": "repo", "full_name": "org/repo"}]
            if "count(DISTINCT c) as commits" in cypher:
                return [{
                    "commits": 10,
                    "commit_files": 25,
                    "pull_requests": 3,
                    "reviews": 2,
                    "review_comments": 4,
                    "files": 5,
                    "issues": 1,
                    "issue_comments": 2,
                }]
            if "RETURN count(DISTINCT d) as developers" in cypher:
                return [{"developers": 4}]
            if "RETURN count(DISTINCT last(rel)) as count" in cypher:
                return [{"count": 65}]
            return []

        self.mock_neo4j.execute_query.side_effect = mock_query

        summary = builder.get_repository_summary("org", "repo")

        self.assertEqual(summary["nodes"]["commits"], 10)
        self.assertEqual(summary["nodes"]["commit_files"], 25)
        self.assertEqual(summary["nodes"]["files"], 5)
        self.assertEqual(summary["relationships_count"], 65)

    @patch("graph.builder.get_supabase_client")
    def test_pr_contains_commits_points_to_same_commit_node(self, mock_get_sb):
        """
        Verifies:
        1. (:PullRequest)-[:CONTAINS]->(:Commit) points to the SAME Commit node (id: sha).
        2. Supports 1 PR containing multiple commits (e.g. Commit A, Commit B).
        3. Preserves Commit AUTHORED_BY Developer separately from PR CREATED_BY Developer.
        4. Preserves Commit CHANGED CommitFile while PR has CHANGED File.
        """
        mock_get_sb.return_value = self.mock_supabase
        mock_github = MagicMock()
        builder = KnowledgeGraphBuilder(neo4j_client=self.mock_neo4j, github_client=mock_github)

        self.mock_supabase.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "name": "platform", "full_name": "org/platform", "owner_login": "org"}],
            "developers": [
                {"id": 101, "github_login": "pr_author", "name": "PR Author"},
                {"id": 102, "github_login": "commit_author_1", "name": "Commit Author 1"},
                {"id": 103, "github_login": "commit_author_2", "name": "Commit Author 2"},
            ],
            "commits": [
                {
                    "id": 1,
                    "sha": "sha_commit_1",
                    "repository_id": 1,
                    "developer_id": 102,
                    "message": "feat: first commit in pr",
                    "committed_at": "2026-03-01T10:00:00Z",
                },
                {
                    "id": 2,
                    "sha": "sha_commit_2",
                    "repository_id": 1,
                    "developer_id": 103,
                    "message": "fix: second commit in pr",
                    "committed_at": "2026-03-01T11:00:00Z",
                }
            ],
            "commit_files": [
                {"repository": "org/platform", "commit_sha": "sha_commit_1", "filename": "src/payment.py", "status": "modified", "additions": 20, "deletions": 5},
                {"repository": "org/platform", "commit_sha": "sha_commit_1", "filename": "src/retry.py", "status": "added", "additions": 40, "deletions": 0},
                {"repository": "org/platform", "commit_sha": "sha_commit_2", "filename": "src/retry.py", "status": "modified", "additions": 2, "deletions": 1},
            ],
            "pull_requests": [
                {
                    "id": 501,
                    "github_pr_number": 23,
                    "repository_id": 1,
                    "developer_id": 101,
                    "title": "Add retry mechanism",
                    "body": "PR description",
                    "state": "closed",
                    "created_at": "2026-03-01T09:00:00Z",
                }
            ],
            "changed_files": [
                {"pull_request_id": 501, "filename": "src/retry.py", "status": "added", "additions": 42, "deletions": 1, "changes": 43}
            ],
            "reviews": [],
            "review_comments": [],
            "issues": [],
            "issue_comments": [],
        }.get(table, [])

        # Mock GitHub PR commits API response returning both commits
        mock_github.get_pull_request_commits.return_value = [
            {"sha": "sha_commit_1", "message": "feat: first commit in pr", "author_login": "commit_author_1"},
            {"sha": "sha_commit_2", "message": "fix: second commit in pr", "author_login": "commit_author_2"},
        ]

        result = builder.build_repository_graph("org", "platform")

        # Verify GitHub client was called for PR #23
        mock_github.get_pull_request_commits.assert_called_once_with(
            owner="org", repo="platform", pull_number=23
        )

        # Inspect all calls to Neo4j execute_query
        executed = self.mock_neo4j.execute_query.call_args_list

        # Check PR CONTAINS Commit relationships
        contains_calls = [
            c for c in executed
            if "MERGE (p)-[:CONTAINS]->(c)" in c[0][0]
        ]
        self.assertEqual(len(contains_calls), 2, "Expected 2 CONTAINS relationships for PR #23")
        self.assertEqual(contains_calls[0][0][1]["pr_id"], "org/platform#23")
        self.assertEqual(contains_calls[0][0][1]["sha"], "sha_commit_1")
        self.assertEqual(contains_calls[1][0][1]["pr_id"], "org/platform#23")
        self.assertEqual(contains_calls[1][0][1]["sha"], "sha_commit_2")

        # Check that Commit nodes use stable sha identity (no separate PRCommit)
        for c in executed:
            if "MERGE (c:Commit {id: $sha})" in c[0][0]:
                sha_param = c[0][1]["sha"]
                self.assertIn(sha_param, ["sha_commit_1", "sha_commit_2"])

        # Check PR CREATED_BY is pr_author (developer 101)
        pr_created_by_calls = [
            c for c in executed
            if "MERGE (p)-[:CREATED_BY]->(d)" in c[0][0]
        ]
        self.assertEqual(len(pr_created_by_calls), 1)
        self.assertEqual(pr_created_by_calls[0][0][1]["dev_login"], "pr_author")

        # Check Commit AUTHORED_BY preserves individual commit authors (developer 102, 103)
        commit_authored_by_calls = [
            c for c in executed
            if "MERGE (c)-[:AUTHORED_BY]->(d)" in c[0][0]
        ]
        authored_pairs = {(c[0][1]["sha"], c[0][1]["dev_login"]) for c in commit_authored_by_calls}
        self.assertIn(("sha_commit_1", "commit_author_1"), authored_pairs)
        self.assertIn(("sha_commit_2", "commit_author_2"), authored_pairs)

        # Check PR CHANGED File vs Commit CHANGED CommitFile separation
        pr_file_calls = [c for c in executed if "MERGE (p)-[:CHANGED]->(f)" in c[0][0]]
        self.assertEqual(len(pr_file_calls), 1)
        self.assertEqual(pr_file_calls[0][0][1]["file_id"], "org/platform:src/retry.py")

        commit_file_calls = [c for c in executed if "MERGE (c)-[:CHANGED]->(cf)" in c[0][0]]
        # 3 commit files (2 for commit 1, 1 for commit 2)
        cf_ids = {c[0][1]["id"] for c in commit_file_calls}
        self.assertIn("org/platform:sha_commit_1:src/payment.py", cf_ids)
        self.assertIn("org/platform:sha_commit_1:src/retry.py", cf_ids)
        self.assertIn("org/platform:sha_commit_2:src/retry.py", cf_ids)


if __name__ == "__main__":
    unittest.main()

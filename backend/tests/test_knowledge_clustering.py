"""
Unit and integration tests for Step 10: Knowledge Document Construction + HDBSCAN Clustering.

Verifies:
1. Knowledge Document Construction:
   - Commit document (deterministic ID, author, message, committed_at)
   - CommitFile document (deterministic ID, diff patch, stats, filename)
   - PullRequest document (deterministic ID, title, description, created_at, state)
   - PRFile document (deterministic ID, stats, PR number, filename)
   - Separation of CommitFile vs PRFile
   - Safe null/missing value handling
   - Original event timestamp preservation
   - Deduplication
2. BGE Embedding Validation:
   - Reusing existing BGE service
   - Vector dimension check (768)
   - NaN / Inf validation
3. HDBSCAN Clustering:
   - Valid integer cluster labels (0, 1, ..., -1 for noise)
   - Handling small datasets (insufficient data)
   - Handling empty datasets (no data)
   - Cluster statistics and document type distributions
4. API Endpoints:
   - POST /knowledge/repositories/{owner}/{repo}/cluster
   - GET /knowledge/repositories/{owner}/{repo}/documents
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from knowledge.clustering import (
    KnowledgeDocument,
    format_commit_document,
    format_commit_file_document,
    format_pull_request_document,
    format_pr_file_document,
    load_knowledge_documents,
    build_embeddings,
    cluster_embeddings,
    build_cluster_summary,
    run_knowledge_clustering,
)


class TestKnowledgeDocumentConstruction(unittest.TestCase):
    """Verifies construction, deterministic IDs, and metadata of Knowledge Documents."""

    def setUp(self):
        self.repo = "sonal-38/smart_payment_platform"
        self.dev_map = {
            1: {"id": 1, "login": "sonal-38", "name": "Sonal Developer"}
        }

    def test_format_commit_document(self):
        commit = {
            "sha": "abc12345",
            "developer_id": 1,
            "message": "Add retry handling for failed payment requests",
            "committed_at": "2026-03-01T10:30:00Z",
            "html_url": "https://github.com/sonal-38/smart_payment_platform/commit/abc12345",
        }
        commit_files = [
            {"filename": "payment/retry.py"},
            {"filename": "payment/service.py"},
        ]

        doc = format_commit_document(commit, self.repo, self.dev_map, commit_files)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "commit:sonal-38/smart_payment_platform:abc12345")
        self.assertEqual(doc.document_type, "commit")
        self.assertEqual(doc.source_id, "abc12345")
        self.assertEqual(doc.repository, self.repo)
        self.assertIn("Add retry handling for failed payment requests", doc.text)
        self.assertIn("Author:\nsonal-38", doc.text)
        self.assertIn("payment/retry.py", doc.text)
        self.assertEqual(doc.metadata["committed_at"], "2026-03-01T10:30:00Z")
        self.assertEqual(doc.metadata["author_login"], "sonal-38")

    def test_format_commit_file_document(self):
        cf = {
            "commit_sha": "abc12345",
            "filename": "payment/retry.py",
            "status": "modified",
            "additions": 25,
            "deletions": 5,
            "changes": 30,
            "patch": "@@ -1,5 +1,10 @@\n+retry_delay = 2.0",
        }
        commit_info = {
            "sha": "abc12345",
            "developer_id": 1,
            "message": "Add retry handling",
            "committed_at": "2026-03-01T10:30:00Z",
        }

        doc = format_commit_file_document(cf, self.repo, commit_info, self.dev_map)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "commit_file:sonal-38/smart_payment_platform:abc12345:payment/retry.py")
        self.assertEqual(doc.document_type, "commit_file")
        self.assertEqual(doc.source_id, "abc12345:payment/retry.py")
        self.assertIn("Changed file:\npayment/retry.py", doc.text)
        self.assertIn("+25 additions, -5 deletions", doc.text)
        self.assertIn("+retry_delay = 2.0", doc.text)
        self.assertEqual(doc.metadata["filename"], "payment/retry.py")
        self.assertEqual(doc.metadata["status"], "modified")
        self.assertEqual(doc.metadata["committed_at"], "2026-03-01T10:30:00Z")

    def test_format_pull_request_document_with_contained_commits(self):
        pr = {
            "github_pr_number": 5,
            "developer_id": 1,
            "title": "Improve payment retry mechanism",
            "body": "This PR adds exponential backoff.",
            "state": "merged",
            "created_at": "2026-03-01T12:00:00Z",
            "merged_at": "2026-03-01T15:00:00Z",
        }
        contained_commits = [
            {"sha": "abc12345", "author_login": "amit", "message": "Add payment retry handler", "date": "2026-03-01T11:00:00Z"},
            {"sha": "def67890", "author_login": "sonal-38", "message": "Fix retry timeout", "date": "2026-03-01T11:30:00Z"},
            {"sha": "ghi13579", "author_login": "rahul", "message": "Add transaction rollback", "date": "2026-03-01T11:45:00Z"},
        ]
        pr_files = [{"filename": "payment/retry.py"}]

        doc = format_pull_request_document(pr, self.repo, self.dev_map, pr_files, contained_commits=contained_commits)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "pr:sonal-38/smart_payment_platform:5")
        self.assertEqual(doc.document_type, "pull_request")
        self.assertEqual(doc.source_id, "5")
        # PR author preserved
        self.assertEqual(doc.metadata["author_login"], "sonal-38")
        # Commit authors preserved separately in text & contained_commit_ids
        self.assertIn("Author:\nsonal-38", doc.text)
        self.assertIn("amit", doc.text)
        self.assertIn("rahul", doc.text)
        self.assertIn("Contained commits:", doc.text)
        self.assertEqual(
            doc.contained_commit_ids,
            ["commit:abc12345", "commit:def67890", "commit:ghi13579"]
        )
        self.assertEqual(doc.metadata["contained_commit_ids"], ["commit:abc12345", "commit:def67890", "commit:ghi13579"])
        self.assertEqual(doc.metadata["commit_count"], 3)
        self.assertEqual(doc.timestamp, "2026-03-01T12:00:00Z")

    def test_format_commit_document_timestamp_and_canonical_id(self):
        commit = {
            "sha": "abc12345",
            "developer_id": 1,
            "message": "Add retry handling",
            "committed_at": "2026-03-01T10:30:00Z",
        }
        doc = format_commit_document(commit, self.repo, self.dev_map)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.timestamp, "2026-03-01T10:30:00Z")
        self.assertEqual(doc.metadata["canonical_id"], "commit:abc12345")
        self.assertEqual(doc.metadata["author_login"], "sonal-38")

    def test_format_pull_request_document(self):
        pr = {
            "github_pr_number": 23,
            "developer_id": 1,
            "title": "Improve payment retry mechanism",
            "body": "This PR adds exponential backoff and jitter to payment retries.",
            "state": "merged",
            "created_at": "2026-03-01T12:00:00Z",
            "merged_at": "2026-03-01T15:00:00Z",
        }
        pr_files = [{"filename": "payment/retry.py"}]

        doc = format_pull_request_document(pr, self.repo, self.dev_map, pr_files)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "pr:sonal-38/smart_payment_platform:23")
        self.assertEqual(doc.document_type, "pull_request")
        self.assertEqual(doc.source_id, "23")
        self.assertIn("Pull Request:\n#23", doc.text)
        self.assertIn("Improve payment retry mechanism", doc.text)
        self.assertEqual(doc.metadata["pr_number"], 23)
        self.assertEqual(doc.metadata["created_at"], "2026-03-01T12:00:00Z")
        self.assertEqual(doc.metadata["merged_at"], "2026-03-01T15:00:00Z")

    def test_format_pr_file_document_distinct_from_commit_file(self):
        cf = {
            "filename": "payment/service.py",
            "status": "modified",
            "additions": 10,
            "deletions": 2,
            "changes": 12,
            "patch": "@@ -20,4 +20,10 @@",
        }
        doc = format_pr_file_document(cf, 23, self.repo)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "pr_file:sonal-38/smart_payment_platform:23:payment/service.py")
        self.assertEqual(doc.document_type, "pr_file")
        self.assertEqual(doc.source_id, "23:payment/service.py")
        self.assertIn("Pull Request:\n#23", doc.text)
        self.assertIn("Changed File:\npayment/service.py", doc.text)

    def test_null_field_handling(self):
        commit = {
            "sha": "xyz999",
            "developer_id": None,
            "message": None,
            "committed_at": None,
        }
        doc = format_commit_document(commit, self.repo, {})
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "commit:sonal-38/smart_payment_platform:xyz999")
        self.assertIn("No commit message provided", doc.text)

        empty_commit = {"sha": ""}
        self.assertIsNone(format_commit_document(empty_commit, self.repo, {}))


class TestLoadKnowledgeDocuments(unittest.TestCase):
    """Verifies loading and deduplication of knowledge documents from Supabase."""

    @patch("knowledge.clustering.get_supabase_client")
    def test_load_knowledge_documents_deduplication(self, mock_get_sb):
        mock_sb = MagicMock()
        mock_get_sb.return_value = mock_sb

        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 10, "full_name": "sonal-38/smart_payment_platform", "name": "smart_payment_platform"}],
            "developers": [{"id": 1, "login": "sonal-38", "name": "Sonal"}],
            "commits": [
                {"sha": "c1", "repository_id": 10, "developer_id": 1, "message": "commit 1", "committed_at": "2026-03-01T00:00:00Z"},
                {"sha": "c2", "repository_id": 10, "developer_id": 1, "message": "commit 2", "committed_at": "2026-03-01T01:00:00Z"},
            ],
            "commit_files": [
                {"repository": "sonal-38/smart_payment_platform", "commit_sha": "c1", "filename": "file1.py", "status": "modified"},
            ],
            "pull_requests": [
                {"id": 201, "github_pr_number": 1, "repository_id": 10, "developer_id": 1, "title": "PR 1", "created_at": "2026-03-01T02:00:00Z"},
            ],
            "changed_files": [
                {"pull_request_id": 201, "filename": "file1.py", "status": "modified"},
            ],
        }.get(table, [])

        docs = load_knowledge_documents("sonal-38", "smart_payment_platform", supabase_client=mock_sb)

        # Expected: 2 commit docs + 1 commit_file doc + 1 pr doc + 1 pr_file doc = 5 docs
        self.assertEqual(len(docs), 5)
        doc_types = [d.document_type for d in docs]
        self.assertEqual(doc_types.count("commit"), 2)
        self.assertEqual(doc_types.count("commit_file"), 1)
        self.assertEqual(doc_types.count("pull_request"), 1)
        self.assertEqual(doc_types.count("pr_file"), 1)

        # Verify all IDs are unique
        doc_ids = [d.id for d in docs]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))

    @patch("knowledge.clustering.get_supabase_client")
    def test_load_knowledge_documents_with_github_pr_commits(self, mock_get_sb):
        mock_sb = MagicMock()
        mock_get_sb.return_value = mock_sb
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 10, "full_name": "sonal-38/smart_payment_platform", "name": "smart_payment_platform"}],
            "developers": [{"id": 1, "login": "sonal-38", "name": "Sonal"}],
            "commits": [
                {"sha": "c1", "repository_id": 10, "developer_id": 1, "message": "commit 1", "committed_at": "2026-03-01T00:00:00Z"},
            ],
            "commit_files": [],
            "pull_requests": [
                {"id": 201, "github_pr_number": 5, "repository_id": 10, "developer_id": 1, "title": "PR 5", "created_at": "2026-03-01T02:00:00Z"},
            ],
            "changed_files": [],
        }.get(table, [])

        mock_gh = MagicMock()
        mock_gh.get_pull_request_commits.return_value = [
            {"sha": "c1", "author_login": "amit", "message": "Sub-commit in PR", "date": "2026-03-01T01:00:00Z"}
        ]

        docs = load_knowledge_documents("sonal-38", "smart_payment_platform", supabase_client=mock_sb, github_client=mock_gh)
        pr_doc = next(d for d in docs if d.document_type == "pull_request")
        self.assertEqual(pr_doc.contained_commit_ids, ["commit:c1"])
        self.assertIn("Sub-commit in PR", pr_doc.text)
        mock_gh.get_pull_request_commits.assert_called_once_with("sonal-38", "smart_payment_platform", 5)


class TestBGEEmbeddingsValidation(unittest.TestCase):
    """Verifies vector generation, dimension validation, and NaN/inf checking."""

    def test_build_embeddings_valid(self):
        docs = [
            KnowledgeDocument(id=f"doc:{i}", document_type="commit", repository="repo", source_id=str(i), text=f"commit text {i}")
            for i in range(5)
        ]
        mock_embedding_service = MagicMock()
        mock_embedding_service.dimension = 768
        mock_embedding_service.embed_documents.return_value = [
            [0.1 * j for j in range(768)] for _ in range(5)
        ]

        embeddings = build_embeddings(docs, embedding_service=mock_embedding_service)
        self.assertEqual(embeddings.shape, (5, 768))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_build_embeddings_empty(self):
        embeddings = build_embeddings([])
        self.assertEqual(embeddings.shape, (0, 768))

    def test_build_embeddings_nan_detection(self):
        docs = [KnowledgeDocument(id="doc:1", document_type="commit", repository="repo", source_id="1", text="text")]
        mock_embedding_service = MagicMock()
        mock_embedding_service.dimension = 768
        bad_vector = [0.0] * 768
        bad_vector[10] = float("nan")
        mock_embedding_service.embed_documents.return_value = [bad_vector]

        with self.assertRaises(ValueError) as ctx:
            build_embeddings(docs, embedding_service=mock_embedding_service)
        self.assertIn("NaN", str(ctx.exception))

    def test_build_embeddings_dimension_mismatch(self):
        docs = [KnowledgeDocument(id="doc:1", document_type="commit", repository="repo", source_id="1", text="text")]
        mock_embedding_service = MagicMock()
        mock_embedding_service.dimension = 768
        mock_embedding_service.embed_documents.return_value = [[0.1] * 512]  # wrong dimension

        with self.assertRaises(ValueError) as ctx:
            build_embeddings(docs, embedding_service=mock_embedding_service)
        self.assertIn("dimension mismatch", str(ctx.exception))


class TestHDBSCANClustering(unittest.TestCase):
    """Verifies HDBSCAN execution, cluster labeling, noise detection, and summary statistics."""

    def test_cluster_embeddings_with_distinct_clusters(self):
        # Create 2 synthetic dense clusters + 1 outlier
        np.random.seed(42)
        cluster_a = np.random.normal(loc=-1.0, scale=0.05, size=(8, 10))
        cluster_b = np.random.normal(loc=1.0, scale=0.05, size=(8, 10))
        outlier = np.random.normal(loc=10.0, scale=0.05, size=(1, 10))
        embeddings = np.vstack([cluster_a, cluster_b, outlier])

        labels = cluster_embeddings(
            embeddings,
            min_cluster_size=5,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        )

        self.assertEqual(len(labels), 17)
        # Verify cluster labels are integers
        self.assertTrue(all(isinstance(int(l), int) for l in labels))
        # Unique labels should have >= 2 clusters or noise
        unique_labels = set(labels)
        self.assertTrue(len(unique_labels) >= 2)

    def test_build_cluster_summary(self):
        docs = [
            KnowledgeDocument(id=f"commit:r:c{i}", document_type="commit", repository="r", source_id=f"c{i}", text=f"Commit {i}")
            for i in range(5)
        ] + [
            KnowledgeDocument(id=f"commit_file:r:cf{i}", document_type="commit_file", repository="r", source_id=f"cf{i}", text=f"CommitFile {i}")
            for i in range(5)
        ] + [
            KnowledgeDocument(id="pr:r:99", document_type="pull_request", repository="r", source_id="99", text="Outlier PR")
        ]
        # Labels: 5 in cluster 0, 5 in cluster 1, 1 as noise (-1)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1])

        summary = build_cluster_summary("r", docs, labels, parameters={"min_cluster_size": 5})

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["document_count"], 11)
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["noise_count"], 1)

        # Check clusters list
        clusters = summary["clusters"]
        self.assertEqual(len(clusters), 3)  # cluster 0, cluster 1, cluster -1

        # Check noise cluster
        noise_cluster = next(c for c in clusters if c["cluster_id"] == -1)
        self.assertEqual(noise_cluster["document_count"], 1)
        self.assertEqual(noise_cluster["document_type_counts"], {"pull_request": 1})

    def test_run_knowledge_clustering_empty(self):
        mock_sb = MagicMock()
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "full_name": "org/empty_repo"}],
            "developers": [],
            "commits": [],
            "commit_files": [],
            "pull_requests": [],
            "changed_files": [],
        }.get(table, [])

        result = run_knowledge_clustering("org", "empty_repo", supabase_client=mock_sb)
        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["document_count"], 0)
        self.assertEqual(result["clusters"], [])

    def test_run_knowledge_clustering_insufficient_data(self):
        mock_sb = MagicMock()
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "full_name": "org/small_repo"}],
            "developers": [],
            "commits": [{"sha": "c1", "repository_id": 1, "message": "msg"}],
            "commit_files": [],
            "pull_requests": [],
            "changed_files": [],
        }.get(table, [])

        result = run_knowledge_clustering("org", "small_repo", min_cluster_size=5, supabase_client=mock_sb)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["document_count"], 1)
        self.assertEqual(result["minimum_required"], 5)
        self.assertIn("Not enough documents", result["message"])

    def test_run_knowledge_clustering_complete_pipeline(self):
        # 12 documents total: 6 payment cluster, 6 auth cluster
        mock_sb = MagicMock()
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "full_name": "sonal-38/smart_payment_platform"}],
            "developers": [{"id": 1, "login": "sonal-38"}],
            "commits": [
                {"sha": f"pay_c{i}", "repository_id": 1, "developer_id": 1, "message": f"Payment commit {i}"}
                for i in range(3)
            ] + [
                {"sha": f"auth_c{i}", "repository_id": 1, "developer_id": 1, "message": f"Auth commit {i}"}
                for i in range(3)
            ],
            "commit_files": [
                {"repository": "sonal-38/smart_payment_platform", "commit_sha": f"pay_c{i}", "filename": f"payment/{i}.py"}
                for i in range(3)
            ] + [
                {"repository": "sonal-38/smart_payment_platform", "commit_sha": f"auth_c{i}", "filename": f"auth/{i}.py"}
                for i in range(3)
            ],
            "pull_requests": [],
            "changed_files": [],
        }.get(table, [])

        mock_embedding_service = MagicMock()
        mock_embedding_service.dimension = 768
        # Create distinct 768-dim vectors
        vectors = []
        for i in range(6):  # payment
            vec = [0.0] * 768
            vec[0] = 5.0 + (i * 0.01)
            vectors.append(vec)
        for i in range(6):  # auth
            vec = [0.0] * 768
            vec[100] = 5.0 + (i * 0.01)
            vectors.append(vec)

        mock_embedding_service.embed_documents.return_value = vectors

        result = run_knowledge_clustering(
            owner="sonal-38",
            repo="smart_payment_platform",
            min_cluster_size=5,
            min_samples=2,
            supabase_client=mock_sb,
            embedding_service=mock_embedding_service,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["document_count"], 12)
        self.assertGreaterEqual(result["cluster_count"], 1)
        self.assertIn("documents", result)
        self.assertEqual(len(result["documents"]), 12)
        self.assertIn("cluster_summary", result)
        self.assertIn("noise_documents", result)
        first_doc = result["documents"][0]
        self.assertIn("cluster_id", first_doc)
        self.assertIn("cluster_probability", first_doc)


class TestKnowledgeAPIEndpoints(unittest.TestCase):
    """Verifies FastAPI knowledge clustering endpoints."""

    @patch("knowledge.clustering.get_supabase_client")
    def test_cluster_api_endpoint_insufficient_data(self, mock_get_sb):
        from api.knowledge import cluster_repository_knowledge

        mock_sb = MagicMock()
        mock_get_sb.return_value = mock_sb
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "full_name": "sonal-38/smart-payment-platform"}],
            "developers": [],
            "commits": [{"sha": "c1", "repository_id": 1, "message": "msg"}],
            "commit_files": [],
            "pull_requests": [],
            "changed_files": [],
        }.get(table, [])

        response = cluster_repository_knowledge(
            owner="sonal-38",
            repo="smart-payment-platform",
            min_cluster_size=5,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self.assertEqual(response["status"], "insufficient_data")
        self.assertEqual(response["document_count"], 1)

    @patch("knowledge.clustering.get_supabase_client")
    def test_documents_preview_api_endpoint(self, mock_get_sb):
        from api.knowledge import inspect_knowledge_documents

        mock_sb = MagicMock()
        mock_get_sb.return_value = mock_sb
        mock_sb.select.side_effect = lambda table, filters=None: {
            "repositories": [{"id": 1, "full_name": "sonal-38/smart-payment-platform"}],
            "developers": [{"id": 1, "login": "sonal-38"}],
            "commits": [{"sha": "c1", "repository_id": 1, "message": "msg"}],
            "commit_files": [],
            "pull_requests": [],
            "changed_files": [],
        }.get(table, [])

        data = inspect_knowledge_documents(owner="sonal-38", repo="smart-payment-platform")
        self.assertEqual(data["total_documents"], 1)
        self.assertEqual(data["document_type_counts"], {"commit": 1})


if __name__ == "__main__":
    unittest.main()

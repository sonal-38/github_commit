"""
Unit tests for Step 11: Knowledge Area Interpretation Pipeline.

Verifies:
1. File-path evidence extraction:
   - Normalization, frequency ranking, file paths from metadata and text.
   - Does not just blindly use folder names.
2. Representative terms extraction:
   - Tokenization (snake_case, camelCase), stopword filtering (English + VCS noise).
   - Weighted frequency and distinctiveness across clusters.
3. Candidate label generation:
   - Dynamic and explainable from actual evidence.
   - NEVER hardcoded by cluster ID (tested with inverted cluster IDs).
   - Fallback to "Unclassified Engineering Area" on insufficient evidence.
4. Representative documents selection:
   - Prefer higher HDBSCAN membership probability and rich document types.
   - Small representative subset.
5. Noise handling:
   - cluster_id = -1 is NEVER turned into a knowledge area.
   - Counted in noise_documents and tracked in noise_document_ids.
6. PR author vs Commit authors separation:
   - PR author preserved separately from commit authors in PR-contained commits.
   - No developer expertise or risk scoring computed.
7. Small datasets and edge cases:
   - Empty documents, single document, all noise, missing fields, missing messages.
   - Zero crashes.
8. Traceability:
   - Every cluster links candidate_label -> representative_documents -> all_document_ids.
9. API Endpoint:
   - POST /knowledge/repositories/{owner}/{repo}/interpret
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from knowledge.clustering import KnowledgeDocument
from knowledge.interpretation import (
    tokenize_text,
    extract_file_paths,
    extract_representative_terms,
    select_representative_documents,
    generate_candidate_label,
    extract_evidence_summary,
    interpret_clusters,
    run_knowledge_interpretation,
)


class TestTokenizeAndFilePaths(unittest.TestCase):
    """Verifies tokenization and file-path extraction logic."""

    def test_tokenize_text(self):
        # CamelCase and snake_case splitting, stopword removal
        raw = "PaymentGateway processed payment_retry and transaction_status in repo for commit"
        tokens = tokenize_text(raw)
        self.assertIn("payment", tokens)
        self.assertIn("gateway", tokens)
        self.assertIn("retry", tokens)
        self.assertIn("transaction", tokens)
        # Stopwords and git noise must be removed
        self.assertNotIn("and", tokens)
        self.assertNotIn("in", tokens)
        self.assertNotIn("for", tokens)
        self.assertNotIn("repo", tokens)
        self.assertNotIn("commit", tokens)

    def test_extract_file_paths(self):
        docs = [
            KnowledgeDocument(
                id="doc1",
                document_type="commit_file",
                metadata={"filename": "payment/retry.py"},
            ),
            KnowledgeDocument(
                id="doc2",
                document_type="commit_file",
                metadata={"filename": "payment/retry.py"},
            ),
            KnowledgeDocument(
                id="doc3",
                document_type="commit_file",
                metadata={"filename": "payment/service.py"},
            ),
            KnowledgeDocument(
                id="doc4",
                document_type="commit",
                text="Changed Files:\n- payment/transaction.py\n- payment/service.py",
                metadata={"changed_files": ["payment/transaction.py", "payment/service.py"]},
            ),
        ]
        files = extract_file_paths(docs)
        self.assertIn("payment/retry.py", files)
        self.assertIn("payment/service.py", files)
        self.assertIn("payment/transaction.py", files)
        # Most frequent file (retry.py) should be first
        self.assertEqual(files[0], "payment/retry.py")


class TestRepresentativeTermsAndLabels(unittest.TestCase):
    """Verifies representative term extraction and candidate label generation."""

    def test_extract_representative_terms(self):
        docs = [
            KnowledgeDocument(
                id="c1",
                document_type="commit",
                text="Commit: Add payment retry mechanism\nAuthor: sonal",
                metadata={"message": "Add payment retry mechanism"},
            ),
            KnowledgeDocument(
                id="pr1",
                document_type="pull_request",
                text="Pull Request #5\nTitle: Fix payment retry timeout and refund transaction",
                metadata={
                    "title": "Fix payment retry timeout and refund transaction",
                    "body": "Ensures failed transactions are safely refunded after retry timeout",
                },
            ),
        ]
        terms = extract_representative_terms(docs, top_k=5)
        self.assertIn("payment", terms)
        self.assertIn("retry", terms)

    def test_candidate_label_not_hardcoded_by_cluster_id(self):
        # Payment terms in cluster 1 must produce payment label, not caching
        terms_payment = ["payment", "retry", "transaction"]
        files_payment = ["payment/retry.py", "payment/service.py"]
        label_payment = generate_candidate_label(terms_payment, files_payment)
        self.assertIn("Payment", label_payment)

        # Caching terms in cluster 0 must produce caching label, not payment
        terms_cache = ["cache", "redis", "invalidation"]
        files_cache = ["cache/redis.py", "cache/service.py"]
        label_cache = generate_candidate_label(terms_cache, files_cache)
        self.assertIn("Cach", label_cache)

        # Inverted cluster testing: cluster ID does not dictate the label
        doc_cache = KnowledgeDocument(
            id="doc_cache",
            document_type="commit",
            text="Commit: Invalidate cache keys in redis",
            metadata={"message": "Invalidate cache keys in redis"},
        )
        result = interpret_clusters(
            repository="test/repo",
            documents=[doc_cache],
            labels=np.array([0]),  # cluster_id = 0
            probabilities=np.array([0.95]),
        )
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(result["clusters"][0]["cluster_id"], 0)
        self.assertIn("Cach", result["clusters"][0]["candidate_label"])

    def test_fallback_unclassified_label_on_empty_evidence(self):
        label = generate_candidate_label([], [])
        self.assertEqual(label, "Unclassified Engineering Area")


class TestRepresentativeDocumentsAndNoise(unittest.TestCase):
    """Verifies representative document selection and noise isolation."""

    def test_select_representative_documents(self):
        d1 = KnowledgeDocument(id="doc:low", document_type="commit", cluster_probability=0.3)
        d2 = KnowledgeDocument(id="doc:high", document_type="commit", cluster_probability=0.95)
        d3 = KnowledgeDocument(id="doc:pr", document_type="pull_request", cluster_probability=0.92)

        rep_docs = select_representative_documents([d1, d2, d3], max_docs=2)
        self.assertIn("doc:high", rep_docs)
        self.assertIn("doc:pr", rep_docs)
        self.assertNotIn("doc:low", rep_docs)

    def test_noise_handling(self):
        docs = [
            KnowledgeDocument(id="doc:noise1", document_type="commit", text="misc commit 1"),
            KnowledgeDocument(id="doc:noise2", document_type="commit", text="misc commit 2"),
            KnowledgeDocument(
                id="doc:payment",
                document_type="commit",
                text="Add payment gateway",
                metadata={"message": "Add payment gateway"},
            ),
        ]
        labels = np.array([-1, -1, 0])
        probabilities = np.array([0.1, 0.2, 0.9])

        result = interpret_clusters(
            repository="sonal-38/smart_payment_platform",
            documents=docs,
            labels=labels,
            probabilities=probabilities,
        )

        self.assertEqual(result["noise_documents"], 2)
        self.assertEqual(len(result["noise_document_ids"]), 2)
        self.assertIn("doc:noise1", result["noise_document_ids"])
        self.assertIn("doc:noise2", result["noise_document_ids"])

        # Only 1 legitimate cluster, cluster_id = -1 must NOT be in clusters list
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(result["clusters"][0]["cluster_id"], 0)
        cluster_ids = [c["cluster_id"] for c in result["clusters"]]
        self.assertNotIn(-1, cluster_ids)


class TestPRAvCommitAuthorSeparation(unittest.TestCase):
    """
    CRITICAL: Verifies PR author and commit authors (including PR contained commits)
    remain completely separated, and no expertise/risk scores are calculated.
    """

    def test_pr_author_and_contained_commit_authors_separated(self):
        pr_doc = KnowledgeDocument(
            id="pr:smart_payment_platform:5",
            document_type="pull_request",
            text="PR #5 Author: sonal-38",
            contained_commit_ids=["commit:c1", "commit:c2", "commit:c3"],
            metadata={
                "pr_number": 5,
                "title": "Payment retry & rollback pipeline",
                "author_login": "sonal-38",
                "contained_commits": [
                    {"sha": "c1", "author_login": "amit", "message": "Add payment retry"},
                    {"sha": "c2", "author_login": "sonal-38", "message": "Fix retry timeout"},
                    {"sha": "c3", "author_login": "rahul", "message": "Add rollback"},
                ],
            },
        )

        commit_doc = KnowledgeDocument(
            id="commit:smart_payment_platform:c4",
            document_type="commit",
            text="Commit c4 Author: vikram",
            metadata={"author_login": "vikram", "message": "Payment metrics"},
        )

        docs = [pr_doc, commit_doc]
        labels = np.array([0, 0])

        result = interpret_clusters(
            repository="sonal-38/smart_payment_platform",
            documents=docs,
            labels=labels,
        )

        cluster0 = result["clusters"][0]
        evidence_summary = cluster0["evidence_summary"]

        # PR author is ONLY sonal-38
        self.assertEqual(evidence_summary["pr_authors"], ["sonal-38"])

        # Commit authors include amit, rahul, sonal-38, vikram
        self.assertIn("amit", evidence_summary["commit_authors"])
        self.assertIn("rahul", evidence_summary["commit_authors"])
        self.assertIn("sonal-38", evidence_summary["commit_authors"])
        self.assertIn("vikram", evidence_summary["commit_authors"])

        # Must NOT claim all work was done by PR author
        self.assertNotEqual(evidence_summary["pr_authors"], evidence_summary["commit_authors"])

        # Must NOT contain forbidden scores
        self.assertNotIn("expertise_score", cluster0)
        self.assertNotIn("knowledge_risk_score", cluster0)
        self.assertNotIn("developer_ranking", cluster0)


class TestSmallDatasetsAndEdgeCases(unittest.TestCase):
    """Verifies safe handling of empty data, single document, all noise, missing fields."""

    def test_empty_documents(self):
        result = interpret_clusters(
            repository="empty/repo",
            documents=[],
            labels=np.array([]),
        )
        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["noise_documents"], 0)

    def test_all_documents_noise(self):
        docs = [
            KnowledgeDocument(id="d1", document_type="commit", text="misc"),
            KnowledgeDocument(id="d2", document_type="commit", text="misc2"),
        ]
        result = interpret_clusters(
            repository="noisy/repo",
            documents=docs,
            labels=np.array([-1, -1]),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["noise_documents"], 2)

    def test_missing_fields_and_messages(self):
        doc = KnowledgeDocument(
            id="d_empty",
            document_type="commit",
            text="",
            metadata={},
        )
        result = interpret_clusters(
            repository="test/repo",
            documents=[doc],
            labels=np.array([0]),
        )
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(result["clusters"][0]["candidate_label"], "Unclassified Engineering Area")
        self.assertEqual(result["clusters"][0]["important_files"], [])
        self.assertEqual(result["clusters"][0]["representative_terms"], [])

    def test_pr_with_no_contained_commits(self):
        doc = KnowledgeDocument(
            id="pr:test:1",
            document_type="pull_request",
            metadata={"title": "Empty PR", "contained_commits": []},
        )
        result = interpret_clusters(
            repository="test/repo",
            documents=[doc],
            labels=np.array([0]),
        )
        self.assertEqual(len(result["clusters"]), 1)


class TestTraceabilityAndOutputStructure(unittest.TestCase):
    """Verifies complete traceability from candidate label -> cluster -> representative docs -> all docs."""

    def test_traceability_links(self):
        docs = [
            KnowledgeDocument(id="c1", document_type="commit", text="payment 1", cluster_probability=0.9),
            KnowledgeDocument(id="c2", document_type="commit", text="payment 2", cluster_probability=0.85),
            KnowledgeDocument(id="c3", document_type="commit", text="payment 3", cluster_probability=0.7),
        ]
        labels = np.array([0, 0, 0])
        result = interpret_clusters(
            repository="test/repo",
            documents=docs,
            labels=labels,
        )

        cluster = result["clusters"][0]
        self.assertEqual(cluster["cluster_id"], 0)
        self.assertIn("representative_documents", cluster)
        self.assertIn("all_document_ids", cluster)
        self.assertEqual(len(cluster["all_document_ids"]), 3)
        # Representative docs must be a subset of all document IDs
        for r_doc in cluster["representative_documents"]:
            self.assertIn(r_doc, cluster["all_document_ids"])


class TestInterpretationAPIEndpoint(unittest.TestCase):
    """Verifies FastAPI endpoint /knowledge/repositories/{owner}/{repo}/interpret."""

    @patch("api.knowledge.run_knowledge_interpretation")
    def test_interpret_api_success(self, mock_interpret):
        from fastapi.testclient import TestClient
        from main import app

        mock_interpret.return_value = {
            "status": "completed",
            "repository": "sonal-38/smart_payment_platform",
            "clusters": [
                {
                    "cluster_id": 0,
                    "candidate_label": "Payment Processing & Retry Handling",
                    "document_count": 5,
                    "representative_documents": ["commit:c1", "pr:1"],
                    "representative_terms": ["payment", "retry", "transaction"],
                    "important_files": ["payment/retry.py", "payment/service.py"],
                    "all_document_ids": ["commit:c1", "pr:1", "commit:c2", "commit_file:c1:f1", "pr_file:1:f1"],
                    "evidence_summary": {
                        "document_type_counts": {"commit": 2, "pull_request": 1, "commit_file": 1, "pr_file": 1},
                        "pr_authors": ["sonal-38"],
                        "commit_authors": ["amit", "sonal-38"],
                    },
                }
            ],
            "noise_documents": 1,
            "noise_document_ids": ["commit:misc"],
            "total_documents": 6,
        }

        client = TestClient(app)
        response = client.post("/knowledge/repositories/sonal-38/smart_payment_platform/interpret")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["repository"], "sonal-38/smart_payment_platform")
        self.assertEqual(len(data["clusters"]), 1)
        self.assertEqual(data["clusters"][0]["candidate_label"], "Payment Processing & Retry Handling")
        self.assertEqual(data["noise_documents"], 1)


if __name__ == "__main__":
    unittest.main()

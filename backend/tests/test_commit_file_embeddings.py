"""
Unit and integration tests for Commit File Vector Embeddings pipeline.

Verifies:
1. DocumentBuilder.build_commit_file_doc formatting, metadata, and deterministic IDs.
2. Parent commit context inheritance (message, developer, timestamp).
3. Diff patch inclusion and safe truncation.
4. DocumentBuilder.build_commit_doc changed files summary extension (Step 9).
5. VectorIndexer incremental embedding (Step 8 & 13) preventing duplicate vectors.
6. Only new commit/commit-file embeddings are generated when new records are added.
7. RAG compatibility with commit_file document types.
"""
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

from vector.document_builder import DocumentBuilder, DocumentItem
from vector.indexer import VectorIndexer


class TestCommitFileDocumentBuilder(unittest.TestCase):
    """Test semantic document construction for commit_files."""

    def setUp(self):
        self.repo_name = "sonal-38/smart-payment-platform"
        self.commit_sha = "abc123456789def0123456789abcdef01234567"
        self.commit_info = {
            "sha": self.commit_sha,
            "message": "Implement payment retry mechanism with exponential backoff",
            "committed_at": "2026-03-01T14:30:00Z",
            "developer_id": 42,
            "author_login": "sonal-38",
            "html_url": f"https://github.com/{self.repo_name}/commit/{self.commit_sha}",
        }
        self.dev_map = {
            42: {
                "id": 42,
                "login": "sonal-38",
                "name": "Sonal Developer",
            }
        }
        self.commit_file = {
            "repository": self.repo_name,
            "commit_sha": self.commit_sha,
            "filename": "payment/retry.py",
            "status": "modified",
            "additions": 35,
            "deletions": 8,
            "changes": 43,
            "patch": "@@ -10,6 +10,12 @@ def process_payment():\n+    retry_count = 0\n+    backoff = 1.5",
            "blob_url": f"https://github.com/{self.repo_name}/blob/{self.commit_sha}/payment/retry.py",
            "raw_url": f"https://github.com/{self.repo_name}/raw/{self.commit_sha}/payment/retry.py",
        }

    def test_build_commit_file_doc_complete(self):
        """Verify complete commit-file document generation."""
        doc = DocumentBuilder.build_commit_file_doc(
            cf=self.commit_file,
            repo_name=self.repo_name,
            commit_info=self.commit_info,
            dev_map=self.dev_map,
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.document_type, "commit_file")
        self.assertEqual(doc.repository, self.repo_name)
        self.assertEqual(doc.developer, "sonal-38")
        self.assertEqual(doc.source_id, f"{self.commit_sha}:payment/retry.py")

        # Step 3: Check deterministic stable key
        expected_key = f"commit_file:{self.repo_name}:{self.commit_sha}:payment/retry.py"
        self.assertEqual(doc.stable_key, expected_key)

        # Re-running must produce the exact same point_id
        doc_repeat = DocumentBuilder.build_commit_file_doc(
            cf=self.commit_file,
            repo_name=self.repo_name,
            commit_info=self.commit_info,
            dev_map=self.dev_map,
        )
        self.assertEqual(doc.point_id, doc_repeat.point_id)

        # Step 4 & 5: Check text content
        self.assertIn(f"Repository: {self.repo_name}", doc.text)
        self.assertIn("Implement payment retry mechanism with exponential backoff", doc.text)
        self.assertIn("Developer:\nsonal-38", doc.text)
        self.assertIn("Changed file:\npayment/retry.py", doc.text)
        self.assertIn("File status:\nmodified", doc.text)
        self.assertIn("+35 additions", doc.text)
        self.assertIn("-8 deletions", doc.text)
        self.assertIn("Diff Patch Preview:", doc.text)
        self.assertIn("+    retry_count = 0", doc.text)

        # Step 6 & 7: Check metadata
        meta = doc.metadata
        self.assertEqual(meta["repository"], self.repo_name)
        self.assertEqual(meta["document_type"], "commit_file")
        self.assertEqual(meta["commit_sha"], self.commit_sha)
        self.assertEqual(meta["filename"], "payment/retry.py")
        self.assertEqual(meta["developer"], "sonal-38")
        self.assertEqual(meta["committed_at"], "2026-03-01T14:30:00Z")
        self.assertEqual(meta["status"], "modified")
        self.assertEqual(meta["additions"], 35)
        self.assertEqual(meta["deletions"], 8)
        self.assertEqual(meta["changes"], 43)
        self.assertTrue(meta["patch_available"])

    def test_patch_truncation(self):
        """Verify that excessively long diff patches are truncated."""
        large_patch = "A" * 2000
        cf_large = dict(self.commit_file, patch=large_patch)
        doc = DocumentBuilder.build_commit_file_doc(
            cf=cf_large,
            repo_name=self.repo_name,
            commit_info=self.commit_info,
            dev_map=self.dev_map,
        )
        self.assertIsNotNone(doc)
        self.assertIn("... [diff truncated]", doc.text)
        self.assertLess(len(doc.text), 2500)

    def test_missing_filename_returns_none(self):
        """Missing or blank filename returns None."""
        cf_invalid = dict(self.commit_file, filename="")
        doc = DocumentBuilder.build_commit_file_doc(
            cf=cf_invalid,
            repo_name=self.repo_name,
            commit_info=self.commit_info,
        )
        self.assertIsNone(doc)

    def test_build_commit_doc_with_changed_files(self):
        """Step 9: Existing commit document builder optionally includes changed files."""
        c_files = [
            {"filename": "payment/service.py"},
            {"filename": "payment/retry.py"},
            {"filename": "database/payment_repository.py"},
        ]
        doc = DocumentBuilder.build_commit_doc(
            commit=self.commit_info,
            repo_name=self.repo_name,
            dev_map=self.dev_map,
            commit_files=c_files,
        )
        self.assertIsNotNone(doc)
        self.assertEqual(doc.document_type, "commit")
        self.assertIn("Changed Files:", doc.text)
        self.assertIn("- payment/service.py", doc.text)
        self.assertIn("- payment/retry.py", doc.text)
        self.assertIn("- database/payment_repository.py", doc.text)

    def test_build_commit_doc_without_changed_files_backwards_compatible(self):
        """Step 9: Existing commit builder without commit_files preserves previous output."""
        doc = DocumentBuilder.build_commit_doc(
            commit=self.commit_info,
            repo_name=self.repo_name,
            dev_map=self.dev_map,
        )
        self.assertIsNotNone(doc)
        self.assertNotIn("Changed Files:", doc.text)


class TestVectorIndexerIncremental(unittest.TestCase):
    """Test VectorIndexer incremental behavior with commit_files."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_documents.side_effect = lambda texts, **kwargs: [
            [0.1] * 768 for _ in texts
        ]

        self.repo_record = {
            "id": 1,
            "name": "smart-payment-platform",
            "owner_login": "sonal-38",
            "full_name": "sonal-38/smart-payment-platform",
        }
        self.commit_records = [
            {
                "id": 10,
                "sha": "sha111",
                "message": "feat: add payment gateway",
                "committed_at": "2026-03-01T10:00:00Z",
                "developer_id": 1,
            }
        ]
        self.commit_file_records = [
            {
                "id": 100,
                "repository": "sonal-38/smart-payment-platform",
                "commit_sha": "sha111",
                "filename": "gateway/stripe.py",
                "status": "added",
                "additions": 100,
                "deletions": 0,
                "changes": 100,
                "patch": "@@ -0,0 +1,100 @@",
            }
        ]
        self.dev_records = [
            {"id": 1, "login": "sonal-38", "name": "Sonal"}
        ]

    def _setup_supabase_mocks(self, existing_embeddings: List[Dict[str, Any]]):
        def mock_select(table, filters=None):
            if table == "repositories":
                return [self.repo_record]
            elif table == "commits":
                return self.commit_records
            elif table == "commit_files":
                return self.commit_file_records
            elif table == "developers":
                return self.dev_records
            elif table in ("pull_requests", "issues", "reviews", "review_comments", "changed_files", "issue_comments"):
                return []
            elif table == "document_embeddings":
                return existing_embeddings
            return []

        self.mock_supabase.select.side_effect = mock_select

    def test_first_index_run_embeds_all_including_commit_files(self):
        """First run: no existing embeddings in Supabase; all docs are embedded."""
        self._setup_supabase_mocks(existing_embeddings=[])

        indexer = VectorIndexer(
            supabase_client=self.mock_supabase,
            embedding_service=self.mock_embeddings,
        )
        result = indexer.index_repository("sonal-38", "smart-payment-platform", incremental=True)

        self.assertEqual(result["indexed"]["commits"], 1)
        self.assertEqual(result["indexed"]["commit_files"], 1)
        self.assertEqual(result["new_vectors"], 2)
        self.assertEqual(result["total_vectors"], 2)

        # Verify embedding service was called with 2 texts (commit + commit_file)
        self.assertEqual(self.mock_embeddings.embed_documents.call_count, 1)
        embedded_texts = self.mock_embeddings.embed_documents.call_args[0][0]
        self.assertEqual(len(embedded_texts), 2)

        # Verify Supabase upsert was called with on_conflict="id"
        self.assertTrue(self.mock_supabase.upsert.called)
        upsert_call = self.mock_supabase.upsert.call_args
        self.assertEqual(upsert_call[1]["on_conflict"], "id")

    def test_second_run_without_new_data_does_not_regenerate_embeddings(self):
        """Step 13: Second run without new GitHub data generates 0 new vectors."""
        # Existing embeddings already contain the 2 stable keys
        existing = [
            {"id": "commit:sonal-38/smart-payment-platform:sha111"},
            {"id": "commit_file:sonal-38/smart-payment-platform:sha111:gateway/stripe.py"},
        ]
        self._setup_supabase_mocks(existing_embeddings=existing)

        indexer = VectorIndexer(
            supabase_client=self.mock_supabase,
            embedding_service=self.mock_embeddings,
        )
        result = indexer.index_repository("sonal-38", "smart-payment-platform", incremental=True)

        # Embeddings should NOT have been generated
        self.assertEqual(self.mock_embeddings.embed_documents.call_count, 0)
        self.assertEqual(result["new_vectors"], 0)
        self.assertEqual(result["total_vectors"], 2)
        self.assertIn("0 new embeddings", result.get("message", ""))

    def test_adding_new_commit_file_only_embeds_new_item(self):
        """Step 13: Adding a new commit-file record only embeds the new item."""
        # Only the parent commit already exists in vector DB
        existing = [
            {"id": "commit:sonal-38/smart-payment-platform:sha111"},
        ]
        self._setup_supabase_mocks(existing_embeddings=existing)

        indexer = VectorIndexer(
            supabase_client=self.mock_supabase,
            embedding_service=self.mock_embeddings,
        )
        result = indexer.index_repository("sonal-38", "smart-payment-platform", incremental=True)

        # Only 1 new vector (the commit_file) was embedded
        self.assertEqual(self.mock_embeddings.embed_documents.call_count, 1)
        embedded_texts = self.mock_embeddings.embed_documents.call_args[0][0]
        self.assertEqual(len(embedded_texts), 1)
        self.assertEqual(result["new_vectors"], 1)
        self.assertEqual(result["total_vectors"], 2)


class TestRAGWithCommitFiles(unittest.TestCase):
    """Step 14: Test RAG integration and source citation for commit_files."""

    @patch("ai.rag_service.GeminiService")
    @patch("ai.rag_service.VectorIndexer")
    @patch("ai.rag_service.SupabaseClient")
    def test_rag_ask_with_commit_file_evidence(self, mock_supa_cls, mock_indexer_cls, mock_gemini_cls):
        from ai.rag_service import RAGService

        mock_indexer = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate_grounded_answer.return_value = (
            "The payment retry mechanism was introduced in payment/retry.py to handle intermittent network failures."
        )

        commit_file_search_result = {
            "query": "Why was the payment retry mechanism introduced?",
            "results": [
                {
                    "score": 0.89,
                    "document_type": "commit_file",
                    "repository": "sonal-38/smart-payment-platform",
                    "developer": "sonal-38",
                    "source_id": "abc123:payment/retry.py",
                    "text": (
                        "Repository: sonal-38/smart-payment-platform\n\n"
                        "Commit:\nImplement payment retry mechanism\n\n"
                        "Developer:\nsonal-38\n\n"
                        "Changed file:\npayment/retry.py\n\n"
                        "File status:\nmodified\n\n"
                        "Changes:\n+35 additions\n-8 deletions"
                    ),
                    "metadata": {
                        "repository": "sonal-38/smart-payment-platform",
                        "document_type": "commit_file",
                        "source_id": "abc123:payment/retry.py",
                        "commit_sha": "abc123",
                        "filename": "payment/retry.py",
                        "developer": "sonal-38",
                        "committed_at": "2026-03-01T14:30:00Z",
                    },
                }
            ],
        }
        mock_indexer.search.return_value = commit_file_search_result

        rag = RAGService(
            supabase_client=mock_supa_cls.return_value,
            vector_indexer=mock_indexer,
            gemini_service=mock_gemini,
        )

        response = rag.answer_question(
            question="Why was the payment retry mechanism introduced?",
            repository="sonal-38/smart-payment-platform",
        )

        self.assertEqual(response["question"], "Why was the payment retry mechanism introduced?")
        self.assertIn("payment/retry.py", response["answer"])
        self.assertEqual(len(response["sources"]), 1)

        source = response["sources"][0]
        self.assertEqual(source["document_type"], "commit_file")
        self.assertEqual(source["source_id"], "abc123:payment/retry.py")
        self.assertEqual(source["developer"], "sonal-38")
        self.assertEqual(source["date"], "2026-03-01T14:30:00Z")


if __name__ == "__main__":
    unittest.main()

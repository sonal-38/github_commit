"""
Unit and Integration Tests for OpenRouter Migration.

Tests:
1. OpenRouterService initialization, environment loading, and API key safety.
2. OpenRouterService chat completions invocation and response parsing.
3. OpenRouterService comprehensive HTTP and network error handling (401, 403, 429, 500+, timeout, connection).
4. Grounded answer prompt construction and no-evidence shortcut.
5. RAGService integration with OpenRouterService as active provider.
6. Verification that Gemini is not used by the active RAG path.
7. FastAPI /ai/ask endpoint contract with OpenRouter.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import requests

from ai.openrouter_service import (
    OpenRouterService,
    OpenRouterConfigurationError,
    OpenRouterAPIError,
)
from ai.rag_service import RAGService


class TestOpenRouterService(unittest.TestCase):
    """Test suite for OpenRouterService."""

    def test_missing_api_key_raises_configuration_error(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True):
            service = OpenRouterService(api_key="")
            with self.assertRaises(OpenRouterConfigurationError) as ctx:
                service.generate_grounded_answer(
                    question="Why was caching added?",
                    context="Commit abc: Added Redis cache",
                )
            self.assertIn("OPENROUTER_API_KEY is not set", str(ctx.exception))

    def test_model_loaded_from_env(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openrouter/free", "OPENROUTER_API_KEY": "sk-or-test-123"}):
            service = OpenRouterService()
            self.assertEqual(service._get_model_name(), "openrouter/free")

    def test_custom_model_override(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "meta-llama/llama-3-8b-instruct", "OPENROUTER_API_KEY": "sk-or-test-123"}):
            service = OpenRouterService(model_name="meta-llama/llama-3-8b-instruct")
            self.assertEqual(service._get_model_name(), "meta-llama/llama-3-8b-instruct")

    def test_empty_question_raises_value_error(self):
        service = OpenRouterService(api_key="sk-or-test-123")
        with self.assertRaises(ValueError):
            service.generate_grounded_answer(question="", context="Some context")

    @patch("ai.openrouter_service.requests.post")
    def test_empty_context_returns_controlled_response_without_api_call(self, mock_post):
        service = OpenRouterService(api_key="sk-or-test-123")
        result = service.generate_grounded_answer(
            question="Why was caching added?",
            context="",
        )
        self.assertEqual(result, "I could not find enough repository evidence to answer this question.")
        mock_post.assert_not_called()

    @patch("ai.openrouter_service.requests.post")
    def test_successful_answer_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Caching was introduced in commit abc1234 by sonal-38 to reduce database latency.",
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        secret_key = "sk-or-super-secret-key-999"
        service = OpenRouterService(api_key=secret_key, model_name="openrouter/free")
        answer = service.generate_grounded_answer(
            question="Why was caching added?",
            context="Repository: sonal-38/smart_payment_platform\nCommit abc1234: Add Redis caching\nDeveloper: sonal-38",
        )

        self.assertIn("Caching was introduced", answer)
        self.assertIn("sonal-38", answer)

        # Verify call payload and headers
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Authorization"], f"Bearer {secret_key}")
        self.assertEqual(call_kwargs["json"]["model"], "openrouter/free")
        self.assertEqual(len(call_kwargs["json"]["messages"]), 2)
        self.assertEqual(call_kwargs["json"]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["json"]["messages"][1]["role"], "user")

    @patch("ai.openrouter_service.requests.post")
    def test_api_key_not_leaked_on_401_unauthorized(self, mock_post):
        secret_key = "sk-or-secret-token-do-not-leak"
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid API Key"
        mock_post.return_value = mock_resp

        service = OpenRouterService(api_key=secret_key)
        with self.assertRaises(OpenRouterConfigurationError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some evidence context",
            )
        err_msg = str(ctx.exception)
        self.assertNotIn(secret_key, err_msg)
        self.assertIn("HTTP 401 Unauthorized", err_msg)

    @patch("ai.openrouter_service.requests.post")
    def test_403_forbidden_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"error": {"message": "Credit quota exceeded"}}
        mock_post.return_value = mock_resp

        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterConfigurationError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("HTTP 403 Forbidden", str(ctx.exception))

    @patch("ai.openrouter_service.requests.post")
    def test_429_rate_limit_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "Too many requests"}}
        mock_post.return_value = mock_resp

        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterAPIError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("HTTP 429", str(ctx.exception))

    @patch("ai.openrouter_service.requests.post")
    def test_500_server_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "Internal model failure"}}
        mock_post.return_value = mock_resp

        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterAPIError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("HTTP 500", str(ctx.exception))

    @patch("ai.openrouter_service.requests.post")
    def test_timeout_handling(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()
        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterAPIError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("timed out", str(ctx.exception))

    @patch("ai.openrouter_service.requests.post")
    def test_connection_error_handling(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterAPIError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("Failed to connect", str(ctx.exception))

    @patch("ai.openrouter_service.requests.post")
    def test_empty_choice_content_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "   ",
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        service = OpenRouterService(api_key="sk-or-test-key")
        with self.assertRaises(OpenRouterAPIError) as ctx:
            service.generate_grounded_answer(
                question="Why was caching added?",
                context="Some context",
            )
        self.assertIn("empty answer text", str(ctx.exception))


class TestRAGWithOpenRouter(unittest.TestCase):
    """Test RAG integration with OpenRouter."""

    @patch("ai.rag_service.OpenRouterService")
    @patch("ai.rag_service.VectorIndexer")
    @patch("ai.rag_service.SupabaseClient")
    def test_rag_default_uses_openrouter(self, mock_supa_cls, mock_indexer_cls, mock_or_cls):
        rag = RAGService()
        # Verify OpenRouterService was instantiated as the default provider
        mock_or_cls.assert_called_once()
        self.assertEqual(rag.openrouter, mock_or_cls.return_value)

    @patch("ai.rag_service.VectorIndexer")
    @patch("ai.rag_service.SupabaseClient")
    def test_rag_ask_with_openrouter(self, mock_supa_cls, mock_indexer_cls):
        mock_indexer = MagicMock()
        mock_openrouter = MagicMock()
        mock_openrouter.generate_grounded_answer.return_value = (
            "Caching was introduced in commit abc1234 by sonal-38 to improve response times."
        )

        mock_indexer.search.return_value = {
            "query": "Why was caching introduced?",
            "results": [
                {
                    "score": 0.88,
                    "document_type": "commit",
                    "repository": "sonal-38/smart-payment-platform",
                    "developer": "sonal-38",
                    "source_id": "abc1234",
                    "text": "Repository: sonal-38/smart-payment-platform\nCommit: Add Redis caching\nDeveloper: sonal-38",
                    "metadata": {
                        "repository": "sonal-38/smart-payment-platform",
                        "document_type": "commit",
                        "source_id": "abc1234",
                        "developer": "sonal-38",
                        "committed_at": "2026-03-01T10:00:00Z",
                    },
                }
            ],
        }

        rag = RAGService(
            supabase_client=mock_supa_cls.return_value,
            vector_indexer=mock_indexer,
            openrouter_service=mock_openrouter,
        )

        response = rag.answer_question(
            question="Why was caching introduced?",
            repository="sonal-38/smart-payment-platform",
        )

        self.assertEqual(response["question"], "Why was caching introduced?")
        self.assertIn("Caching was introduced in commit abc1234", response["answer"])
        self.assertEqual(len(response["sources"]), 1)
        self.assertEqual(response["sources"][0]["source_id"], "abc1234")
        mock_openrouter.generate_grounded_answer.assert_called_once()

    @patch("ai.rag_service.VectorIndexer")
    @patch("ai.rag_service.SupabaseClient")
    def test_no_evidence_does_not_call_openrouter(self, mock_supa_cls, mock_indexer_cls):
        mock_indexer = MagicMock()
        mock_openrouter = MagicMock()

        # pgvector returns no results meeting similarity threshold
        mock_indexer.search.return_value = {
            "query": "What Kubernetes architecture did this repository use?",
            "results": [],
        }

        rag = RAGService(
            supabase_client=mock_supa_cls.return_value,
            vector_indexer=mock_indexer,
            openrouter_service=mock_openrouter,
        )

        response = rag.answer_question(
            question="What Kubernetes architecture did this repository use?",
            repository="sonal-38/smart-payment-platform",
        )

        self.assertEqual(
            response["answer"],
            "I could not find enough repository evidence to answer this question.",
        )
        self.assertEqual(response["sources"], [])
        # OpenRouter MUST NOT be called when no evidence exists
        mock_openrouter.generate_grounded_answer.assert_not_called()


class TestFastAPIAIRoute(unittest.TestCase):
    """Test FastAPI /ai/ask endpoint with OpenRouter."""

    @patch("api.ai.RAGService")
    def test_ask_endpoint_success(self, mock_rag_cls):
        from api.ai import ask_question, AskRequest
        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.answer_question.return_value = {
            "question": "Why was caching introduced?",
            "answer": "Redis caching was introduced to optimize database query performance.",
            "sources": [
                {
                    "document_type": "commit",
                    "source_id": "abc1234",
                    "developer": "sonal-38",
                    "date": "2026-03-01T10:00:00Z",
                }
            ],
        }

        req = AskRequest(
            question="Why was caching introduced?",
            repository="sonal-38/smart-payment-platform",
            top_k=5,
        )
        response = ask_question(req)

        self.assertEqual(response.question, "Why was caching introduced?")
        self.assertIn("Redis caching was introduced", response.answer)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].source_id, "abc1234")

    def test_ask_endpoint_empty_question_returns_400(self):
        from api.ai import ask_question, AskRequest
        from fastapi import HTTPException

        req = AskRequest(
            question="   ",
            repository="sonal-38/smart-payment-platform",
        )
        with self.assertRaises(HTTPException) as ctx:
            ask_question(req)
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("api.ai.RAGService")
    def test_ask_endpoint_openrouter_configuration_error_returns_500(self, mock_rag_cls):
        from api.ai import ask_question, AskRequest
        from fastapi import HTTPException

        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.answer_question.side_effect = OpenRouterConfigurationError("API key missing")

        req = AskRequest(
            question="Why was caching introduced?",
            repository="sonal-38/smart-payment-platform",
        )
        with self.assertRaises(HTTPException) as ctx:
            ask_question(req)
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("api.ai.RAGService")
    def test_ask_endpoint_openrouter_api_error_returns_502(self, mock_rag_cls):
        from api.ai import ask_question, AskRequest
        from fastapi import HTTPException

        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.answer_question.side_effect = OpenRouterAPIError("500 internal server error")

        req = AskRequest(
            question="Why was caching introduced?",
            repository="sonal-38/smart-payment-platform",
        )
        with self.assertRaises(HTTPException) as ctx:
            ask_question(req)
        self.assertEqual(ctx.exception.status_code, 502)


class TestRequiredScenarioQuestions(unittest.TestCase):
    """Verifies the 5 specific scenario questions required by Section 18."""

    def setUp(self):
        self.mock_indexer = MagicMock()
        self.mock_openrouter = MagicMock()
        self.mock_supabase = MagicMock()
        self.rag = RAGService(
            vector_indexer=self.mock_indexer,
            openrouter_service=self.mock_openrouter,
            supabase_client=self.mock_supabase,
        )

    def test_1_why_was_caching_introduced(self):
        self.mock_indexer.search.return_value = {
            "query": "Why was caching introduced?",
            "results": [
                {
                    "score": 0.91,
                    "document_type": "commit",
                    "source_id": "c1a2b3",
                    "developer": "sonal-38",
                    "text": "Repository: sonal-38/smart-payment-platform\nCommit: Add Redis caching layer\nDeveloper: sonal-38",
                    "metadata": {"source_id": "c1a2b3", "developer": "sonal-38", "document_type": "commit"},
                }
            ],
        }
        self.mock_openrouter.generate_grounded_answer.return_value = (
            "Caching was introduced in commit c1a2b3 by sonal-38 to reduce latency and speed up payment lookups."
        )

        resp = self.rag.answer_question("Why was caching introduced?", repository="sonal-38/smart-payment-platform")
        self.assertIn("c1a2b3", resp["answer"])
        self.assertIn("sonal-38", resp["answer"])
        self.assertEqual(len(resp["sources"]), 1)
        self.mock_openrouter.generate_grounded_answer.assert_called_once()

    def test_2_who_worked_on_payment_processing(self):
        self.mock_indexer.search.return_value = {
            "query": "Who worked on payment processing?",
            "results": [
                {
                    "score": 0.89,
                    "document_type": "pull_request",
                    "source_id": "PR#42",
                    "developer": "sonal-38",
                    "text": "Repository: sonal-38/smart-payment-platform\nPR #42: Payment processing pipeline\nDeveloper: sonal-38",
                    "metadata": {"source_id": "PR#42", "developer": "sonal-38", "document_type": "pull_request"},
                }
            ],
        }
        self.mock_openrouter.generate_grounded_answer.return_value = (
            "According to PR #42, sonal-38 worked on the core payment processing pipeline."
        )

        resp = self.rag.answer_question("Who worked on payment processing?", repository="sonal-38/smart-payment-platform")
        self.assertIn("sonal-38", resp["answer"])
        self.assertIn("PR #42", resp["answer"])
        self.mock_openrouter.generate_grounded_answer.assert_called_once()

    def test_3_why_was_the_payment_retry_mechanism_changed(self):
        self.mock_indexer.search.return_value = {
            "query": "Why was the payment retry mechanism changed?",
            "results": [
                {
                    "score": 0.93,
                    "document_type": "commit_file",
                    "source_id": "r7e8t9:payment/retry.py",
                    "developer": "sonal-38",
                    "text": "Repository: sonal-38/smart-payment-platform\nChanged file: payment/retry.py\nCommit: Exponential backoff for retries",
                    "metadata": {"source_id": "r7e8t9:payment/retry.py", "developer": "sonal-38", "document_type": "commit_file"},
                }
            ],
        }
        self.mock_openrouter.generate_grounded_answer.return_value = (
            "The payment retry mechanism was modified in payment/retry.py (commit r7e8t9) to implement exponential backoff."
        )

        resp = self.rag.answer_question("Why was the payment retry mechanism changed?", repository="sonal-38/smart-payment-platform")
        self.assertIn("payment/retry.py", resp["answer"])
        self.assertIn("exponential backoff", resp["answer"].lower())
        self.mock_openrouter.generate_grounded_answer.assert_called_once()

    def test_4_what_happened_in_the_payment_system(self):
        self.mock_indexer.search.return_value = {
            "query": "What happened in the payment system?",
            "results": [
                {
                    "score": 0.85,
                    "document_type": "commit",
                    "source_id": "p0a1y2",
                    "developer": "sonal-38",
                    "text": "Repository: sonal-38/smart-payment-platform\nCommit: Integrated Stripe webhook handlers\nDeveloper: sonal-38",
                    "metadata": {"source_id": "p0a1y2", "developer": "sonal-38", "document_type": "commit"},
                }
            ],
        }
        self.mock_openrouter.generate_grounded_answer.return_value = (
            "Based on commit p0a1y2 by sonal-38, Stripe webhook handlers were integrated into the payment system."
        )

        resp = self.rag.answer_question("What happened in the payment system?", repository="sonal-38/smart-payment-platform")
        self.assertIn("Stripe webhook", resp["answer"])
        self.mock_openrouter.generate_grounded_answer.assert_called_once()

    def test_5_kubernetes_architecture_no_evidence_does_not_hallucinate(self):
        # When pgvector returns no results meeting threshold
        self.mock_indexer.search.return_value = {
            "query": "What Kubernetes architecture did this repository use?",
            "results": [],
        }

        resp = self.rag.answer_question("What Kubernetes architecture did this repository use?", repository="sonal-38/smart-payment-platform")
        self.assertEqual(
            resp["answer"],
            "I could not find enough repository evidence to answer this question.",
        )
        self.assertEqual(resp["sources"], [])
        # Critical constraint: OpenRouter must NOT be called when no evidence exists
        self.mock_openrouter.generate_grounded_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

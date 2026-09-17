"""
OpenRouter Service for AI Digital Shadow.

Handles communication with OpenRouter's OpenAI-compatible chat completions API:
https://openrouter.ai/api/v1/chat/completions

Provides grounded answer generation, prompt composition, and robust error handling.
Does NOT expose or log the API key.
"""
import os
import logging
from typing import Optional, Dict, Any
import requests

try:
    from dotenv import load_dotenv
    # Ensure environment variables from backend/.env or root .env are loaded
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)


class OpenRouterConfigurationError(Exception):
    """Raised when OPENROUTER_API_KEY is missing or invalid."""
    pass


class OpenRouterAPIError(Exception):
    """Raised when the OpenRouter API returns an error or fails."""
    pass


class OpenRouterService:
    """
    Manages LLM invocations via OpenRouter's chat completions API
    using Bearer token authentication with strict key safety and error abstraction.
    """

    DEFAULT_MODEL = "openrouter/free"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 60,
    ):
        raw_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
        self.model_name = (
            model_name
            or os.getenv("OPENROUTER_MODEL")
            or self.DEFAULT_MODEL
        ).strip().strip("\"' \t\r\n\u200b\ufeff")
        self.timeout = timeout

    def _ensure_api_key(self) -> str:
        """Validates and refreshes the OpenRouter API key from constructor or environment."""
        raw_key = self.api_key or os.getenv("OPENROUTER_API_KEY", "")
        clean_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")

        if not clean_key or clean_key == "your_openrouter_api_key":
            # Try reloading dotenv explicitly in case it was modified
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
                raw_key = os.getenv("OPENROUTER_API_KEY", "")
                clean_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
            except Exception:
                pass

        if not clean_key or clean_key == "your_openrouter_api_key":
            raise OpenRouterConfigurationError(
                "OpenRouter configuration missing: OPENROUTER_API_KEY is not set in backend/.env. "
                "Please add OPENROUTER_API_KEY=your_key_here to backend/.env."
            )
        self.api_key = clean_key
        return self.api_key

    def _get_model_name(self) -> str:
        """Returns the configured model name from constructor, env, or default."""
        return (
            self.model_name
            or os.getenv("OPENROUTER_MODEL", "").strip().strip("\"' \t\r\n\u200b\ufeff")
            or self.DEFAULT_MODEL
        )

    def generate_grounded_answer(
        self,
        question: str,
        context: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generates a strictly grounded answer based ONLY on the provided repository evidence
        using OpenRouter's chat completions endpoint.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            return "I could not find enough repository evidence to answer this question."

        api_key = self._ensure_api_key()
        model = self._get_model_name()

        default_system_prompt = (
            "You are an AI engineering knowledge assistant for a software codebase.\n"
            "Answer the user's question using ONLY the supplied repository evidence below.\n"
            "Strict Guidelines:\n"
            "1. Do not invent, extrapolate, or hallucinate facts not present in the evidence.\n"
            "2. If the retrieved evidence does not contain enough information to answer the question with certainty, "
            "explicitly state that the available repository evidence does not provide enough details.\n"
            "3. Cite or mention the specific evidence sources (such as commit SHA, PR number, issue number, file, or developer) "
            "supporting each key point.\n"
            "4. Distinguish verified facts from reasonable developer interpretation.\n"
            "5. Never claim that a developer made a decision or wrote a feature unless the retrieved evidence explicitly supports it.\n"
            "6. Keep your answer clear, concise, and structured."
        )

        sys_prompt = system_instruction or default_system_prompt

        user_content = (
            f"=== RETRIEVED REPOSITORY EVIDENCE ===\n\n"
            f"{context.strip()}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question.strip()}\n\n"
            f"=== GROUNDED ANSWER ==="
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-digital-shadow",
            "X-Title": "AI Digital Shadow",
        }

        try:
            resp = requests.post(
                self.API_URL,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise OpenRouterAPIError(
                f"Request to OpenRouter API timed out after {self.timeout} seconds."
            )
        except requests.exceptions.ConnectionError:
            raise OpenRouterAPIError(
                "Failed to connect to OpenRouter API. Please check your network connection."
            )
        except requests.exceptions.RequestException as e:
            # Do not include raw exception if it might contain authorization header
            error_type = type(e).__name__
            raise OpenRouterAPIError(
                f"Network error communicating with OpenRouter API ({error_type})."
            )

        # Handle specific HTTP status codes without exposing API key
        if resp.status_code == 400:
            error_detail = self._extract_error_detail(resp)
            raise OpenRouterAPIError(f"OpenRouter API returned HTTP 400 (Bad Request): {error_detail}")

        elif resp.status_code == 401:
            logger.error("OpenRouter 401 Unauthorized response")
            raise OpenRouterConfigurationError(
                "OpenRouter API authentication failed (HTTP 401 Unauthorized). "
                "Please verify your OPENROUTER_API_KEY in backend/.env."
            )

        elif resp.status_code == 403:
            error_detail = self._extract_error_detail(resp)
            raise OpenRouterConfigurationError(
                f"OpenRouter API access denied (HTTP 403 Forbidden): {error_detail}. "
                "Please check your OPENROUTER_API_KEY permissions and account balance."
            )

        elif resp.status_code == 429:
            error_detail = self._extract_error_detail(resp)
            raise OpenRouterAPIError(
                f"OpenRouter API rate limit or credit quota exceeded (HTTP 429 Too Many Requests): {error_detail}. "
                "Please wait a moment before retrying."
            )

        elif resp.status_code >= 500:
            error_detail = self._extract_error_detail(resp)
            raise OpenRouterAPIError(
                f"OpenRouter API server error (HTTP {resp.status_code}): {error_detail}. Please try again later."
            )

        elif resp.status_code != 200:
            error_detail = self._extract_error_detail(resp)
            raise OpenRouterAPIError(
                f"OpenRouter API returned unexpected HTTP status {resp.status_code}: {error_detail}"
            )

        # Parse JSON response
        try:
            data = resp.json()
        except ValueError:
            raise OpenRouterAPIError("Failed to parse OpenRouter API JSON response.")

        # Check for error payload in a 200 response (some gateways return error objects)
        if "error" in data:
            err_msg = data["error"].get("message") or str(data["error"])
            raise OpenRouterAPIError(f"OpenRouter API returned an error: {err_msg}")

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterAPIError("OpenRouter API returned no choices in response.")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content")

        if content is None:
            raise OpenRouterAPIError("OpenRouter API message content was missing.")

        answer_text = str(content).strip()
        if not answer_text:
            raise OpenRouterAPIError("OpenRouter API returned empty answer text.")

        return answer_text

    @staticmethod
    def _extract_error_detail(resp: requests.Response) -> str:
        """Safely extract error message from response without exposing secrets."""
        try:
            data = resp.json()
            if isinstance(data, dict):
                if "error" in data:
                    err = data["error"]
                    if isinstance(err, dict):
                        return err.get("message", str(err))
                    return str(err)
                if "message" in data:
                    return str(data["message"])
        except Exception:
            pass
        # Fallback to sanitized snippet of resp.text (up to 200 chars)
        text = (resp.text or "")[:200].strip()
        return text if text else "No error details provided by OpenRouter."

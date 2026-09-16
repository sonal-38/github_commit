"""
Gemini Service for AI Digital Shadow.

Handles communication with the Google Gemini API directly using standard REST endpoints
with the 'x-goog-api-key' authentication header.
Provides strictly grounded generation, prompt composition, and robust error handling.
"""
import os
import logging
from typing import Optional
import requests

try:
    from dotenv import load_dotenv
    # Ensure environment variables from backend/.env or root .env are loaded
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)


class GeminiConfigurationError(Exception):
    """Raised when GEMINI_API_KEY is missing or invalid."""
    pass


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error or fails."""
    pass


class GeminiService:
    """
    Manages Gemini LLM invocations via direct Google AI REST API calls
    using x-goog-api-key authentication with API key safety and error abstraction.
    """

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)

    def _ensure_api_key(self) -> str:
        """Validates and refreshes the Gemini API key from environment."""
        # Always re-read from os.getenv to catch changes without requiring a full server restart
        raw_key = os.getenv("GEMINI_API_KEY", "") or self.api_key or ""
        self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")

        if not self.api_key:
            # Try reloading dotenv explicitly in case it was modified
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
                raw_key = os.getenv("GEMINI_API_KEY", "")
                self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
            except Exception:
                pass

        if not self.api_key:
            raise GeminiConfigurationError(
                "Gemini configuration missing: GEMINI_API_KEY is not set in backend/.env. "
                "Please add GEMINI_API_KEY=your_key_here to backend/.env."
            )
        return self.api_key

    def generate_grounded_answer(
        self,
        question: str,
        context: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generates a strictly grounded answer based ONLY on the provided context
        using direct REST API calls with the x-goog-api-key header.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            return "I could not find enough repository evidence to answer this question."

        api_key = self._ensure_api_key()

        default_system_prompt = (
            "You are an AI engineering knowledge assistant for a software codebase.\n"
            "Answer the user's question using ONLY the supplied repository evidence below.\n"
            "Strict Guidelines:\n"
            "1. Do not invent, extrapolate, or hallucinate facts not present in the evidence.\n"
            "2. If the retrieved evidence does not contain enough information to answer the question with certainty, "
            "explicitly state that the available repository evidence does not provide enough details.\n"
            "3. Cite or mention the specific evidence sources (such as commit SHA, PR number, issue number, or developer) "
            "supporting each key point.\n"
            "4. Distinguish verified facts from reasonable developer interpretation.\n"
            "5. Never claim that a developer made a decision or wrote a feature unless the retrieved evidence explicitly supports it.\n"
            "6. Keep your answer clear, concise, and structured."
        )

        sys_prompt = system_instruction or default_system_prompt

        full_prompt = (
            f"=== RETRIEVED REPOSITORY EVIDENCE ===\n\n"
            f"{context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question.strip()}\n\n"
            f"=== GROUNDED ANSWER ==="
        )

        # Google official Gemini API documentation:
        # Gemini API requests accept API keys via x-goog-api-key or ?key=.
        # For new Google AI Studio Authentication Keys (AQ.), try x-goog-api-key, ?key=, and Authorization: Bearer
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        
        payload = {
            "system_instruction": {
                "parts": [{"text": sys_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # If 401 occurs, try ?key= query parameter (some gateways only inspect URL query params for AQ keys)
            if resp.status_code == 401:
                query_url = f"{url}?key={api_key}"
                retry_resp = requests.post(
                    query_url, 
                    json=payload, 
                    headers={"Content-Type": "application/json"}, 
                    timeout=30
                )
                if retry_resp.status_code == 200:
                    resp = retry_resp
                elif retry_resp.status_code != 401:
                    resp = retry_resp
                    
            # If still 401 and it is an AQ. or OAuth token, try Authorization: Bearer
            if resp.status_code == 401 and (api_key.startswith("AQ.") or api_key.startswith("ya29.")):
                bearer_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                bearer_resp = requests.post(
                    url,
                    json=payload,
                    headers=bearer_headers,
                    timeout=30
                )
                if bearer_resp.status_code == 200:
                    resp = bearer_resp
                elif bearer_resp.status_code != 401:
                    resp = bearer_resp
        except requests.exceptions.Timeout:
            raise GeminiAPIError("Request to Gemini API timed out after 30 seconds.")
        except requests.exceptions.ConnectionError as e:
            raise GeminiAPIError(f"Failed to connect to Gemini API: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise GeminiAPIError(f"Network error communicating with Gemini API: {str(e)}")

        # Handle specific HTTP status codes without exposing API key
        if resp.status_code == 400:
            resp_text = resp.text
            if "API_KEY_INVALID" in resp_text or "INVALID_ARGUMENT" in resp_text and "key" in resp_text.lower():
                raise GeminiConfigurationError("Invalid GEMINI_API_KEY provided in backend/.env.")
            raise GeminiAPIError(f"Gemini API returned HTTP 400 (Bad Request): {resp_text}")

        elif resp.status_code == 401:
            logger.error("Gemini 401 response: %s", resp.text)
            raise GeminiConfigurationError(
                f"Gemini API authentication failed (HTTP 401 Unauthorized): {resp.text}. "
                f"Key length: {len(api_key)}, Key prefix: {api_key[:4] if len(api_key)>=4 else 'short'}... "
                "Please check your GEMINI_API_KEY in backend/.env."
            )

        elif resp.status_code == 403:
            raise GeminiConfigurationError(
                "Gemini API access denied (HTTP 403 Forbidden). "
                "Please check your GEMINI_API_KEY permissions and ensure the Generative Language API is enabled."
            )

        elif resp.status_code == 429:
            raise GeminiAPIError(
                "Gemini API rate limit exceeded (HTTP 429 Too Many Requests). "
                "Please wait a moment before retrying."
            )

        elif resp.status_code >= 500:
            raise GeminiAPIError(
                f"Gemini API server error (HTTP {resp.status_code}). Please try again later."
            )

        elif resp.status_code != 200:
            raise GeminiAPIError(
                f"Gemini API returned unexpected HTTP status {resp.status_code}: {resp.text}"
            )

        # Parse JSON response
        try:
            data = resp.json()
        except ValueError as e:
            raise GeminiAPIError(f"Failed to parse Gemini API JSON response: {str(e)}")

        # Check prompt feedback (blocked prompt, safety ratings, etc.)
        prompt_feedback = data.get("promptFeedback", {})
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise GeminiAPIError(f"Gemini request was blocked by safety filters (reason: {block_reason}).")

        candidates = data.get("candidates", [])
        if not candidates:
            raise GeminiAPIError("Gemini API returned no response candidates.")

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason")
        if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
            logger.warning("Gemini candidate finishReason: %s", finish_reason)

        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise GeminiAPIError("Gemini API candidate response contained no text parts.")

        text_parts = [p.get("text", "") for p in parts if "text" in p]
        answer_text = "".join(text_parts).strip()

        if not answer_text:
            raise GeminiAPIError("Gemini API candidate text was empty.")

        return answer_text

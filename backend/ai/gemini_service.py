"""
Gemini Service for AI Digital Shadow.

Handles communication with the Google Gemini API using google-genai SDK
or standard Google AI REST endpoints as fallback.
Provides strictly grounded generation, prompt composition, and error handling.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiConfigurationError(Exception):
    """Raised when GEMINI_API_KEY is missing or invalid."""
    pass


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error or fails."""
    pass


class GeminiService:
    """
    Manages Gemini LLM invocations with API key safety and error abstraction.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        """Lazy initialization of the official Google GenAI client."""
        if not self.api_key:
            raise GeminiConfigurationError(
                "Gemini configuration missing: GEMINI_API_KEY is not set in backend/.env. "
                "Please add GEMINI_API_KEY=your_key_here to backend/.env."
            )

        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                # If google-genai is not installed, we can fall back to google.generativeai or requests
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.api_key)
                    self._client = legacy_genai.GenerativeModel(self.model_name)
                    self._is_legacy = True
                    return self._client
                except ImportError:
                    pass
                self._client = "REST_FALLBACK"

        return self._client

    def generate_grounded_answer(
        self,
        question: str,
        context: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generates a strictly grounded answer based ONLY on the provided context.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            return "I could not find enough repository evidence to answer this question."

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

        client = self._get_client()

        # Path 1: Official google-genai SDK
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            try:
                from google.genai import types
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        temperature=0.2,  # Low temperature for factuality & groundness
                    ),
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
                raise GeminiAPIError("Gemini returned an empty response.")
            except Exception as e:
                err_msg = str(e)
                if "API_KEY" in err_msg or "401" in err_msg or "unauthenticated" in err_msg.lower():
                    raise GeminiConfigurationError("Gemini API authentication failed. Check your GEMINI_API_KEY.")
                raise GeminiAPIError(f"Gemini generation error: {err_msg}")

        # Path 2: Legacy google.generativeai SDK
        if getattr(self, "_is_legacy", False):
            try:
                combined_prompt = f"{sys_prompt}\n\n{full_prompt}"
                response = client.generate_content(
                    combined_prompt,
                    generation_config={"temperature": 0.2},
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
                raise GeminiAPIError("Gemini returned an empty response.")
            except Exception as e:
                err_msg = str(e)
                if "API_KEY" in err_msg or "401" in err_msg:
                    raise GeminiConfigurationError("Gemini API authentication failed. Check your GEMINI_API_KEY.")
                raise GeminiAPIError(f"Gemini generation error: {err_msg}")

        # Path 3: Direct HTTP REST fallback (zero extra third-party SDK dependencies required)
        return self._generate_via_rest(sys_prompt, full_prompt)

    def _generate_via_rest(self, system_instruction: str, prompt_text: str) -> str:
        """
        Fallback REST caller directly to Google Gemini API endpoint via requests.
        Ensures the service operates even if google-genai library installation is pending.
        """
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": prompt_text}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }

        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        except requests.exceptions.RequestException as e:
            raise GeminiAPIError(f"Network error communicating with Gemini API: {str(e)}")

        if resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
            raise GeminiConfigurationError("Invalid GEMINI_API_KEY provided in backend/.env.")
        elif resp.status_code == 403:
            raise GeminiConfigurationError("Gemini API access denied. Check your GEMINI_API_KEY permissions.")
        elif resp.status_code != 200:
            raise GeminiAPIError(f"Gemini API returned HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise GeminiAPIError("Gemini API returned no response candidates.")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise GeminiAPIError("Gemini API candidate has no text parts.")
            return parts[0].get("text", "").strip()
        except (ValueError, KeyError, IndexError) as e:
            raise GeminiAPIError(f"Failed to parse Gemini API response: {str(e)}")

"""
Gemini Embedding service using the official google-genai SDK.

Model: 'gemini-embedding-001'
Supports asymmetric retrieval:
- RETRIEVAL_DOCUMENT: Used when embedding repository artifacts to be indexed in Supabase pgvector.
- RETRIEVAL_QUERY: Used when embedding natural language queries to search against pgvector.

Uses GEMINI_API_KEY from environment variables (backend/.env).
Includes a direct Google Generative Language REST fallback for maximum resilience.
"""
import os
import logging
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_VECTOR_DIMENSION = 768


class EmbeddingError(Exception):
    """Raised when embedding model loading or vector generation fails."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmbeddingService:
    """
    Manages Gemini text embedding generation using gemini-embedding-001 with task_type differentiation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_VECTOR_DIMENSION,
    ):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
        self.model_name = model_name
        self._dimension = dimension
        self._client = None

    def _get_api_key(self) -> str:
        """Retrieves and sanitizes the GEMINI_API_KEY."""
        if not self.api_key:
            raw_key = os.getenv("GEMINI_API_KEY", "")
            self.api_key = raw_key.strip().strip("\"' \t\r\n\u200b\ufeff")
        if not self.api_key:
            raise EmbeddingError(
                "Gemini configuration missing: GEMINI_API_KEY is not set in backend/.env. "
                "Please add GEMINI_API_KEY=your_key_here to backend/.env.",
                status_code=500,
            )
        return self.api_key

    def _get_client(self):
        """Lazy initialization of the official Google GenAI client."""
        key = self._get_api_key()
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
            except ImportError:
                self._client = "REST_FALLBACK"
        return self._client

    @property
    def dimension(self) -> int:
        """Returns the configured vector dimension (default 768 for gemini-embedding-001)."""
        return self._dimension

    def embed_document(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a repository document using task_type='RETRIEVAL_DOCUMENT'.
        """
        return self.embed_text(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a search query using task_type='RETRIEVAL_QUERY'.
        """
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> List[float]:
        """
        Embeds a single string into a dense vector using Gemini embedContent.
        Supports task_type ('RETRIEVAL_QUERY' or 'RETRIEVAL_DOCUMENT').
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty or whitespace text", status_code=400)

        results = self.embed_batch([text.strip()], task_type=task_type, batch_size=1)
        if not results:
            raise EmbeddingError("Embedding generation produced no vector output")
        return results[0]

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 20,
    ) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of texts using Gemini embedContent.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t and t.strip() else "[empty]" for t in texts]
        client = self._get_client()

        all_embeddings: List[List[float]] = []

        # Process in batches to respect API limits
        for i in range(0, len(cleaned_texts), batch_size):
            chunk = cleaned_texts[i : i + batch_size]

            # Path 1: Official google-genai SDK
            if client != "REST_FALLBACK" and hasattr(client, "models") and hasattr(client.models, "embed_content"):
                try:
                    from google.genai import types
                    config = types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    )
                    # When chunk has multiple items, contents can be a list
                    response = client.models.embed_content(
                        model=self.model_name,
                        contents=chunk if len(chunk) > 1 else chunk[0],
                        config=config,
                    )
                    extracted = self._extract_sdk_embeddings(response, expected_count=len(chunk))
                    all_embeddings.extend(extracted)
                    continue
                except Exception as sdk_err:
                    logger.warning(
                        "google-genai SDK embed_content failed (%s), falling back to REST: %s",
                        type(sdk_err).__name__,
                        str(sdk_err),
                    )

            # Path 2: REST Fallback
            rest_embeddings = self._embed_via_rest(chunk, task_type=task_type)
            all_embeddings.extend(rest_embeddings)

        return all_embeddings

    def _extract_sdk_embeddings(self, response: Any, expected_count: int) -> List[List[float]]:
        """
        Extracts float list vectors from the google-genai EmbedContentResponse object.
        """
        # Case A: response has .embeddings (list of ContentEmbedding)
        if hasattr(response, "embeddings") and response.embeddings:
            extracted = []
            for item in response.embeddings:
                if hasattr(item, "values"):
                    extracted.append(list(item.values))
                elif isinstance(item, (list, tuple)):
                    extracted.append(list(item))
            if extracted:
                return extracted

        # Case B: single embedding returned via .embedding
        if hasattr(response, "embedding") and response.embedding:
            emb = response.embedding
            if hasattr(emb, "values"):
                return [list(emb.values)]
            elif isinstance(emb, (list, tuple)):
                return [list(emb)]

        raise EmbeddingError("Unable to extract vector values from google-genai response")

    def _embed_via_rest(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Direct REST fallback to Google Generative Language API embedContent endpoint.
        """
        try:
            import requests
        except ImportError:
            raise EmbeddingError("Neither google-genai nor requests library is installed.")

        key = self._get_api_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={key}"

        results: List[List[float]] = []

        for text in texts:
            payload: Dict[str, Any] = {
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": task_type,
                "outputDimensionality": self._dimension,
            }

            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=20,
                )
            except requests.exceptions.RequestException as e:
                raise EmbeddingError(f"Network error communicating with Gemini Embeddings API: {str(e)}")

            if resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
                raise EmbeddingError("Invalid GEMINI_API_KEY provided in backend/.env.", status_code=401)
            elif resp.status_code == 403:
                raise EmbeddingError("Gemini API access denied. Check your GEMINI_API_KEY permissions.", status_code=403)
            elif resp.status_code != 200:
                raise EmbeddingError(
                    f"Gemini Embeddings API error HTTP {resp.status_code}: {resp.text}",
                    status_code=502,
                )

            try:
                data = resp.json()
                embedding_obj = data.get("embedding", {})
                values = embedding_obj.get("values", [])
                if not values:
                    raise EmbeddingError("Gemini Embeddings API returned empty vector values.")
                results.append([float(v) for v in values])
            except (ValueError, KeyError) as e:
                raise EmbeddingError(f"Failed to parse Gemini Embeddings API response: {str(e)}")

        return results

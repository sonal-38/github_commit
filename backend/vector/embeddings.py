"""
Embedding service using local Sentence Transformers with BAAI/bge-base-en-v1.5.

Model: 'BAAI/bge-base-en-v1.5'
Produces 768-dimensional normalized dense vectors.
Loads model lazily to avoid unnecessary startup overhead.
No external API keys or HF_TOKEN required.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_VECTOR_DIMENSION = 768


class EmbeddingError(Exception):
    """Raised when embedding model loading or vector generation fails."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmbeddingService:
    """
    Manages SentenceTransformer embedding model lifecycle and transforms text into 768-dim vectors.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._dimension = DEFAULT_VECTOR_DIMENSION

    def _get_model(self):
        """Lazy load the BAAI/bge-base-en-v1.5 SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension()
                elif hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Please install it via "
                    "pip install sentence-transformers",
                    status_code=500,
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}': {str(e)}",
                    status_code=500,
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the output vector dimension for the active model (768)."""
        return self._dimension

    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """
        Embeds a single string into a 768-dimensional normalized dense vector.
        Accepts optional kwargs for compatibility with previous callers.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty or whitespace text", status_code=400)

        results = self.embed_batch([text.strip()], batch_size=1)
        if not results:
            raise EmbeddingError("Embedding generation produced no vector output")
        return results[0]

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of document strings into 768-dimensional normalized dense vectors.
        """
        return self.embed_batch(texts, batch_size=batch_size)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Generates 768-dimensional normalized dense vectors for a list of texts in batches.
        Normalized embeddings ensure direct compatibility with cosine distance search (1 - <=>).
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t and t.strip() else "[empty]" for t in texts]
        model = self._get_model()

        try:
            embeddings = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Convert numpy array to list of floats for pgvector/JSON serialization
            return [vec.tolist() for vec in embeddings]
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings for batch of {len(texts)} texts: {str(e)}",
                status_code=500,
            )

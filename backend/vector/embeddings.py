"""
Embedding service using lightweight sentence-transformers.

Uses a standard, fast, CPU-friendly model ('sentence-transformers/all-MiniLM-L6-v2' by default)
which produces 384-dimensional dense vectors.
Loads model lazily to avoid startup overhead.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIMENSION = 384


class EmbeddingError(Exception):
    """Raised when embedding model loading or vector generation fails."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmbeddingService:
    """
    Manages embedding model lifecycle and transforms text into numeric vectors.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def _get_model(self):
        """Lazy load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Please install it via "
                    "pip install sentence-transformers"
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}': {str(e)}"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the output vector dimension for the active model."""
        if self._dimension is None:
            try:
                model = self._get_model()
                self._dimension = model.get_sentence_embedding_dimension()
            except Exception:
                # Standard dimension for all-MiniLM-L6-v2
                return DEFAULT_VECTOR_DIMENSION
        return self._dimension or DEFAULT_VECTOR_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a vector.
        Rejects empty or purely whitespace strings.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty or whitespace text", status_code=400)

        results = self.embed_batch([text.strip()])
        if not results:
            raise EmbeddingError("Embedding generation produced no vector output")
        return results[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of texts in batches.
        Empty texts are replaced with a minimal placeholder to maintain index alignment.
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
            # Convert numpy array to list of floats for JSON / Qdrant serialization
            return [vec.tolist() for vec in embeddings]
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings for batch of {len(texts)} texts: {str(e)}"
            )

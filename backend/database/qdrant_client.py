"""
Qdrant Cloud client integration.

Loads QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION_NAME from environment variables.
Provides connection validation, collection management, and vector upsert/search methods.
Credentials and API keys are never exposed in error responses.
"""
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class QdrantConfigurationError(Exception):
    """Raised when Qdrant credentials or configuration are missing or invalid."""
    pass


class QdrantConnectionError(Exception):
    """Raised when connection to Qdrant Cloud fails."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class QdrantOperationError(Exception):
    """Raised when a Qdrant operation (upsert, search, collection creation) fails."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class QdrantDatabaseClient:
    """
    Client for interacting with Qdrant Cloud vector database.
    Provides lazy client initialization, collection verification, and vector operations.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.url = (url or os.getenv("QDRANT_URL", "")).strip()
        self.api_key = (api_key or os.getenv("QDRANT_API_KEY", "")).strip()
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION_NAME", "digital_shadow")
        ).strip()

        if not self.url:
            raise QdrantConfigurationError(
                "Qdrant configuration missing: QDRANT_URL must be set in backend/.env"
            )
        if not self.api_key:
            raise QdrantConfigurationError(
                "Qdrant configuration missing: QDRANT_API_KEY must be set in backend/.env"
            )
        if not self.collection_name:
            self.collection_name = "digital_shadow"

        self._client = None

    def get_client(self):
        """Lazy initialization of the Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=20,
                )
            except ImportError:
                raise QdrantConfigurationError(
                    "qdrant-client package is not installed. Please install it via "
                    "pip install qdrant-client"
                )
            except Exception as e:
                raise QdrantConnectionError(
                    f"Failed to initialize Qdrant Cloud client: {str(e)}"
                )
        return self._client

    def verify_connectivity(self) -> bool:
        """Verifies active connectivity to the Qdrant Cloud instance."""
        try:
            client = self.get_client()
            # Attempt to fetch collections to verify network and credentials
            client.get_collections()
            return True
        except QdrantConfigurationError:
            raise
        except Exception as e:
            raise QdrantConnectionError(
                f"Failed to connect to Qdrant Cloud. Please verify QDRANT_URL and QDRANT_API_KEY. Error: {str(e)}"
            )

    def ensure_collection(self, vector_dimension: int) -> bool:
        """
        Ensures the collection exists in Qdrant with the appropriate vector dimension and metric.
        If collection exists: reuses it safely without recreation.
        If not exists: creates it with Cosine distance.
        """
        client = self.get_client()
        try:
            from qdrant_client.http import models
            from qdrant_client.http.exceptions import UnexpectedResponse

            # Check existing collections safely
            collection_exists = False
            try:
                collection_info = client.get_collection(collection_name=self.collection_name)
                collection_exists = True
            except (UnexpectedResponse, Exception):
                collection_exists = False

            if not collection_exists:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
            return True
        except Exception as e:
            raise QdrantOperationError(
                f"Failed to ensure Qdrant collection '{self.collection_name}': {str(e)}"
            )

    def upsert_points(self, points: List[Any]) -> bool:
        """
        Upserts a batch of vector points into the target collection.
        Uses idempotent upsert so existing point IDs are updated, preventing duplicates.
        """
        if not points:
            return True

        client = self.get_client()
        try:
            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            return True
        except Exception as e:
            raise QdrantOperationError(
                f"Failed to upsert points into Qdrant collection '{self.collection_name}': {str(e)}"
            )

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        repository_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the Qdrant collection.
        Optionally filters by repository metadata.
        """
        client = self.get_client()
        try:
            from qdrant_client.http import models

            qdrant_filter = None
            if repository_filter:
                repo_clean = repository_filter.strip()
                # Match either the full identifier or normalized repo name
                qdrant_filter = models.Filter(
                    should=[
                        models.FieldCondition(
                            key="repository",
                            match=models.MatchValue(value=repo_clean),
                        ),
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=repo_clean),
                        ),
                    ]
                )

            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            formatted_results = []
            for hit in search_results:
                payload = hit.payload or {}
                formatted_results.append({
                    "score": round(float(hit.score), 4),
                    "document_type": payload.get("document_type", "unknown"),
                    "repository": payload.get("repository", ""),
                    "developer": payload.get("developer", "unknown"),
                    "source_id": payload.get("source_id", ""),
                    "text": payload.get("text", ""),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k not in ["text", "document_type", "repository", "developer", "source_id"]
                    },
                })
            return formatted_results
        except Exception as e:
            raise QdrantOperationError(
                f"Failed to execute semantic search in Qdrant: {str(e)}"
            )

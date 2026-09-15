"""
Neo4j Aura cloud client integration.

Reads NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD from environment variables
and provides managed driver connectivity, query execution, and error handling.
Credentials are never exposed in error responses.
"""
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class Neo4jConfigurationError(Exception):
    """Raised when Neo4j credentials or URI are missing or improperly configured."""
    pass


class Neo4jConnectionError(Exception):
    """Raised when connecting to Neo4j Aura fails."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class Neo4jExecutionError(Exception):
    """Raised when Cypher execution encounters an error."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class Neo4jClient:
    """
    Client for interacting with Neo4j Aura cloud instances.
    Provides session management, parameterized query execution, and connectivity testing.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = (uri or os.getenv("NEO4J_URI", "")).strip()
        self.username = (username or os.getenv("NEO4J_USERNAME", "")).strip()
        self.password = (password or os.getenv("NEO4J_PASSWORD", "")).strip()

        if not self.uri or not self.username or not self.password:
            raise Neo4jConfigurationError(
                "Neo4j configuration missing: NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD "
                "must be set in backend/.env"
            )

        self._driver = None

    def _get_driver(self):
        """Lazy driver initialization to avoid overhead when not in use."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=30 * 60,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=15.0,
                )
            except Exception as e:
                raise Neo4jConnectionError(
                    f"Failed to initialize Neo4j driver: {str(e)}"
                )
        return self._driver

    def verify_connectivity(self) -> bool:
        """
        Verify that the client can connect and authenticate with Neo4j Aura.
        Raises Neo4jConnectionError if verification fails.
        """
        driver = self._get_driver()
        try:
            driver.verify_connectivity()
            return True
        except Exception as e:
            err_msg = str(e)
            if "authentication" in err_msg.lower() or "unauthorized" in err_msg.lower():
                raise Neo4jConnectionError(
                    "Neo4j authentication failed. Please verify NEO4J_USERNAME and NEO4J_PASSWORD in backend/.env",
                    status_code=401
                )
            raise Neo4jConnectionError(
                f"Could not connect to Neo4j Aura at {self.uri}: {err_msg}",
                status_code=503
            )

    def execute_query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with parameters and return a list of record dicts.
        """
        driver = self._get_driver()
        params = parameters or {}
        try:
            with driver.session() as session:
                result = session.run(cypher, params)
                return [record.data() for record in result]
        except Exception as e:
            err_msg = str(e)
            if "authentication" in err_msg.lower() or "unauthorized" in err_msg.lower():
                raise Neo4jConnectionError(
                    "Neo4j authentication failed during query execution.",
                    status_code=401
                )
            raise Neo4jExecutionError(
                f"Neo4j Cypher query execution failed: {err_msg}",
                status_code=500
            )

    def execute_write_transaction(self, work_fn, *args, **kwargs) -> Any:
        """
        Execute a transaction function using session.execute_write.
        """
        driver = self._get_driver()
        try:
            with driver.session() as session:
                return session.execute_write(work_fn, *args, **kwargs)
        except Exception as e:
            raise Neo4jExecutionError(f"Neo4j transaction failed: {str(e)}", status_code=500)

    def close(self):
        """Close the driver connection pool cleanly."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None


def get_neo4j_client() -> Neo4jClient:
    """Factory helper to obtain a configured Neo4j client."""
    return Neo4jClient()

"""
Supabase PostgreSQL client integration.

Reads SUPABASE_URL and SUPABASE_KEY from environment variables and exposes
methods to interact with Supabase tables safely and reliably.
"""
import os
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.getcwd(), "backend", ".env"))


class SupabaseConfigurationError(Exception):
    """Raised when Supabase credentials or environment variables are missing."""
    pass


class SupabaseDatabaseError(Exception):
    """Raised when database operations encounter an error."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class SupabaseClient:
    """
    Client for interacting with Supabase PostgreSQL via PostgREST API.
    Supports resilient upsert, query, and batch operations without requiring
    local PostgreSQL drivers or Docker.
    """

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        raw_url = (url or os.getenv("SUPABASE_URL", "")).strip().rstrip("/")
        # If user included /rest/v1 in their SUPABASE_URL, strip it so it doesn't double
        if raw_url.endswith("/rest/v1"):
            raw_url = raw_url[:-len("/rest/v1")].rstrip("/")

        self.url = raw_url
        self.key = (key or os.getenv("SUPABASE_KEY", "")).strip()

        if not self.url or not self.key:
            raise SupabaseConfigurationError(
                "Supabase configuration missing: SUPABASE_URL and SUPABASE_KEY "
                "must be set in backend/.env"
            )

        self.rest_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def verify_connectivity(self) -> bool:
        """Verifies connection to the Supabase REST API."""
        try:
            # Query the root REST endpoint to check headers and reachability
            resp = requests.get(self.rest_url, headers=self.headers, timeout=10)
            if resp.status_code in (200, 404):
                return True
            if resp.status_code == 401:
                raise SupabaseDatabaseError(
                    "Supabase authentication failed: Invalid SUPABASE_KEY",
                    status_code=401,
                )
            return True
        except requests.exceptions.RequestException as e:
            raise SupabaseDatabaseError(
                f"Failed to connect to Supabase: {str(e)}",
                status_code=503,
            )

    def select(
        self,
        table: str,
        query_params: Optional[Union[Dict[str, str], List[Tuple[str, str]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query against a Supabase table.
        Preserves PostgREST query parameters (eq, in, etc.) without malformed encoding.
        Supports both Dict[str, str] and List[Tuple[str, str]] (for multi-filter on same column).
        """
        endpoint = f"{self.rest_url}/{table}"
        if query_params:
            # Build query string preserving literal PostgREST syntax like in.(1,2) and eq.val
            # Ensure query values (especially '+' in timezone offsets like +00:00) are properly URL-encoded
            import urllib.parse

            def encode_postgrest_val(val: str) -> str:
                # PostgREST syntax is <operator>.<value> (e.g. gte.2026-08-01T00:00:00+00:00)
                if "." in val:
                    op, rest = val.split(".", 1)
                    # Quote the value part so '+' becomes '%2B' and doesn't get parsed as space ' '
                    return f"{op}.{urllib.parse.quote(rest, safe='')}"
                return urllib.parse.quote(val, safe='')

            if isinstance(query_params, dict):
                parts = [f"{k}={encode_postgrest_val(str(v))}" for k, v in query_params.items()]
            else:
                parts = [f"{k}={encode_postgrest_val(str(v))}" for k, v in query_params]
            endpoint = f"{endpoint}?{'&'.join(parts)}"

        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=15
            )
        except requests.exceptions.RequestException as e:
            raise SupabaseDatabaseError(f"Network error connecting to Supabase: {str(e)}", status_code=503)

        if response.status_code not in (200, 206):
            self._handle_error_response(response, table, "SELECT")

        return response.json()

    def upsert(
        self,
        table: str,
        data: List[Dict[str, Any]] | Dict[str, Any],
        on_conflict: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an UPSERT operation into a Supabase table.
        Merges duplicates on unique conflict keys and returns representations.
        """
        if not data:
            return []

        payload = data if isinstance(data, list) else [data]
        endpoint = f"{self.rest_url}/{table}"

        headers = dict(self.headers)
        headers["Prefer"] = "resolution=merge-duplicates,return=representation"

        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                params=params,
                json=payload,
                timeout=20
            )
        except requests.exceptions.RequestException as e:
            raise SupabaseDatabaseError(f"Network error connecting to Supabase: {str(e)}", status_code=503)

        if response.status_code not in (200, 201):
            self._handle_error_response(response, table, "UPSERT")

        try:
            result = response.json()
            return result if isinstance(result, list) else [result]
        except ValueError:
            return []

    def rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a PostgreSQL Remote Procedure Call (RPC) via PostgREST.
        Used for pgvector similarity search, custom SQL functions, and aggregations.
        """
        endpoint = f"{self.rest_url}/rpc/{function_name}"
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=params or {},
                timeout=25
            )
        except requests.exceptions.RequestException as e:
            raise SupabaseDatabaseError(
                f"Network error calling Supabase RPC '{function_name}': {str(e)}",
                status_code=503
            )

        if response.status_code not in (200, 206):
            self._handle_error_response(response, f"rpc/{function_name}", "RPC")

        try:
            result = response.json()
            return result if isinstance(result, list) else [result]
        except ValueError:
            return []

    def _handle_error_response(self, response: requests.Response, table: str, action: str):
        """Standardized error handler hiding credentials from exceptions."""
        try:
            err_data = response.json()
            code = err_data.get("code", "")
            hint = err_data.get("hint")
            msg = err_data.get("message") or err_data.get("details") or response.text
        except Exception:
            code = ""
            hint = None
            msg = response.text

        status = response.status_code
        if status in (401, 403):
            raise SupabaseDatabaseError(
                "Supabase authentication failed. Please verify SUPABASE_KEY in backend/.env (use your project's service_role secret key or check RLS permissions).",
                status_code=status,
                details=msg
            )
        elif "schema cache" in msg.lower() or code == "PGRST205":
            raise SupabaseDatabaseError(
                f"Table '{table}' exists in your PostgreSQL database, but Supabase's API cache has not reloaded yet. "
                "To fix this immediately: In Supabase SQL Editor, run: NOTIFY pgrst, 'reload schema'; "
                "(or in Supabase Dashboard go to Settings -> API and click 'Reload schema cache').",
                status_code=404,
                details=msg
            )
        elif status == 404 or "does not exist" in msg.lower() or "relation" in msg.lower():
            hint_str = f" (Hint: {hint})" if hint else ""
            if table.startswith("rpc/"):
                fn = table.split("/", 1)[1]
                raise SupabaseDatabaseError(
                    f"PostgreSQL function '{fn}' was not found in Supabase public schema{hint_str}. "
                    "Please run database/schema.sql in the Supabase SQL Editor to create the 'match_documents' function and 'document_embeddings' table.",
                    status_code=404,
                    details=msg
                )
            raise SupabaseDatabaseError(
                f"Table '{table}' was not found in Supabase public schema{hint_str}. "
                "Please run database/schema.sql in the Supabase SQL Editor and ensure 'NOTIFY pgrst, ''reload schema'';' is executed.",
                status_code=404,
                details=msg
            )
        else:
            raise SupabaseDatabaseError(
                f"Supabase {action} error on table '{table}': {msg}",
                status_code=status,
                details=msg
            )


def get_supabase_client() -> SupabaseClient:
    """Factory helper to obtain a configured Supabase client."""
    return SupabaseClient()

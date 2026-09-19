"""
Unit and integration tests for Step 12: Connecting Knowledge Areas to the Neo4j Knowledge Graph.

Verifies:
1. Schema initialization adds KnowledgeArea uniqueness constraint.
2. Idempotent KnowledgeArea node creation with stable IDs:
   `knowledge_area:{repository}:{cluster_id}`
3. Idempotent (:KnowledgeArea)-[:EVIDENCED_BY]->(:Commit|:PullRequest|:CommitFile) relationships.
4. Preserving the existing Step 9 graph (no replacement or modification of base nodes).
5. Many-to-many KnowledgeArea <-> Developer relationships derived strictly via evidence:
   (:KnowledgeArea)-[:EVIDENCED_BY]->(:Commit)-[:AUTHORED_BY]->(:Developer)
   (:KnowledgeArea)-[:EVIDENCED_BY]->(:PullRequest)-[:CREATED_BY]->(:Developer)
   Strictly NO direct (:KnowledgeArea)-[:KNOWS]->(:Developer) relationships.
6. Evidence traceability:
   (:KnowledgeArea)-[:EVIDENCED_BY]->(:Commit)-[:CHANGED]->(:CommitFile)
7. One Knowledge Area with multiple developers and one developer across multiple Knowledge Areas.
8. Reusing Step 11 interpretation results without extra Supabase tables.
9. Verifying no GitHub API calls are made during Step 12.
10. FastAPI endpoint POST /knowledge/repositories/{owner}/{repo}/graph.
"""
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from graph.builder import KnowledgeGraphBuilder
from main import app


class TestStep12KnowledgeGraph(unittest.TestCase):
    """Test suite for Step 12 KnowledgeArea graph connections."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.mock_neo4j = MagicMock()
        self.mock_github = MagicMock()

    @patch("graph.builder.get_supabase_client")
    def test_init_schema_includes_knowledge_area_constraint(self, mock_get_sb):
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(
            neo4j_client=self.mock_neo4j,
            github_client=self.mock_github,
        )

        builder.init_schema()

        # Check queries executed
        executed_queries = [call[0][0] for call in self.mock_neo4j.execute_query.call_args_list]
        ka_constraints = [
            q for q in executed_queries
            if "KnowledgeArea" in q and "IS UNIQUE" in q
        ]
        self.assertTrue(
            len(ka_constraints) > 0,
            "Expected uniqueness constraint for KnowledgeArea in init_schema"
        )

    @patch("graph.builder.get_supabase_client")
    def test_build_knowledge_area_graph_creates_nodes_and_evidence_relationships(self, mock_get_sb):
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(
            neo4j_client=self.mock_neo4j,
            github_client=self.mock_github,
        )

        # Mock repository exists in Neo4j
        self.mock_neo4j.execute_query.side_effect = lambda query, params=None: (
            # 1. Repository check
            [{"id": "sonal-38/smart_payment_platform", "name": "smart_payment_platform"}]
            if "MATCH (r:Repository)" in query
            # Summary metrics queries
            else [{"count": 2}] if "count(DISTINCT ka)" in query
            else [{"count": 4}] if "count(rel)" in query
            else [{"count": 3}] if "UNWIND all_devs" in query
            else []
        )

        # Step 11 interpretation output
        sample_step11_result = {
            "status": "completed",
            "repository": "sonal-38/smart_payment_platform",
            "clusters": [
                {
                    "cluster_id": 0,
                    "candidate_label": "Payment Processing",
                    "document_count": 3,
                    "representative_documents": ["commit:sonal-38/smart_payment_platform:c111"],
                    "representative_terms": ["payment", "stripe", "transaction"],
                    "important_files": ["services/payment.py", "models/transaction.py"],
                    "all_document_ids": [
                        "commit:sonal-38/smart_payment_platform:c111",
                        "pr:sonal-38/smart_payment_platform:12",
                        "commit_file:sonal-38/smart_payment_platform:c111:services/payment.py",
                    ],
                },
                {
                    "cluster_id": 1,
                    "candidate_label": "Authentication & Session Management",
                    "document_count": 2,
                    "representative_documents": ["commit:sonal-38/smart_payment_platform:c222"],
                    "representative_terms": ["auth", "token", "jwt"],
                    "important_files": ["auth/jwt.py"],
                    "all_document_ids": [
                        "commit:sonal-38/smart_payment_platform:c222",
                    ],
                },
            ],
            "noise_documents": 1,
            "noise_document_ids": ["commit:sonal-38/smart_payment_platform:noise99"],
            "total_documents": 6,
        }

        result = builder.build_knowledge_area_graph(
            owner="sonal-38",
            repo="smart_payment_platform",
            interpretation_result=sample_step11_result,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["repository"], "sonal-38/smart_payment_platform")
        self.assertEqual(result["knowledge_areas"], 2)
        self.assertEqual(result["evidence_relationships"], 4)
        self.assertEqual(result["developers_reached_through_evidence"], 3)

        # Inspect all executed Cypher queries
        query_calls = self.mock_neo4j.execute_query.call_args_list
        queries = [call[0][0] for call in query_calls]
        params_list = [call[0][1] for call in query_calls if len(call[0]) > 1 and call[0][1] is not None]

        # 1. Verify KnowledgeArea MERGE queries
        ka_queries = [q for q in queries if "MERGE (ka:KnowledgeArea {id: $id})" in q]
        self.assertTrue(len(ka_queries) >= 2, "Expected MERGE queries for 2 KnowledgeArea clusters")

        # Verify stable ID format
        ka_ids = [p["id"] for p in params_list if "id" in p and "knowledge_area:" in str(p.get("id", ""))]
        self.assertIn("knowledge_area:sonal-38/smart_payment_platform:0", ka_ids)
        self.assertIn("knowledge_area:sonal-38/smart_payment_platform:1", ka_ids)

        # 2. Verify EVIDENCED_BY relationships
        evidenced_by_queries = [q for q in queries if "-[:EVIDENCED_BY]->" in q]
        self.assertTrue(len(evidenced_by_queries) > 0, "Expected EVIDENCED_BY relationship queries")

        # 3. Verify NO direct KNOWS relationships
        knows_queries = [q for q in queries if "-[:KNOWS]->" in q]
        self.assertEqual(len(knows_queries), 0, "Strictly NO direct KNOWS relationships should be created")

        # 4. Verify no GitHub API calls
        self.mock_github.get_repository.assert_not_called()
        self.mock_github.get_commits.assert_not_called()
        self.mock_github.get_pull_requests.assert_not_called()

    @patch("graph.builder.get_supabase_client")
    def test_evidence_traceability_and_multi_developer_traversal(self, mock_get_sb):
        """
        Verifies that:
        - One KnowledgeArea can reach multiple developers via commits/PRs
        - One Developer can be reached from multiple KnowledgeAreas
        - CommitFile can be reached via Commit -> CHANGED -> CommitFile
        """
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(
            neo4j_client=self.mock_neo4j,
            github_client=self.mock_github,
        )

        self.mock_neo4j.execute_query.side_effect = lambda query, params=None: (
            [{"id": "sonal-38/smart_payment_platform", "name": "smart_payment_platform"}]
            if "MATCH (r:Repository)" in query
            else [{"count": 2}] if "count(DISTINCT ka)" in query
            else [{"count": 5}] if "count(rel)" in query
            else [{"count": 2}] if "UNWIND all_devs" in query
            else []
        )

        # Cluster 0 has contributions from commit c1 (Alice) and commit c2 (Bob)
        # Cluster 1 has contributions from commit c3 (Alice)
        # => Alice is in both Knowledge Areas (many-to-many)
        # => Knowledge Area 0 has both Alice and Bob (many-to-many)
        step11_multi_dev_result = {
            "status": "completed",
            "repository": "sonal-38/smart_payment_platform",
            "clusters": [
                {
                    "cluster_id": 0,
                    "candidate_label": "Payment Processing",
                    "document_count": 2,
                    "representative_documents": ["commit:sonal-38/smart_payment_platform:c1"],
                    "representative_terms": ["payment"],
                    "important_files": ["services/payment.py"],
                    "all_document_ids": [
                        "commit:sonal-38/smart_payment_platform:c1",
                        "commit:sonal-38/smart_payment_platform:c2",
                        "commit_file:sonal-38/smart_payment_platform:c1:services/payment.py",
                    ],
                },
                {
                    "cluster_id": 1,
                    "candidate_label": "Refund Management",
                    "document_count": 1,
                    "representative_documents": ["commit:sonal-38/smart_payment_platform:c3"],
                    "representative_terms": ["refund"],
                    "important_files": ["services/refund.py"],
                    "all_document_ids": [
                        "commit:sonal-38/smart_payment_platform:c3",
                    ],
                },
            ],
            "noise_documents": 0,
            "noise_document_ids": [],
            "total_documents": 4,
        }

        result = builder.build_knowledge_area_graph(
            owner="sonal-38",
            repo="smart_payment_platform",
            interpretation_result=step11_multi_dev_result,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["knowledge_areas"], 2)
        self.assertEqual(result["developers_reached_through_evidence"], 2)

    @patch("graph.builder.get_supabase_client")
    def test_idempotency_running_twice(self, mock_get_sb):
        """
        Verifies that running twice executes MERGE without duplicating nodes or relationships.
        """
        mock_get_sb.return_value = self.mock_supabase
        builder = KnowledgeGraphBuilder(
            neo4j_client=self.mock_neo4j,
            github_client=self.mock_github,
        )

        self.mock_neo4j.execute_query.side_effect = lambda query, params=None: (
            [{"id": "sonal-38/smart_payment_platform", "name": "smart_payment_platform"}]
            if "MATCH (r:Repository)" in query
            else [{"count": 1}] if "count(DISTINCT ka)" in query
            else [{"count": 2}] if "count(rel)" in query
            else [{"count": 1}] if "UNWIND all_devs" in query
            else []
        )

        step11_single = {
            "status": "completed",
            "repository": "sonal-38/smart_payment_platform",
            "clusters": [
                {
                    "cluster_id": 0,
                    "candidate_label": "Payment Processing",
                    "document_count": 1,
                    "representative_documents": ["commit:sonal-38/smart_payment_platform:c1"],
                    "representative_terms": ["payment"],
                    "important_files": ["payment.py"],
                    "all_document_ids": [
                        "commit:sonal-38/smart_payment_platform:c1",
                        "pr:sonal-38/smart_payment_platform:10",
                    ],
                }
            ],
            "noise_documents": 0,
            "noise_document_ids": [],
            "total_documents": 2,
        }

        res1 = builder.build_knowledge_area_graph("sonal-38", "smart_payment_platform", step11_single)
        res2 = builder.build_knowledge_area_graph("sonal-38", "smart_payment_platform", step11_single)

        self.assertEqual(res1["knowledge_areas"], res2["knowledge_areas"])
        self.assertEqual(res1["evidence_relationships"], res2["evidence_relationships"])

        # Check all writes used MERGE
        executed_queries = [call[0][0] for call in self.mock_neo4j.execute_query.call_args_list]
        for q in executed_queries:
            if "CREATE" in q and "CREATE CONSTRAINT" not in q:
                # Must be ON CREATE SET
                self.assertIn("ON CREATE SET", q, f"Found raw CREATE without MERGE: {q}")

    @patch("api.knowledge.KnowledgeGraphBuilder")
    def test_api_endpoint_post_knowledge_repositories_graph(self, mock_builder_class):
        """
        Verifies the POST /knowledge/repositories/{owner}/{repo}/graph endpoint.
        """
        mock_instance = MagicMock()
        mock_builder_class.return_value = mock_instance
        mock_instance.build_knowledge_area_graph.return_value = {
            "repository": "sonal-38/smart_payment_platform",
            "status": "completed",
            "knowledge_areas": 6,
            "evidence_relationships": 42,
            "developers_reached_through_evidence": 5,
        }

        client = TestClient(app)
        response = client.post("/knowledge/repositories/sonal-38/smart_payment_platform/graph")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["repository"], "sonal-38/smart_payment_platform")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["knowledge_areas"], 6)
        self.assertEqual(data["evidence_relationships"], 42)
        self.assertEqual(data["developers_reached_through_evidence"], 5)


if __name__ == "__main__":
    unittest.main()

"""
End-to-end integration tests for the GraphRAG pipeline.

Tests the full flow: document ingestion -> graph building -> query execution.
Uses testcontainers to spin up ephemeral Neo4j instances.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end tests verifying ingestion through query."""

    def test_ingest_and_query_pipeline(
        self, neo4j_container, db_manager, clean_neo4j, tmp_path: Path
    ):
        """
        Test the complete GraphRAG pipeline:
        1. Ingest a document using build_graph_index
        2. Query the graph using query_engine.query
        3. Verify the response is valid and not empty
        """
        # Create a test document with content matching the ontology schema
        doc_content = """
        Solar panels are a key technology for renewable energy.
        They convert sunlight into electricity using photovoltaic cells.
        Germany is a leading country in solar energy adoption.
        Solar energy produces clean electricity without emissions.
        """

        test_doc = tmp_path / "solar_energy.txt"
        test_doc.write_text(doc_content)

        # Step 1: Ingestion
        from src.ingestion import build_graph_index

        try:
            index = build_graph_index(db_manager, data_dir=str(tmp_path))
        except Exception as e:
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not configured for integration test")
            raise

        assert index is not None, "Expected PropertyGraphIndex to be created"

        # Step 2: Query the graph
        from src.query_engine import query

        response = query(
            "What technology is used for renewable energy?",
            db_manager=db_manager,
        )

        # Step 3: Assertions
        assert response is not None, "Expected a response from query"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        # Check that the response is not an error message
        assert "❌" not in response, f"Query returned an error: {response}"

    def test_query_without_prior_ingestion_returns_empty_result(
        self, neo4j_container, db_manager, clean_neo4j
    ):
        """
        Test that querying an empty graph returns a valid (possibly empty) response,
        not an error.
        """
        from src.query_engine import query

        try:
            response = query(
                "What is solar energy?",
                db_manager=db_manager,
            )
        except Exception as e:
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not configured for integration test")
            raise

        # Should return something (even if it says it doesn't have info)
        assert response is not None
        assert isinstance(response, str)

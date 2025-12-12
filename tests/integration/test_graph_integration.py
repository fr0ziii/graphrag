"""
Integration tests for GraphRAG knowledge graph building.

These tests use testcontainers to spin up ephemeral Neo4j instances.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
class TestGraphIntegration:
    """Integration tests for the graph building pipeline."""

    def test_neo4j_connection_via_testcontainer(self, neo4j_container, neo4j_driver):
        """
        Basic test to verify Neo4j testcontainer is working.
        """
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 as value")
            assert result.single()["value"] == 1

    def test_document_ingestion_creates_nodes(
        self, neo4j_container, neo4j_driver, db_manager, clean_neo4j, tmp_path: Path
    ):
        """
        Test that ingesting a document creates expected nodes in Neo4j.

        This is a smoke test to verify the basic ingestion pipeline works.
        """
        # Create a simple test document
        doc_content = """
        Solar panels are a key technology for renewable energy.
        They convert sunlight into electricity using photovoltaic cells.
        Germany is a leading country in solar energy adoption.
        """

        test_doc = tmp_path / "test_doc.txt"
        test_doc.write_text(doc_content)

        # Import here to use the patched environment variables
        from src.ingestion import build_graph_index

        # Build the graph (this uses the env vars set by neo4j_container fixture)
        try:
            build_graph_index(db_manager, data_dir=str(tmp_path))
        except Exception as e:
            # Skip if OpenAI API is not configured
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not configured for integration test")
            raise

        # Verify nodes were created
        with neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]

            assert node_count > 0, "Expected at least one node to be created"

    def test_graph_has_relationships(
        self, neo4j_container, neo4j_driver, db_manager, clean_neo4j, tmp_path: Path
    ):
        """
        Test that ingested documents create relationships, not just nodes.
        """
        doc_content = """
        Wind turbines produce electricity.
        Denmark uses wind energy extensively.
        Vestas develops wind turbine technology.
        """

        test_doc = tmp_path / "wind_energy.txt"
        test_doc.write_text(doc_content)

        from src.ingestion import build_graph_index

        try:
            build_graph_index(db_manager, data_dir=str(tmp_path))
        except Exception as e:
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not configured for integration test")
            raise

        # Verify relationships were created
        with neo4j_driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]

            assert rel_count > 0, "Expected at least one relationship to be created"


@pytest.mark.integration
class TestDocumentTracking:
    """Integration tests for document tracking (idempotency)."""

    def test_document_exists_by_hash_returns_false_for_nonexistent(
        self, neo4j_container, neo4j_driver, db_manager, clean_neo4j
    ):
        """Test that document_exists_by_hash returns False when hash doesn't exist."""
        result = db_manager.document_exists_by_hash("nonexistent_hash_12345")
        assert result is False

    def test_create_document_node_and_verify(
        self, neo4j_container, neo4j_driver, db_manager, clean_neo4j
    ):
        """Test creating a Document node and verifying it exists."""
        test_hash = "abc123def456"
        test_filename = "test_document.txt"
        test_timestamp = "2025-01-01T00:00:00Z"

        # Initially should not exist
        assert db_manager.document_exists_by_hash(test_hash) is False

        # Create the document node
        db_manager.create_document_node(test_filename, test_hash, test_timestamp)

        # Now should exist
        assert db_manager.document_exists_by_hash(test_hash) is True

        # Verify properties in Neo4j
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {hash: $hash}) RETURN d", {"hash": test_hash}
            )
            record = result.single()
            assert record is not None
            doc_node = record["d"]
            assert doc_node["filename"] == test_filename
            assert doc_node["ingested_at"] == test_timestamp

    def test_create_document_node_is_idempotent(
        self, neo4j_container, neo4j_driver, db_manager, clean_neo4j
    ):
        """Test that creating same document twice doesn't create duplicates."""
        test_hash = "unique_hash_xyz"

        # Create twice
        db_manager.create_document_node("file.txt", test_hash, "2025-01-01T00:00:00Z")
        db_manager.create_document_node("file.txt", test_hash, "2025-01-02T00:00:00Z")

        # Should only have one node
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {hash: $hash}) RETURN count(d) as count",
                {"hash": test_hash},
            )
            count = result.single()["count"]
            assert count == 1

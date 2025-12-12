"""
Database connection module for Neo4j.

This module provides a centralized GraphDatabaseManager to handle Neo4j connections
and operations, implementing the Dependency Injection pattern.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore, Neo4jPropertyGraphStore
from neo4j import Driver, GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


VALID_NEO4J_SCHEMES = (
    "bolt://",
    "bolt+s://",
    "bolt+ssc://",
    "neo4j://",
    "neo4j+s://",
    "neo4j+ssc://",
)


class Neo4jConfig:
    """Configuration for Neo4j connection."""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self._validate_uri_format()

    def _validate_uri_format(self) -> None:
        """Validate that NEO4J_URI has a valid scheme. Raises ValueError if invalid."""
        if not self.uri:
            raise ValueError("NEO4J_URI cannot be empty")
        if not any(self.uri.startswith(scheme) for scheme in VALID_NEO4J_SCHEMES):
            raise ValueError(
                f"NEO4J_URI must start with a valid scheme: {', '.join(VALID_NEO4J_SCHEMES)}. "
                f"Got: '{self.uri}'"
            )

    def validate(self) -> bool:
        """Check if all required configuration is present."""
        return all([self.uri, self.username, self.password])


class GraphDatabaseManager:
    """
    Manages Neo4j database connections and operations.

    This class encapsulates all database interaction logic, allowing for
    dependency injection and easier testing.
    """

    def __init__(self, config: Neo4jConfig | None = None):
        """
        Initialize the database manager.

        Args:
            config: Optional Neo4jConfig instance. If None, created from environment.
        """
        self.config = config or Neo4jConfig()

    def get_driver(self) -> Driver:
        """
        Create and return a Neo4j driver instance.

        Returns:
            Driver: Neo4j driver for direct Cypher queries.

        Raises:
            ValueError: If required environment variables are missing.
            ConnectionError: If unable to connect to Neo4j.
        """
        if not self.config.validate():
            raise ValueError(
                "Missing required Neo4j environment variables. "
                "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )

        try:
            driver = GraphDatabase.driver(
                self.config.uri, auth=(self.config.username, self.config.password)
            )
            # Verify connection
            driver.verify_connectivity()
            # logger.info("Successfully connected to Neo4j at %s", self.config.uri)
            return driver
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e

    def get_graph_store(self) -> Neo4jGraphStore:
        """
        Create and return a Neo4j graph store for LlamaIndex.

        Returns:
            Neo4jGraphStore: Connected graph store instance.

        Raises:
            ConnectionError: If unable to connect to Neo4j.
            ValueError: If required environment variables are missing.
        """
        if not self.config.validate():
            raise ValueError(
                "Missing required Neo4j environment variables. "
                "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )

        try:
            graph_store = Neo4jGraphStore(
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password,
            )
            logger.info(
                "Successfully connected to Neo4j graph store at %s", self.config.uri
            )
            return graph_store
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e

    def get_property_graph_store(self) -> Neo4jPropertyGraphStore:
        """
        Create and return a Neo4j property graph store for LlamaIndex PropertyGraphIndex.

        Returns:
            Neo4jPropertyGraphStore: Connected property graph store instance.

        Raises:
            ConnectionError: If unable to connect to Neo4j.
            ValueError: If required environment variables are missing.
        """
        if not self.config.validate():
            raise ValueError(
                "Missing required Neo4j environment variables. "
                "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )

        try:
            graph_store = Neo4jPropertyGraphStore(
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password,
            )
            logger.info(
                "Successfully connected to Neo4j property graph store at %s",
                self.config.uri,
            )
            return graph_store
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e

    @contextmanager
    def session(self):
        """
        Context manager for Neo4j sessions.

        Usage:
            with db_manager.session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 10")
        """
        driver = self.get_driver()
        try:
            with driver.session() as session:
                yield session
        finally:
            driver.close()

    def execute_query(self, query: str, parameters: dict | None = None) -> list[Any]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of query result records.
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

    def check_connection(self) -> tuple[bool, str]:
        """
        Check if Neo4j is accessible.

        Returns:
            Tuple of (is_connected, status_message)
        """
        if not self.config.validate():
            return False, "Missing Neo4j credentials in .env"

        try:
            driver = self.get_driver()
            driver.close()
            return True, f"Connected to {self.config.uri}"
        except Exception as e:
            return False, f"Connection failed: {str(e)[:50]}..."

    # =============================================================================
    # DOCUMENT TRACKING FOR IDEMPOTENCY
    # =============================================================================

    def document_exists_by_hash(self, doc_hash: str) -> bool:
        """
        Check if a Document node with the given hash already exists.

        Args:
            doc_hash: SHA-256 hash of the document text content.

        Returns:
            True if a Document with this hash exists, False otherwise.
        """
        query = "MATCH (d:Document {hash: $hash}) RETURN d LIMIT 1"
        results = self.execute_query(query, {"hash": doc_hash})
        return len(results) > 0

    def create_document_node(
        self, filename: str, doc_hash: str, ingested_at: str
    ) -> None:
        """
        Create a Document node to track an ingested file.

        Args:
            filename: Original filename of the document.
            doc_hash: SHA-256 hash of the document text content.
            ingested_at: ISO format timestamp of ingestion.
        """
        query = """
        MERGE (d:Document {hash: $hash})
        SET d.filename = $filename, d.ingested_at = $ingested_at
        """
        self.execute_query(
            query, {"filename": filename, "hash": doc_hash, "ingested_at": ingested_at}
        )
        logger.info("Created/updated Document node for %s", filename)

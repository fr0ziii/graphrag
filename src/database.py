"""
Database connection module for Neo4j.

This module provides a centralized Neo4j connection that can be imported
and reused across all modules.
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


class Neo4jConfig:
    """Configuration for Neo4j connection."""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")

    def validate(self) -> bool:
        """Check if all required configuration is present."""
        return all([self.uri, self.username, self.password])


# Singleton config instance
_config = Neo4jConfig()


def get_config() -> Neo4jConfig:
    """Get the Neo4j configuration."""
    return _config


def get_neo4j_driver() -> Driver:
    """
    Create and return a Neo4j driver instance.

    Returns:
        Driver: Neo4j driver for direct Cypher queries.

    Raises:
        ValueError: If required environment variables are missing.
        ConnectionError: If unable to connect to Neo4j.
    """
    config = get_config()

    if not config.validate():
        raise ValueError(
            "Missing required Neo4j environment variables. "
            "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )

    try:
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password)
        )
        # Verify connection
        driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j at %s", config.uri)
        return driver
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", str(e))
        raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e


def get_neo4j_graph_store() -> Neo4jGraphStore:
    """
    Create and return a Neo4j graph store for LlamaIndex.

    Returns:
        Neo4jGraphStore: Connected graph store instance.

    Raises:
        ConnectionError: If unable to connect to Neo4j.
        ValueError: If required environment variables are missing.
    """
    config = get_config()

    if not config.validate():
        raise ValueError(
            "Missing required Neo4j environment variables. "
            "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )

    try:
        graph_store = Neo4jGraphStore(
            url=config.uri,
            username=config.username,
            password=config.password,
        )
        logger.info("Successfully connected to Neo4j graph store at %s", config.uri)
        return graph_store
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", str(e))
        raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e


def get_neo4j_property_graph_store() -> Neo4jPropertyGraphStore:
    """
    Create and return a Neo4j property graph store for LlamaIndex PropertyGraphIndex.

    This is the newer API that supports schema-driven extraction with
    SchemaLLMPathExtractor. Use this for structured knowledge graph extraction
    with predefined entity and relationship types.

    Returns:
        Neo4jPropertyGraphStore: Connected property graph store instance.

    Raises:
        ConnectionError: If unable to connect to Neo4j.
        ValueError: If required environment variables are missing.
    """
    config = get_config()

    if not config.validate():
        raise ValueError(
            "Missing required Neo4j environment variables. "
            "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )

    try:
        graph_store = Neo4jPropertyGraphStore(
            url=config.uri,
            username=config.username,
            password=config.password,
        )
        logger.info(
            "Successfully connected to Neo4j property graph store at %s", config.uri
        )
        return graph_store
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", str(e))
        raise ConnectionError(f"Unable to connect to Neo4j: {str(e)}") from e


@contextmanager
def neo4j_session():
    """
    Context manager for Neo4j sessions.

    Usage:
        with neo4j_session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 10")
    """
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            yield session
    finally:
        driver.close()


def execute_query(query: str, parameters: dict | None = None) -> list[Any]:
    """
    Execute a Cypher query and return results.

    Args:
        query: Cypher query string.
        parameters: Optional query parameters.

    Returns:
        List of query result records.
    """
    with neo4j_session() as session:
        result = session.run(query, parameters or {})
        return list(result)


def check_connection() -> tuple[bool, str]:
    """
    Check if Neo4j is accessible.

    Returns:
        Tuple of (is_connected, status_message)
    """
    config = get_config()

    if not config.validate():
        return False, "Missing Neo4j credentials in .env"

    try:
        driver = get_neo4j_driver()
        driver.close()
        return True, f"Connected to {config.uri}"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:50]}..."

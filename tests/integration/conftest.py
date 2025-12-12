"""
Integration test fixtures using testcontainers.
"""

import os

import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer


@pytest.fixture(scope="module")
def neo4j_container():
    """
    Spin up an ephemeral Neo4j container for integration tests.

    Uses module scope to reuse the container across tests in the same file,
    improving test execution speed.
    
    Note: Uses neo4j image with APOC plugin enabled, which is required
    by LlamaIndex's Neo4jPropertyGraphStore.
    """
    # Use Neo4j with APOC plugin enabled
    neo4j = Neo4jContainer("neo4j:5.15.0")
    
    # Enable APOC plugin (required by LlamaIndex Neo4j integration)
    neo4j.with_env("NEO4J_PLUGINS", '["apoc"]')
    neo4j.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*")
    neo4j.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*")
    
    neo4j.start()
    
    # Set environment variables for the test session
    os.environ["NEO4J_URI"] = neo4j.get_connection_url()
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = neo4j.password
    
    yield neo4j
    
    neo4j.stop()


@pytest.fixture
def neo4j_driver(neo4j_container):
    """Provide a Neo4j driver connected to the test container."""
    driver = GraphDatabase.driver(
        neo4j_container.get_connection_url(),
        auth=("neo4j", neo4j_container.password),
    )
    yield driver
    driver.close()


@pytest.fixture
def clean_neo4j(neo4j_driver):
    """Clean all nodes and relationships before each test."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
    # Cleanup after test as well
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


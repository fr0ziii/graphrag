# 1. Use Neo4j as the Graph Database

Date: 2025-12-12

## Status

Accepted

## Context

The project requires a mechanism to store and query the knowledge graph generated from ingested documents. The system must support:
-   Production-ready persistence (surviving restarts).
-   Complex querying capabilities for multi-hop reasoning (RAG).
-   Scalability beyond simple demo datasets.

We evaluated two primary options:
1.  **NetworkX:** A Python library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. It operates primarily in-memory.
2.  **Neo4j:** A native graph database platform designed to store, manage, and query data in the form of graph structures.

## Decision

We will use **Neo4j** as the graph database backend.

## Consequences

### Positive
-   **Persistence:** Data is stored on disk and persists across application restarts, unlike NetworkX which typically requires serialization/deserialization steps.
-   **Scalability:** Neo4j allows for handling much larger graphs than can fit in memory.
-   **Query Language:** Access to **Cypher**, a declarative graph query language, enabling powerful pattern matching and traversal operations essential for GraphRAG.
-   **Ecosystem:** Integration with LlamaIndex `Neo4jGraphStore` and visualization tools like Neo4j Bloom (or comparable web UIs).

### Negative
-   **Infrastructure Complexity:** Requires running a separate service (Docker container), which increases the development environment complexity compared to a pure Python library.
-   **Resources:** Higher memory and CPU footprint than a simple in-memory structure for very small graphs.

### Compliance
-   The project uses `docker-compose` to orchestrate the Neo4j instance.
-   The application connects via the `neo4j` Python driver.

"""
Graph Analysis module using Neo4j Graph Data Science (GDS).

This module provides a post-ingestion enrichment pipeline that computes:
1. PageRank - Node importance/centrality score
2. Louvain Community Detection - Cluster identification

Results are written back to Neo4j nodes as persistent properties:
- pageRankScore: Float indicating relative node importance
- communityId: Integer identifying the node's community cluster

Usage:
    Standalone: python -m src.analysis
    Integrated: Import and call run_analysis() from ingestion.py
"""

import logging
import os

from dotenv import load_dotenv
from graphdatascience import GraphDataScience

from src.database import GraphDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Graph projection name (used in GDS catalog)
GRAPH_NAME = "graphrag_analysis"


def get_gds_client(db_manager: GraphDatabaseManager) -> GraphDataScience:
    """
    Create and return a Graph Data Science client instance.

    Uses the same configuration as the main database module for consistency.

    Args:
        db_manager: Configured GraphDatabaseManager instance.

    Returns:
        GraphDataScience: Connected GDS client instance.

    Raises:
        ConnectionError: If unable to connect to Neo4j GDS.
    """
    config = db_manager.config

    if not config.validate():
        raise ValueError(
            "Missing required Neo4j environment variables. "
            "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )

    try:
        # Get database name from env or use default
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        gds = GraphDataScience(
            config.uri,
            auth=(config.username, config.password),
            database=database,
        )

        # Verify connection by checking GDS version
        version = gds.version()
        logger.info("Connected to Neo4j GDS version: %s", version)

        return gds

    except Exception as e:
        logger.error("Failed to connect to Neo4j GDS: %s", str(e))
        raise ConnectionError(f"Unable to connect to Neo4j GDS: {str(e)}") from e


def drop_graph_if_exists(gds: GraphDataScience, graph_name: str) -> None:
    """
    Drop a projected graph from the GDS catalog if it exists.

    Args:
        gds: GraphDataScience client instance.
        graph_name: Name of the graph to drop.
    """
    if gds.graph.exists(graph_name).exists:
        logger.info("Dropping existing graph projection: %s", graph_name)
        gds.graph.drop(graph_name)


def project_graph(gds: GraphDataScience, graph_name: str):
    """
    Project all nodes and relationships into GDS in-memory graph.

    This creates a native projection that includes all node labels and
    relationship types present in the database.

    Args:
        gds: GraphDataScience client instance.
        graph_name: Name for the projected graph.

    Returns:
        Graph: The projected GDS graph object.
    """
    # Drop existing projection if present
    drop_graph_if_exists(gds, graph_name)

    logger.info("Projecting graph into GDS catalog as '%s'...", graph_name)

    # Project all nodes and relationships using native projection
    # Using "*" captures all node labels and relationship types
    G, result = gds.graph.project(
        graph_name,
        "*",  # All node labels
        "*",  # All relationship types
    )

    logger.info(
        "Graph projected: %d nodes, %d relationships",
        result["nodeCount"],
        result["relationshipCount"],
    )

    return G


def run_pagerank(gds: GraphDataScience, G, write_property: str = "pageRankScore"):
    """
    Execute PageRank algorithm and write results to Neo4j.

    PageRank measures the relative importance of nodes based on the structure
    of incoming links. Nodes with more/higher-quality connections get higher scores.

    Args:
        gds: GraphDataScience client instance.
        G: Projected GDS graph object.
        write_property: Name of the property to write results to.

    Returns:
        dict: Algorithm execution result statistics.
    """
    logger.info("Running PageRank algorithm (writeProperty='%s')...", write_property)

    result = gds.pageRank.write(
        G,
        writeProperty=write_property,
        maxIterations=20,
        dampingFactor=0.85,
    )

    logger.info(
        "PageRank complete: %d nodes processed in %d ms",
        result["nodePropertiesWritten"],
        result["computeMillis"],
    )

    # Log centrality distribution for insight
    stats = result.get("centralityDistribution", {})
    if stats:
        logger.info(
            "  Score distribution: min=%.4f, max=%.4f, mean=%.4f",
            stats.get("min", 0),
            stats.get("max", 0),
            stats.get("mean", 0),
        )

    return result


def run_louvain(gds: GraphDataScience, G, write_property: str = "communityId"):
    """
    Execute Louvain community detection and write results to Neo4j.

    Louvain is a modularity-based community detection algorithm that identifies
    clusters of densely connected nodes. Each node is assigned a community ID.

    Note: Using Louvain instead of Leiden because Leiden may require additional
    licensing in some Neo4j configurations (Community Edition).

    Args:
        gds: GraphDataScience client instance.
        G: Projected GDS graph object.
        write_property: Name of the property to write results to.

    Returns:
        dict: Algorithm execution result statistics.
    """
    logger.info("Running Louvain community detection (writeProperty='%s')...", write_property)

    result = gds.louvain.write(
        G,
        writeProperty=write_property,
        maxIterations=10,
        maxLevels=10,
    )

    logger.info(
        "Louvain complete: %d communities detected in %d ms",
        result["communityCount"],
        result["computeMillis"],
    )
    logger.info("  Modularity: %.4f", result.get("modularity", 0))

    return result


def run_analysis(
    db_manager: GraphDatabaseManager | None = None,
    pagerank_property: str = "pageRankScore",
    community_property: str = "communityId",
) -> dict:
    """
    Execute the complete graph analysis pipeline.

    This is the main entry point for graph enrichment. It:
    1. Connects to Neo4j GDS
    2. Projects the graph into memory
    3. Runs PageRank for node importance
    4. Runs Louvain for community detection
    5. Writes results back to the database
    6. Cleans up the in-memory projection

    Args:
        db_manager: Configured GraphDatabaseManager instance. If None, creates a new one.
        pagerank_property: Property name for PageRank scores.
        community_property: Property name for community IDs.

    Returns:
        dict: Summary of analysis results.
    """
    logger.info("=" * 60)
    logger.info("Starting Graph Enrichment Analysis")
    logger.info("=" * 60)

    if db_manager is None:
        db_manager = GraphDatabaseManager()

    # Initialize GDS client
    gds = get_gds_client(db_manager)

    try:
        # Step 1: Project graph
        G = project_graph(gds, GRAPH_NAME)

        # Check if graph has nodes
        node_count = G.node_count()
        if node_count == 0:
            logger.warning("Graph is empty! Run ingestion first.")
            return {"status": "empty", "nodes": 0}

        # Step 2: Run PageRank
        pagerank_result = run_pagerank(gds, G, pagerank_property)

        # Step 3: Run Louvain
        louvain_result = run_louvain(gds, G, community_property)

        # Summary
        summary = {
            "status": "success",
            "nodes_processed": node_count,
            "pagerank": {
                "property": pagerank_property,
                "nodes_written": pagerank_result["nodePropertiesWritten"],
                "compute_ms": pagerank_result["computeMillis"],
            },
            "louvain": {
                "property": community_property,
                "communities_found": louvain_result["communityCount"],
                "modularity": louvain_result.get("modularity", 0),
                "compute_ms": louvain_result["computeMillis"],
            },
        }

        logger.info("-" * 60)
        logger.info("Analysis complete! Properties written to Neo4j:")
        logger.info("  - %s (PageRank importance scores)", pagerank_property)
        logger.info("  - %s (Community cluster IDs)", community_property)

        return summary

    finally:
        # Clean up: drop the in-memory graph projection
        drop_graph_if_exists(gds, GRAPH_NAME)
        logger.info("Cleaned up in-memory graph projection")


if __name__ == "__main__":
    # Run analysis when executed directly
    print("=" * 60)
    print("GraphRAG Graph Enrichment Pipeline")
    print("=" * 60)
    print("\nAlgorithms:")
    print("  1. PageRank - Node importance scoring")
    print("  2. Louvain - Community detection")
    print("-" * 60)

    try:
        result = run_analysis()

        if result["status"] == "success":
            print("\n✅ Graph enrichment complete!")
            print(f"   Nodes processed: {result['nodes_processed']}")
            print(f"   Communities found: {result['louvain']['communities_found']}")
            print("\nVerification queries for Neo4j Browser:")
            print("  MATCH (n) WHERE n.pageRankScore IS NOT NULL")
            print("  RETURN n.id, n.pageRankScore, n.communityId")
            print("  ORDER BY n.pageRankScore DESC LIMIT 10")
        elif result["status"] == "empty":
            print("\n⚠️  Graph is empty. Run ingestion first:")
            print("   uv run python -m src.ingestion")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise

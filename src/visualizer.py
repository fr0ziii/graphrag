"""
Graph Visualizer module for rendering Neo4j data with Pyvis.

This module provides functionality to extract graph data from Neo4j
and generate interactive HTML visualizations using Pyvis.

Note: All HTML generation is performed in-memory without disk I/O.
"""

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pyvis.network import Network

from src.database import execute_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def fetch_graph_data(limit: int = 100) -> dict[str, Any]:
    """
    Fetch nodes and relationships from Neo4j, including GDS-enriched properties.

    Args:
        limit: Maximum number of relationships to fetch.

    Returns:
        Dictionary containing nodes and edges data with pageRankScore and communityId.
    """
    # Cypher query to get nodes, relationships, and analytics properties
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m,
           n.pageRankScore AS source_pr,
           n.communityId AS source_community,
           m.pageRankScore AS target_pr,
           m.communityId AS target_community
    LIMIT $limit
    """

    nodes = {}
    edges = []

    try:
        results = execute_query(query, {"limit": limit})

        for record in results:
            # Extract source node with analytics properties
            source = record["n"]
            source_id = str(source.element_id)
            source_pr = record.get("source_pr")
            source_community = record.get("source_community")

            if source_id not in nodes:
                # Get node label and name/id property
                labels = list(source.labels)
                label = labels[0] if labels else "Node"
                name = source.get("id", source.get("name", source_id[:8]))

                # Build tooltip with analytics info
                tooltip = f"{label}: {name}"
                if source_pr is not None:
                    tooltip += f"\nPageRank: {source_pr:.4f}"
                if source_community is not None:
                    tooltip += f"\nCommunity: {source_community}"

                nodes[source_id] = {
                    "id": source_id,
                    "label": str(name),
                    "title": tooltip,
                    "group": label,
                    "pageRankScore": source_pr,
                    "communityId": source_community,
                }

            # Extract target node with analytics properties
            target = record[\"m\"]
            target_id = str(target.element_id)
            target_pr = record.get(\"target_pr\")
            target_community = record.get(\"target_community\")

            if target_id not in nodes:
                labels = list(target.labels)
                label = labels[0] if labels else \"Node\"
                name = target.get(\"id\", target.get(\"name\", target_id[:8]))

                # Build tooltip with analytics info
                tooltip = f\"{label}: {name}\"
                if target_pr is not None:
                    tooltip += f\"\\nPageRank: {target_pr:.4f}\"
                if target_community is not None:
                    tooltip += f\"\\nCommunity: {target_community}\"

                nodes[target_id] = {
                    \"id\": target_id,
                    \"label\": str(name),
                    \"title\": tooltip,
                    \"group\": label,
                    \"pageRankScore\": target_pr,
                    \"communityId\": target_community,
                }

            # Extract relationship
            rel = record["r"]
            edges.append({
                "from": source_id,
                "to": target_id,
                "label": rel.type,
                "title": rel.type,
            })

        logger.info("Fetched %d nodes and %d edges from Neo4j",
                   len(nodes), len(edges))

    except Exception as e:
        logger.error("Failed to fetch graph data: %s", str(e))
        raise

    return {"nodes": list(nodes.values()), "edges": edges}


def generate_graph_html(
    height: str = "600px",
    width: str = "100%",
    bgcolor: str = "#0e1117",
    font_color: str = "white",
    limit: int = 100,
    color_by_community: bool = True,
) -> str:
    """
    Generate an interactive HTML visualization of the knowledge graph.

    Args:
        height: Height of the visualization canvas.
        width: Width of the visualization canvas.
        bgcolor: Background color of the graph.
        font_color: Color of node labels.
        limit: Maximum number of relationships to display.
        color_by_community: If True, color nodes by communityId; else by entity type.

    Returns:
        str: HTML string containing the interactive graph visualization.
    """
    # Fetch graph data
    graph_data = fetch_graph_data(limit=limit)

    if not graph_data["nodes"]:
        return """
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            color: #888;
            font-family: sans-serif;
        ">
            <div style="text-align: center;">
                <h3>📊 No Graph Data Found</h3>
                <p>Run the ingestion script first to populate the knowledge graph.</p>
                <code>python src/ingestion.py</code>
            </div>
        </div>
        """

    # Create Pyvis network
    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Configure physics for better visualization
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            },
            "shape": "dot",
            "size": 20,
            "borderWidth": 2
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "font": {
                "size": 10,
                "align": "middle"
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    # Color palette for different node groups/communities
    colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4",
        "#ffeaa7", "#dfe6e9", "#fd79a8", "#00b894",
        "#6c5ce7", "#fdcb6e", "#e17055", "#74b9ff",
        "#a29bfe", "#fab1a0", "#81ecec", "#ffeaa7",
    ]

    # Track groups/communities for color assignment
    groups = {}
    communities = {}
    color_index = 0

    # Calculate PageRank range for node sizing
    pageranks = [n.get("pageRankScore", 0.0) for n in graph_data["nodes"]]
    max_pagerank = max(pageranks) if pageranks else 1.0
    min_pagerank = min(pageranks) if pageranks else 0.0
    pagerank_range = max_pagerank - min_pagerank if max_pagerank > min_pagerank else 1.0

    # Check if any nodes have community data
    has_community_data = any(
        n.get("communityId") is not None for n in graph_data["nodes"]
    )

    # Add nodes with size based on PageRank
    for node in graph_data["nodes"]:
        # Calculate node size based on PageRank (scale from 15 to 50)
        pagerank = node.get("pageRankScore", 0.0)
        normalized_pagerank = (pagerank - min_pagerank) / pagerank_range if pagerank_range > 0 else 0
        node_size = 15 + (normalized_pagerank * 35)  # Range: 15-50 pixels

        # Determine color based on community or entity type
        if color_by_community and has_community_data:
            community_id = node.get("communityId")
            if community_id is not None:
                if community_id not in communities:
                    communities[community_id] = colors[len(communities) % len(colors)]
                node_color = communities[community_id]
            else:
                node_color = "#888888"  # Gray for nodes without community
        else:
            # Fall back to entity type coloring
            group = node.get("group", "default")
            if group not in groups:
                groups[group] = colors[color_index % len(colors)]
                color_index += 1
            node_color = groups[group]

        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("title", node["label"]),
            color=node_color,
            size=node_size,
        )

    # Add edges
    for edge in graph_data["edges"]:
        net.add_edge(
            edge["from"],
            edge["to"],
            title=edge.get("title", ""),
            label=edge.get("label", ""),
        )

    # Generate HTML string directly in memory (no disk I/O)
    html_content = net.generate_html()

    logger.info(
        "Generated graph visualization with %d nodes and %d edges",
        len(graph_data["nodes"]),
        len(graph_data["edges"]),
    )

    return html_content


if __name__ == "__main__":
    # Test visualization
    print("=" * 60)
    print("GraphRAG Visualizer Test")
    print("=" * 60)

    try:
        html = generate_graph_html()

        # Save to file for testing
        output_path = Path("test_graph.html")
        output_path.write_text(html, encoding="utf-8")

        print(f"\n✅ Graph visualization saved to {output_path}")
        print("   Open this file in a browser to view the graph.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

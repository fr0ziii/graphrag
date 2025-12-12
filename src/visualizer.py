"""
Graph Visualizer module for rendering Neo4j data with Pyvis.

This module provides functionality to extract graph data from Neo4j
and generate interactive HTML visualizations using Pyvis.
"""

import os
import logging
from typing import Dict, Any
import tempfile

from dotenv import load_dotenv
from pyvis.network import Network

from src.database import execute_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def fetch_graph_data(limit: int = 100) -> Dict[str, Any]:
    """
    Fetch nodes and relationships from Neo4j.
    
    Args:
        limit: Maximum number of relationships to fetch.
        
    Returns:
        Dictionary containing nodes and edges data.
    """
    # Cypher query to get nodes and relationships
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT $limit
    """
    
    nodes = {}
    edges = []
    
    try:
        results = execute_query(query, {"limit": limit})
        
        for record in results:
            # Extract source node
            source = record["n"]
            source_id = str(source.element_id)
            if source_id not in nodes:
                # Get node label and name/id property
                labels = list(source.labels)
                label = labels[0] if labels else "Node"
                name = source.get("id", source.get("name", source_id[:8]))
                nodes[source_id] = {
                    "id": source_id,
                    "label": str(name),
                    "title": f"{label}: {name}",
                    "group": label,
                }
            
            # Extract target node
            target = record["m"]
            target_id = str(target.element_id)
            if target_id not in nodes:
                labels = list(target.labels)
                label = labels[0] if labels else "Node"
                name = target.get("id", target.get("name", target_id[:8]))
                nodes[target_id] = {
                    "id": target_id,
                    "label": str(name),
                    "title": f"{label}: {name}",
                    "group": label,
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
) -> str:
    """
    Generate an interactive HTML visualization of the knowledge graph.
    
    Args:
        height: Height of the visualization canvas.
        width: Width of the visualization canvas.
        bgcolor: Background color of the graph.
        font_color: Color of node labels.
        limit: Maximum number of relationships to display.
        
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
    
    # Color palette for different node groups
    colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4",
        "#ffeaa7", "#dfe6e9", "#fd79a8", "#00b894",
        "#6c5ce7", "#fdcb6e", "#e17055", "#74b9ff",
    ]
    
    # Track groups for color assignment
    groups = {}
    color_index = 0
    
    # Add nodes
    for node in graph_data["nodes"]:
        group = node.get("group", "default")
        if group not in groups:
            groups[group] = colors[color_index % len(colors)]
            color_index += 1
        
        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("title", node["label"]),
            color=groups[group],
        )
    
    # Add edges
    for edge in graph_data["edges"]:
        net.add_edge(
            edge["from"],
            edge["to"],
            title=edge.get("title", ""),
            label=edge.get("label", ""),
        )
    
    # Generate HTML
    # Use a temporary file to get the HTML content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        temp_path = f.name
    
    try:
        net.save_graph(temp_path)
        with open(temp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    logger.info("Generated graph visualization with %d nodes and %d edges",
               len(graph_data["nodes"]), len(graph_data["edges"]))
    
    return html_content


if __name__ == "__main__":
    # Test visualization
    print("=" * 60)
    print("GraphRAG Visualizer Test")
    print("=" * 60)
    
    try:
        html = generate_graph_html()
        
        # Save to file for testing
        output_path = "test_graph.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"\n✅ Graph visualization saved to {output_path}")
        print("   Open this file in a browser to view the graph.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

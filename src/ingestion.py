"""
Ingestion module for building the Knowledge Graph Index from documents.

This module reads documents from the data folder and creates a Knowledge Graph
Index stored in Neo4j for GraphRAG functionality.
"""

import os
import logging

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from src.database import get_neo4j_graph_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def build_graph_index(
    data_dir: str = "data",
    max_triplets_per_chunk: int = 2,
    include_embeddings: bool = True,
) -> KnowledgeGraphIndex:
    """
    Build a Knowledge Graph Index from documents in the specified directory.
    
    This function reads all documents from the data directory, extracts knowledge
    triplets (subject-predicate-object), and stores them in Neo4j for later retrieval.
    
    Args:
        data_dir: Path to the directory containing source documents.
        max_triplets_per_chunk: Maximum number of triplets to extract per text chunk.
                               Lower values create cleaner, more focused graphs.
        include_embeddings: Whether to include vector embeddings for hybrid search.
        
    Returns:
        KnowledgeGraphIndex: The constructed knowledge graph index.
        
    Raises:
        FileNotFoundError: If the data directory doesn't exist or is empty.
        ConnectionError: If unable to connect to Neo4j.
    """
    # Validate data directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load documents
    logger.info("Loading documents from %s...", data_dir)
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    if not documents:
        raise FileNotFoundError(f"No documents found in {data_dir}")
    
    logger.info("Loaded %d document(s)", len(documents))
    
    # Initialize LLM and embedding model
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
    )
    
    # Connect to Neo4j using shared database module
    graph_store = get_neo4j_graph_store()
    
    # Create storage context with Neo4j backend
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # Build the Knowledge Graph Index
    logger.info("Building Knowledge Graph Index (this may take a while)...")
    logger.info("Settings: max_triplets_per_chunk=%d, include_embeddings=%s",
                max_triplets_per_chunk, include_embeddings)
    
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=include_embeddings,
        llm=llm,
        embed_model=embed_model,
        show_progress=True,
    )
    
    logger.info("Knowledge Graph Index built successfully!")
    return index


if __name__ == "__main__":
    # Run ingestion when executed directly
    print("=" * 60)
    print("GraphRAG Ingestion Pipeline")
    print("=" * 60)
    
    try:
        index = build_graph_index()
        print("\n✅ Ingestion complete! Knowledge graph is ready in Neo4j.")
        print("   You can explore it at http://localhost:7474")
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise

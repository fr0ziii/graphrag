"""
Query Engine module for loading and querying the Knowledge Graph.

This module provides functionality to connect to an existing Knowledge Graph
in Neo4j and create a query engine for answering questions.
"""

import logging

from dotenv import load_dotenv
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from src.database import get_neo4j_graph_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_query_engine(
    response_mode: str = "tree_summarize",
    verbose: bool = False,
) -> BaseQueryEngine:
    """
    Load the Knowledge Graph Index from Neo4j and create a query engine.
    
    This function connects to an existing Neo4j instance containing the
    knowledge graph and reconstructs the index without re-ingesting documents.
    
    Args:
        response_mode: The response synthesis mode. Options include:
            - "tree_summarize": Recursively summarizes chunks (best for context)
            - "compact": Compact the chunks and synthesize
            - "simple_summarize": Simple summarization
        verbose: Whether to print verbose query information.
        
    Returns:
        BaseQueryEngine: A query engine ready to answer questions.
        
    Raises:
        ConnectionError: If unable to connect to Neo4j.
        RuntimeError: If the knowledge graph is empty or not initialized.
    """
    # Initialize LLM and embedding model
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
    )
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
    )
    
    # Connect to Neo4j using shared database module
    graph_store = get_neo4j_graph_store()
    
    # Create storage context from existing graph store
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # Reconstruct the index from existing data (no re-ingestion)
    # Pass empty nodes list to load from existing graph store
    logger.info("Loading Knowledge Graph Index from Neo4j...")
    
    index = KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
    )
    
    # Create and configure the query engine
    query_engine = index.as_query_engine(
        response_mode=response_mode,
        verbose=verbose,
        include_text=True,
    )
    
    logger.info("Query engine ready with response_mode='%s'", response_mode)
    return query_engine


def query(question: str, engine: BaseQueryEngine | None = None) -> str:
    """
    Query the knowledge graph with a natural language question.
    
    Args:
        question: The question to ask.
        engine: Optional pre-initialized query engine. If None, creates a new one.
        
    Returns:
        str: The generated response from the knowledge graph.
    """
    if engine is None:
        engine = get_query_engine()
    
    logger.info("Processing query: %s", question)
    response = engine.query(question)
    
    return str(response)


if __name__ == "__main__":
    # Test the query engine
    print("=" * 60)
    print("GraphRAG Query Engine Test")
    print("=" * 60)
    
    try:
        engine = get_query_engine(verbose=True)
        
        test_question = "What are the main types of renewable energy?"
        print(f"\nQuestion: {test_question}")
        print("-" * 40)
        
        answer = query(test_question, engine)
        print(f"\nAnswer: {answer}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

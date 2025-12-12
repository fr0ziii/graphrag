"""
Query Engine module for loading and querying the Knowledge Graph.

This module provides functionality to connect to an existing Knowledge Graph
in Neo4j and create a query engine for answering questions.

Features:
- Async query support via LlamaIndex's aquery() method
- Robust error handling for Neo4j and OpenAI failures
- User-friendly error messages for common failure modes
"""

import logging

from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex, StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from neo4j.exceptions import ServiceUnavailable
from openai import RateLimitError

from src.config import settings
from src.database import GraphDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_query_engine(
    db_manager: GraphDatabaseManager | None = None,
    response_mode: str = "tree_summarize",
    verbose: bool = False,
) -> BaseQueryEngine:
    """
    Load the Knowledge Graph Index from Neo4j and create a query engine.

    This function connects to an existing Neo4j instance containing the
    knowledge graph and reconstructs the index without re-ingesting documents.

    Args:
        db_manager: Configured GraphDatabaseManager instance. If None, a new one is created.
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
    if db_manager is None:
        db_manager = GraphDatabaseManager()

    # Initialize LLM and embedding model
    llm = OpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_base=settings.llm.api_base,
    )
    embed_model = OpenAIEmbedding(
        model=settings.embedding.model,
        dimensions=settings.embedding.dimensions,
    )

    # Connect to Neo4j using shared database module
    graph_store = db_manager.get_graph_store()

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


def query(
    question: str,
    engine: BaseQueryEngine | None = None,
    db_manager: GraphDatabaseManager | None = None,
) -> str:
    """
    Query the knowledge graph with a natural language question.

    This function includes robust error handling for common failure modes:
    - Neo4j connection issues (ServiceUnavailable)
    - OpenAI rate limiting (RateLimitError)
    - General exceptions

    Args:
        question: The question to ask.
        engine: Optional pre-initialized query engine. If None, creates a new one.
        db_manager: Optional GraphDatabaseManager to create engine if engine is None.

    Returns:
        str: The generated response, or a user-friendly error message on failure.
    """
    try:
        if engine is None:
            engine = get_query_engine(db_manager=db_manager)

        logger.info("Processing query: %s", question)
        response = engine.query(question)
        return str(response)

    except ServiceUnavailable as e:
        logger.error("Neo4j connection lost: %s", str(e))
        return (
            "⚠️ **Database Connection Lost**\n\n"
            "Unable to reach the Neo4j database. Please ensure:\n"
            "1. Neo4j is running: `docker compose up -d`\n"
            "2. The database has finished starting (wait ~30 seconds)\n"
            "3. Check connection settings in `.env`"
        )

    except RateLimitError as e:
        logger.error("OpenAI rate limit exceeded: %s", str(e))
        return (
            "⚠️ **Rate Limit Exceeded**\n\n"
            "OpenAI API rate limit reached. Please:\n"
            "1. Wait a moment before trying again\n"
            "2. Check your API quota at https://platform.openai.com/usage\n"
            "3. Consider upgrading your OpenAI plan if this persists"
        )

    except Exception as e:
        logger.error("Query failed with unexpected error: %s", str(e))
        return f"❌ **Query Failed**\n\nAn unexpected error occurred: {str(e)}"


async def async_query(
    question: str,
    engine: BaseQueryEngine | None = None,
    db_manager: GraphDatabaseManager | None = None,
) -> str:
    """
    Asynchronously query the knowledge graph with a natural language question.

    Uses LlamaIndex's native async support via `aquery()` for non-blocking
    operations. This is ideal for use in async web frameworks or when
    concurrent queries are needed.

    Args:
        question: The question to ask.
        engine: Optional pre-initialized query engine. If None, creates a new one.
        db_manager: Optional GraphDatabaseManager to create engine if engine is None.

    Returns:
        str: The generated response, or a user-friendly error message on failure.

    Note:
        The query engine creation (`get_query_engine()`) is still synchronous
        as LlamaIndex's KnowledgeGraphIndex doesn't support async initialization.
        For best performance, create the engine once and pass it to this function.
    """
    try:
        if engine is None:
            engine = get_query_engine(db_manager=db_manager)

        logger.info("Processing async query: %s", question)
        response = await engine.aquery(question)
        return str(response)

    except ServiceUnavailable as e:
        logger.error("Neo4j connection lost: %s", str(e))
        return (
            "⚠️ **Database Connection Lost**\n\n"
            "Unable to reach the Neo4j database. Please ensure:\n"
            "1. Neo4j is running: `docker compose up -d`\n"
            "2. The database has finished starting (wait ~30 seconds)\n"
            "3. Check connection settings in `.env`"
        )

    except RateLimitError as e:
        logger.error("OpenAI rate limit exceeded: %s", str(e))
        return (
            "⚠️ **Rate Limit Exceeded**\n\n"
            "OpenAI API rate limit reached. Please:\n"
            "1. Wait a moment before trying again\n"
            "2. Check your API quota at https://platform.openai.com/usage\n"
            "3. Consider upgrading your OpenAI plan if this persists"
        )

    except Exception as e:
        logger.error("Async query failed with unexpected error: %s", str(e))
        return f"❌ **Query Failed**\n\nAn unexpected error occurred: {str(e)}"


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

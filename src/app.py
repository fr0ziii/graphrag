"""
Streamlit Application for GraphRAG Demo.

This application provides a chat interface to query the knowledge graph
and a visualization explorer to interact with the graph structure.

Features:
- Non-blocking UI with timeout-protected connection checks
- In-memory graph visualization (no temp files)
- Graceful error handling for external service failures
"""

import concurrent.futures
import os
import sys
from pathlib import Path

# Add project root to Python path for Streamlit compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.database import GraphDatabaseManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GraphRAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #1e3a5f;
    }
    .assistant-message {
        background-color: #1a1a2e;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_manager() -> GraphDatabaseManager:
    """
    Get or create the GraphDatabaseManager instance.
    Cached to ensure a single instance across the session.
    """
    return GraphDatabaseManager()


def check_openai_key() -> tuple[bool, str]:
    """
    Check if OpenAI API key is configured.

    Returns:
        Tuple of (is_configured, status_message)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-") and len(api_key) > 20:
        return True, "API key configured"
    return False, "Missing or invalid API key"


def check_connection_with_timeout(db_manager: GraphDatabaseManager, timeout: float = 5.0) -> tuple[bool, str]:
    """
    Check Neo4j connection with a timeout to prevent blocking.

    This function wraps the database connection check in a ThreadPoolExecutor
    to ensure the main thread is not blocked indefinitely if Neo4j is
    unresponsive or slow to start.

    Args:
        db_manager: The database manager instance.
        timeout: Maximum seconds to wait for connection check (default: 5.0)

    Returns:
        Tuple of (is_connected, status_message)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(db_manager.check_connection)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False, f"Connection timed out after {timeout}s"
        except Exception as e:
            return False, f"Connection check failed: {str(e)[:50]}"


@st.cache_resource
def get_cached_query_engine(_db_manager: GraphDatabaseManager):
    """
    Cache the query engine to avoid recreating on each interaction.
    Using _db_manager to prevent Streamlit from hashing the object.
    """
    from src.query_engine import get_query_engine
    return get_query_engine(db_manager=_db_manager)


def render_sidebar(db_manager: GraphDatabaseManager):
    """Render the sidebar with configuration status."""
    st.sidebar.title("🧠 GraphRAG Demo")
    st.sidebar.markdown("---")

    # Connection Status Section
    st.sidebar.subheader("📡 Connection Status")

    # Neo4j Status (using timeout-protected check)
    neo4j_connected, neo4j_status = check_connection_with_timeout(db_manager, timeout=5.0)
    if neo4j_connected:
        st.sidebar.success(f"✅ Neo4j: {neo4j_status}")
    else:
        st.sidebar.error(f"❌ Neo4j: {neo4j_status}")

    # OpenAI Status
    openai_configured, openai_status = check_openai_key()
    if openai_configured:
        st.sidebar.success(f"✅ OpenAI: {openai_status}")
    else:
        st.sidebar.error(f"❌ OpenAI: {openai_status}")

    st.sidebar.markdown("---")

    # Info Section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **GraphRAG** combines the power of:
    - 📊 **Knowledge Graphs** for structured relationships
    - 🔍 **Vector Search** for semantic similarity
    - 🤖 **LLMs** for natural language understanding

    This creates more accurate and contextual responses than traditional RAG.
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with LlamaIndex, Neo4j, and Streamlit")

    return neo4j_connected, openai_configured


def render_chat_tab(db_manager: GraphDatabaseManager):
    """Render the chat interface tab."""
    st.header("💬 Chat with Knowledge Graph")
    st.markdown("Ask questions about your data. The system uses graph-based retrieval for better context.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about renewable energy..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using query function with built-in error handling
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying knowledge graph..."):
                from src.query_engine import query
                engine = get_cached_query_engine(db_manager)
                # The query function now returns user-friendly error messages
                # instead of raising exceptions
                response_text = query(prompt, engine)

            st.markdown(response_text)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def render_graph_tab(db_manager: GraphDatabaseManager):
    """Render the graph explorer tab."""
    st.header("📊 Knowledge Graph Explorer")
    st.markdown("Interactive visualization of the knowledge graph stored in Neo4j.")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("🔄 Refresh Graph"):
            # Clear all cached graph HTML keys
            keys_to_remove = [k for k in st.session_state if k.startswith("graph_html_")]
            for key in keys_to_remove:
                st.session_state.pop(key, None)
            st.rerun()

    with col2:
        node_limit = st.selectbox(
            "Max Nodes",
            options=[50, 100, 200],
            index=1,
            help="Limit the number of relationships to display for performance.",
        )

    # Generate or use cached graph HTML (in-memory, no temp files)
    cache_key = f"graph_html_{node_limit}"

    if cache_key not in st.session_state:
        with st.spinner("🔄 Generating graph visualization..."):
            try:
                from src.visualizer import generate_graph_html
                # generate_graph_html now returns HTML string directly (no disk I/O)
                st.session_state[cache_key] = generate_graph_html(
                    db_manager=db_manager,
                    height="650px",
                    limit=node_limit,
                )
            except Exception as e:
                st.error(f"Failed to load graph: {str(e)}")
                st.info("Make sure Neo4j is running and the knowledge graph has been populated using `python -m src.ingestion`")
                return

    # Display the graph (HTML string rendered via components.html)
    components.html(
        st.session_state[cache_key],
        height=700,
        scrolling=True,
    )

    # Help text
    with st.expander("📖 How to use the graph"):
        st.markdown("""
        - **Zoom**: Scroll to zoom in/out
        - **Pan**: Click and drag on empty space
        - **Move nodes**: Click and drag nodes
        - **Hover**: See node/edge details
        - **Select**: Click on a node to highlight it

        The graph shows entities (nodes) and their relationships (edges)
        extracted from your documents by the LLM.
        """)


def main():
    """Main application entry point."""
    # Initialize Database Manager
    db_manager = get_db_manager()

    # Render sidebar and get connection status
    neo4j_connected, openai_configured = render_sidebar(db_manager)

    # Main content area with tabs
    tab_chat, tab_graph = st.tabs(["💬 Chat", "📊 Graph Explorer"])

    with tab_chat:
        if not neo4j_connected or not openai_configured:
            st.warning("⚠️ Please configure Neo4j and OpenAI connections to use the chat feature.")
            st.markdown("""
            ### Quick Setup:
            1. Copy `.env.example` to `.env`
            2. Add your OpenAI API key
            3. Start Neo4j: `docker compose up -d`
            4. Run ingestion: `python src/ingestion.py`
            """)
        else:
            render_chat_tab(db_manager)

    with tab_graph:
        if not neo4j_connected:
            st.warning("⚠️ Please start Neo4j to view the graph visualization.")
            st.code("docker compose up -d", language="bash")
        else:
            render_graph_tab(db_manager)


if __name__ == "__main__":
    main()

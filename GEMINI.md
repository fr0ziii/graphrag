# simple-graphrag-demo

## Project Overview
This project is a production-ready implementation of **GraphRAG** (Retrieval-Augmented Generation using Knowledge Graphs). It demonstrates how to combine LlamaIndex, Neo4j, and OpenAI to create a Retrieval-Augmented Generation system that uses knowledge graphs for more accurate and contextual AI responses, specifically enabling multi-hop reasoning.

**Key Technologies:**
- **Language:** Python 3.10+
- **Graph Database:** Neo4j (via Docker)
- **Framework:** LlamaIndex
- **LLM:** OpenAI (GPT-4o/Turbo recommended)
- **UI:** Streamlit
- **Package Manager:** uv

## Architecture
The system follows a standard RAG pipeline enhanced with Knowledge Graph capabilities:
1.  **Ingestion:** Documents in `data/` are processed by LlamaIndex to extract entities and relationships, which are stored in Neo4j. **Idempotent:** Each document's SHA-256 hash is stored as a `Document` node; re-running ingestion skips already-processed files.
2.  **Storage:** Neo4j serves as the Graph Store.
3.  **Querying:** A Query Engine uses "Tree Summarize" to traverse the graph and retrieve context for user queries.
4.  **Interface:** A Streamlit app provides a Chat UI and an interactive Graph Explorer (using Pyvis).

## Setup & Running

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose
- OpenAI API Key

### Quick Start
1.  **Environment Setup:**
    ```bash
    cp .env.example .env
    # Edit .env to add OPENAI_API_KEY and Neo4j credentials
    ```

2.  **Start Infrastructure (Neo4j):**
    ```bash
    docker compose up -d
    # Wait ~30s for Neo4j to be ready at http://localhost:7474
    ```

3.  **Install Dependencies:**
    ```bash
    uv sync
    # OR: pip install -r requirements.txt
    ```

4.  **Ingest Data:**
    Build the knowledge graph from documents in `data/`:
    ```bash
    uv run python src/ingestion.py
    ```

5.  **Run Application:**
    Launch the Streamlit UI:
    ```bash
    uv run streamlit run src/app.py
    ```
    Access at http://localhost:8501

## Key Files & Directories

-   **`src/`**: Source code directory.
    -   **`app.py`**: Main Streamlit application entry point. Handles UI rendering and session state.
    -   **`ingestion.py`**: Script to load documents, extract entities, and populate the Neo4j graph.
    -   **`query_engine.py`**: Initializes the LlamaIndex query engine with Neo4j context.
    -   **`visualizer.py`**: Logic for generating Pyvis interactive graph visualizations.
    -   **`database.py`**: Centralized Neo4j connection management.
-   **`data/`**: Directory for source text documents (e.g., `renewable_energy.txt`).
-   **`docker-compose.yml`**: Defines the Neo4j service configuration.
-   **`pyproject.toml`**: Project configuration, dependencies (uv), and tool settings (Ruff).

## Development Conventions

-   **Dependency Management:** The project uses `uv` for fast dependency resolution. `pyproject.toml` is the source of truth.
-   **Linting & Formatting:** Code quality is enforced by `ruff`.
    -   Configured in `pyproject.toml`.
    -   Run checks: `uv run ruff check .`
    -   Format code: `uv run ruff format .`
-   **Pre-commit:** A `.pre-commit-config.yaml` is present. Ensure hooks are installed if contributing.
-   **Graph Density:** Adjustable via `max_triplets_per_chunk` in `src/ingestion.py`.
-   **LLM Configuration:** Model parameters (e.g., `gpt-4o`) are defined in `src/ingestion.py` and `src/query_engine.py`.

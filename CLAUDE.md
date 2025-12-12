# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GraphRAG (Retrieval-Augmented Generation using Knowledge Graphs) demo that combines Neo4j property graphs with LlamaIndex for semantic search and LLM-powered question answering. The project demonstrates schema-driven knowledge graph extraction from documents.

## Essential Commands

### Setup and Installation
```bash
# Install dependencies with uv (preferred)
uv sync

# Or with pip
pip install -r requirements.txt

# Start Neo4j database
docker compose up -d

# Verify Neo4j is running (wait ~30s after starting)
docker compose ps

# Set up environment
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
```

### Development Workflow
```bash
# Run ingestion pipeline (builds knowledge graph from data/ directory)
uv run python src/ingestion.py

# Test query engine standalone
uv run python src/query_engine.py

# Launch Streamlit application
uv run streamlit run src/app.py

# Install pre-commit hooks
pre-commit install

# Run pre-commit manually
pre-commit run --all-files
```

### Code Quality
```bash
# Lint with Ruff (auto-fix enabled)
ruff check --fix .

# Format with Ruff
ruff format .

# Run both linter and formatter
pre-commit run --all-files
```

## Architecture

### Core Data Flow

1. **Ingestion Pipeline** (`src/ingestion.py`):
   - Reads documents from `data/` directory using `SimpleDirectoryReader`
   - Uses `SchemaLLMPathExtractor` with strict schema enforcement to extract knowledge triplets
   - Stores entities and relationships in Neo4j via `Neo4jPropertyGraphStore`
   - Enforces predefined ontology: TECHNOLOGY, CONCEPT, LOCATION, METRIC, ORGANIZATION, MATERIAL (entities) and USES, PRODUCES, LOCATED_IN, AFFECTS, HAS_METRIC, DEVELOPED_BY (relationships)

2. **Query Engine** (`src/query_engine.py`):
   - Loads existing `KnowledgeGraphIndex` from Neo4j without re-ingestion
   - Uses `tree_summarize` response mode for context-aware answers
   - Combines graph traversal with vector similarity search

3. **Streamlit App** (`src/app.py`):
   - Chat interface for natural language queries
   - Interactive graph visualization using Pyvis
   - Connection status monitoring for Neo4j and OpenAI

### Module Responsibilities

- **`database.py`**: Centralized Neo4j connection management. Provides:
  - `get_neo4j_driver()` for direct Cypher queries
  - `get_neo4j_graph_store()` for LlamaIndex `KnowledgeGraphIndex` (legacy)
  - `get_neo4j_property_graph_store()` for `PropertyGraphIndex` with schema extraction
  - `execute_query()` for executing Cypher queries
  - `check_connection()` for health checks

- **`ingestion.py`**: Schema-driven knowledge graph extraction. Key features:
  - Uses `PropertyGraphIndex` (not `KnowledgeGraphIndex`)
  - `SchemaLLMPathExtractor` with `strict=True` enforces ontology
  - Entity normalization to Title Case reduces duplicates
  - Configurable `max_triplets_per_chunk` and `num_workers`

- **`query_engine.py`**: Loads existing graph index for querying. Uses `KnowledgeGraphIndex` (note: different from ingestion which uses `PropertyGraphIndex`)

- **`visualizer.py`**: Fetches graph data via Cypher and renders with Pyvis. Supports color-coded node groups and interactive exploration.

- **`app.py`**: Streamlit UI with two tabs (Chat and Graph Explorer). Implements query engine caching with `@st.cache_resource`.

### Important Patterns

**Two Graph Index Types Used:**
- **Ingestion** uses `PropertyGraphIndex` with `Neo4jPropertyGraphStore` for schema-driven extraction
- **Query** uses `KnowledgeGraphIndex` with `Neo4jGraphStore` to load existing data
- This dual-store approach is intentional: PropertyGraphIndex for structured writes, KnowledgeGraphIndex for flexible reads

**Schema Enforcement:**
The `VALIDATION_SCHEMA` dict in `ingestion.py` maps entity types to allowed outgoing relationships. Only triplets matching this schema are extracted when `strict=True`.

**Environment Variables:**
All modules use `python-dotenv` to load from `.env`:
- `OPENAI_API_KEY`: Required for LLM and embeddings
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Neo4j connection

**Python Path Setup:**
`app.py` adds project root to `sys.path` for Streamlit compatibility (line 13).

## Modifying the Knowledge Graph

### Changing the Schema

To add new entity or relationship types:

1. Update `ENTITY_TYPES` and `RELATION_TYPES` Literals in `src/ingestion.py`
2. Update `VALIDATION_SCHEMA` to define valid entity-relationship combinations
3. Re-run ingestion to rebuild graph with new schema

### Adding New Data

1. Place `.txt` files in `data/` directory
2. Run `uv run python src/ingestion.py`
3. Graph automatically updates in Neo4j

### Adjusting Graph Density

In `src/ingestion.py`, modify `build_graph_index()` parameters:
- `max_triplets_per_chunk`: Lower (1-2) for focused graphs, higher (5-10) for comprehensive
- `num_workers`: Parallel extraction workers (default 4)
- `normalize_entities`: Toggle Title Case normalization (default True)

## Troubleshooting

**Neo4j Connection Errors:**
```bash
# Check container status
docker compose ps

# View logs
docker compose logs neo4j

# Restart
docker compose restart neo4j
```

**Empty Graph Visualization:**
Ensure ingestion has run successfully. Verify in Neo4j Browser (http://localhost:7474):
```cypher
MATCH (n) RETURN DISTINCT labels(n) AS entity_types
MATCH ()-[r]->() RETURN DISTINCT type(r) AS relation_types
```

**Ingestion Fails with Schema Errors:**
Check that all relationships in `VALIDATION_SCHEMA` reference defined `RELATION_TYPES`, and all keys are in `ENTITY_TYPES`.

## Development Notes

- Ruff is configured in `pyproject.toml` with strict rules: pyflakes, pycodestyle, isort, flake8-bugbear, pyupgrade, bandit
- Pre-commit hooks enforce trailing whitespace, YAML validity, large file checks, and private key detection
- Line length: 88 characters (Black-compatible)
- Target Python version: 3.10+
- Dependency management with `uv` (modern, fast alternative to pip/pip-tools)

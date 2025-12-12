# 🧠 Simple GraphRAG Demo

A production-ready implementation of **GraphRAG** (Retrieval-Augmented Generation using Knowledge Graphs) designed for developer portfolios. This project demonstrates how to combine the power of knowledge graphs with LLMs for more accurate and contextual AI responses.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Neo4j](https://img.shields.io/badge/neo4j-5.x-green.svg)
![LlamaIndex](https://img.shields.io/badge/llamaindex-0.11+-orange.svg)

## 🎯 What is GraphRAG?

Traditional RAG (Retrieval-Augmented Generation) uses vector similarity to find relevant documents. **GraphRAG** enhances this by:

| Traditional RAG | GraphRAG |
|----------------|----------|
| Vector similarity search | Graph traversal + Vector search |
| Independent document chunks | Connected entity relationships |
| "What's similar?" | "What's related?" |
| Good for simple queries | Excellent for complex, multi-hop reasoning |

### Why Knowledge Graphs Matter

Knowledge graphs capture **entities** (people, concepts, events) and their **relationships** in a structured format. This enables:

- **Multi-hop reasoning**: "What companies does Elon Musk run that produce electric vehicles?"
- **Contextual understanding**: Related concepts are explicitly connected
- **Explainability**: You can trace the path from question to answer

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Documents     │────▶│  LlamaIndex      │────▶│   Neo4j         │
│   (data/)       │     │  (KG Extraction) │     │   (Graph Store) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│   Streamlit     │◀────│  Query Engine    │◀─────────────┘
│   (Chat UI)     │     │  (Tree Summarize)│
└─────────────────┘     └──────────────────┘
```

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- OpenAI API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/simple-graphrag-demo.git
cd simple-graphrag-demo
```

### Step 2: Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Required environment variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |

### Step 3: Start Neo4j

```bash
docker compose up -d
```

Wait ~30 seconds for Neo4j to fully start. You can check the status at [http://localhost:7474](http://localhost:7474).

### Step 4: Install Dependencies

Using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:

```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Step 5: Run Data Ingestion

This step reads documents from `data/` and builds the knowledge graph in Neo4j:

```bash
uv run python src/ingestion.py
```

**Expected output:**
```
============================================================
GraphRAG Ingestion Pipeline
============================================================
Loading documents from data...
Loaded 1 document(s)
Building Knowledge Graph Index (this may take a while)...
✅ Ingestion complete! Knowledge graph is ready in Neo4j.
```

### Step 6: Launch the Application

```bash
uv run streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📁 Project Structure

```
simple-graphrag-demo/
│
├── data/
│   └── renewable_energy.txt   # Sample knowledge base
│
├── src/
│   ├── database.py            # Centralized Neo4j connection
│   ├── ingestion.py           # Document loading & KG building
│   ├── query_engine.py        # Query engine initialization
│   ├── visualizer.py          # Neo4j to Pyvis visualization
│   └── app.py                 # Streamlit application
│
├── .env.example               # Environment template
├── docker-compose.yml         # Neo4j container config
├── pyproject.toml             # uv/pip dependencies
├── requirements.txt           # Legacy pip dependencies
└── README.md                  # This file
```

## 🎨 Features

### 💬 Chat Interface
- Natural language Q&A over your knowledge graph
- Conversation history with session state
- Tree-summarize response mode for better context

### 📊 Graph Explorer
- Interactive visualization with Pyvis
- Zoom, pan, and drag nodes
- Color-coded entity types
- Relationship labels on edges

## 🔧 Customization

### Adding Your Own Data

1. Place your text files in the `data/` directory
2. Re-run the ingestion: `uv run python src/ingestion.py`
3. Refresh the app

### Adjusting Graph Density

In `src/ingestion.py`, modify `max_triplets_per_chunk`:
- Lower values (1-2): Cleaner, more focused graphs
- Higher values (5-10): Denser, more comprehensive graphs

### Using Different LLMs

Modify the `OpenAI()` initialization in `src/ingestion.py` and `src/query_engine.py`:

```python
llm = OpenAI(
    model="gpt-4o",  # or "gpt-4-turbo"
    temperature=0,
)
```

## 🐛 Troubleshooting

### Neo4j Connection Failed
```bash
# Check if Neo4j is running
docker compose ps

# View logs
docker compose logs neo4j

# Restart if needed
docker compose restart neo4j
```

### Empty Graph Visualization
- Ensure you've run `python src/ingestion.py` first
- Check Neo4j browser at http://localhost:7474 to verify data exists

### OpenAI API Errors
- Verify your API key is correctly set in `.env`
- Check your OpenAI account has available credits

## 📚 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Neo4j Knowledge Graph Guide](https://neo4j.com/developer/graph-database/)
- [GraphRAG Explained (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

## 📄 License

MIT License - feel free to use this project in your portfolio!

---

Built with ❤️ using LlamaIndex, Neo4j, Streamlit, and Pyvis

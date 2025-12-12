"""
Schema-Driven Ingestion for GraphRAG Knowledge Graph.

This module implements a structured knowledge graph extraction pipeline using
LlamaIndex's PropertyGraphIndex with SchemaLLMPathExtractor. Unlike naive
extraction that allows the LLM to hallucinate arbitrary node/relationship types,
this approach enforces a fixed ontology for the Renewable Energy domain.

SCHEMA ENFORCEMENT STRATEGY:
----------------------------
1. ONTOLOGY DEFINITION: We define allowed entity types (TECHNOLOGY, CONCEPT,
   LOCATION, METRIC, ORGANIZATION, MATERIAL) and relationship types (USES,
   PRODUCES, LOCATED_IN, AFFECTS, HAS_METRIC, DEVELOPED_BY) using Python Literals.

2. VALIDATION SCHEMA: A dictionary mapping entity types to their allowed
   relationships constrains which triplets can be extracted.

3. STRICT MODE: SchemaLLMPathExtractor with strict=True rejects any triplets
   that don't conform to the schema, eliminating "spaghetti graph" noise.

4. ENTITY NORMALIZATION: Text is preprocessed to normalize entity names to
   Title Case, reducing duplicates like "solar energy" vs "Solar Energy".
"""

import logging
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.database import get_neo4j_property_graph_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =============================================================================
# ONTOLOGY DEFINITION: Fixed Schema for Renewable Energy Domain
# =============================================================================

# Allowed Entity Types (use uppercase by convention)
ENTITY_TYPES = Literal[
    "TECHNOLOGY",
    "CONCEPT",
    "LOCATION",
    "METRIC",
    "ORGANIZATION",
    "MATERIAL",
]

# Allowed Relationship Types
RELATION_TYPES = Literal[
    "USES",
    "PRODUCES",
    "LOCATED_IN",
    "AFFECTS",
    "HAS_METRIC",
    "DEVELOPED_BY",
]

# Validation Schema: Maps entity types to their allowed outgoing relationships.
# This constrains which triplets are valid (e.g., TECHNOLOGY can USES, PRODUCES, etc.)
VALIDATION_SCHEMA = {
    "TECHNOLOGY": ["USES", "PRODUCES", "LOCATED_IN", "HAS_METRIC", "DEVELOPED_BY"],
    "CONCEPT": ["AFFECTS", "USES", "PRODUCES"],
    "LOCATION": ["LOCATED_IN"],
    "METRIC": ["HAS_METRIC"],
    "ORGANIZATION": ["DEVELOPED_BY", "USES", "LOCATED_IN"],
    "MATERIAL": ["USES", "PRODUCES"],
}


# =============================================================================
# ENTITY NORMALIZATION: Preprocessing to reduce duplicates
# =============================================================================


def normalize_text(text: str) -> str:
    """
    Normalize text to improve entity consistency.

    Transformations:
    - Collapse multiple whitespace into single spaces
    - Strip leading/trailing whitespace
    - Convert to Title Case (reduces "solar energy" vs "Solar Energy" duplicates)

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip and convert to title case
    return text.strip().title()


def preprocess_documents(documents: list[Document]) -> list[Document]:
    """
    Preprocess documents to normalize entity names before extraction.

    This step applies Title Case normalization to document content,
    reducing the chance of duplicate entities with case variations.

    Args:
        documents: List of LlamaIndex Document objects.

    Returns:
        List of preprocessed Document objects.
    """
    processed = []
    for doc in documents:
        # Create a new document with normalized text
        # Note: We normalize the text content, not metadata
        normalized_text = normalize_text(doc.text)
        processed.append(
            Document(
                text=normalized_text,
                metadata=doc.metadata,
                doc_id=doc.doc_id,
            )
        )
    return processed


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================


def build_graph_index(
    data_dir: str = "data",
    max_triplets_per_chunk: int = 10,
    num_workers: int = 4,
    normalize_entities: bool = True,
) -> PropertyGraphIndex:
    """
    Build a PropertyGraphIndex with schema-driven extraction.

    This function reads documents, optionally normalizes them, and extracts
    knowledge triplets constrained by the predefined ontology schema.

    Args:
        data_dir: Path to the directory containing source documents.
        max_triplets_per_chunk: Maximum triplets to extract per text chunk.
        num_workers: Number of parallel workers for extraction.
        normalize_entities: Whether to normalize text to Title Case.

    Returns:
        PropertyGraphIndex: The constructed knowledge graph index.

    Raises:
        FileNotFoundError: If the data directory doesn't exist or is empty.
        ConnectionError: If unable to connect to Neo4j.
    """
    # Validate data directory
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load documents
    logger.info("Loading documents from %s...", data_dir)
    documents = SimpleDirectoryReader(data_dir).load_data()

    if not documents:
        raise FileNotFoundError(f"No documents found in {data_dir}")

    logger.info("Loaded %d document(s)", len(documents))

    # Optional: Normalize document text for entity consistency
    if normalize_entities:
        logger.info("Normalizing document text for entity consistency...")
        documents = preprocess_documents(documents)

    # Initialize LLM and embedding model
    llm = OpenAI(
        model="gpt-4o-mini",  # Better extraction quality than gpt-3.5-turbo
        temperature=0,
    )
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
    )

    # Connect to Neo4j PropertyGraphStore (required for PropertyGraphIndex)
    graph_store = get_neo4j_property_graph_store()

    # Create the schema-enforced extractor
    # strict=True ensures only schema-conforming triplets are extracted
    logger.info("Configuring SchemaLLMPathExtractor with strict schema enforcement...")
    logger.info("  Entity types: TECHNOLOGY, CONCEPT, LOCATION, METRIC, ORGANIZATION, MATERIAL")
    logger.info("  Relation types: USES, PRODUCES, LOCATED_IN, AFFECTS, HAS_METRIC, DEVELOPED_BY")

    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        kg_validation_schema=VALIDATION_SCHEMA,
        strict=True,  # CRITICAL: Reject triplets outside the schema
        num_workers=num_workers,
        max_triplets_per_chunk=max_triplets_per_chunk,
    )

    # Build the PropertyGraphIndex with schema-driven extraction
    logger.info("Building PropertyGraphIndex (this may take a while)...")
    logger.info(
        "Settings: max_triplets=%d, workers=%d, strict=True",
        max_triplets_per_chunk,
        num_workers,
    )

    index = PropertyGraphIndex.from_documents(
        documents,
        embed_model=embed_model,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    logger.info("PropertyGraphIndex built successfully with schema enforcement!")
    return index


if __name__ == "__main__":
    # Run ingestion when executed directly
    print("=" * 60)
    print("GraphRAG Schema-Driven Ingestion Pipeline")
    print("=" * 60)
    print("\nOntology:")
    print("  Entity Types: TECHNOLOGY, CONCEPT, LOCATION, METRIC, ORGANIZATION, MATERIAL")
    print("  Relation Types: USES, PRODUCES, LOCATED_IN, AFFECTS, HAS_METRIC, DEVELOPED_BY")
    print("-" * 60)

    try:
        index = build_graph_index()
        print("\n✅ Ingestion complete! Schema-enforced knowledge graph is ready.")
        print("   Explore it at http://localhost:7474")
        print("\nVerification queries to run in Neo4j Browser:")
        print("  MATCH (n) RETURN DISTINCT labels(n) AS entity_types")
        print("  MATCH ()-[r]->() RETURN DISTINCT type(r) AS relation_types")
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise

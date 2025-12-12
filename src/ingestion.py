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

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.config import get_entity_literal, get_ontology, get_relation_literal
from src.database import (
    create_document_node,
    document_exists_by_hash,
    get_neo4j_property_graph_store,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# =============================================================================
# DOCUMENT HASHING FOR IDEMPOTENCY
# =============================================================================


def compute_document_hash(text: str) -> str:
    """
    Compute SHA-256 hash of document text content.

    Used to uniquely identify documents for idempotent ingestion.
    Documents with the same text content will produce the same hash.

    Args:
        text: Document text content.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
) -> PropertyGraphIndex | None:
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
        PropertyGraphIndex: The constructed knowledge graph index,
            or None if all documents were already ingested.

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

    # =========================================================================
    # IDEMPOTENCY CHECK: Skip documents that have already been ingested
    # =========================================================================
    documents_to_process = []
    document_hashes: dict[str, str] = {}  # filename -> hash mapping for persistence

    for doc in documents:
        # Compute hash from original text (before normalization)
        doc_hash = compute_document_hash(doc.text)
        filename = doc.metadata.get("file_name", doc.doc_id or "unknown")

        if document_exists_by_hash(doc_hash):
            logger.info("Skipping file %s, already ingested", filename)
        else:
            documents_to_process.append(doc)
            document_hashes[filename] = doc_hash

    # Early return if all documents were already ingested
    if not documents_to_process:
        logger.info("All %d document(s) already ingested. Nothing to do.", len(documents))
        # Return None to signal no new processing was done
        return None

    logger.info(
        "Processing %d new document(s) (%d skipped)",
        len(documents_to_process),
        len(documents) - len(documents_to_process),
    )

    # Use filtered list for further processing
    documents = documents_to_process

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

    # Load ontology from external configuration
    ontology = get_ontology()

    # Create the schema-enforced extractor
    # strict=True ensures only schema-conforming triplets are extracted
    logger.info(
        "Configuring SchemaLLMPathExtractor for domain '%s' (v%s) with strict schema enforcement...",
        ontology.domain,
        ontology.version,
    )
    logger.info("  Entity types: %s", ", ".join(ontology.entity_types))
    logger.info("  Relation types: %s", ", ".join(ontology.relation_types))

    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=get_entity_literal(ontology),
        possible_relations=get_relation_literal(ontology),
        kg_validation_schema=ontology.validation_schema,
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

    # =========================================================================
    # PERSIST DOCUMENT NODES: Track ingested documents for idempotency
    # =========================================================================
    ingestion_time = datetime.now(timezone.utc).isoformat()
    for filename, doc_hash in document_hashes.items():
        create_document_node(filename, doc_hash, ingestion_time)

    logger.info("Persisted %d Document nodes for tracking", len(document_hashes))

    return index


if __name__ == "__main__":
    # Run ingestion when executed directly
    # Load ontology first to display info
    ontology = get_ontology()

    print("=" * 60)
    print("GraphRAG Schema-Driven Ingestion Pipeline")
    print("=" * 60)
    print(f"\nDomain: {ontology.domain} (v{ontology.version})")
    print("\nOntology loaded from: config/ontology.yaml")
    print(f"  Entity Types: {', '.join(ontology.entity_types)}")
    print(f"  Relation Types: {', '.join(ontology.relation_types)}")
    print("-" * 60)

    try:
        index = build_graph_index()

        # Handle case where all documents were already ingested
        if index is None:
            print("\n✅ All documents already ingested. No new processing needed.")
            print("   Run with a fresh database to re-ingest.")
        else:
            print("\n✅ Ingestion complete! Schema-enforced knowledge graph is ready.")
            print("   Explore it at http://localhost:7474")
        print("\nVerification queries to run in Neo4j Browser:")
        print("  MATCH (n) RETURN DISTINCT labels(n) AS entity_types")
        print("  MATCH ()-[r]->() RETURN DISTINCT type(r) AS relation_types")

        # Optional: Run graph analytics post-ingestion
        print("\n" + "-" * 60)
        run_analytics = input(
            "Run graph analytics (PageRank + Community Detection)? [y/N]: "
        )
        if run_analytics.lower() == "y":
            print("\nStarting graph analytics pipeline...")
            from src.analysis import run_analysis

            result = run_analysis()
            if result.get("status") == "success":
                print("\n✅ Graph analytics complete!")
                print(f"   Communities found: {result['louvain']['communities_found']}")
                print("   Properties written: pageRankScore, communityId")
            else:
                print("\n⚠️  Analytics skipped - graph may be empty")
        else:
            print("\nSkipping analytics. Run later with:")
            print("  uv run python -m src.analysis")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise

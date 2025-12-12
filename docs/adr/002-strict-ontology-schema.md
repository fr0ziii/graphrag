# 2. Use Strict Ontology Schema for Extraction

Date: 2025-12-12

## Status

Accepted

## Context

In initial iterations, we used "open extraction" where the LLM was free to determine entity types and relationship labels based on the text. This led to:
-   **Spaghetti Graphs:** A proliferation of unique entity types (e.g., "Company", "Business", "Corp", "Firm") making querying difficult.
-   **Noise:** irrelevant or hallucinated relationships.
-   **Lack of Structure:** Difficulty in writing consistent Cypher queries because the schema was unpredictable.

To support a robust RAG system, we need a predictable graph structure.

## Decision

We will use **Strict Schema Enforcement** for the knowledge graph extraction. Specifically, we will use LlamaIndex's `SchemaLLMPathExtractor` (or equivalent logic) with `strict=True`.

The ontology is explicitly defined in `config/ontology.yaml` and loaded at runtime. Invalid entities or relationships that do not conform to the allowed types will be rejected or coerced by the LLM (guided by prompts).

## Consequences

### Positive
-   **Query Reliability:** We can write Cypher queries targeting specific labels (e.g., `MATCH (t:Technology)-[:AFFECTS]->(c:Concept)`) with high confidence.
-   **Data Quality:** Reduced noise and duplication in the graph.
-   **Domain Consistency:** Forces the graph to align with the specific domain model relevant to the application (e.g., focusing on Technologies and Concepts).

### Negative
-   **Reduced Flexibility:** The system will not automatically discover novel entity types outside the pre-defined ontology. New domains require updating the `ontology.yaml`.
-   **Extraction "Loss":** Information that doesn't fit the schema might be ignored during the extraction phase.

### Compliance
-   All ingestion logic must load the ontology from configuration.
-   The PromptTemplate for extraction must include the strict schema instructions.

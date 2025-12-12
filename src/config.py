"""
Configuration Manager for GraphRAG Ontology.

This module loads and validates the ontology configuration from an external
YAML file, enabling domain-agnostic knowledge graph extraction without
modifying Python code.

Usage:
    from src.config import get_ontology

    ontology = get_ontology()
    print(ontology.entity_types)  # ['TECHNOLOGY', 'CONCEPT', ...]
    print(ontology.validation_schema)  # {'TECHNOLOGY': ['USES', ...], ...}
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Configure logging
logger = logging.getLogger(__name__)

# Default path to ontology configuration
DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent / "config" / "ontology.yaml"


class OntologyConfigError(Exception):
    """Raised when ontology configuration is invalid or cannot be loaded."""

    pass


class OntologyConfig(BaseModel):
    """
    Pydantic model for validating ontology configuration.

    Attributes:
        domain: Human-readable domain name (e.g., "Renewable Energy").
        version: Schema version string for tracking changes.
        entity_types: List of allowed entity types (node labels).
        relation_types: List of allowed relation types (edge types).
        validation_schema: Maps entity types to their allowed outgoing relations.
    """

    model_config = ConfigDict(frozen=True)  # Make immutable after creation

    domain: str
    version: str
    entity_types: list[str]
    relation_types: list[str]
    validation_schema: dict[str, list[str]]

    @field_validator("entity_types", "relation_types")
    @classmethod
    def validate_non_empty_list(cls, v: list[str], info) -> list[str]:
        """Ensure entity and relation type lists are not empty."""
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    @field_validator("entity_types", "relation_types")
    @classmethod
    def validate_uppercase(cls, v: list[str]) -> list[str]:
        """Ensure all types are uppercase for consistency."""
        return [item.upper() for item in v]

    @model_validator(mode="after")
    def validate_schema_consistency(self) -> "OntologyConfig":
        """
        Validate that validation_schema references only defined types.

        Ensures:
        - All keys in validation_schema are in entity_types
        - All values in validation_schema are in relation_types
        """
        entity_set = set(self.entity_types)
        relation_set = set(self.relation_types)

        for entity_type, allowed_relations in self.validation_schema.items():
            # Check entity type exists
            if entity_type.upper() not in entity_set:
                raise ValueError(
                    f"validation_schema references undefined entity type: '{entity_type}'. "
                    f"Defined types: {self.entity_types}"
                )

            # Check all relation types exist
            for relation in allowed_relations:
                if relation.upper() not in relation_set:
                    raise ValueError(
                        f"validation_schema['{entity_type}'] references undefined "
                        f"relation type: '{relation}'. Defined types: {self.relation_types}"
                    )

        return self


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> OntologyConfig:
    """
    Load and validate ontology configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        OntologyConfig: Validated configuration object.

    Raises:
        OntologyConfigError: If file cannot be read or has invalid format.
    """
    path = Path(path)

    if not path.exists():
        raise OntologyConfigError(
            f"Ontology configuration file not found: {path}\n"
            f"Please create the file or check the path."
        )

    try:
        with path.open(encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OntologyConfigError(
            f"Invalid YAML format in ontology configuration: {path}\n"
            f"YAML Error: {e}"
        ) from e

    if raw_config is None:
        raise OntologyConfigError(f"Ontology configuration file is empty: {path}")

    try:
        config = OntologyConfig.model_validate(raw_config)
    except Exception as e:
        raise OntologyConfigError(
            f"Invalid ontology configuration in {path}:\n{e}"
        ) from e

    logger.info("Loaded ontology configuration for domain: %s (v%s)", config.domain, config.version)

    return config


@lru_cache(maxsize=1)
def get_ontology(path: str | Path | None = None) -> OntologyConfig:
    """
    Get the ontology configuration (singleton with caching).

    This function caches the loaded configuration to avoid repeated file I/O.
    Call with a different path to load a different configuration (cache will
    be bypassed for non-default paths).

    Args:
        path: Optional path to ontology YAML. Uses default if not specified.

    Returns:
        OntologyConfig: The validated ontology configuration.
    """
    if path is None:
        path = DEFAULT_ONTOLOGY_PATH
    return load_ontology(path)


def get_entity_literal(ontology: OntologyConfig) -> type:
    """
    Generate a Literal type for entity types compatible with SchemaLLMPathExtractor.

    Args:
        ontology: The loaded ontology configuration.

    Returns:
        A Literal type containing all entity types.

    Example:
        >>> ontology = get_ontology()
        >>> EntityType = get_entity_literal(ontology)
        >>> # EntityType is now Literal["TECHNOLOGY", "CONCEPT", ...]
    """
    return Literal[tuple(ontology.entity_types)]  # type: ignore[valid-type]


def get_relation_literal(ontology: OntologyConfig) -> type:
    """
    Generate a Literal type for relation types compatible with SchemaLLMPathExtractor.

    Args:
        ontology: The loaded ontology configuration.

    Returns:
        A Literal type containing all relation types.

    Example:
        >>> ontology = get_ontology()
        >>> RelationType = get_relation_literal(ontology)
        >>> # RelationType is now Literal["USES", "PRODUCES", ...]
    """
    return Literal[tuple(ontology.relation_types)]  # type: ignore[valid-type]


# =============================================================================
# CLI: Test configuration loading when run directly
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAG Ontology Configuration Loader")
    print("=" * 60)

    try:
        config = get_ontology()
        print("\n✅ Successfully loaded ontology configuration!")
        print(f"\nDomain: {config.domain}")
        print(f"Version: {config.version}")
        print(f"\nEntity Types ({len(config.entity_types)}):")
        for et in config.entity_types:
            print(f"  - {et}")
        print(f"\nRelation Types ({len(config.relation_types)}):")
        for rt in config.relation_types:
            print(f"  - {rt}")
        print("\nValidation Schema:")
        for entity, relations in config.validation_schema.items():
            print(f"  {entity}: {', '.join(relations)}")
    except OntologyConfigError as e:
        print(f"\n❌ Configuration Error:\n{e}")
        exit(1)

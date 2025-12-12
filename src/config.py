"""
Configuration Manager for GraphRAG.

This module provides a centralized configuration system using Pydantic Settings
and handles the loading of the ontology configuration.

Usage:
    from src.config import settings, get_ontology

    # Access settings
    print(settings.llm.model)
    print(settings.ingestion.max_triplets_per_chunk)

    # Access ontology
    ontology = get_ontology()
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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
            f"Invalid YAML format in ontology configuration: {path}\nYAML Error: {e}"
        ) from e

    if raw_config is None:
        raise OntologyConfigError(f"Ontology configuration file is empty: {path}")

    try:
        config = OntologyConfig.model_validate(raw_config)
    except Exception as e:
        raise OntologyConfigError(
            f"Invalid ontology configuration in {path}:\n{e}"
        ) from e

    logger.info(
        "Loaded ontology configuration for domain: %s (v%s)",
        config.domain,
        config.version,
    )

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
    """
    return Literal[tuple(ontology.entity_types)]  # type: ignore[valid-type]


def get_relation_literal(ontology: OntologyConfig) -> type:
    """
    Generate a Literal type for relation types compatible with SchemaLLMPathExtractor.

    Args:
        ontology: The loaded ontology configuration.

    Returns:
        A Literal type containing all relation types.
    """
    return Literal[tuple(ontology.relation_types)]  # type: ignore[valid-type]


# =============================================================================
# New Configuration System (Pydantic Settings)
# =============================================================================


class LLMConfig(BaseModel):
    """Configuration for Language Models."""

    model: str = Field(default="gpt-4o-mini", description="OpenAI model identifier")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    api_key: SecretStr | None = Field(
        default=None, description="OpenAI API Key provided implicitly via env"
    )
    api_base: str | None = Field(default=None, description="Custom API base URL")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API key format for OpenAI compatibility."""
        if v is None:
            return v
        key_value = v.get_secret_value()
        if not key_value:
            raise ValueError("API key cannot be empty if provided")
        if not key_value.startswith("sk-"):
            raise ValueError(
                "OpenAI API key must start with 'sk-'. "
                "Please check your OPENAI_API_KEY or GRAPHRAG_LLM__API_KEY environment variable."
            )
        return v

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str | None) -> str | None:
        """Validate API base URL structure."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"api_base must be a valid URL starting with 'http://' or 'https://'. Got: '{v}'"
            )
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for Embedding Models."""

    model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier",
    )
    dimensions: int | None = Field(default=1536, description="Embedding dimensions")


class IngestionConfig(BaseModel):
    """Configuration for the Ingestion Pipeline."""

    max_triplets_per_chunk: int = Field(
        default=10, description="Max triplets to extract per chunk"
    )
    num_workers: int = Field(default=4, description="Parallel workers for extraction")
    normalize_entities: bool = Field(
        default=True, description="Whether to normalize entities to title case"
    )


class Settings(BaseSettings):
    """
    Main Application Settings.

    Loads from environment variables with the prefix 'GRAPHRAG_'.
    Example: GRAPHRAG_LLM__MODEL="gpt-4" will override settings.llm.model
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAPHRAG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


# Singleton settings instance
settings = Settings()


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAG Configuration")
    print("=" * 60)

    # Print settings (secrets are automatically obscured)
    print("\n[LLM Settings]")
    print(f"  Model: {settings.llm.model}")
    print(f"  Temperature: {settings.llm.temperature}")
    print(f"  API Base: {settings.llm.api_base}")

    print("\n[Embedding Settings]")
    print(f"  Model: {settings.embedding.model}")

    print("\n[Ingestion Settings]")
    print(f"  Max Triplets/Chunk: {settings.ingestion.max_triplets_per_chunk}")
    print(f"  Workers: {settings.ingestion.num_workers}")
    print(f"  Normalize Entities: {settings.ingestion.normalize_entities}")

    try:
        config = get_ontology()
        print(f"\n[Ontology] Loaded: {config.domain} (v{config.version})")
    except OntologyConfigError as e:
        print(f"\n[Ontology] Not loaded: {e}")

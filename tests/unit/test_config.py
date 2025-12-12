"""
Unit tests for src/config.py configuration loading.
"""

import pytest
from pathlib import Path

from src.config import (
    load_ontology,
    OntologyConfig,
    OntologyConfigError,
)


class TestLoadOntology:
    """Test suite for the load_ontology function."""

    def test_load_valid_yaml(self, tmp_yaml_file: Path):
        """Test loading a valid ontology YAML file."""
        config = load_ontology(tmp_yaml_file)

        assert isinstance(config, OntologyConfig)
        assert config.domain == "Test Domain"
        assert config.version == "1.0"
        assert "ENTITY_A" in config.entity_types
        assert "RELATES_TO" in config.relation_types

    def test_file_not_found(self, tmp_path: Path):
        """Test error when YAML file doesn't exist."""
        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(OntologyConfigError) as exc_info:
            load_ontology(missing_file)

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_yaml_syntax(self, tmp_path: Path, invalid_yaml_syntax: str):
        """Test error when YAML has syntax errors."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(invalid_yaml_syntax)

        with pytest.raises(OntologyConfigError) as exc_info:
            load_ontology(yaml_file)

        assert "yaml" in str(exc_info.value).lower()

    def test_empty_yaml_file(self, tmp_path: Path):
        """Test error when YAML file is empty."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(OntologyConfigError) as exc_info:
            load_ontology(yaml_file)

        assert "empty" in str(exc_info.value).lower()

    def test_missing_required_field(self, tmp_path: Path):
        """Test error when required field is missing."""
        yaml_content = """
domain: "Test"
version: "1.0"
entity_types:
  - ENTITY_A
# Missing relation_types and validation_schema
"""
        yaml_file = tmp_path / "incomplete.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(OntologyConfigError):
            load_ontology(yaml_file)

    def test_empty_entity_types(self, tmp_path: Path):
        """Test error when entity_types list is empty."""
        yaml_content = """
domain: "Test"
version: "1.0"
entity_types: []
relation_types:
  - RELATES_TO
validation_schema:
  ENTITY_A:
    - RELATES_TO
"""
        yaml_file = tmp_path / "empty_entities.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(OntologyConfigError) as exc_info:
            load_ontology(yaml_file)

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_invalid_schema_reference(self, tmp_path: Path):
        """Test error when validation_schema references undefined type."""
        yaml_content = """
domain: "Test"
version: "1.0"
entity_types:
  - ENTITY_A
relation_types:
  - RELATES_TO
validation_schema:
  UNDEFINED_ENTITY:
    - RELATES_TO
"""
        yaml_file = tmp_path / "invalid_schema.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(OntologyConfigError) as exc_info:
            load_ontology(yaml_file)

        assert "undefined" in str(exc_info.value).lower()


class TestOntologyConfigValidation:
    """Test suite for OntologyConfig Pydantic validation."""

    def test_uppercase_normalization(self):
        """Test that entity and relation types are uppercased."""
        config = OntologyConfig(
            domain="Test",
            version="1.0",
            entity_types=["entity_a", "Entity_B"],
            relation_types=["relates_to"],
            validation_schema={"ENTITY_A": ["RELATES_TO"]},
        )

        assert config.entity_types == ["ENTITY_A", "ENTITY_B"]
        assert config.relation_types == ["RELATES_TO"]

    def test_immutable_config(self):
        """Test that config is immutable (frozen)."""
        config = OntologyConfig(
            domain="Test",
            version="1.0",
            entity_types=["ENTITY_A"],
            relation_types=["RELATES_TO"],
            validation_schema={"ENTITY_A": ["RELATES_TO"]},
        )

        with pytest.raises(Exception):  # ValidationError for frozen model
            config.domain = "Modified"

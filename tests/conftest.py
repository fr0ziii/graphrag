"""
Shared pytest fixtures for GraphRAG tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_yaml_content() -> str:
    """Valid ontology YAML content for testing."""
    return '''
domain: "Test Domain"
version: "1.0"
entity_types:
  - ENTITY_A
  - ENTITY_B
relation_types:
  - RELATES_TO
  - CONNECTS
validation_schema:
  ENTITY_A:
    - RELATES_TO
  ENTITY_B:
    - CONNECTS
'''


@pytest.fixture
def invalid_yaml_syntax() -> str:
    """YAML with syntax errors."""
    return '''
domain: "Test"
entity_types:
  - ENTITY_A
  - [invalid yaml here
'''


@pytest.fixture
def tmp_yaml_file(tmp_path: Path, sample_yaml_content: str) -> Path:
    """Create a temporary valid YAML file."""
    yaml_file = tmp_path / "test_ontology.yaml"
    yaml_file.write_text(sample_yaml_content)
    return yaml_file

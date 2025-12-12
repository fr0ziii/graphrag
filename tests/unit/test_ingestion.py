"""
Unit tests for src/ingestion.py normalize_text function.
"""

import pytest

from src.ingestion import normalize_text


class TestNormalizeText:
    """Test suite for the normalize_text function."""

    def test_basic_normalization(self):
        """Test basic string normalization to title case."""
        assert normalize_text("solar energy") == "Solar Energy"
        assert normalize_text("WIND POWER") == "Wind Power"
        assert normalize_text("hydroelectric dam") == "Hydroelectric Dam"

    def test_whitespace_collapse(self):
        """Test that multiple whitespace is collapsed to single space."""
        assert normalize_text("solar   energy") == "Solar Energy"
        assert normalize_text("wind\t\tpower") == "Wind Power"
        assert normalize_text("hydro\n\nelectric") == "Hydro Electric"

    def test_whitespace_strip(self):
        """Test leading and trailing whitespace is stripped."""
        assert normalize_text("  solar energy  ") == "Solar Energy"
        assert normalize_text("\twind power\n") == "Wind Power"

    def test_empty_string(self):
        """Test empty string returns empty string."""
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        """Test whitespace-only string returns empty string."""
        assert normalize_text("   ") == ""
        assert normalize_text("\t\n") == ""

    def test_already_normalized(self):
        """Test already normalized text is unchanged."""
        assert normalize_text("Solar Energy") == "Solar Energy"

    def test_single_word(self):
        """Test single word normalization."""
        assert normalize_text("technology") == "Technology"
        assert normalize_text("TECHNOLOGY") == "Technology"

    def test_mixed_case_preservation(self):
        """Test title case conversion handles mixed input."""
        assert normalize_text("sOlAr EnErGy") == "Solar Energy"

    def test_special_characters(self):
        """Test text with special characters."""
        assert normalize_text("solar-energy") == "Solar-Energy"
        assert normalize_text("wind_power") == "Wind_Power"


class TestComputeDocumentHash:
    """Test suite for the compute_document_hash function."""

    def test_deterministic_output(self):
        """Test that same input always produces same hash."""
        from src.ingestion import compute_document_hash

        text = "This is a sample document about renewable energy."
        hash1 = compute_document_hash(text)
        hash2 = compute_document_hash(text)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Test that different inputs produce different hashes."""
        from src.ingestion import compute_document_hash

        hash1 = compute_document_hash("Document A")
        hash2 = compute_document_hash("Document B")
        assert hash1 != hash2

    def test_hash_format(self):
        """Test that output is a valid SHA-256 hex string."""
        from src.ingestion import compute_document_hash

        result = compute_document_hash("test")
        # SHA-256 produces 64 character hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        """Test hashing empty string produces consistent result."""
        from src.ingestion import compute_document_hash

        result = compute_document_hash("")
        # Known SHA-256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_unicode_handling(self):
        """Test that unicode characters are handled correctly."""
        from src.ingestion import compute_document_hash

        # Text with unicode characters
        text = "Energía renovable y tecnología 日本語"
        hash_result = compute_document_hash(text)
        # Should produce valid hash without error
        assert len(hash_result) == 64

    def test_whitespace_sensitivity(self):
        """Test that different whitespace produces different hashes."""
        from src.ingestion import compute_document_hash

        hash1 = compute_document_hash("renewable energy")
        hash2 = compute_document_hash("renewable  energy")
        hash3 = compute_document_hash(" renewable energy ")
        # All should be different since content differs
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

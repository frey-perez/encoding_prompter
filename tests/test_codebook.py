"""Tests for the codebook module."""

import json
import tempfile
from pathlib import Path

import pytest

from encoding_prompter.codebook import Codebook, Construct


def test_construct_creation() -> None:
    """Test basic construct creation."""
    construct = Construct(
        name="Test Construct",
        definition="A test definition",
        examples=["Example 1", "Example 2"],
    )

    assert construct.name == "Test Construct"
    assert construct.definition == "A test definition"
    assert len(construct.examples) == 2


def test_construct_to_string() -> None:
    """Test string representation of construct."""
    construct = Construct(
        name="Emotional Awareness",
        definition="Recognition of emotions",
        examples=["I felt sad", "I noticed anxiety"],
    )

    result = construct.to_string()

    assert "Construct: Emotional Awareness" in result
    assert "Definition: Recognition of emotions" in result
    assert "Examples:" in result
    assert "I felt sad" in result


def test_construct_to_string_no_examples() -> None:
    """Test string representation without examples."""
    construct = Construct(
        name="Simple Construct",
        definition="A definition",
    )

    result = construct.to_string()

    assert "Examples:" not in result


def test_codebook_from_json() -> None:
    """Test loading codebook from JSON file."""
    data = {
        "constructs": [
            {
                "name": "Construct A",
                "definition": "Definition A",
                "examples": ["Ex 1", "Ex 2"],
            },
            {
                "name": "Construct B",
                "definition": "Definition B",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        filepath = f.name

    try:
        codebook = Codebook.from_file(filepath)

        assert len(codebook) == 2
        assert codebook.constructs[0].name == "Construct A"
        assert codebook.constructs[1].name == "Construct B"
    finally:
        Path(filepath).unlink()


def test_codebook_from_json_list_format() -> None:
    """Test loading codebook from JSON list format."""
    data = [
        {"name": "Construct A", "definition": "Definition A"},
        {"name": "Construct B", "definition": "Definition B"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        filepath = f.name

    try:
        codebook = Codebook.from_file(filepath)
        assert len(codebook) == 2
    finally:
        Path(filepath).unlink()


def test_codebook_from_csv() -> None:
    """Test loading codebook from CSV file."""
    csv_content = """name,definition,examples
                    Construct A,Definition A,Ex1; Ex2
                    Construct B,Definition B,
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        filepath = f.name

    try:
        codebook = Codebook.from_file(filepath)

        assert len(codebook) == 2
        assert codebook.constructs[0].name == "Construct A"
        assert len(codebook.constructs[0].examples) == 2
    finally:
        Path(filepath).unlink()


def test_codebook_from_txt_structured() -> None:
    """Test loading codebook from structured TXT format."""
    txt_content = """CONSTRUCT: Emotional Awareness
                    DEFINITION: Recognition of emotional states
                    EXAMPLES: I felt sad; I noticed anxiety

                    CONSTRUCT: Self-Reflection
                    DEFINITION: Examining one's thoughts
                    EXAMPLES: I realized that; Looking back
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(txt_content)
        filepath = f.name

    try:
        codebook = Codebook.from_file(filepath)

        assert len(codebook) == 2
        assert codebook.constructs[0].name == "Emotional Awareness"
        assert len(codebook.constructs[0].examples) == 2
    finally:
        Path(filepath).unlink()


def test_codebook_file_not_found() -> None:
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Codebook.from_file("nonexistent_file.json")


def test_codebook_unsupported_format() -> None:
    """Test error for unsupported file format."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported codebook format"):
            Codebook.from_file(filepath)
    finally:
        Path(filepath).unlink()


def test_codebook_iteration() -> None:
    """Test iterating over codebook constructs."""
    constructs = [
        Construct("A", "Def A"),
        Construct("B", "Def B"),
    ]
    codebook = Codebook(constructs)

    names = [c.name for c in codebook]

    assert names == ["A", "B"]

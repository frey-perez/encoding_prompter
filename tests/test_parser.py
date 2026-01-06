"""Tests for the parser module."""

import pandas as pd

from encoding_prompter.parser import (
    EncodingInstance,
    instances_to_dataframe,
    merge_results,
    parse_response,
)


def test_parse_single_instance() -> None:
    """Test parsing a single construct instance."""
    response = """DOC_ID: interview1.txt
SPEAKER_ID: PA-001
CONSTRUCT: Emotional Awareness
QUOTE: I felt really anxious about the situation
CONFIDENCE: 2
"""

    instances = parse_response(response)

    assert len(instances) == 1
    inst = instances[0]
    assert inst.doc_id == "interview1.txt"
    assert inst.speaker_id == "PA-001"
    assert inst.construct == "Emotional Awareness"
    assert "anxious" in inst.quote
    assert inst.confidence == 2


def test_parse_multiple_instances() -> None:
    """Test parsing multiple construct instances."""
    response = """DOC_ID: interview1.txt
SPEAKER_ID: PA-001
CONSTRUCT: Emotional Awareness
QUOTE: I felt anxious
CONFIDENCE: 2

DOC_ID: interview1.txt
SPEAKER_ID: PA-001
CONSTRUCT: Self-Reflection
QUOTE: I realized that I need to change
CONFIDENCE: 1
"""

    instances = parse_response(response)

    assert len(instances) == 2
    assert instances[0].construct == "Emotional Awareness"
    assert instances[1].construct == "Self-Reflection"


def test_parse_with_default_doc_id() -> None:
    """Test using default doc_id when not in response."""
    response = """SPEAKER_ID: PA-001
CONSTRUCT: Emotional Awareness
QUOTE: I felt anxious
CONFIDENCE: 2
"""

    instances = parse_response(response, default_doc_id="default_doc")

    assert len(instances) == 1


def test_parse_alternative_format() -> None:
    """Test parsing with space instead of underscore."""
    response = """DOC ID: interview1.txt
SPEAKER ID: PA-001
CONSTRUCT: Emotional Awareness
QUOTE: I felt anxious
CONFIDENCE: 2
"""

    instances = parse_response(response)

    assert len(instances) == 1
    assert instances[0].doc_id == "interview1.txt"


def test_parse_empty_response() -> None:
    """Test parsing an empty response."""
    instances = parse_response("")

    assert len(instances) == 0


def test_parse_invalid_confidence() -> None:
    """Test handling invalid confidence value."""
    response = """DOC_ID: test.txt
SPEAKER_ID: PA-001
CONSTRUCT: Test
QUOTE: Test quote
CONFIDENCE: invalid
"""

    instances = parse_response(response)

    assert len(instances) == 1
    assert instances[0].confidence == -1


def test_parse_multiline_quote() -> None:
    """Test parsing quotes that span multiple lines."""
    response = """DOC_ID: test.txt
SPEAKER_ID: PA-001
CONSTRUCT: Emotional Awareness
QUOTE: I felt really anxious about the whole situation and didn't know what to do
CONFIDENCE: 2
"""

    instances = parse_response(response)

    assert len(instances) == 1
    assert "anxious" in instances[0].quote


def test_basic_conversion() -> None:
    """Test converting instances to DataFrame."""
    instances = [
        EncodingInstance(
            doc_id="doc1",
            speaker_id="PA-001",
            construct="Construct A",
            quote="Quote 1",
            confidence=2,
        ),
        EncodingInstance(
            doc_id="doc1",
            speaker_id="PA-001",
            construct="Construct B",
            quote="Quote 2",
            confidence=1,
        ),
    ]

    df = instances_to_dataframe(instances)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "doc_id",
        "speaker_id",
        "construct",
        "quote",
        "confidence",
    ]
    assert df.iloc[0]["construct"] == "Construct A"
    assert df.iloc[1]["construct"] == "Construct B"


def test_empty_instances() -> None:
    """Test handling empty instance list."""
    df = instances_to_dataframe([])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == [
        "doc_id",
        "speaker_id",
        "construct",
        "quote",
        "confidence",
    ]


def test_merge_multiple_dataframes() -> None:
    """Test merging multiple result DataFrames."""
    df1 = pd.DataFrame(
        {
            "doc_id": ["doc1"],
            "speaker_id": ["PA-001"],
            "construct": ["A"],
            "quote": ["Quote 1"],
            "confidence": [2],
        }
    )
    df2 = pd.DataFrame(
        {
            "doc_id": ["doc2"],
            "speaker_id": ["PA-002"],
            "construct": ["B"],
            "quote": ["Quote 2"],
            "confidence": [1],
        }
    )

    merged = merge_results([df1, df2])

    assert len(merged) == 2
    assert merged.iloc[0]["doc_id"] == "doc1"
    assert merged.iloc[1]["doc_id"] == "doc2"

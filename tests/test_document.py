"""Tests for the document module."""

import tempfile
from pathlib import Path

import pytest

from encoding_prompter.document import Document, DocumentLoader


def test_document_creation() -> None:
    """Test basic document creation."""
    doc = Document(
        doc_id="test_doc",
        content="Test content here",
        speakers=["SPEAKER-001"],
        source_path="/path/to/file.txt",
    )

    assert doc.doc_id == "test_doc"
    assert "Test content" in doc.content
    assert len(doc.speakers) == 1


def test_get_text_with_speakers() -> None:
    """Test getting content with speaker info."""
    doc = Document(
        doc_id="test",
        content="SPEAKER-001: Hello\nSPEAKER-002: Hi",
        speakers=["SPEAKER-001", "SPEAKER-002"],
        source_path="<string>",
    )

    result = doc.get_text_with_speakers()

    assert "SPEAKER-001" in result
    assert "SPEAKER-002" in result


def test_load_txt_file() -> None:
    """Test loading a TXT file."""
    content = """SPEAKERS
                TH-001, PA-001

                TH-001  
                How are you feeling today?

                PA-001  
                I'm feeling quite anxious about the situation.
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        filepath = f.name

    try:
        documents = DocumentLoader.load(filepath)

        assert len(documents) == 1
        doc = documents[0]
        assert "TH-001" in doc.speakers
        assert "PA-001" in doc.speakers
        assert "anxious" in doc.content
    finally:
        Path(filepath).unlink()


def test_load_csv_with_headers() -> None:
    """Test loading a CSV file with speaker/text columns."""
    csv_content = """speaker,text
                    TH-001,How are you feeling?
                    PA-001,I'm feeling anxious.
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        filepath = f.name

    try:
        documents = DocumentLoader.load(filepath)

        assert len(documents) == 1
        doc = documents[0]
        assert "TH-001" in doc.content or "TH-001" in doc.speakers
        assert "anxious" in doc.content
    finally:
        Path(filepath).unlink()


def test_load_directory() -> None:
    """Test loading all documents from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "doc1.txt").write_text("Content 1")
        (Path(tmpdir) / "doc2.txt").write_text("Content 2")
        (Path(tmpdir) / "ignored.pdf").write_text("Should be ignored")

        documents = DocumentLoader.load(tmpdir)

        assert len(documents) == 2
        doc_ids = {d.doc_id for d in documents}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids


def test_load_from_string() -> None:
    """Test creating a document from a string."""
    text = """TH-001
            Tell me about yourself.

            PA-001
            I've been feeling anxious lately.
"""

    doc = DocumentLoader.load_from_string(text, doc_id="inline_test")

    assert doc.doc_id == "inline_test"
    assert "TH-001" in doc.speakers
    assert "PA-001" in doc.speakers
    assert doc.source_path == "<string>"


def test_extract_speakers_pattern_1() -> None:
    """Test speaker extraction with pattern 'XX-NNN  '."""
    content = """TH-001
                Hello

                PA-001
                Hi there
"""

    speakers = DocumentLoader._extract_speakers(content)

    assert "TH-001" in speakers
    assert "PA-001" in speakers


def test_extract_speakers_pattern_2() -> None:
    """Test speaker extraction with 'SPEAKER:' pattern."""
    content = """Interviewer: Hello
Participant: Hi there
"""

    speakers = DocumentLoader._extract_speakers(content)

    assert "Interviewer" in speakers
    assert "Participant" in speakers


def test_extract_speakers_from_header() -> None:
    """Test speaker extraction from SPEAKERS section."""
    content = """SPEAKERS
                Alice, Bob, Charlie

                Alice  
                Hello everyone
"""

    speakers = DocumentLoader._extract_speakers(content)

    assert "Alice" in speakers
    assert "Bob" in speakers
    assert "Charlie" in speakers


def test_file_not_found() -> None:
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load("nonexistent_file.txt")


def test_unsupported_format() -> None:
    """Test error for unsupported file format."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported document format"):
            DocumentLoader.load(filepath)
    finally:
        Path(filepath).unlink()


def test_empty_directory() -> None:
    """Test error when directory has no valid documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only unsupported files
        (Path(tmpdir) / "ignored.pdf").write_text("PDF content")

        with pytest.raises(ValueError, match="No valid documents found"):
            DocumentLoader.load(tmpdir)

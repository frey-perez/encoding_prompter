"""Document loading utilities for interview text files.

This module provides the DocumentLoader class and Document dataclass for
loading interview documents from various file formats (TXT, CSV) and
directories.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents a loaded interview document to be used with prompt.

    Attributes
    ----------
        doc_id: Unique identifier for the document (typically filename).
        content: The full text content of the document.
        speakers: List of speaker IDs found in the document.
        source_path: Path to the source file.

    """

    doc_id: str
    content: str
    speakers: list[str]
    source_path: str

    def get_text_with_speakers(self) -> str:
        """Return the document content with speaker information preserved.

        Returns
        -------
            The full document content as a string.

        """
        return self.content


class DocumentLoader:
    """Loads interview documents from files or directories.

    Handles loading documents from TXT and CSV files, with automatic
    speaker detection for interview transcripts.
    """

    # Common patterns for speaker identification in transcripts
    SPEAKER_PATTERNS = [
        # Pattern: "SPEAKER_ID  " or "SPEAKER_ID\t" at start of line
        r"^([A-Z]{2,}-\d{3})\s{2,}",
        # Pattern: "SPEAKER:" at start of line
        r"^([A-Za-z0-9_-]+):\s*",
        # Pattern: "[SPEAKER]" at start of line
        r"^\[([A-Za-z0-9_-]+)\]\s*",
    ]

    @classmethod
    def load(cls, path: str | Path) -> list[Document]:
        """Load documents from a file or directory.

        Args:
        ----
            path: Path to a single file or directory containing documents.

        Returns:
        -------
            List of Document objects.

        Raises:
        ------
            FileNotFoundError: If the path does not exist.
            ValueError: If no valid documents are found.

        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_dir():
            return cls._load_directory(path)
        else:
            return [cls._load_file(path)]

    @classmethod
    def _load_directory(cls, directory: Path) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
        ----
            directory: Path to the directory.

        Returns:
        -------
            List of Document objects.

        """
        documents = []
        supported_extensions = {".txt", ".csv"}

        for filepath in sorted(directory.iterdir()):
            if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
                try:
                    documents.append(cls._load_file(filepath))
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")

        if not documents:
            raise ValueError(
                f"No valid documents found in directory: {directory}. "
                "Supported formats: .txt, .csv"
            )

        return documents

    @classmethod
    def _load_file(cls, filepath: Path) -> Document:
        """Load a single document from a file.

        Args:
        ----
            filepath: Path to the file.

        Returns:
        -------
            A Document object.

        Raises:
        ------
            ValueError: If the file format is not supported.


        """
        extension = filepath.suffix.lower()

        if extension == ".txt":
            return cls._load_txt(filepath)
        elif extension == ".csv":
            return cls._load_csv(filepath)
        else:
            raise ValueError(
                f"Unsupported document format: {extension}. "
                "Supported formats: .txt, .csv"
            )

    @classmethod
    def _load_txt(cls, filepath: Path) -> Document:
        """Load a document from a TXT file.

        Handles interview transcript format with speaker labels.

        Args:
        ----
            filepath: Path to the txt file.

        Returns:
        -------
            A Document object.

        """
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        doc_id = filepath.stem
        speakers = cls._extract_speakers(content)

        return Document(
            doc_id=doc_id,
            content=content,
            speakers=speakers,
            source_path=str(filepath),
        )

    @classmethod
    def _load_csv(cls, filepath: Path) -> Document:
        """Load a document from a CSV file.

        Expected CSV format options:
        1. Columns: speaker, text (or similar)
        2. Single column with transcript text

        Args:
        ----
            filepath: Path to the CSV file.

        Returns:
        -------
            A Document object.

        """
        rows = []
        speakers = set()

        with open(filepath, encoding="utf-8", newline="") as f:
            sample = f.read(2048)
            f.seek(0)

            if any(
                term in sample.lower()
                for term in ["speaker", "text", "content", "utterance"]
            ):
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty")

                fieldnames_lower = {fn.lower().strip(): fn for fn in reader.fieldnames}

                speaker_col = None
                text_col = None

                for key in fieldnames_lower:
                    if key in ["speaker", "speaker_id", "participant", "id"]:
                        speaker_col = fieldnames_lower[key]
                    elif key in [
                        "text",
                        "content",
                        "utterance",
                        "transcript",
                        "message",
                    ]:
                        text_col = fieldnames_lower[key]

                if text_col is None:
                    text_col = reader.fieldnames[0]

                for row in reader:
                    if speaker_col and row.get(speaker_col):
                        speaker = row[speaker_col].strip()
                        speakers.add(speaker)
                        rows.append(f"{speaker}: {row.get(text_col, '').strip()}")
                    else:
                        rows.append(row.get(text_col, "").strip())

                content = "\n".join(rows)
            else:
                content = sample + f.read()

        if not speakers:
            speakers = set(cls._extract_speakers(content))

        return Document(
            doc_id=filepath.stem,
            content=content,
            speakers=list(speakers),
            source_path=str(filepath),
        )

    @classmethod
    def _extract_speakers(cls, content: str) -> list[str]:
        """Extract speaker IDs from transcript content.

        Args:
        ----
            content: The transcript text content.

        Returns:
        -------
            List of unique speaker IDs found in the content.

        """
        speakers = set()

        # looking for speakers in header
        speakers_match = re.search(r"SPEAKERS\s*\n([^\n]+)", content)
        if speakers_match:
            # parse speaker list (comma or space separated)
            speaker_line = speakers_match.group(1)
            for speaker in re.split(r"[,\s]+", speaker_line):
                speaker = speaker.strip()
                if speaker and re.match(r"^[A-Za-z0-9_-]+$", speaker):
                    speakers.add(speaker)

        for pattern in cls.SPEAKER_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                speakers.add(match.group(1))

        return sorted(speakers)

    @classmethod
    def load_from_string(cls, text: str, doc_id: str = "inline") -> Document:
        """Create a Document from a string.

        Args:
        ----
            text: The document text content.
            doc_id: Identifier for the document.

        Returns:
        -------
            A Document object.

        """
        speakers = cls._extract_speakers(text)
        return Document(
            doc_id=doc_id,
            content=text,
            speakers=speakers,
            source_path="<string>",
        )

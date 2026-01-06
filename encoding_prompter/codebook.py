"""Codebook loading and parsing utilities.

This module provides the Codebook class for loading structured codebooks from
various file formats (txt, CSV, JSON) that define psychological constructs
with their definitions and examples.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Construct:
    """Represents a single psychological construct from a codebook.

    Attributes
    ----------
        name: The name of the construct.
        definition: A description or definition of the construct.
        examples: Optional list of example instances of the construct.

    """

    name: str
    definition: str
    examples: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert the construct to a formatted string for prompt inclusion.

        Returns
        -------
            A formatted string representation of the construct.

        """
        result = f"Construct: {self.name}\nDefinition: {self.definition}"
        if self.examples:
            examples_str = "\n  - ".join(self.examples)
            result += f"\nExamples:\n  - {examples_str}"
        return result


class Codebook:
    """A collection of psychological constructs loaded from a file.

    The Codebook class handles loading construct definitions from various
    file formats and provides methods for converting them into strings that
    can be incorporated into prommpts.

    Attributes
    ----------
        constructs: List of Construct objects loaded from the codebook file.
        source_path: Path to the source file the codebook was loaded from.

    """

    def __init__(
        self, constructs: list[Construct], source_path: str | None = None
    ) -> None:
        """Initialize a Codebook with a list of constructs.

        Args:
        ----
            constructs: List of Construct objects.
            source_path: Optional path to the source file.

        """
        self.constructs = constructs
        self.source_path = source_path

    @classmethod
    def from_file(cls, filepath: str | Path) -> "Codebook":
        """Load a codebook from a file.

        Automatically detects the file format based on extension and parses
        accordingly.

        Args:
        ----
            filepath: Path to the codebook file (.txt, .csv, or .json).

        Returns:
        -------
            A Codebook instance populated with constructs from the file.

        Raises:
        ------
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.

        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Codebook file not found: {filepath}")

        extension = filepath.suffix.lower()

        if extension == ".json":
            return cls._from_json(filepath)
        elif extension == ".csv":
            return cls._from_csv(filepath)
        elif extension == ".txt":
            return cls._from_txt(filepath)
        else:
            raise ValueError(
                f"Unsupported codebook format: {extension}. "
                "Supported formats: .json, .csv, .txt"
            )

    @classmethod
    def _from_json(cls, filepath: Path) -> "Codebook":
        """Load a codebook from a JSON file.

        Args:
        ----
            filepath: Path to the JSON file.

        Returns:
        -------
            A Codebook instance.

        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        constructs = []

        if isinstance(data, dict) and "constructs" in data:
            construct_list = data["constructs"]
            for item in construct_list:
                constructs.append(
                    Construct(
                        name=item["name"],
                        definition=item.get("definition", ""),
                        examples=item.get("examples", []),
                    )
                )
        elif isinstance(data, list):
            for item in data:
                constructs.append(
                    Construct(
                        name=item["name"],
                        definition=item.get("definition", ""),
                        examples=item.get("examples", []),
                    )
                )
        elif isinstance(data, dict):
            for name, details in data.items():
                constructs.append(
                    Construct(
                        name=name,
                        definition=details.get("definition", ""),
                        examples=details.get("examples", []),
                    )
                )
        else:
            raise ValueError(
                "JSON codebook format not recognized. Supported formats:\n"
                "1. {'constructs': [{'name': '...', 'definition': '...'}]}\n"
                "2. [{'name': '...', 'definition': '...'}]\n"
                "3. {'construct_name': {'definition': '...', 'examples': [...]}}"
            )

        return cls(constructs, source_path=str(filepath))

    @classmethod
    def _from_csv(cls, filepath: Path) -> "Codebook":
        """Load a codebook from a CSV file.

        Args:
        ----
            filepath: Path to the CSV file.

        Returns:
        -------
            A Codebook instance.

        """
        constructs = []

        with open(filepath, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty")

            for row in reader:
                row_lower = {k.lower().strip(): v for k, v in row.items()}

                name = row_lower.get("name", row_lower.get("construct", ""))
                definition = row_lower.get(
                    "definition", row_lower.get("description", "")
                )
                examples_str = row_lower.get("examples", row_lower.get("example", ""))

                if examples_str:
                    examples = [e.strip() for e in examples_str.split(";") if e.strip()]
                else:
                    examples = []

                if name:
                    constructs.append(
                        Construct(
                            name=name.strip(),
                            definition=definition.strip(),
                            examples=examples,
                        )
                    )

        return cls(constructs, source_path=str(filepath))

    @classmethod
    def _from_txt(cls, filepath: Path) -> "Codebook":
        """Load a codebook from a txt file.

        Args:
        ----
            filepath: Path to the txt file.

        Returns:
        -------
            A Codebook instance.

        """
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        constructs = []

        if "CONSTRUCT:" in content.upper():
            constructs = cls._parse_structured_txt(content)
        else:
            constructs = cls._parse_simple_txt(content)

        return cls(constructs, source_path=str(filepath))

    @staticmethod
    def _parse_structured_txt(content: str) -> list[Construct]:
        """Parse structured txt format with CONSTRUCT:/DEFINITION:/EXAMPLES: markers."""
        constructs = []
        current_name = None
        current_definition = ""
        current_examples = []

        for line in content.split("\n"):
            line_upper = line.upper().strip()
            line_stripped = line.strip()

            if line_upper.startswith("CONSTRUCT:"):
                if current_name:
                    constructs.append(
                        Construct(
                            name=current_name,
                            definition=current_definition.strip(),
                            examples=current_examples,
                        )
                    )

                current_name = line_stripped[len("CONSTRUCT:") :].strip()
                current_definition = ""
                current_examples = []

            elif line_upper.startswith("DEFINITION:"):
                current_definition = line_stripped[len("DEFINITION:") :].strip()

            elif line_upper.startswith("EXAMPLES:"):
                examples_str = line_stripped[len("EXAMPLES:") :].strip()
                current_examples = [
                    e.strip() for e in examples_str.split(";") if e.strip()
                ]

        if current_name:
            constructs.append(
                Construct(
                    name=current_name,
                    definition=current_definition.strip(),
                    examples=current_examples,
                )
            )

        return constructs

    @staticmethod
    def _parse_simple_txt(content: str) -> list[Construct]:
        """Parse simple txt format with blank line separators."""
        constructs = []
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if len(lines) >= 1:
                name = lines[0]
                definition = " ".join(lines[1:]) if len(lines) > 1 else ""
                constructs.append(Construct(name=name, definition=definition))

        return constructs

    def to_string(self) -> str:
        """Convert the entire codebook to a formatted string for prompts.

        Returns
        -------
            A formatted string containing all constructs.

        """
        construct_strings = [c.to_string() for c in self.constructs]
        return "\n\n".join(construct_strings)

    def __len__(self) -> int:
        """Return the number of constructs in the codebook."""
        return len(self.constructs)

    def __iter__(self):  # noqa: ANN204
        """Iterate over constructs in the codebook."""
        return iter(self.constructs)

    def __repr__(self) -> str:
        """Return a string representation of the codebook."""
        return (
            f"Codebook(constructs={len(self.constructs)}, source='{self.source_path}')"
        )

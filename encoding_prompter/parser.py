"""Response parsing utilities for LLM output.

This module provides functions for parsing structured LLM responses into
pandas DataFrames with extracted construct instances.
"""

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class EncodingInstance:
    """Represents a single identified construct instance.

    Attributes
    ----------
        doc_id: Document identifier.
        speaker_id: Speaker identifier (if applicable).
        construct: Name of the identified construct.
        quote: Exact quote from the text.
        confidence: Confidence/ordinal score (0, 1, or 2).

    """

    doc_id: str
    speaker_id: str
    construct: str
    quote: str
    confidence: int


def parse_response(
    response_text: str, default_doc_id: str = ""
) -> list[EncodingInstance]:
    """Parse an LLM response into a list of EncodingInstance objects.

    Args:
    ----
        response_text: The raw text response from the LLM.
        default_doc_id: Document ID to use for all instances. This overrides
            any doc_id returned by the LLM to ensure accuracy.

    Returns:
    -------
        List of EncodingInstance objects extracted from the response.

    """
    instances = []

    # Split response into individual instance blocks
    # Each block should have DOC_ID, SPEAKER_ID, CONSTRUCT, QUOTE, CONFIDENCE
    blocks = re.split(r"\n(?=DOC_ID:|DOC ID:)", response_text, flags=re.IGNORECASE)

    for block in blocks:
        if not block.strip():
            continue

        instance = _parse_block(block, default_doc_id)
        if instance and instance.construct:
            # Always override with the actual doc_id we know is correct
            if default_doc_id:
                instance.doc_id = default_doc_id
            instances.append(instance)

    return instances


def _parse_block(block: str, default_doc_id: str) -> EncodingInstance | None:
    """Parse a single instance block from the response.

    Args:
    ----
        block: Text block containing one construct instance.
        default_doc_id: Default document ID to use if not found.

    Returns:
    -------
        EncodingInstance or None if parsing fails.

    """
    patterns = {
        "doc_id": r"DOC[_\s]?ID:\s*(.+?)(?=\n|SPEAKER|$)",
        "speaker_id": r"SPEAKER[_\s]?ID:\s*(.+?)(?=\n|CONSTRUCT|$)",
        "construct": r"CONSTRUCT:\s*(.+?)(?=\n|QUOTE|$)",
        "quote": r"QUOTE:\s*(.+?)(?=\n(?:CONFIDENCE|DOC)|$)",
        "confidence": r"CONFIDENCE:\s*(\d+)",
    }

    values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if match:
            values[field] = match.group(1).strip()

    if "construct" not in values:
        return None

    confidence = -1
    if "confidence" in values:
        try:
            confidence = int(values["confidence"])
        except ValueError:
            num_match = re.search(r"\d+", values["confidence"])
            if num_match:
                confidence = int(num_match.group())

    return EncodingInstance(
        doc_id=values.get("doc_id", default_doc_id),
        speaker_id=values.get("speaker_id", ""),
        construct=values.get("construct", ""),
        quote=values.get("quote", ""),
        confidence=confidence,
    )


def instances_to_dataframe(instances: list[EncodingInstance]) -> pd.DataFrame:
    """Convert a list of EncodingInstance objects to a pandas DataFrame for analysis.

    Args:
    ----
        instances: List of EncodingInstance objects.

    Returns:
    -------
        DataFrame with columns: doc_id, speaker_id, construct, quote, confidence

    """
    if not instances:
        return pd.DataFrame(
            columns=["doc_id", "speaker_id", "construct", "quote", "confidence"]
        )

    data = [
        {
            "doc_id": inst.doc_id,
            "speaker_id": inst.speaker_id,
            "construct": inst.construct,
            "quote": inst.quote,
            "confidence": inst.confidence,
        }
        for inst in instances
    ]

    return pd.DataFrame(data)


def merge_results(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple result DataFrames into a single DataFrame.

    Args:
    ----
        dataframes: List of DataFrames to merge.

    Returns:
    -------
        A single merged DataFrame.

    """
    if not dataframes:
        return pd.DataFrame(
            columns=["doc_id", "speaker_id", "construct", "quote", "confidence"]
        )

    return pd.concat(dataframes, ignore_index=True)

"""Prompt templates for psychological construct encoding."""

DEFAULT_PROMPT = """You are analyzing an interview transcript to identify and extract instances of psychological constructs.

Document ID: {doc_id}
Speakers in this document: {speakers}

Text to analyze:
{text}

Codebook of constructs:
{codebook}

Instructions:
1. Identify which constructs from the codebook appear in the text
2. For each construct found, extract ALL instances where it appears
3. For each instance, provide:
   - Document ID (use exactly: {doc_id})
   - Speaker ID (use the EXACT speaker ID from the transcript, e.g., {speakers})
   - The construct name (use the exact name from the codebook)
   - An exact quote from the text
   - Provide an ordinal score (0=construct is not mentioned or is negated, 1 = indirect mention or not clear, 2 = clear and prototypical mention of the construct) as to whether the interview clearly mentions the construct, according to its definition and examples.

Format your response EXACTLY like this:
DOC_ID: {doc_id}
SPEAKER_ID: [use exact speaker ID from transcript]
CONSTRUCT: [construct name]
QUOTE: [exact quote from text]
CONFIDENCE: [score]

(Continue for all instances of all constructs found)

Your response:"""  # noqa: E501


DEFAULT_SCORING_CRITERIA = """Provide an ordinal score (0=construct is not mentioned or is negated, 1 = indirect mention or not clear, 2 = clear and prototypical mention of the construct) as to whether the interview clearly mentions the construct, according to its definition and examples."""  # noqa: E501


def create_custom_prompt(
    base_prompt: str | None = None, scoring_criteria: str | None = None
) -> str:
    """Create a customized prompt with optional scoring criteria replacement.

    Args:
    ----
        base_prompt: Custom base prompt template. If None, uses DEFAULT_PROMPT.
            Must contain {text} and {codebook} placeholders.
        scoring_criteria: Custom scoring criteria string to replace the default
            scoring instructions. Only used if base_prompt is None.

    Returns:
    -------
        The prompt string with any customizations applied.

    Raises:
    ------
        ValueError: If base_prompt is provided but missing required placeholders.

    """
    if base_prompt is not None:
        if "{text}" not in base_prompt or "{codebook}" not in base_prompt:
            raise ValueError(
                "Custom prompt must contain {text} and {codebook} placeholders"
            )
        return base_prompt

    if scoring_criteria is not None:
        return DEFAULT_PROMPT.replace(DEFAULT_SCORING_CRITERIA, scoring_criteria)

    return DEFAULT_PROMPT

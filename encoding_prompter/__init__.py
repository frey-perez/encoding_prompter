"""Encoding Prompter: LLM-based psychological construct encoding from interview text.

This package provides tools for encoding psychological constructs found within
interview text using LLMs and structured codebooks.

Example usage:
    from encoding_prompter import EncodingPrompter

    prompter = EncodingPrompter(api_key="your-api-key")
    results = prompter.encode(
        documents="path/to/interviews/",
        codebook="path/to/codebook.json"
    )
"""


__version__ = "0.1.0"

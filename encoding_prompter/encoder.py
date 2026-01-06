"""Main encoder module for psychological construct encoding.

This module provides the EncodingPrompter class, which is the primary
interface for encoding psychological constructs from interview text
using LLMs and codebooks.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from encoding_prompter.codebook import Codebook
from encoding_prompter.document import Document, DocumentLoader
from encoding_prompter.llm_client import DEFAULT_MODEL, LLMClient, LLMResponse
from encoding_prompter.parser import (
    instances_to_dataframe,
    merge_results,
    parse_response,
)
from encoding_prompter.prompts import create_custom_prompt


class EncodingPrompter:
    """Main interface for LLM-based psychological construct encoding.

    The EncodingPrompter handles the complete pipeline of loading documents
    and codebooks, sending prompts to an LLM, and parsing the results into
    a structured DataFrame.

    Attributes
    ----------
        client: The LLMClient instance for API communication.
        model: The model identifier being used.

    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the EncodingPrompter.

        Args:
        ----
            api_key: OpenRouter API key. If None, will look for
                OPENROUTER_API_KEY environment variable.
            model: Model identifier string for OpenRouter.
                Defaults to a free Llama model.

        """
        self.client = LLMClient(api_key=api_key, model=model)
        self.model = model

    def encode(
        self,
        documents: str | Path | list[Document],
        codebook: str | Path | Codebook,
        prompt_template: str | None = None,
        scoring_criteria: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        show_progress: bool = True,
        on_document_complete: Callable[[str, pd.DataFrame], None] | None = None,
    ) -> pd.DataFrame:
        """Encode psychological constructs from documents using a codebook.

        This method processes one or more documents, sending each to the LLM
        with the codebook to identify construct instances.

        Args:
        ----
            documents: Path to a document file, directory of documents, or
                a list of pre-loaded Document objects.
            codebook: Path to a codebook file or a pre-loaded Codebook object.
            prompt_template: Custom prompt template. Must contain {text} and
                {codebook} placeholders. If None, uses the default prompt.
            scoring_criteria: Custom scoring criteria string to replace just
                the scoring instructions in the default prompt. Only used
                if prompt_template is None.
            max_tokens: Maximum tokens for LLM response.
            temperature: LLM sampling temperature.
            show_progress: Whether to show a progress bar.
            on_document_complete: Optional callback function called after each
                document is processed. Receives (doc_id, results_df).

        Returns:
        -------
            DataFrame with columns:
                - doc_id: Document identifier
                - speaker_id: Speaker identifier (if applicable)
                - construct: Name of the identified construct
                - quote: Exact quote from the text
                - confidence: Ordinal score (0, 1, or 2)

        Raises:
        ------
            FileNotFoundError: If document or codebook paths don't exist.
            ValueError: If documents or codebook format is invalid.

        """
        docs = self._load_documents(documents)

        cb = self._load_codebook(codebook)

        prompt = create_custom_prompt(
            base_prompt=prompt_template,
            scoring_criteria=scoring_criteria,
        )
        all_results = []
        doc_iterator = (
            tqdm(docs, desc="Processing documents") if show_progress else docs
        )

        for doc in doc_iterator:
            result_df = self._process_document(
                document=doc,
                codebook=cb,
                prompt_template=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            all_results.append(result_df)

            if on_document_complete:
                on_document_complete(doc.doc_id, result_df)

        return merge_results(all_results)

    def encode_single(
        self,
        text: str,
        codebook: str | Path | Codebook,
        doc_id: str = "inline",
        prompt_template: str | None = None,
        scoring_criteria: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        """Encode a single text string.

        Convenience method for encoding a single piece of text without
        loading from a file.

        Args:
        ----
            text: The text content to analyze.
            codebook: Path to a codebook file or a pre-loaded Codebook object.
            doc_id: Identifier for this document.
            prompt_template: Custom prompt template.
            scoring_criteria: Custom scoring criteria string.
            max_tokens: Maximum tokens for LLM response.
            temperature: LLM sampling temperature.

        Returns:
        -------
            DataFrame with encoding results.

        """
        doc = DocumentLoader.load_from_string(text, doc_id=doc_id)

        return self.encode(
            documents=[doc],
            codebook=codebook,
            prompt_template=prompt_template,
            scoring_criteria=scoring_criteria,
            max_tokens=max_tokens,
            temperature=temperature,
            show_progress=False,
        )

    def _load_documents(self, documents: str | Path | list[Document]) -> list[Document]:
        """Load documents from various input formats.

        Args:
        ----
            documents: File path, directory path, or list of Documents.

        Returns:
        -------
            List of Document objects.

        """
        if isinstance(documents, list):
            return documents
        return DocumentLoader.load(documents)

    def _load_codebook(self, codebook: str | Path | Codebook) -> Codebook:
        """Load a codebook from various input formats.

        Args:
        ----
            codebook: File path or Codebook object.

        Returns:
        -------
            Codebook object.

        """
        if isinstance(codebook, Codebook):
            return codebook
        return Codebook.from_file(codebook)

    def _process_document(
        self,
        document: Document,
        codebook: Codebook,
        prompt_template: str,
        max_tokens: int,
        temperature: float,
    ) -> pd.DataFrame:
        """Process a single document through the LLM.

        Args:
        ----
            document: The document to process.
            codebook: The codebook to use.
            prompt_template: The prompt template.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
        -------
            DataFrame with results for this document.

        """
        speakers_str = ", ".join(document.speakers) if document.speakers else "Unknown"

        formatted_prompt = prompt_template.format(
            text=document.content,
            codebook=codebook.to_string(),
            doc_id=document.doc_id,
            speakers=speakers_str,
        )

        response = self.client.complete(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        instances = parse_response(
            response_text=response.content,
            default_doc_id=document.doc_id,
        )

        return instances_to_dataframe(instances)

    def get_raw_response(
        self,
        document: str | Path | Document,
        codebook: str | Path | Codebook,
        prompt_template: str | None = None,
        scoring_criteria: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Get the raw LLM response for a single document.

        Useful for debugging or when you want to inspect the raw output
        before parsing.

        Args:
        ----
            document: Path to a document file or a Document object.
            codebook: Path to a codebook file or a Codebook object.
            prompt_template: Custom prompt template.
            scoring_criteria: Custom scoring criteria string.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
        -------
            LLMResponse object containing the raw response.

        """
        if isinstance(document, Document):
            doc = document
        else:
            docs = DocumentLoader.load(document)
            doc = docs[0]

        cb = self._load_codebook(codebook)

        prompt = create_custom_prompt(
            base_prompt=prompt_template,
            scoring_criteria=scoring_criteria,
        )

        formatted_prompt = prompt.format(
            text=doc.content,
            codebook=cb.to_string(),
            doc_id=doc.doc_id,
            speakers=", ".join(doc.speakers) if doc.speakers else "Unknown",
        )

        return self.client.complete(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def preview_prompt(
        self,
        document: str | Path | Document,
        codebook: str | Path | Codebook,
        prompt_template: str | None = None,
        scoring_criteria: str | None = None,
    ) -> str:
        """Preview the formatted prompt that would be sent to the LLM.

        Useful for debugging and understanding what the model will receive.

        Args:
        ----
            document: Path to a document file or a Document object.
            codebook: Path to a codebook file or a Codebook object.
            prompt_template: Custom prompt template.
            scoring_criteria: Custom scoring criteria string.

        Returns:
        -------
            The formatted prompt string.

        """
        if isinstance(document, Document):
            doc = document
        else:
            docs = DocumentLoader.load(document)
            doc = docs[0]

        cb = self._load_codebook(codebook)

        prompt = create_custom_prompt(
            base_prompt=prompt_template,
            scoring_criteria=scoring_criteria,
        )

        return prompt.format(
            text=doc.content,
            codebook=cb.to_string(),
            doc_id=doc.doc_id,
            speakers=", ".join(doc.speakers) if doc.speakers else "Unknown",
        )

    def __repr__(self) -> str:
        """Return string representation of the prompter."""
        return f"EncodingPrompter(model='{self.model}')"

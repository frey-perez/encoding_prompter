"""LLM client for interacting with OpenRouter API."""

import os
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


@dataclass
class LLMResponse:
    """Response from the LLM.

    Attributes
    ----------
        content: The text content of the response.
        model: The model that generated the response.
        usage: Token usage information (if available).
        raw_response: The raw API response.

    """

    content: str
    model: str
    usage: dict[str, int] | None
    raw_response: dict[str, Any] | None


class LLMClient:
    """Client for interacting with LLMs through OpenRouter.

    OpenRouter provides a unified API for LLM models, including both free and
    paid options.

    Attributes
    ----------
        api_key: The OpenRouter API key.
        model: The model identifier to use.
        base_url: The OpenRouter API base URL.

    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the LLM client.

        Args:
        ----
            api_key: OpenRouter API key. If None, will look for
                OPENROUTER_API_KEY environment variable.
            model: Model identifier string. Defaults to a free model.

        Raises:
        ------
            ValueError: If no API key is provided or found in environment.

        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set "
                "OPENROUTER_API_KEY environment variable."
            )

        self.model = model
        self.base_url = self.OPENROUTER_BASE_URL

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
        ----
            prompt: The prompt text to send.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-2.0).

        Returns:
        -------
            LLMResponse containing the model's response.

        Raises:
        ------
            requests.RequestException: If the API request fails.
            ValueError: If the response format is unexpected.

        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/encoding-prompter",
            "X-Title": "Encoding Prompter",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Unexpected format: {data}")

        content = data["choices"][0].get("message", {}).get("content", "")

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            raw_response=data,
        )

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"LLMClient(model='{self.model}')"


def get_available_models() -> list[str]:
    """Return a NON-exhaustive list of commonly used models on OpenRouter.

    Returns
    -------
        List of modle names, free and paid options .

    """
    return [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/devstral-2512:free",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "openai/gpt-oss-120b:free",
        "openai/gpt-5-nano",
        "meta-llama/llama-3-8b-instruct",
    ]

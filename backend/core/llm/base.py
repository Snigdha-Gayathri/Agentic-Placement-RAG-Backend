"""Abstract base class for language model providers."""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM call.

    Attributes:
        text: The generated text.
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the completion.
        latency_ms: Wall-clock time for the API call in milliseconds.
        model: The model identifier that served the request.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class BaseLLM(abc.ABC):
    """Abstract interface for large-language-model providers.

    Concrete subclasses must implement :meth:`generate`.
    """

    @abc.abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The user query / prompt content.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            An :class:`LLMResponse` with generated text and metadata.
        """
        ...

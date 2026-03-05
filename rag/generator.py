"""
Generator Module
LLM-based answer generation from retrieved context.
Supports Groq, OpenAI, and Ollama (local) backends.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Tunable generation parameters."""
    model: str = "llama3-8b-8192"
    temperature: float = 0.2        # Low = factual/deterministic; high = creative
    max_tokens: int = 1024
    top_p: float = 0.9
    stream: bool = False


@dataclass
class RAGResponse:
    """Structured output from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def format_sources(self) -> str:
        """Human-readable source list."""
        lines = []
        for i, src in enumerate(self.sources, 1):
            meta = src.get("metadata", {})
            score = src.get("similarity_score", 0)
            source_file = meta.get("source", "unknown")
            page = meta.get("page", "")
            page_str = f" (page {int(page)+1})" if page != "" else ""
            lines.append(f"  [{i}] {source_file}{page_str} — similarity: {score:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions \
using ONLY the provided context. 

Rules:
- Base your answer strictly on the context provided.
- If the context does not contain enough information, say so clearly.
- Be concise but complete.
- Do not hallucinate or add information beyond the context.
- When referencing specific facts, indicate which source they came from."""

RAG_PROMPT_TEMPLATE = """{system}

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


def build_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Construct a RAG prompt from a question and retrieved document chunks.

    Each chunk is prefixed with its source for citation awareness.
    """
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        header = f"[Source {i}: {source}" + (f", page {int(page)+1}" if page != "" else "") + "]"
        context_blocks.append(f"{header}\n{doc['content']}")

    context = "\n\n".join(context_blocks)
    return RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base for LLM provider backends."""

    @abstractmethod
    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        ...

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        ...


class GroqBackend(LLMBackend):
    """
    Groq Cloud backend. Free tier available at console.groq.com.
    Supported models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
    """

    def __init__(self, api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set. Export it or pass api_key=...")
        self._client = Groq(api_key=key)

    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        response = self._client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        usage = response.usage
        return RAGResponse(
            answer=response.choices[0].message.content.strip(),
            model=config.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OpenAIBackend(LLMBackend):
    """
    OpenAI backend. Requires OPENAI_API_KEY.
    Supported models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    """

    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        self._client = OpenAI(api_key=key)

    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        response = self._client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        usage = response.usage
        return RAGResponse(
            answer=response.choices[0].message.content.strip(),
            model=config.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaBackend(LLMBackend):
    """
    Ollama backend — runs LLMs locally. No API key required.
    Install Ollama and pull a model: ollama pull llama3
    Default model: llama3
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        try:
            from ollama import Client
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")
        self._client = Client(host=base_url)

    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        response = self._client.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": config.temperature, "num_predict": config.max_tokens},
        )
        return RAGResponse(
            answer=response["message"]["content"].strip(),
            model=config.model,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        for chunk in self._client.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            delta = chunk["message"]["content"]
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# High-level Generator
# ---------------------------------------------------------------------------

BACKENDS = {
    "groq": GroqBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


class RAGGenerator:
    """
    Orchestrates prompt construction and LLM generation.

    Usage:
        generator = RAGGenerator(provider="groq")
        response = generator.generate(question, retrieved_docs)
        print(response.answer)
        print(response.format_sources())
    """

    def __init__(
        self,
        provider: str = "groq",
        config: Optional[GenerationConfig] = None,
        **backend_kwargs,
    ):
        """
        Args:
            provider: One of 'groq', 'openai', 'ollama'.
            config:   GenerationConfig instance (uses defaults if None).
            **backend_kwargs: Passed to the backend (e.g. api_key=...).
        """
        if provider not in BACKENDS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(BACKENDS)}")

        self.provider = provider
        self.config = config or GenerationConfig()
        self.backend: LLMBackend = BACKENDS[provider](**backend_kwargs)
        logger.info(f"RAGGenerator ready | provider={provider} | model={self.config.model}")

    def generate(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> RAGResponse:
        """
        Generate an answer grounded in the retrieved documents.

        Args:
            question:       User's question.
            retrieved_docs: Output of RAGRetriever.retrieve().

        Returns:
            RAGResponse with answer, sources, and token usage.
        """
        if not retrieved_docs:
            logger.warning("No documents retrieved — answer will be unsupported.")

        prompt = build_prompt(question, retrieved_docs)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        response = self.backend.complete(prompt, self.config)
        response.sources = retrieved_docs

        logger.info(
            f"Generated answer | tokens: {response.total_tokens} "
            f"(prompt={response.prompt_tokens}, completion={response.completion_tokens})"
        )
        return response

    def stream_generate(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Iterator[str]:
        """
        Stream the answer token-by-token. Useful for chat UIs.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        prompt = build_prompt(question, retrieved_docs)
        yield from self.backend.stream(prompt, self.config)
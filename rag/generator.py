"""
Generator Module
LLM-based answer generation from retrieved context.
Supports Groq, OpenAI, Ollama (local), and HuggingFace backends.
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
    temperature: float = 0.2
    max_tokens: int = 256          # FIXED: was 1024, reduced to 256 for TinyLlama
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
    @abstractmethod
    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        ...

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        ...


class GroqBackend(LLMBackend):
    """Groq Cloud backend. Free tier at console.groq.com."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set.")
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
    """OpenAI backend. Requires OPENAI_API_KEY."""

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
    """Ollama backend — runs LLMs locally. No API key required."""

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


class HuggingFaceBackend(LLMBackend):
    """
    HuggingFace Transformers backend — runs models locally.
    Works on HuggingFace Spaces CPU. No API key required.
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float32,
            device=-1,  # -1 = CPU explicitly
        )
        self._model_name = model_name

    def complete(self, prompt: str, config: GenerationConfig) -> RAGResponse:
        # FIXED: Truncate prompt to 1024 tokens before passing to model
        # This prevents the "2453 > 2048" warning and indexing errors
        inputs = self._pipe.tokenizer(
            prompt,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        truncated_prompt = self._pipe.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=True
        )

        output = self._pipe(
            truncated_prompt,
            max_new_tokens=256,        # FIXED: hardcoded 256, no conflict with max_length
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )
        answer = output[0]["generated_text"]
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        return RAGResponse(answer=answer, model=self._model_name)

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        response = self.complete(prompt, config)
        yield response.answer


# ---------------------------------------------------------------------------
# High-level Generator
# ---------------------------------------------------------------------------

BACKENDS = {
    "groq": GroqBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
    "huggingface": HuggingFaceBackend,
}


class RAGGenerator:
    """Orchestrates prompt construction and LLM generation."""

    def __init__(
        self,
        provider: str = "huggingface",
        config: Optional[GenerationConfig] = None,
        **backend_kwargs,
    ):
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
        prompt = build_prompt(question, retrieved_docs)
        yield from self.backend.stream(prompt, self.config)
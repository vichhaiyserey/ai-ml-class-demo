"""Minimal OpenAI-compatible LLM client.

Works with:
  * OpenAI  (base_url default)
  * Ollama  (http://localhost:11434/v1)  -> e.g. qwen2.5-coder
  * LM Studio / vLLM / any OpenAI-compatible server

If no API key / server is configured, the client degrades gracefully and
returns an offline response so the demo still works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


SYSTEM_DEFAULT = "You are a helpful, concise assistant for teaching AI/ML and cybersecurity."


@dataclass
class LLMClient:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    temperature: float = 0.3

    def __post_init__(self):
        self._client = None

    # ---------- availability ----------
    @property
    def is_live(self) -> bool:
        return bool(self.api_key) or bool(self.base_url)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package not installed") from exc
        kwargs = {"api_key": self.api_key or "ollama", "temperature": self.temperature}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # ---------- core ----------
    def chat(self, messages: list[dict], system: str | None = None) -> str:
        messages = self._with_system(messages, system)
        if not self.is_live:
            return self._offline(messages)
        client = self._get_client()
        resp = client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def ask(self, prompt: str, system: str | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system)

    def generate_code(self, task: str, language: str = "python") -> str:
        system = f"You are an expert {language} developer. Return clean, runnable code with brief comments."
        return self.ask(task, system=system)

    def rag_answer(self, question: str, context: str) -> str:
        system = (
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not present, say so. Cite sources like [file]."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self.ask(prompt, system=system)

    # ---------- helpers ----------
    @staticmethod
    def _with_system(messages: list[dict], system: str | None) -> list[dict]:
        if system and (not messages or messages[0].get("role") != "system"):
            return [{"role": "system", "content": system}, *messages]
        return messages

    @staticmethod
    def _offline(messages: list[dict]) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "[offline mode] No LLM endpoint configured.\n"
            "Set LLM_API_KEY or LLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama/Qwen).\n\n"
            f"You asked: {last}"
        )

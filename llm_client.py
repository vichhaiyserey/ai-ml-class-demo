"""Minimal OpenAI-compatible LLM client.

Works with:
  * OpenAI  (base_url default)
  * Ollama  (http://localhost:11434/v1)  -> e.g. qwen2.5-coder
  * LM Studio / vLLM / any OpenAI-compatible server

If no API key / server is configured, the client degrades gracefully and
returns an offline response so the demo still works.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
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
        kwargs = {"api_key": self.api_key or "ollama"}
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
        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
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


# --------------------------------------------------------------------------- #
# Ollama native API helpers (used by the educational playground)
# --------------------------------------------------------------------------- #
def _ollama_root(base_url: str | None) -> str:
    """Turn an OpenAI-style base URL into the Ollama server root."""
    base = (base_url or "http://localhost:11434/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def list_ollama_models(base_url: str | None = None,
                       timeout: float = 3.0) -> list[dict]:
    """Return installed Ollama models as [{'name', 'size'}]; [] if unavailable."""
    try:
        with urllib.request.urlopen(
            f"{_ollama_root(base_url)}/api/tags", timeout=timeout
        ) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [
        {"name": m.get("name") or m.get("model", ""), "size": m.get("size", 0)}
        for m in data.get("models", [])
    ]


def ollama_chat(model: str, messages: list[dict], base_url: str | None = None,
                options: dict | None = None, keep_alive: str | int = "5m",
                timeout: float = 600.0) -> dict:
    """Chat via Ollama's native /api/chat, returning content plus timing metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {},
        "keep_alive": keep_alive,
    }
    request = urllib.request.Request(
        f"{_ollama_root(base_url)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        data = json.load(resp)
    elapsed = time.perf_counter() - started

    eval_count = int(data.get("eval_count", 0) or 0)
    eval_ns = data.get("eval_duration", 0) or 0
    toks_per_s = eval_count / (eval_ns / 1e9) if eval_ns else 0.0
    return {
        "content": (data.get("message") or {}).get("content", ""),
        "eval_count": eval_count,
        "prompt_eval_count": int(data.get("prompt_eval_count", 0) or 0),
        "total_seconds": elapsed,
        "tokens_per_second": toks_per_s,
        "done_reason": data.get("done_reason", ""),
    }


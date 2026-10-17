"""Lightweight retrieval-augmented generation (RAG) engine.

Uses TF-IDF for retrieval so it needs no heavy embedding dependencies.
Can optionally hand retrieved context to an OpenAI-compatible LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str
    score: float = 0.0


class RAGEngine:
    """Simple chunk-and-retrieve RAG over plain text documents."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None

    # ---------- indexing ----------
    def add_text(self, text: str, source: str = "document"):
        for piece in self._split(text):
            self.chunks.append(Chunk(text=piece, source=source))

    def add_file(self, path: str | Path):
        path = Path(path)
        self.add_text(path.read_text(encoding="utf-8", errors="ignore"), source=path.name)

    def build(self):
        if not self.chunks:
            raise ValueError("No documents indexed. Call add_text/add_file first.")
        corpus = [c.text for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    # ---------- retrieval ----------
    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        if self.matrix is None:
            raise ValueError("Index not built. Call build() first.")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in order:
            c = self.chunks[i]
            results.append(Chunk(text=c.text, source=c.source, score=float(sims[i])))
        return results

    def build_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        blocks = [f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in results]
        return "\n\n".join(blocks)

    # ---------- helpers ----------
    def _split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # Prefer splitting on blank lines / sections, then sliding window.
        sections = re.split(r"\n\s*\n", text)
        pieces: list[str] = []
        for section in sections:
            words = section.split()
            if len(words) <= self.chunk_size:
                pieces.append(section.strip())
                continue
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(words), step):
                window = words[start:start + self.chunk_size]
                if window:
                    pieces.append(" ".join(window))
        return [p for p in pieces if p]


def load_default_rag(data_dir: str | Path = "data") -> RAGEngine:
    """Convenience loader that indexes the bundled sample documents."""
    engine = RAGEngine()
    data_dir = Path(data_dir)
    for name in ("trivy_scan.txt",):
        path = data_dir / name
        if path.exists():
            engine.add_file(path)
    engine.build()
    return engine

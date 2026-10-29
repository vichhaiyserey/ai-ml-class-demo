"""Lightweight retrieval-augmented generation (RAG) engine.

Uses TF-IDF for retrieval so it needs no heavy embedding dependencies.
Can optionally hand retrieved context to an OpenAI-compatible LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]


@dataclass
class Chunk:
    text: str
    source: str
    score: float = 0.0


@dataclass
class Vulnerability:
    cve: str
    package: str
    severity: str
    fixed_version: str = ""
    title: str = ""
    source: str = ""


class RAGEngine:
    """Simple chunk-and-retrieve RAG over plain text documents."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80,
                 split_on: str = "section"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_on = split_on
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
    def retrieve(self, query: str, top_k: int = 3,
                 min_score: float = 0.0) -> list[Chunk]:
        if self.matrix is None:
            raise ValueError("Index not built. Call build() first.")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        order = np.argsort(sims)[::-1]
        results = []
        for i in order:
            if len(results) >= top_k:
                break
            if float(sims[i]) < min_score:
                continue
            c = self.chunks[i]
            results.append(Chunk(text=c.text, source=c.source, score=float(sims[i])))
        return results

    def build_context(self, query: str, top_k: int = 3,
                      min_score: float = 0.0) -> str:
        results = self.retrieve(query, top_k=top_k, min_score=min_score)
        blocks = [f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in results]
        return "\n\n".join(blocks)

    # ---------- helpers ----------
    def _split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self.split_on == "line":
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
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


# --------------------------------------------------------------------------- #
# Trivy report parsing
# --------------------------------------------------------------------------- #
_CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
_SEV_RE = re.compile(r"\b(CRITICAL|HIGH|MEDIUM|LOW|UNKNOWN)\b", re.IGNORECASE)
_PKG_RE = re.compile(r"\bin\s+([A-Za-z0-9_.@/-]+)", re.IGNORECASE)
_FIX_RE = re.compile(r"([A-Za-z0-9_.:@/-]+)\s+to\s+([A-Za-z0-9_.:@/-]+)")


def is_trivy_json(raw: str) -> bool:
    return raw.lstrip().startswith("{")


def _parse_trivy_json(raw: str) -> list[Vulnerability]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    out: list[Vulnerability] = []
    for result in data.get("Results", []) or []:
        for v in result.get("Vulnerabilities", []) or []:
            out.append(Vulnerability(
                cve=str(v.get("VulnerabilityID", "")).upper(),
                package=str(v.get("PkgName", "")),
                severity=str(v.get("Severity", "UNKNOWN")).upper(),
                fixed_version=str(v.get("FixedVersion", "") or ""),
                title=str(v.get("Title", "") or ""),
            ))
    return out


def _parse_trivy_text(text: str) -> list[Vulnerability]:
    fixes: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip().lower().startswith("remediation"):
            for pkg, ver in _FIX_RE.findall(line):
                fixes[pkg.lower()] = ver
    out: list[Vulnerability] = []
    for line in text.splitlines():
        cve = _CVE_RE.search(line)
        if not cve:
            continue
        sev = _SEV_RE.search(line)
        pkg = _PKG_RE.search(line)
        pkg_name = pkg.group(1) if pkg else ""
        out.append(Vulnerability(
            cve=cve.group(0).upper(),
            package=pkg_name,
            severity=sev.group(1).upper() if sev else "UNKNOWN",
            fixed_version=fixes.get(pkg_name.lower(), "").rstrip("."),
        ))
    return out


def parse_trivy(raw: str) -> list[Vulnerability]:
    """Parse either a plain-text or JSON Trivy report into vulnerabilities."""
    return _parse_trivy_json(raw) if is_trivy_json(raw) else _parse_trivy_text(raw)


def trivy_json_to_text(raw: str) -> str:
    """Convert a JSON Trivy report to indexable text (passes text through)."""
    if not is_trivy_json(raw):
        return raw
    vulns = _parse_trivy_json(raw)
    if not vulns:
        return raw
    lines = ["=== Trivy Scan Report ===", "Found vulnerabilities:"]
    for v in vulns:
        fix = f" (fix: {v.fixed_version})" if v.fixed_version else ""
        lines.append(f"- {v.cve} in {v.package}:{v.severity.lower()}{fix}")
    return "\n".join(lines)


def filter_by_severity(text: str, allowed: set[str]) -> str:
    """Keep only lines whose severity is in ``allowed`` (header lines kept)."""
    if not allowed or set(SEVERITIES) <= allowed:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        sev = _SEV_RE.search(line)
        if sev and sev.group(1).upper() not in allowed:
            continue
        kept.append(line)
    return "\n".join(kept)


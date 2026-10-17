"""Exercise 08 - Retrieval-Augmented Generation over a Trivy-style scan report.

Run:
    python exercises/08_rag_trivy.py

The RAG index is built from data/trivy_scan.txt. With no LLM configured the
script still shows the retrieved context (offline mode).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_engine import load_default_rag
from llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    engine = load_default_rag(ROOT / "data")
    print(f"Indexed {len(engine.chunks)} chunk(s).")

    questions = [
        "Which CVE is critical?",
        "What should I update to fix the Python issue?",
        "How many high severity issues are there?",
    ]

    llm = LLMClient()
    for q in questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        context = engine.build_context(q, top_k=2)
        print("\n[retrieved context]")
        print(context)
        print("\n[A]")
        print(llm.rag_answer(q, context))


if __name__ == "__main__":
    main()
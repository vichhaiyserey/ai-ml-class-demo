"""Exercise 07 - LLM chat & code generation.

Run:
    python exercises/07_llm_chat.py

Configure an endpoint with env vars, e.g. local Ollama (Qwen):
    set LLM_BASE_URL=http://localhost:11434/v1
    set LLM_MODEL=qwen2.5-coder:7b
Or OpenAI:
    set LLM_API_KEY=sk-...
    set LLM_MODEL=gpt-4o-mini
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import LLMClient


def main():
    llm = LLMClient()
    print(f"Live endpoint: {llm.is_live}  (model={llm.model})")

    print("\n--- Normal chat ---")
    print(llm.ask("Explain overfitting in one sentence."))

    print("\n--- Coding with Qwen ---")
    code = llm.generate_code("Write a Python function that computes precision and recall from two lists.")
    print(code)


if __name__ == "__main__":
    main()
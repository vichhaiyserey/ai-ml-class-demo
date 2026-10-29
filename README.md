# AI/ML Class Demo

An interactive, offline-friendly **Streamlit** web app that teaches a practical
AI/ML workflow in Python — from language basics all the way to LLM chat and a
small Retrieval-Augmented Generation (RAG) demo grounded in a Trivy scan report.

## Features

| # | Section | Covers |
|---|---------|--------|
| 1 | Python basics | types, truthiness, loops, functions, collections, comprehensions, errors, files |
| 2 | NumPy | arrays & reshape, indexing/slicing, broadcasting, aggregations, matrix ops, randomness, visualisations |
| 3 | Pandas | Series/DataFrame, describe, filter, parsing scan output |
| 4 | Visualization | chart gallery, distributions & skew, pairplots, heatmaps/clustermaps, feature selection, stacked charts, BI dashboard |
| 5 | Regression | linear regression + MSE / R² |
| 6 | Classification | logistic regression / KNN + confusion matrix |
| 7 | Pipeline + Pickle | `Pipeline`, joblib save → load → predict |
| 8 | ML Workflow | guided steps: select → clean → filter/features → explore → train → pipeline & predict |
| 9 | LLM Chat | OpenAI-compatible chat & code-gen (works with local Qwen) |
| 10 | LLM Playground | temperature / top-p / max-tokens explained, live speed metrics |
| 11 | RAG (Trivy) | TF-IDF retrieval over a scan report + grounded answers |
| 12 | RAG (summary) | query-focused summarizer over notes + tuning guide and key lessons |

## Project layout

```
ai_ml_demo/
├── app.py                  # Streamlit app (all 12 sections)
├── llm_client.py           # OpenAI-compatible LLM client (graceful offline mode)
├── rag_engine.py           # Lightweight TF-IDF RAG engine
├── requirements.txt
├── data/
│   ├── sample_data.csv     # tiny tabular dataset
│   ├── customers.csv       # messy churn dataset for the ML Workflow page
│   ├── trivy_scan.txt      # sample security scan report for RAG
│   └── notes/              # weekly team notes for the RAG summarizer page
├── exercises/              # standalone scripts (run one at a time)
│   ├── 01_basic_python.py
│   ├── 02_numpy.py
│   ├── 03_pandas.py
│   ├── 04_visualization.py
│   ├── 05_ml_models.py
│   ├── 06_pipeline_pickle.py
│   ├── 07_llm_chat.py
│   └── 08_rag_trivy.py
├── notebooks/              # Jupyter notebooks for hands-on practice
│   ├── 01_ml_workflow.ipynb
│   └── 02_llm_and_rag.ipynb
└── models/                 # saved .joblib artifacts (git-ignored)
```

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the web app

```powershell
streamlit run app.py
```

Then open <http://localhost:8501>.

## Run the standalone exercises

```powershell
python exercises\01_basic_python.py
python exercises\02_numpy.py
python exercises\03_pandas.py
python exercises\05_ml_models.py
python exercises\06_pipeline_pickle.py
python exercises\07_llm_chat.py
python exercises\08_rag_trivy.py
```

## Run the notebooks

```powershell
pip install jupyter          # once
jupyter notebook notebooks
```

- `01_ml_workflow.ipynb` — load `customers.csv`, clean it, engineer features, train,
  evaluate, and save a `Pipeline` (mirrors the **ML Workflow** app page).
- `02_llm_and_rag.ipynb` — chat with your local model, explore generation settings,
  and run RAG over `trivy_scan.txt` (mirrors the **LLM / RAG** app pages).

Notebooks run from the `notebooks/` folder and read data from `../data`.

## Enabling the LLM sections

The LLM chat and RAG answer-generation work in **offline mode** by default
(they still show retrieved context). To go live, point the app at any
OpenAI-compatible endpoint.

Local **Ollama** running Qwen:

```powershell
ollama pull qwen2.5-coder:7b
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL    = "qwen2.5-coder:7b"
streamlit run app.py
```

Or OpenAI:

```powershell
$env:LLM_API_KEY = "sk-..."
$env:LLM_MODEL   = "gpt-4o-mini"
streamlit run app.py
```

The same env vars work for the standalone exercise scripts.

## Notes

- Everything runs locally; no data leaves your machine unless you configure an
  external LLM endpoint.
- Saved models land in `models/` and are intentionally git-ignored — regenerate
  them from the app or `exercises/06_pipeline_pickle.py`.
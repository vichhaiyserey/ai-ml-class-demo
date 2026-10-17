"""Interactive AI/ML teaching web app (Streamlit).

Run:
    .venv\\Scripts\\streamlit run app.py

Sections:
  1. Home
  2. Python basics
  3. NumPy
  4. Pandas
  5. Visualization
  6. Regression
  7. Classification
  8. Pipeline + Pickle (save/load/evaluate)
  9. LLM Chat (OpenAI-compatible / local Qwen)
 10. RAG over a Trivy scan
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from llm_client import LLMClient  # noqa: E402
from rag_engine import load_default_rag  # noqa: E402

st.set_page_config(page_title="AI/ML Class Demo", page_icon="🤖", layout="wide")

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Home
# --------------------------------------------------------------------------- #
def page_home():
    st.title("🤖 AI/ML Class Demo")
    st.markdown(
        """
Welcome! This interactive app walks through a practical AI/ML workflow in Python:

| # | Section | What you learn |
|---|---------|----------------|
| 1 | Python basics | types, f-strings, comprehensions, lambdas |
| 2 | NumPy | arrays, matrices, vectorised math |
| 3 | Pandas | Series/DataFrame, indexing, parsing scan data |
| 4 | Visualization | matplotlib, seaborn, plotly |
| 5 | Regression | linear regression + metrics |
| 6 | Classification | logistic regression / KNN + metrics |
| 7 | Pipeline + Pickle | train → save → load → evaluate |
| 8 | LLM Chat | OpenAI-compatible chat & code-gen (Qwen) |
| 9 | RAG | question-answering over a Trivy scan report |

Use the sidebar to navigate. Everything runs locally and offline; the LLM
sections light up automatically when you configure an endpoint.
"""
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Python", "3.14")
    col2.metric("Scikit-learn", "1.9.1")
    col3.metric("Streamlit", "1.66.0")


# --------------------------------------------------------------------------- #
# 2. Python basics
# --------------------------------------------------------------------------- #
def page_python():
    st.title("🐍 Python basics")
    st.caption("Small, high-leverage ideas that pay off everywhere.")

    st.subheader("Types & f-strings")
    st.code(
        """x = 10
y = 3.14
name = "AI/ML"
print(f"{name}: x={x} ({type(x).__name__}), y={y} ({type(y).__name__})")""",
        language="python",
    )
    x, y, name = 10, 3.14, "AI/ML"
    st.success(f"{name}: x={x} ({type(x).__name__}), y={y} ({type(y).__name__})")

    st.subheader("List comprehensions")
    st.code("squares = [n**2 for n in range(10)]", language="python")
    n = st.slider("range(n)", 1, 20, 10)
    st.write([i ** 2 for i in range(n)])

    st.subheader("Lambdas & map")
    st.code("double = lambda v: v * 2\ndoubled = list(map(double, range(5)))", language="python")
    double = lambda v: v * 2  # noqa: E731
    st.write(list(map(double, range(5))))

    st.subheader("Dictionaries & Counter")
    st.code(
        """from collections import Counter
Counter("mississippi").most_common(3)""",
        language="python",
    )
    from collections import Counter
    st.write(Counter("mississippi").most_common(3))


# --------------------------------------------------------------------------- #
# 3. NumPy
# --------------------------------------------------------------------------- #
def page_numpy():
    st.title("🔢 NumPy")
    st.caption("Fast, vectorised numerical computing.")

    st.subheader("Create & reshape")
    flat = np.arange(12)
    st.code("np.arange(12).reshape(3, 4)", language="python")
    st.write(flat.reshape(3, 4))

    st.subheader("Vectorised operations")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    st.code("a + b, np.dot(a, b)", language="python")
    st.write({"a+b": (a + b).tolist(), "dot": int(np.dot(a, b))})

    st.subheader("Random stats")
    data = np.random.default_rng(42).normal(size=1000)
    st.code("data.mean(), data.std()", language="python")
    st.write({"mean": round(float(data.mean()), 3), "std": round(float(data.std()), 3)})
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="#4C78A8")
    ax.set_title("Normal distribution (n=1000)")
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# 4. Pandas
# --------------------------------------------------------------------------- #
def page_pandas():
    st.title("🐼 Pandas")
    st.caption("Tabular data: load, inspect, transform.")

    csv_path = DATA_DIR / "sample_data.csv"
    df = pd.read_csv(csv_path)
    st.subheader("sample_data.csv")
    st.dataframe(df, width='stretch')

    st.subheader("Describe")
    st.dataframe(df.describe(), width='stretch')

    st.subheader("Filter & aggregate")
    thr = st.slider("feature1 >", float(df.feature1.min()), float(df.feature1.max()), 6.0)
    filtered = df[df.feature1 > thr]
    st.write(f"{len(filtered)} rows where feature1 > {thr}")
    st.dataframe(filtered, width='stretch')

    st.subheader("Parse a Trivy-style report")
    text = (DATA_DIR / "trivy_scan.txt").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if "CVE" in line and " in " in line:
            cve, rest = line.split(" in ")
            tool, severity = rest.strip().split(":")
            rows.append({"CVE": cve.strip("- "), "Tool": tool, "Severity": severity})
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')


# --------------------------------------------------------------------------- #
# 5. Visualization
# --------------------------------------------------------------------------- #
def page_visualization():
    st.title("📊 Visualization")
    st.caption("matplotlib for control, seaborn for stats, plotly for interactivity.")

    st.subheader("Plotly - Trivy severities")
    sev = pd.DataFrame(
        {"Severity": ["critical", "high", "medium", "low"], "Count": [1, 3, 2, 4]}
    )
    st.plotly_chart(px.bar(sev, x="Severity", y="Count", color="Severity"), width='stretch')

    st.subheader("Seaborn - Iris")
    iris = load_iris(as_frame=True).frame
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x="petal length (cm)", y="petal width (cm)",
                    hue="target", palette="deep", ax=ax)
    st.pyplot(fig)

    st.subheader("Matplotlib - line")
    xs = np.linspace(0, 2 * np.pi, 100)
    fig2, ax2 = plt.subplots()
    ax2.plot(xs, np.sin(xs), label="sin")
    ax2.plot(xs, np.cos(xs), label="cos")
    ax2.legend()
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# 6. Regression
# --------------------------------------------------------------------------- #
def page_regression():
    st.title("📈 Regression")
    st.caption("Predict a continuous value with linear regression.")

    n = st.slider("samples", 50, 500, 150, 10)
    noise = st.slider("noise", 0.0, 50.0, 10.0, 1.0)
    X, y = make_regression(n_samples=n, n_features=1, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mean_squared_error(y_test, pred):.2f}")
    c2.metric("R²", f"{r2_score(y_test, pred):.3f}")
    c3.metric("Coef", f"{model.coef_[0]:.2f}")

    order = np.argsort(X_test.ravel())
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="actual", alpha=0.6)
    ax.plot(X_test[order], pred[order], color="red", label="predicted")
    ax.legend()
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# 7. Classification
# --------------------------------------------------------------------------- #
def page_classification():
    st.title("🎯 Classification")
    st.caption("Predict a class label and inspect the confusion matrix.")

    dataset = st.radio("Dataset", ["Iris (3 classes)", "Synthetic (binary)"], horizontal=True)

    if dataset.startswith("Iris"):
        data = load_iris()
        X, y = data.data, data.target
        target_names = list(data.target_names)
        model = KNeighborsClassifier(n_neighbors=st.slider("k", 1, 15, 3))
    else:
        X, y = make_classification(n_samples=300, n_features=4, random_state=42)
        target_names = ["class 0", "class 1"]
        model = LogisticRegression(max_iter=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    st.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    st.pyplot(fig)

    st.text("Classification report")
    st.code(classification_report(y_test, pred, target_names=target_names, zero_division=0))


# --------------------------------------------------------------------------- #
# 8. Pipeline + Pickle
# --------------------------------------------------------------------------- #
def build_churn_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "sample_data.csv")
    X = df[["feature1", "feature2"]]
    y = df["target"]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    return pipe, X, y


def page_pipeline_pickle():
    st.title("💾 Pipeline + Pickle")
    st.caption("Train → persist with joblib → reload → evaluate on test data.")

    if st.button("Train & save pipeline"):
        pipe, X, y = build_churn_pipeline()
        path = MODELS_DIR / "churn_pipeline.joblib"
        joblib.dump(pipe, path)
        st.success(f"Saved pipeline to {path.relative_to(ROOT)}")

    model_path = MODELS_DIR / "churn_pipeline.joblib"
    st.subheader("Saved models")
    saved = sorted(MODELS_DIR.glob("*.joblib"))
    if not saved:
        st.info("No saved models yet - click 'Train & save pipeline'.")
    for m in saved:
        st.write(f"- {m.name}  ({m.stat().st_size} bytes)")

    st.subheader("Load & predict")
    if model_path.exists():
        pipe = joblib.load(model_path)
        df = pd.read_csv(DATA_DIR / "sample_data.csv")
        X = df[["feature1", "feature2"]]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        st.metric("Reloaded model accuracy on test split", f"{acc:.3f}")

        st.write("Try a live prediction:")
        c1, c2 = st.columns(2)
        f1 = c1.number_input("feature1", value=6.0)
        f2 = c2.number_input("feature2", value=3.2)
        if st.button("Predict"):
            pred = int(pipe.predict(pd.DataFrame({"feature1": [f1], "feature2": [f2]}))[0])
            st.success(f"Predicted target: {pred}")
    else:
        st.info("Save a model first to enable loading & prediction.")


# --------------------------------------------------------------------------- #
# 9. LLM chat
# --------------------------------------------------------------------------- #
def page_llm():
    st.title("💬 LLM Chat")
    st.caption("OpenAI-compatible chat & code generation. Works with local Ollama/Qwen.")

    with st.expander("Endpoint settings", expanded=not _llm_configured()):
        model = st.text_input("Model", value="qwen2.5-coder:7b")
        base_url = st.text_input("Base URL", value="http://localhost:11434/v1",
                                 help="Leave blank for OpenAI. Ollama uses :11434/v1")
        api_key = st.text_input("API key", value="", type="password",
                                help="Ollama ignores this; any non-empty value works.")

    llm = LLMClient(model=model,
                    base_url=base_url or None,
                    api_key=api_key or None)
    st.write("Status:", "🟢 live" if llm.is_live else "⚪ offline (set a base URL or key)")

    mode = st.radio("Mode", ["Chat", "Code generation"], horizontal=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear conversation"):
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if mode == "Chat":
        prompt = st.chat_input("Ask something...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = llm.chat(st.session_state.messages)
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        task = st.text_area("What should the code do?",
                            "Write a Python function that cleans a list of Trivy CVEs.")
        if st.button("Generate code"):
            with st.spinner("Generating..."):
                st.code(llm.generate_code(task), language="python")


def _llm_configured() -> bool:
    import os
    return bool(os.getenv("LLM_API_KEY") or os.getenv("LLM_BASE_URL"))


# --------------------------------------------------------------------------- #
# 10. RAG
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Indexing documents...")
def get_rag():
    return load_default_rag(DATA_DIR)


def page_rag():
    st.title("🛡️ RAG over a Trivy scan")
    st.caption("Retrieve relevant context, then answer questions grounded in it.")

    engine = get_rag()
    st.write(f"Indexed **{len(engine.chunks)}** chunk(s) from the scan report.")

    with st.expander("View source document"):
        st.text((DATA_DIR / "trivy_scan.txt").read_text(encoding="utf-8"))

    query = st.text_input("Question", "Which CVE is critical and what should I update?")
    if query:
        context = engine.build_context(query, top_k=3)
        st.subheader("Retrieved context")
        st.code(context)

        with st.expander("Scores"):
            st.dataframe(
                pd.DataFrame(
                    [{"source": c.source, "score": round(c.score, 4), "text": c.text}
                     for c in engine.retrieve(query, top_k=3)]
                ),
                width='stretch',
            )

        st.subheader("Answer")
        llm = LLMClient()
        st.write(llm.rag_answer(query, context))


PAGES = {
    "Home": page_home,
    "Python basics": page_python,
    "NumPy": page_numpy,
    "Pandas": page_pandas,
    "Visualization": page_visualization,
    "Regression": page_regression,
    "Classification": page_classification,
    "Pipeline + Pickle": page_pipeline_pickle,
    "LLM Chat": page_llm,
    "RAG (Trivy)": page_rag,
}


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("Local, offline-friendly demo. LLM sections auto-enable with an endpoint.")
    PAGES[choice]()


if __name__ == "__main__":
    main()

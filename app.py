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
  9. ML Workflow (guided: data → clean → features → model → pipeline)
 10. LLM Chat (OpenAI-compatible / local Qwen)
 11. LLM Playground (generation settings, explained)
 12. RAG over a Trivy scan
 13. RAG for summarization (query-focused summarizer + tuning lessons)
"""
from __future__ import annotations

import io
import os
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
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from llm_client import LLMClient, list_ollama_models, ollama_chat  # noqa: E402
from rag_engine import (  # noqa: E402
    SEVERITIES,
    RAGEngine,
    filter_by_severity,
    parse_trivy,
    trivy_json_to_text,
)

st.set_page_config(page_title="AI/ML Class Demo", page_icon="🤖", layout="wide")

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
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
| 1 | Python basics | types, logic, loops, functions, collections, errors, files |
| 2 | NumPy | arrays, indexing, broadcasting, aggregations, linear algebra, randomness |
| 3 | Pandas | Series/DataFrame, indexing, parsing scan data |
| 4 | Visualization | charts, distributions & skew, pairs, heatmaps, feature selection, BI dashboard |
| 5 | Regression | linear regression + metrics |
| 6 | Classification | logistic regression / KNN + metrics |
| 7 | Pipeline + Pickle | train → save → load → evaluate |
| 8 | ML Workflow | guided: data → clean → features → model → pipeline |
| 9 | LLM Chat | OpenAI-compatible chat & code-gen (Qwen) |
| 10 | LLM Playground | temperature, top-p, max tokens explained |
| 11 | RAG | question-answering over a Trivy scan report |
| 12 | RAG (summary) | query-focused summarizer + what to tune |

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
    st.caption("Small, high-leverage ideas that pay off everywhere in AI/ML.")
    st.markdown(
        "Each tab has a short idea, the code, a live result, and a "
        "👉 note explaining **why it matters**. Read them in order the first time."
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Types & truth", "Logic & loops", "Functions",
        "Collections", "Comprehensions", "Errors & files",
    ])

    # ---------------- Types & truth ----------------
    with t1:
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
        st.caption("👉 f-strings drop variables (and even expressions) straight into text — perfect for reports and logs.")

        st.subheader("Truthiness: what counts as false")
        st.code(
            """bool([]), bool(0), bool(""), bool(None)   # all False
bool([1]), bool(-3), bool("hi")             # all True""",
            language="python",
        )
        st.write({
            "bool([])": bool([]), "bool(0)": bool(0),
            'bool("")': bool(""), "bool(None)": bool(None),
            "bool([1])": bool([1]), "bool(-3)": bool(-3),
        })
        st.caption("👉 Empty containers, `0`, `\"\"` and `None` are **falsy**. So `if rows:` is a clean way to say “is the list non-empty?”.")

        st.subheader("Numbers, integer vs float, rounding")
        st.code(
            """7 / 2      # 3.5   (true division)
7 // 2     # 3     (floor division)
7 % 2      # 1     (remainder)
2 ** 10    # 1024  (power)
round(3.14159, 2)   # 3.14""",
            language="python",
        )
        st.write({
            "7 / 2": 7 / 2, "7 // 2": 7 // 2, "7 % 2": 7 % 2,
            "2 ** 10": 2 ** 10, "round(3.14159, 2)": round(3.14159, 2),
        })
        decimals = st.slider("round(value, n)", 0, 4, 2)
        st.write(f"round(3.14159, {decimals}) = {round(3.14159, decimals)}")
        st.caption("👉 Metrics are floats; `round()` keeps display tidy, `//` and `%` split data into batches and folds.")

        st.subheader("is vs ==")
        st.code(
            """result = None
result is None     # True  — use 'is' with None
[1, 2] == [1, 2]   # True  — use '==' to compare values""",
            language="python",
        )
        result = None
        st.write({"result is None": result is None, "[1,2] == [1,2]": [1, 2] == [1, 2]})
        st.caption("👉 `==` compares **values**, `is` compares **identity**. Use `is None` to test “no value yet”.")

    # ---------------- Logic & loops ----------------
    with t2:
        st.subheader("if / elif / else")
        score = st.slider("Model accuracy (%)", 0, 100, 82)
        if score >= 90:
            label = "excellent"
        elif score >= 75:
            label = "good"
        elif score >= 50:
            label = "needs work"
        else:
            label = "poor"
        st.code(
            """if score >= 90:   label = "excellent"
elif score >= 75: label = "good"
elif score >= 50: label = "needs work"
else:             label = "poor\"""",
            language="python",
        )
        st.write(f"score = {score} -> **{label}**")
        st.caption("👉 Conditions are checked top-to-bottom; the first match wins. Order matters!")

        st.subheader("and / or / not")
        st.code(
            """ready = is_loaded and not has_errors
allowed = is_admin or is_owner""",
            language="python",
        )
        st.write({
            "True and False": True and False,
            "True or False": True or False,
            "not True": not True,
        })
        st.caption("👉 Combine checks without nesting `if`s. `not` flips a boolean.")

        st.subheader("for loops: range, enumerate, zip")
        st.code(
            """models = ["logreg", "knn", "svc"]

for i, m in enumerate(models, start=1):   # index + value
    print(i, m)

pairs = zip(["age", "income"], [34, 52000])   # two lists together
print(dict(pairs))""",
            language="python",
        )
        models = ["logreg", "knn", "svc"]
        st.write([f"{i}. {m}" for i, m in enumerate(models, start=1)])
        st.write(dict(zip(["age", "income", "score"], [34, 52000, 0.8])))
        st.caption("👉 `enumerate` gives you the position; `zip` walks two lists side by side. Both remove error-prone index bookkeeping.")

        st.subheader("while, break, continue")
        st.code(
            """total, i = 0, 0
while total < 100:        # keep going until enough
    i += 1
    total += i

for n in range(10):
    if n % 2 == 0:        # skip the even numbers
        continue
    if n > 7:             # stop once n passes 7
        break""",
            language="python",
        )
        total, i = 0, 0
        while total < 100:
            i += 1
            total += i
        odds = []
        for n in range(10):
            if n % 2 == 0:
                continue
            if n > 7:
                break
            odds.append(n)
        st.write({"while stopped at i": i, "sum reached": total, "odds collected": odds})
        st.caption("👉 `while` repeats until a condition flips; `break` exits early, `continue` skips to the next round.")

    # ---------------- Functions ----------------
    with t3:
        st.subheader("def, parameters, return")
        st.code(
            """def mean(values):
    return sum(values) / len(values)

mean([2, 4, 6])   # 4.0""",
            language="python",
        )
        st.write("mean([2, 4, 6]) =", sum([2, 4, 6]) / len([2, 4, 6]))
        st.caption("👉 Functions name a calculation so you can reuse and test it instead of copying code.")

        st.subheader("Default values")
        st.code(
            """def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Ada")             # "Hello, Ada!"
greet("Ada", "Hi")       # "Hi, Ada!\"""",
            language="python",
        )
        greeting = st.text_input("greeting", "Hello")
        st.write(f"{greeting}, Ada!")
        st.caption("👉 Defaults make common calls short. Note: **never** use a mutable default like `[]` — it is shared between calls.")

        st.subheader("*args and **kwargs (flexible arguments)")
        st.code(
            """def total(*nums):          # any number of positional args
    return sum(nums)

def describe(**info):      # any number of named args
    return info

total(1, 2, 3)                 # 6
describe(model="knn", k=5)     # {'model': 'knn', 'k': 5}""",
            language="python",
        )
        st.write({
            "total(1, 2, 3)": sum((1, 2, 3)),
            "describe(...)": {"model": "knn", "k": 5},
        })
        st.caption("👉 `*args`/`**kwargs` let one function accept whatever you pass — the same trick Streamlit and sklearn use.")

        st.subheader("Type hints & docstrings (self-documenting code)")
        st.code(
            '''def fahrenheit_to_celsius(f: float) -> float:
    """Convert a temperature from Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9''',
            language="python",
        )
        st.write("fahrenheit_to_celsius(212) =", (212 - 32) * 5 / 9)
        st.caption("👉 Hints don't enforce types, but they make code readable and give editors autocomplete.")

        st.subheader("Scope (where names live)")
        st.code(
            """x = "global"

def show():
    x = "local"      # a NEW variable, does not change the global one
    return x

show(), x    # ('local', 'global')""",
            language="python",
        )
        st.write({"inside": "local", "outside": "global"})
        st.caption("👉 Functions can read outside names but assigning inside creates a local one. This is the #1 surprise for beginners.")

    # ---------------- Collections ----------------
    with t4:
        st.subheader("Slicing: grab parts of a sequence")
        st.code(
            """nums = list(range(10))   # [0, 1, ..., 9]
nums[2:5]     # [2, 3, 4]
nums[:3]      # [0, 1, 2]
nums[-2:]     # [8, 9]
nums[::2]     # [0, 2, 4, 6, 8]
nums[::-1]    # reversed""",
            language="python",
        )
        nums = list(range(10))
        st.write({
            "nums[2:5]": nums[2:5], "nums[:3]": nums[:3],
            "nums[-2:]": nums[-2:], "nums[::2]": nums[::2],
            "nums[::-1]": nums[::-1],
        })
        st.caption("👉 Slicing `[start:stop:step]` is how you split data into train/test or reverse a list without loops.")

        st.subheader("Pick the right container")
        st.code(
            """list  = [1, 2, 2, 3]      # ordered, allows duplicates
tuple = (1, 2, 3)         # ordered, cannot change
set   = {1, 2, 3}         # unique, unordered
dict  = {"a": 1, "b": 2}  # key -> value lookup""",
            language="python",
        )
        st.write({
            "list": [1, 2, 2, 3],
            "set removes duplicates": sorted({1, 2, 2, 3, 3}),
            "set overlap  {1,2,3} & {2,3,4}": sorted({1, 2, 3} & {2, 3, 4}),
            "dict lookup": {"a": 1, "b": 2}["b"],
        })
        st.caption("👉 Use a `set` for fast “is it in here?” checks and de-duplication; a `dict` for lookups; a `list` for order.")

        st.subheader("Handy methods: get, items, append")
        st.code(
            """counts = {"high": 3, "low": 5}
counts.get("medium", 0)   # 0 — safe default, no crash
list(counts.items())      # [('high', 3), ('low', 5)]

from collections import Counter
Counter("mississippi").most_common(3)""",
            language="python",
        )
        from collections import Counter
        st.write({
            'counts.get("medium", 0)': 0,
            "counts.items()": list({"high": 3, "low": 5}.items()),
            "Counter top-3": Counter("mississippi").most_common(3),
        })
        st.caption("👉 `.get(key, default)` avoids `KeyError`; `Counter` counts things in one line — great for class labels.")

        st.subheader("sorted with a key (and unpacking)")
        st.code(
            """scores = {"knn": 0.91, "logreg": 0.88, "svc": 0.93}
sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

first, *rest = [1, 2, 3, 4]   # first=1, rest=[2, 3, 4]
a, b = b, a                   # swap in one line""",
            language="python",
        )
        st.write(dict(sorted({"knn": 0.91, "logreg": 0.88, "svc": 0.93}.items(),
                             key=lambda kv: kv[1], reverse=True)))
        first, *rest = [1, 2, 3, 4]
        a, b = 1, 2
        a, b = b, a
        st.write({"first": first, "rest": rest, "swapped": (a, b)})
        st.caption("👉 `key=lambda` is how you sort records by a field — ranking models, sorting CVEs by severity, and more.")

    # ---------------- Comprehensions ----------------
    with t5:
        st.subheader("List comprehension (a loop that builds a list)")
        st.code("squares = [n ** 2 for n in range(6)]", language="python")
        k = st.slider("range(n)", 1, 20, 6)
        st.write([i ** 2 for i in range(k)])
        st.caption("👉 Same result as a `for` loop with `.append`, but shorter and faster to read.")

        st.subheader("With a condition (filter)")
        st.code("[n for n in range(20) if n % 3 == 0]", language="python")
        st.write([n for n in range(20) if n % 3 == 0])
        st.caption("👉 Reads like a filter: “take n, but only if …”. Replaces many loops used to clean data.")

        st.subheader("if / else inside a comprehension")
        st.code('["even" if n % 2 == 0 else "odd" for n in range(5)]', language="python")
        st.write(["even" if n % 2 == 0 else "odd" for n in range(5)])
        st.caption("👉 Transform each item into one of two forms — handy for labels.")

        st.subheader("Dict & set comprehensions")
        st.code(
            """{n: n ** 2 for n in range(5)}          # dict
{abs(x) for x in [-3, -1, 2, 2, 3]}    # set (dedupes)""",
            language="python",
        )
        st.write({
            "dict": {n: n ** 2 for n in range(5)},
            "set": sorted({abs(v) for v in [-3, -1, 2, 2, 3]}),
        })
        st.caption("👉 Build lookup tables or unique feature sets in a single expression.")

        st.subheader("Lambdas & map (a function with no name)")
        st.code(
            """double = lambda v: v * 2
list(map(double, range(5)))   # [0, 2, 4, 6, 8]""",
            language="python",
        )
        double = lambda v: v * 2  # noqa: E731
        st.write(list(map(double, range(5))))
        st.caption("👉 Lambdas are tiny throwaway functions — mostly used as the `key=` in `sorted`.")

    # ---------------- Errors & files ----------------
    with t6:
        st.subheader("try / except: survive bad data")
        st.code(
            """values = ["10", "7", "oops", "3"]
cleaned = []
for v in values:
    try:
        cleaned.append(int(v))
    except ValueError:
        print(f"skipping {v!r}")   # one bad value won't crash the loop""",
            language="python",
        )
        values = ["10", "7", "oops", "3"]
        cleaned, skipped = [], []
        for v in values:
            try:
                cleaned.append(int(v))
            except ValueError:
                skipped.append(v)
        st.write({"cleaned": cleaned, "skipped": skipped})
        st.caption("👉 Real-world data is messy. Wrap risky parsing so **one bad row** doesn’t crash the whole job.")

        st.subheader("Catch specific errors, add else / finally")
        st.code(
            """try:
    result = 10 / divisor
except ZeroDivisionError:
    result = None
else:
    print("ok:", result)      # runs only if no error
finally:
    print("done")             # always runs""",
            language="python",
        )
        st.caption("👉 Catch the **specific** error you expect; bare `except:` hides real bugs. `finally` is for cleanup.")

        st.subheader("Reading files safely with 'with'")
        st.code(
            '''from pathlib import Path

path = Path("data") / "trivy_scan.txt"
with open(path, encoding="utf-8") as f:   # auto-closes the file
    lines = [ln.strip() for ln in f if ln.strip()]

print(len(lines), "lines")''',
            language="python",
        )
        scan_path = DATA_DIR / "trivy_scan.txt"
        if scan_path.exists():
            with open(scan_path, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            st.write(f"{len(lines)} lines read")
            st.code("\n".join(lines))
        st.caption("👉 `with` closes the file for you, even if an error happens. This is how you load CSVs, logs and scan reports.")


# --------------------------------------------------------------------------- #
# 3. NumPy
# --------------------------------------------------------------------------- #
def page_numpy():
    st.title("🔢 NumPy")
    st.caption("The numeric engine under pandas, scikit-learn and PyTorch — fast arrays without Python loops.")
    st.markdown(
        "NumPy stores numbers in compact **arrays** and applies an operation to *all* "
        "elements at once (**vectorisation**). That one idea is the biggest speed trick in "
        "data science — and it is the mental model behind tensors in deep learning."
    )

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Create & reshape", "Index & slice", "Vectorised math",
        "Aggregations", "Matrix ops", "Randomness", "Visualise",
    ])

    # ---------------- Create & reshape ----------------
    with t1:
        st.subheader("Ways to build an array")
        st.code(
            """np.zeros((2, 3))        # 2×3 of 0.0  (pre-allocate a buffer)
np.ones(4)             # [1. 1. 1. 1.]
np.full((2, 2), 7)     # every cell = 7
np.arange(0, 10, 2)    # [0 2 4 6 8]   (start, stop, step)
np.linspace(0, 1, 5)   # [0. 0.25 0.5 0.75 1.]  (n evenly spaced)
np.eye(3)              # 3×3 identity matrix""",
            language="python",
        )
        c1, c2 = st.columns(2)
        c1.write({
            "zeros((2,3))": np.zeros((2, 3)).tolist(),
            "arange(0,10,2)": np.arange(0, 10, 2).tolist(),
            "linspace(0,1,5)": np.linspace(0, 1, 5).round(2).tolist(),
        })
        c2.write({
            "full((2,2),7)": np.full((2, 2), 7).tolist(),
            "eye(3)": np.eye(3, dtype=int).tolist(),
        })
        st.caption("👉 Real life: `zeros`/`ones` pre-allocate weight buffers and one-hot masks; "
                   "`linspace` builds the x-axis of a plot or a learning-rate schedule.")

        st.subheader("Reshape: same data, new shape")
        st.code("np.arange(12).reshape(3, 4)   # row-major (C order)", language="python")
        rc1, rc2 = st.columns(2)
        rows = rc1.slider("rows", 1, 12, 3)
        cols = rc2.slider("cols", 1, 12, 4)
        if rows * cols == 12:
            st.dataframe(pd.DataFrame(np.arange(12).reshape(rows, cols)), width="stretch")
            st.caption("👉 Real life: a flat list of 784 pixels becomes a 28×28 image; a batch of "
                       "images is `(batch, height, width, channels)`.")
        else:
            st.info(f"rows × cols must equal 12 (got {rows}×{cols}).")

    # ---------------- Index & slice ----------------
    with t2:
        st.subheader("1-D indexing")
        st.code(
            """a = np.array([10, 20, 30, 40, 50])
a[0]      # 10          (first)
a[-1]     # 50          (last)
a[1:4]    # [20 30 40]  (slice)
a[::2]    # [10 30 50]  (every 2nd)
a[::-1]   # reversed""",
            language="python",
        )
        a = np.array([10, 20, 30, 40, 50])
        st.write({"a[0]": int(a[0]), "a[-1]": int(a[-1]), "a[1:4]": a[1:4].tolist(),
                  "a[::2]": a[::2].tolist(), "a[::-1]": a[::-1].tolist()})

        st.subheader("Boolean masks: filter without a loop")
        st.code(
            """mask = a > 25
a[mask]                 # [30 40 50]
np.where(a > 25, a, 0)  # [0 0 30 40 50]""",
            language="python",
        )
        mask = a > 25
        st.write({"mask": mask.tolist(), "a[a>25]": a[mask].tolist(),
                  "np.where(a>25, a, 0)": np.where(a > 25, a, 0).tolist()})
        st.caption("👉 Real life: keep only samples above a threshold, select one column, or split "
                   "data with `X[train_idx]` / `X[test_idx]`.")

        st.subheader("2-D slicing: rows and columns")
        st.code(
            """m = np.arange(12).reshape(3, 4)
m[1, :]        # row 1
m[:, 2]        # column 2
m[0:2, 2:4]    # top-right block""",
            language="python",
        )
        m = np.arange(12).reshape(3, 4)
        st.dataframe(pd.DataFrame(m), width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.write("row 1"); c1.write(m[1, :])
        c2.write("column 2"); c2.write(m[:, 2])
        c3.write("block [0:2, 2:4]")
        c3.dataframe(pd.DataFrame(m[0:2, 2:4]), width="stretch")

    # ---------------- Vectorised math ----------------
    with t3:
        st.subheader("Whole-array maths (no loops)")
        st.code(
            """a = np.array([1.0, 4.0, 9.0])
a + 1              # add to every element
np.sqrt(a)         # element-wise square root
np.log(a)          # natural log
np.clip(a, 0, 5)   # squeeze values into [0, 5]""",
            language="python",
        )
        a = np.array([1.0, 4.0, 9.0])
        st.write({"a+1": (a + 1).tolist(), "sqrt(a)": np.sqrt(a).round(2).tolist(),
                  "clip(a,0,5)": np.clip(a, 0, 5).tolist()})

        st.subheader("Broadcasting: different shapes, one operation")
        st.code(
            """col = np.array([[1], [2], [3]])    # shape (3, 1)
row = np.array([10, 20, 30, 40])   # shape (4,)
col + row                          # -> shape (3, 4) automatically""",
            language="python",
        )
        col = np.array([[1], [2], [3]])
        row = np.array([10, 20, 30, 40])
        st.dataframe(pd.DataFrame(col + row), width="stretch")
        st.caption("👉 Real life: add a bias vector to a batch, or standardise features with "
                   "`(X - mean) / std` — broadcasting does it with no loop.")

        st.subheader("Why vectorise? It's often 10-100× faster")
        st.code(
            """# slow: a Python loop
out = [x * 2 for x in data]

# fast: one NumPy call (runs in C)
out = data * 2""",
            language="python",
        )
        n = st.slider("array size", 10_000, 1_000_000, 200_000, 10_000)
        data = np.arange(n, dtype=float)
        import time as _time
        t0 = _time.perf_counter()
        _py = [x * 2 for x in data]
        t_loop = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        _np = data * 2
        t_vec = _time.perf_counter() - t0
        c1, c2, c3 = st.columns(3)
        c1.metric("Python loop", f"{t_loop * 1000:.0f} ms")
        c2.metric("NumPy vector", f"{t_vec * 1000:.1f} ms")
        c3.metric("Speed-up", f"{t_loop / max(t_vec, 1e-9):.0f}×")

    # ---------------- Aggregations ----------------
    with t4:
        st.subheader("Summaries in one call")
        st.code(
            """data.sum(), data.mean(), data.std()
data.min(), data.max(), np.median(data)
np.percentile(data, 90)""",
            language="python",
        )
        data = np.random.default_rng(0).normal(50, 10, 1000)
        st.write({"mean": round(float(data.mean()), 2), "std": round(float(data.std()), 2),
                  "min": round(float(data.min()), 2), "max": round(float(data.max()), 2),
                  "median": round(float(np.median(data)), 2),
                  "p90": round(float(np.percentile(data, 90)), 2)})

        st.subheader("axis: summarise down or across")
        st.code(
            """m.sum()          # one number (all elements)
m.sum(axis=0)    # per column (collapse rows)
m.sum(axis=1)    # per row    (collapse columns)""",
            language="python",
        )
        m = np.arange(1, 13).reshape(3, 4)
        st.dataframe(pd.DataFrame(m), width="stretch")
        st.write({"sum axis=0 (per column)": m.sum(axis=0).tolist(),
                  "sum axis=1 (per row)": m.sum(axis=1).tolist()})
        st.code(
            """scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
scores.argmax(axis=1)   # [1 0]  predicted class per row""",
            language="python",
        )
        scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        st.write({"argmax per row": scores.argmax(axis=1).tolist()})
        st.caption("👉 Real life: `loss.mean()` over a batch; per-feature mean/std for scaling; "
                   "`argmax` turns model scores into a predicted class.")

        st.subheader("Running totals & sorting")
        st.code(
            """np.cumsum([1, 2, 3, 4])   # [1 3 6 10]
np.sort(m, axis=1)        # sort each row""",
            language="python",
        )
        st.write({"cumsum": np.cumsum([1, 2, 3, 4]).tolist(),
                  "sort axis=1": np.sort(m, axis=1).tolist()})

    # ---------------- Matrix ops ----------------
    with t5:
        st.subheader("Transpose, product, dot")
        st.code(
            """A = np.array([[1, 2], [3, 4]])
A.T          # transpose
A @ B        # matrix product (preferred)
np.dot(A, B) # same as A @ B
A * B        # element-wise, NOT matrix product""",
            language="python",
        )
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        c1, c2, c3 = st.columns(3)
        c1.write("A.T"); c1.dataframe(pd.DataFrame(A.T), width="stretch")
        c2.write("A @ B"); c2.dataframe(pd.DataFrame(A @ B), width="stretch")
        c3.write("A * B (element-wise)"); c3.dataframe(pd.DataFrame(A * B), width="stretch")

        st.subheader("Linear algebra with np.linalg")
        st.code(
            """np.linalg.inv(A)         # inverse
np.linalg.det(A)         # determinant
np.linalg.solve(A, b)    # solve A x = b  (more stable than inv)""",
            language="python",
        )
        b = np.array([5, 11])
        st.write({"det(A)": round(float(np.linalg.det(A)), 3),
                  "solve(A, b)": np.linalg.solve(A, b).round(3).tolist()})

        st.subheader("A tiny neural layer in two lines")
        st.code(
            """rng = np.random.default_rng(1)
X = rng.normal(size=(4, 3))    # 4 samples, 3 features
W = rng.normal(size=(3, 2))    # 3 inputs -> 2 outputs
b = rng.normal(size=(2,))

out = np.maximum(X @ W + b, 0)   # 'relu' forward pass, whole batch at once""",
            language="python",
        )
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4, 3))
        W = rng.normal(size=(3, 2))
        bb = rng.normal(size=(2,))
        c1, c2 = st.columns(2)
        c1.write("X @ W + b"); c1.dataframe(pd.DataFrame((X @ W + bb).round(2)), width="stretch")
        c2.write("relu(X @ W + b)")
        c2.dataframe(pd.DataFrame(np.maximum(X @ W + bb, 0).round(2)), width="stretch")
        st.caption("👉 Real life: this **is** a dense layer in a neural network. Deep learning is "
                   "mostly `matrix @ weights + bias`, then a non-linear function.")

    # ---------------- Randomness ----------------
    with t6:
        st.subheader("A seed makes randomness reproducible")
        st.code(
            """rng = np.random.default_rng(42)   # fixed seed = repeatable results
rng.normal(size=5)        # bell-shaped values
rng.uniform(0, 1, 5)      # between 0 and 1
rng.integers(0, 10, 5)    # random whole numbers
rng.choice(["a", "b", "c"], 5)
rng.permutation(5)        # shuffled 0..4""",
            language="python",
        )
        seed = st.number_input("seed", value=42, step=1)
        rng = np.random.default_rng(int(seed))
        st.write({
            "normal": rng.normal(size=5).round(2).tolist(),
            "uniform": rng.uniform(0, 1, 5).round(2).tolist(),
            "integers(0,10)": rng.integers(0, 10, 5).tolist(),
            "choice": rng.choice(["a", "b", "c"], 5).tolist(),
        })
        hist_rng = np.random.default_rng(int(seed))
        samples = hist_rng.normal(0, 1, 5000)
        fig, ax = plt.subplots()
        ax.hist(samples, bins=40, color="#4C78A8")
        ax.set_title("5,000 samples from a standard normal")
        st.pyplot(fig)
        st.caption("👉 Real life: initialise neural-network weights, shuffle rows before a "
                   "train/test split, build dropout masks, and run Monte-Carlo simulations. "
                   "Same seed → same result (essential for reproducible experiments).")

    # ---------------- Visualise ----------------
    with t7:
        st.subheader("Plot a function")
        xs = np.linspace(0, 2 * np.pi, 200)
        fig, ax = plt.subplots()
        ax.plot(xs, np.sin(xs), label="sin")
        ax.plot(xs, np.cos(xs), label="cos")
        ax.legend()
        st.pyplot(fig)
        st.code("xs = np.linspace(0, 2*np.pi, 200)\nplt.plot(xs, np.sin(xs))", language="python")

        st.subheader("See a distribution (histogram)")
        dist = np.random.default_rng(7).normal(100, 15, 2000)
        fig, ax = plt.subplots()
        ax.hist(dist, bins=40, color="#72B7B2")
        ax.set_title("Feature values (~normal)")
        st.pyplot(fig)
        st.caption("👉 Real life: sanity-check feature ranges before training and spot skew, "
                   "gaps and outliers.")

        st.subheader("Scatter: relationship between two features")
        srng = np.random.default_rng(3)
        x = srng.normal(size=400)
        y = 0.8 * x + srng.normal(scale=0.5, size=400)
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=12, alpha=0.6)
        ax.set_xlabel("feature x")
        ax.set_ylabel("feature y")
        st.pyplot(fig)
        st.caption("👉 Real life: correlation checks, clustering, and seeing whether classes separate.")

        st.subheader("Heatmap: visualise a correlation matrix")
        mat = np.random.default_rng(5).normal(size=(8, 24))
        corr = np.corrcoef(mat)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation between 8 features")
        st.pyplot(fig)
        st.caption("👉 Real life: a feature-correlation matrix is exactly this heatmap — a fast way "
                   "to spot redundant features before modeling.")

    st.divider()
    st.subheader("Why NumPy matters for AI/ML")
    st.markdown(
        """
- **Speed:** vectorised C loops replace slow Python loops — often **10-100× faster**.
- **Everything is built on it:** pandas, scikit-learn, and PyTorch/TensorFlow tensors all use the same array idea.
- **The language of models:** features are 1-D arrays, a batch of samples is 2-D, images are 3-D, and a forward pass is mostly `X @ W + b`.
- **Reproducibility:** setting a random seed makes experiments repeatable.
- **Rule of thumb:** if you catch yourself writing a `for` loop over numbers, there is almost always a faster NumPy expression.
        """
    )


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
def _viz_customers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "customers.csv").drop_duplicates()
    df["contract"] = df["contract"].str.title()
    for col in ("monthly_charges", "total_charges"):
        df[col] = df[col].fillna(df[col].median())
    return df


def page_visualization():
    st.title("📊 Visualization")
    st.caption("matplotlib for control, seaborn for statistics, plotly for interactivity — all three, done well.")
    sns.set_theme(style="whitegrid", context="notebook")
    st.markdown(
        "Charts are how you **think** about data and how you **convince** others. "
        "A good chart answers one question. The tabs below walk from single plots up to a "
        "full BI-style team dashboard."
    )

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Chart gallery", "Distributions & skew", "Relationships & pairs",
        "Heatmaps", "Feature selection", "Stacked & composition", "BI dashboard",
    ])

    # ---------------- Chart gallery ----------------
    with t1:
        st.subheader("Bar: compare categories")
        sev = pd.DataFrame(
            {"Severity": ["critical", "high", "medium", "low"], "Count": [1, 3, 2, 4]}
        )
        fig = px.bar(sev, x="Severity", y="Count", color="Severity", text="Count",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("👉 Real life: severity counts in a scan, sales by region, errors per service.")

        st.subheader("Combo: bars + a trend line on a second axis")
        months = pd.date_range("2024-01-01", periods=12, freq="MS")
        rev = pd.DataFrame({
            "month": months,
            "revenue": np.linspace(80, 160, 12) + np.random.default_rng(1).normal(0, 6, 12),
            "tickets": np.linspace(400, 250, 12).astype(int),
        })
        fig = px.bar(rev, x="month", y="revenue", color_discrete_sequence=["#8ecae6"])
        fig.add_scatter(x=rev["month"], y=rev["tickets"], mode="lines+markers",
                        name="support tickets", yaxis="y2")
        fig.update_layout(
            template="plotly_white",
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            yaxis_title="revenue", legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("👉 Real life: revenue bars with a metric line is the classic exec chart.")

        st.subheader("Stacked area: composition over time")
        stack = pd.DataFrame(
            np.abs(np.random.default_rng(2).normal(size=(12, 3)).cumsum(axis=0)) + 20,
            columns=["Mobile", "Web", "Tablet"],
        )
        stack["month"] = months
        fig = px.area(stack.melt("month", var_name="channel", value_name="users"),
                      x="month", y="users", color="channel",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Donut: share of a whole")
        fig = px.pie(sev, names="Severity", values="Count", hole=0.55,
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("👉 Real life: a donut reads better than a pie — the hole leaves room for a headline number.")

    # ---------------- Distributions & skew ----------------
    with t2:
        d = _viz_customers()
        st.subheader("Histogram + KDE: shape of one feature")
        fig, ax = plt.subplots()
        sns.histplot(d["monthly_charges"], kde=True, bins=30, color="#219ebc", ax=ax)
        ax.set_title("monthly_charges")
        st.pyplot(fig)

        st.subheader("Skewness: how lopsided is it?")
        cols = ["monthly_charges", "total_charges", "tenure_months", "support_calls"]
        st.write({c: round(float(d[c].skew()), 2) for c in cols})
        st.caption("👉 0 = symmetric. |skew| > 1 is strongly lopsided — many models do better once it's fixed.")

        st.subheader("Fixing skew with a log transform")
        skewed = pd.Series(np.random.default_rng(0).lognormal(mean=0, sigma=0.9, size=2000))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.histplot(skewed, bins=40, color="#e76f51", ax=ax[0])
        ax[0].set_title(f"raw  (skew = {skewed.skew():.2f})")
        sns.histplot(np.log1p(skewed), bins=40, color="#2a9d8f", ax=ax[1])
        ax[1].set_title(f"log1p  (skew = {np.log1p(skewed).skew():.2f})")
        st.pyplot(fig)

        st.subheader("Box & violin: distribution by group")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        sns.boxplot(data=d, x="churn", y="monthly_charges", hue="churn",
                    palette="Set2", legend=False, ax=ax[0])
        ax[0].set_title("box")
        sns.violinplot(data=d, x="churn", y="monthly_charges", hue="churn",
                       palette="Set2", legend=False, inner="quart", ax=ax[1])
        ax[1].set_title("violin")
        st.pyplot(fig)
        st.caption("👉 Real life: compare a metric across segments. Boxes show quartiles and outliers; "
                   "violins reveal the full shape.")

    # ---------------- Relationships & pairs ----------------
    with t3:
        iris = load_iris(as_frame=True).frame
        st.subheader("Scatter + regression line")
        fig, ax = plt.subplots()
        sns.regplot(data=iris, x="petal length (cm)", y="petal width (cm)", ax=ax,
                    scatter_kws={"alpha": 0.5}, line_kws={"color": "#e76f51"})
        st.pyplot(fig)
        st.caption("👉 Real life: does feature A move with feature B? The line + band estimates the trend.")

        st.subheader("Paired chart (pairplot): every feature against every other")
        g = sns.pairplot(iris, hue="target", corner=True, palette="husl",
                         plot_kws={"alpha": 0.6, "s": 18})
        st.pyplot(g.figure)
        st.caption("👉 Real life: the fastest way to spot which features separate classes before modelling.")

        st.subheader("Bubble chart: a third dimension through size")
        gap = px.data.gapminder().query("year == 2007")
        fig = px.scatter(gap, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                         hover_name="country", log_x=True, size_max=55)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("👉 Real life: size encodes a magnitude (population, revenue) without a third axis.")

    # ---------------- Heatmaps ----------------
    with t4:
        d = _viz_customers()
        num = d[["tenure_months", "monthly_charges", "total_charges", "support_calls", "is_senior"]]
        corr = num.corr()
        st.subheader("Correlation heatmap")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        st.pyplot(fig)
        st.caption("👉 Real life: drop one of any pair with |corr| near 1 — redundant features add noise.")

        st.subheader("Clustermap: let the colours group themselves")
        g = sns.clustermap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, figsize=(6.5, 6))
        st.pyplot(g.fig)

    # ---------------- Feature selection ----------------
    with t5:
        st.subheader("Which features actually matter?")
        d = _viz_customers()
        y = (d["churn"] == "Yes").astype(int)
        X = pd.get_dummies(
            d[["tenure_months", "monthly_charges", "total_charges", "support_calls",
               "is_senior", "contract", "internet_service"]],
            columns=["contract", "internet_service"],
        )
        corr_target = X.corrwith(y).sort_values()
        rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation with target (a fast filter)**")
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.barh(corr_target.index, corr_target.values,
                    color=["#e76f51" if v < 0 else "#2a9d8f" for v in corr_target])
            ax.axvline(0, color="black", lw=0.8)
            st.pyplot(fig)
        with c2:
            st.markdown("**Random-forest importance (captures non-linearity)**")
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.barh(imp.index, imp.values, color="#457b9d")
            st.pyplot(fig)
        st.caption("👉 Real life: drop near-useless features to train faster and reduce overfitting. "
                   "Use both views — linear correlation and tree importance can disagree.")

    # ---------------- Stacked & composition ----------------
    with t6:
        d = _viz_customers()
        ct = d.groupby(["contract", "churn"]).size().reset_index(name="customers")
        st.subheader("Stacked bar: churn by contract type")
        fig = px.bar(ct, x="contract", y="customers", color="churn", barmode="stack",
                     color_discrete_sequence=["#2a9d8f", "#e76f51"])
        fig.update_layout(template="plotly_white", legend_title="churn")
        st.plotly_chart(fig, width="stretch")

        st.subheader("100% stacked: compare proportions, not totals")
        prop = ct.copy()
        prop["share"] = prop["customers"] / prop.groupby("contract")["customers"].transform("sum")
        fig = px.bar(prop, x="contract", y="share", color="churn", barmode="stack",
                     color_discrete_sequence=["#2a9d8f", "#e76f51"])
        fig.update_layout(template="plotly_white", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("👉 Real life: 'which segment churns most?' is a proportions question — normalise it.")

        st.subheader("Funnel: how many survive each stage?")
        funnel = pd.DataFrame({
            "stage": ["Visits", "Sign-ups", "Activated", "Paying", "Retained"],
            "n": [10000, 4200, 2600, 1400, 900],
        })
        fig = px.funnel(funnel, x="n", y="stage")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    # ---------------- BI dashboard ----------------
    with t7:
        st.subheader("BI-style team dashboard")
        st.caption("Headline KPIs first, then drill-down charts — the everyday shape of a business dashboard.")
        d = _viz_customers()
        churn_rate = (d["churn"] == "Yes").mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{len(d):,}")
        c2.metric("Churn rate", f"{churn_rate:.1%}")
        c3.metric("Avg monthly", f"${d['monthly_charges'].mean():.0f}")
        c4.metric("Avg tenure", f"{d['tenure_months'].mean():.0f} mo")

        g1, g2 = st.columns(2)
        with g1:
            ct = d.groupby(["contract", "churn"]).size().reset_index(name="customers")
            fig = px.bar(ct, x="contract", y="customers", color="churn", barmode="stack",
                         title="Churn by contract", color_discrete_sequence=["#2a9d8f", "#e76f51"])
            fig.update_layout(template="plotly_white", height=320, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            sns.kdeplot(data=d, x="tenure_months", hue="churn", fill=True, ax=ax,
                        palette={"No": "#2a9d8f", "Yes": "#e76f51"})
            ax.set_title("Tenure by churn")
            st.pyplot(fig)

        g3, g4 = st.columns(2)
        with g3:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            sns.scatterplot(data=d, x="monthly_charges", y="total_charges", hue="churn",
                            alpha=0.6, ax=ax, palette={"No": "#2a9d8f", "Yes": "#e76f51"})
            ax.set_title("Charges vs spend")
            st.pyplot(fig)
        with g4:
            corr = d[["tenure_months", "monthly_charges", "total_charges",
                      "support_calls", "is_senior"]].corr()
            fig, ax = plt.subplots(figsize=(5, 3.6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar=False)
            ax.set_title("Correlation")
            st.pyplot(fig)
        st.caption("👉 Real life: this is the layout a BI analyst ships in Power BI / Tableau / Streamlit — "
                   "KPIs on top, drill-downs below. Same data, one screen, a clear story.")

    st.divider()
    st.subheader("Choosing the right chart")
    st.markdown(
        """
| Question | Chart |
|---|---|
| How do categories compare? | **bar** |
| How does something change over time? | **line / area** |
| What's the distribution of one feature? | **histogram + KDE**, **box**, **violin** |
| Do two features relate? | **scatter / regplot** |
| How do many features relate at once? | **pairplot**, **heatmap**, **clustermap** |
| What's the composition of a whole? | **stacked bar**, **donut**, **funnel** |
| Which features should I keep? | **feature-importance / correlation bars** |
        """
    )
    st.caption("👉 Golden rule: pick the simplest chart that answers your one question — then label it clearly.")


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
# 9. ML Workflow (guided, clickable end-to-end pipeline)
# --------------------------------------------------------------------------- #
MW_STEPS = [
    "1 · Select data",
    "2 · Clean",
    "3 · Filter & features",
    "4 · Explore",
    "5 · Train & evaluate",
    "6 · Pipeline & predict",
]

MW_MODELS = {
    "Logistic regression": lambda: LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def _mw_text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.kind in "OUS"]


def _mw_load_source(source: str) -> pd.DataFrame | None:
    if source.startswith("Bundled"):
        path = DATA_DIR / "customers.csv"
        if not path.exists():
            st.error(f"Missing {path.name} in the data folder.")
            return None
        return pd.read_csv(path)
    if source.startswith("Iris"):
        return load_iris(as_frame=True).frame.rename(columns={"target": "species"})
    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    return pd.read_csv(upload) if upload is not None else None


def _mw_nav(step: int) -> None:
    cols = st.columns(len(MW_STEPS))
    for i, (col, label) in enumerate(zip(cols, MW_STEPS)):
        marker = "🟢" if i < step else ("🔵" if i == step else "⚪")
        if col.button(f"{marker} {label}", key=f"mw_nav_{i}", width="stretch"):
            st.session_state.mw_step = i
            st.rerun()
    back, nxt, _ = st.columns([1, 1, 4])
    if step > 0 and back.button("⬅ Back", key="mw_back"):
        st.session_state.mw_step = step - 1
        st.rerun()
    if step < len(MW_STEPS) - 1 and nxt.button("Next ➡", key="mw_next", type="primary"):
        st.session_state.mw_step = step + 1
        st.rerun()


def page_ml_workflow():
    st.title("🧭 ML Workflow")
    st.caption("Six guided steps: raw data → clean → features → explore → model → reusable pipeline. "
               "Click any step to jump around.")
    if "mw_step" not in st.session_state:
        st.session_state.mw_step = 0
    step = st.session_state.mw_step
    _mw_nav(step)
    st.divider()

    # ---------------- Step 1: select data ----------------
    if step == 0:
        st.subheader("Step 1 · Select your data")
        st.markdown(
            "Every project starts with a **question** and a **table of data**. "
            "The bundled dataset predicts whether a customer will **churn** (cancel)."
        )
        source = st.radio(
            "Data source",
            ["Bundled: customers.csv", "Iris (built-in)", "Upload a CSV"],
            horizontal=True, key="mw_source",
        )
        df = _mw_load_source(source)
        if df is None:
            st.info("Pick a dataset (or upload a CSV) to continue.")
            return
        st.session_state.mw_raw = df
        st.success(f"Loaded **{df.shape[0]} rows × {df.shape[1]} columns**.")
        st.dataframe(df.head(), width="stretch")
        st.write("Column types & missing values")
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
        })
        st.dataframe(info, width="stretch")
        st.caption("👉 Real data is messy: note the missing values and duplicate rows. "
                   "We fix those next — using code like this:")
        with st.expander("Show the pandas code"):
            st.code(
                'import pandas as pd\n'
                'df = pd.read_csv("data/customers.csv")\n'
                "df.head()\n"
                "df.isna().sum()      # count missing values per column\n"
                "df.duplicated().sum()",
                language="python",
            )

    # ---------------- Steps 2-6 need loaded data ----------------
    raw = st.session_state.get("mw_raw")
    if step != 0 and raw is None:
        st.warning("Go back to Step 1 and load a dataset first.")
        return

    # ---------------- Step 2: clean ----------------
    if step == 1:
        st.subheader("Step 2 · Clean the data")
        st.markdown("Cleaning makes the data **consistent and usable**. Choose what to fix:")
        c1, c2 = st.columns(2)
        drop_dupes = c1.checkbox("Remove duplicate rows", value=True)
        drop_constant = c1.checkbox("Remove constant columns", value=True)
        lower_text = c2.checkbox("Trim & lowercase text columns", value=True)
        missing = c2.selectbox(
            "Missing values",
            ["Drop rows with missing", "Fill numeric: median", "Fill numeric: mean"],
        )

        df = raw.copy()
        log = []
        if drop_dupes:
            before = len(df)
            df = df.drop_duplicates()
            log.append(f"removed {before - len(df)} duplicate rows")
        if lower_text:
            for col in _mw_text_columns(df):
                df[col] = df[col].astype(str).str.strip().str.lower()
            log.append("trimmed & lowercased text columns")
        if missing.startswith("Drop"):
            before = len(df)
            df = df.dropna()
            log.append(f"dropped {before - len(df)} rows with missing values")
        else:
            how = "median" if "median" in missing else "mean"
            num_cols = df.select_dtypes(include="number").columns
            fill = df[num_cols].median() if how == "median" else df[num_cols].mean()
            df[num_cols] = df[num_cols].fillna(fill)
            log.append(f"filled numeric gaps with the {how}")
        if drop_constant:
            const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            df = df.drop(columns=const)
            if const:
                log.append(f"dropped constant columns: {const}")

        st.session_state.mw_clean = df
        st.success("Cleaned! " + ("; ".join(log) if log else "No changes selected."))
        m1, m2 = st.columns(2)
        m1.metric("Rows now", len(df))
        m2.metric("Columns now", df.shape[1])
        st.dataframe(df.head(), width="stretch")
        st.caption("👉 Every choice here changes what the model can learn. "
                   "Try 'drop' vs 'fill' and compare accuracy in Step 5.")

    # ---------------- Step 3: filter & features ----------------
    clean = st.session_state.get("mw_clean")
    if step == 2:
        st.subheader("Step 3 · Filter rows & choose features")
        if clean is None:
            st.warning("Do Step 2 (Clean) first.")
            return
        df = clean.copy()

        st.markdown("**Filter rows** (like a SQL `WHERE`)")
        filter_col = st.selectbox("Filter column (optional)", ["<none>"] + df.columns.tolist())
        if filter_col != "<none>":
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                lo, hi = float(df[filter_col].min()), float(df[filter_col].max())
                thr = st.slider(f"Keep rows where {filter_col} ≥", lo, hi, lo)
                df = df[df[filter_col] >= thr]
            else:
                vals = sorted(df[filter_col].dropna().unique().tolist())
                keep = st.multiselect(f"Keep values of {filter_col}", vals, default=vals)
                df = df[df[filter_col].isin(keep)]
            st.caption(f"{len(df)} rows remain after filtering.")

        target = st.selectbox(
            "Target column (what we predict)", df.columns.tolist(),
            index=len(df.columns) - 1,
        )
        feature_options = [c for c in df.columns if c != target]
        features = st.multiselect("Feature columns (the inputs)", feature_options,
                                  default=feature_options)
        encode = st.checkbox("One-hot encode text features", value=True)

        if not features:
            st.warning("Pick at least one feature column.")
            return

        X = df[features].copy()
        y = df[target].copy()
        if encode:
            text_cols = _mw_text_columns(X)
            X = pd.get_dummies(X, columns=text_cols)

        classes = None
        if not pd.api.types.is_numeric_dtype(y):
            classes = list(y.astype("category").cat.categories)
            y = y.astype("category").cat.codes

        st.session_state.mw_X = X
        st.session_state.mw_y = y
        st.session_state.mw_target = target
        st.session_state.mw_classes = classes
        st.session_state.mw_df = df

        st.success(f"Prepared **{X.shape[0]} rows** with **{X.shape[1]} model inputs** "
                   f"to predict `{target}`.")
        st.dataframe(X.head(), width="stretch")
        st.caption("👉 One-hot encoding turns categories like `contract` into 0/1 columns. "
                   "Avoid using an `id` column as a feature — it leaks no real signal.")

    # ---------------- Step 4: explore ----------------
    X = st.session_state.get("mw_X")
    y = st.session_state.get("mw_y")
    if step == 3:
        st.subheader("Step 4 · Explore (EDA)")
        if X is None or y is None:
            st.warning("Finish Step 3 first.")
            return
        st.markdown("**Class balance** — if one class is very rare, accuracy can mislead.")
        st.bar_chart(y.value_counts().sort_index())

        num = X.select_dtypes(include="number")
        if not num.empty:
            col = st.selectbox("Histogram of", num.columns.tolist())
            fig, ax = plt.subplots()
            ax.hist(num[col].dropna(), bins=20, color="#4C78A8")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
            if num.shape[1] >= 2:
                st.markdown("**Correlation heatmap** — which inputs move together?")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.heatmap(num.corr(), annot=num.shape[1] <= 8, cmap="coolwarm",
                            fmt=".2f", ax=ax2)
                st.pyplot(fig2)
        st.caption("👉 EDA happens **before** modeling: it reveals imbalance, skew and "
                   "redundant features you might drop.")

    # ---------------- Step 5: train & evaluate ----------------
    if step == 4:
        st.subheader("Step 5 · Train & evaluate")
        if X is None or y is None:
            st.warning("Finish Step 3 first.")
            return
        c1, c2, c3 = st.columns(3)
        model_name = c1.selectbox("Model", list(MW_MODELS))
        test_pct = c2.slider("Test set size (%)", 10, 50, 25, 5)
        scale = c3.checkbox("Standardize features", value=True,
                            help="Scaling helps distance/linear models, not trees.")

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct / 100, random_state=42, stratify=stratify,
        )
        steps = [("scaler", StandardScaler())] if scale else []
        pipe = Pipeline(steps + [("clf", MW_MODELS[model_name]())])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.metric("Accuracy on unseen test data", f"{acc:.3f}")
        st.caption(f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows.")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        st.pyplot(fig)
        st.code(classification_report(y_test, pred, zero_division=0))

        st.session_state.mw_pipe = pipe
        st.caption("👉 We score on the **test** set the model never saw — that's the only "
                   "honest estimate of real-world performance.")

    # ---------------- Step 6: pipeline, save & predict ----------------
    if step == 5:
        st.subheader("Step 6 · Reusable pipeline, save & predict")
        pipe = st.session_state.get("mw_pipe")
        if pipe is None:
            st.warning("Train a model in Step 5 first.")
            return

        st.markdown("A **Pipeline** bundles preprocessing **and** the model, so new data gets "
                    "the exact same treatment automatically.")
        st.code(
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "import joblib\n\n"
            "pipe = Pipeline([('scaler', StandardScaler()),\n"
            "                 ('clf', LogisticRegression())])\n"
            "pipe.fit(X_train, y_train)\n"
            "joblib.dump(pipe, 'models/ml_workflow.joblib')",
            language="python",
        )
        st.write("Pipeline steps:", " → ".join(name for name, _ in pipe.steps))

        if st.button("💾 Save this pipeline", key="mw_save"):
            path = MODELS_DIR / "ml_workflow.joblib"
            joblib.dump(pipe, path)
            st.success(f"Saved to {path.relative_to(ROOT)}")

        st.markdown("**Live prediction** — feed a new row to the pipeline:")
        show_cols = X.columns.tolist()[:8]
        if len(X.columns) > 8:
            st.caption(f"Showing the first 8 of {len(X.columns)} inputs for space.")
        ncol = st.columns(min(4, len(show_cols)))
        inputs = {}
        for i, col in enumerate(show_cols):
            default = 0.0
            inputs[col] = ncol[i % len(ncol)].number_input(col, value=default,
                                                           key=f"mw_in_{col}")
        if st.button("🔮 Predict", key="mw_predict", type="primary"):
            row = pd.DataFrame([inputs]).reindex(columns=X.columns, fill_value=0)
            result = pipe.predict(row)[0]
            classes = st.session_state.get("mw_classes")
            label = classes[int(result)] if classes else result
            st.success(f"Predicted `{st.session_state.mw_target}` = **{label}**")
        st.caption("👉 Because the pipeline bundles the steps, deployment is just "
                   "`pipe.predict(new_row)` — no risk of forgetting to scale the data.")


# --------------------------------------------------------------------------- #
# 10. LLM chat
# --------------------------------------------------------------------------- #
def page_llm():
    st.title("💬 LLM Chat")
    st.caption("OpenAI-compatible chat & code generation. Works with local Ollama/Qwen.")

    with st.expander("Endpoint settings", expanded=not _llm_configured()):
        model = st.text_input("Model", value="qwen2.5-coder:1.5b")
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
    return bool(os.getenv("LLM_API_KEY") or os.getenv("LLM_BASE_URL"))


# --------------------------------------------------------------------------- #
# 11. LLM Playground (learn how generation settings change the output)
# --------------------------------------------------------------------------- #
KEEP_ALIVE_CHOICES = {
    "5 minutes": "5m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "Unload now": 0,
}

MODEL_GUIDE = """
| Best for | Model | Size | Rough speed on a CPU laptop |
|---|---|---|---|
| Quick code | `qwen2.5-coder:1.5b` | 1 GB | fast (~15 tok/s) |
| General chat | `qwen2.5:3b` | 2 GB | medium (~8 tok/s) |
| Reasoning | `phi3.5:3.8b` | 2.2 GB | medium (~6 tok/s) |
| Higher quality | `qwen2.5-coder:7b` | 4.7 GB | slow (~3 tok/s) |
"""


def page_playground():
    st.title("🧪 LLM Playground")
    st.caption("Change one setting at a time and watch how the answer and the speed change.")

    with st.expander("📚 Start here: what do these settings mean?", expanded=True):
        st.markdown(
            """
A language model writes its answer **one token at a time** (a token is roughly ¾ of a word).
Every setting below decides *how* it chooses the next token.

- **Temperature** — randomness. `0` always picks the most likely token (focused, repeatable).
  Higher values (up to `2`) make the output more surprising and creative.
- **Max tokens** — the longest answer allowed. Smaller = shorter *and* faster.
- **Top-p** — keep only the most likely tokens whose total probability reaches `p`, then pick among them.
- **Top-k** — only ever consider the `k` most likely tokens at each step.
- **Context size** — how many tokens of earlier text the model can keep in mind at once.
- **Keep in memory** — how long the model stays loaded so the next run starts instantly.

The **tokens/second** number shows how fast your machine is generating. On a CPU-only
laptop, small models and short answers are dramatically faster.
            """
        )

    with st.expander("🤔 Which model should I pick?", expanded=False):
        st.markdown(MODEL_GUIDE)
        st.caption("Rule of thumb: use the smallest model that answers well enough for your task.")

    base_default = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    with st.expander("🔌 Model & connection", expanded=True):
        base_url = st.text_input("Ollama URL", value=base_default,
                                 help="Where your local model server runs.")
        models = list_ollama_models(base_url)
        if models:
            labels = [f"{m['name']}  ({m['size'] / 1e9:.1f} GB)" for m in models]
            picked = st.selectbox(
                "Installed model", range(len(labels)),
                format_func=lambda i: labels[i],
                help="These are the models you have already downloaded.",
            )
            model = models[picked]["name"]
        else:
            st.info("Couldn't list local models — is Ollama running? Type a model name below.")
            model = st.text_input("Model", value=os.getenv("LLM_MODEL", "qwen2.5-coder:1.5b"))

    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                            help="0 = focused and repeatable, 2 = very random/creative.")
    max_tokens = c2.slider("Max tokens (answer length)", 32, 1024, 256, 32,
                           help="The model stops after this many tokens. Lower is faster.")
    top_p = c3.slider("Top-p", 0.1, 1.0, 0.9, 0.05,
                      help="Nucleus sampling: consider only the most likely tokens summing to p.")

    c4, c5, c6 = st.columns(3)
    top_k = c4.slider("Top-k", 1, 100, 40, 1,
                      help="Only sample from the k most likely tokens at each step.")
    num_ctx = c5.select_slider("Context size (tokens)", [1024, 2048, 4096, 8192], value=2048,
                               help="How much conversation the model can remember at once.")
    keep_choice = c6.selectbox("Keep model in memory", list(KEEP_ALIVE_CHOICES),
                               help="Staying loaded avoids a slow reload on the next run.")

    system = st.text_input("System prompt (optional)", "You are a friendly, concise tutor.")
    prompt = st.text_area("Your prompt", "Explain what a for-loop does in two sentences.")

    st.caption("Try this: set **Temperature** to `0`, run twice with the same prompt, "
               "and compare — a repeatable model gives almost the same answer. "
               "Then raise it to `1.5` and run again.")

    if st.button("▶️ Run", type="primary"):
        if not model.strip() or not prompt.strip():
            st.warning("Enter a model and a prompt first.")
            return
        messages: list[dict] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": num_ctx,
        }
        with st.spinner(f"Running {model}..."):
            try:
                result = ollama_chat(
                    model, messages, base_url=base_url, options=options,
                    keep_alive=KEEP_ALIVE_CHOICES[keep_choice],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                return

        m1, m2, m3 = st.columns(3)
        m1.metric("Speed", f"{result['tokens_per_second']:.1f} tok/s",
                  help="Tokens generated per second — higher is faster.")
        m2.metric("Tokens generated", result["eval_count"])
        m3.metric("Total time", f"{result['total_seconds']:.1f}s")

        st.subheader("Response")
        st.markdown(result["content"] or "_(empty response)_")
        st.caption(
            f"Prompt tokens: {result['prompt_eval_count']} · "
            f"stop reason: {result['done_reason'] or 'n/a'}"
        )
        with st.expander("What the model actually received"):
            st.json(messages)



# --------------------------------------------------------------------------- #
# 12. RAG
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _build_rag(docs: tuple[tuple[str, str], ...],
               chunk_size: int, overlap: int, line_mode: bool) -> RAGEngine:
    engine = RAGEngine(chunk_size=chunk_size, overlap=overlap,
                       split_on="line" if line_mode else "section")
    for name, text in docs:
        engine.add_text(text, source=name)
    engine.build()
    return engine


def _load_documents(source: str) -> list[tuple[str, str]]:
    """Return a list of (name, raw_text) documents for the chosen source."""
    if source == "Bundled scan(s)":
        available = sorted(p.name for p in DATA_DIR.glob("*.txt"))
        default = ["trivy_scan.txt"] if "trivy_scan.txt" in available else available[:1]
        chosen = st.multiselect("Bundled files", available, default=default)
        return [(n, (DATA_DIR / n).read_text(encoding="utf-8", errors="ignore"))
                for n in chosen]
    if source == "Upload report(s)":
        files = st.file_uploader(
            "Trivy report(s)", type=["txt", "json", "log", "md"],
            accept_multiple_files=True,
        )
        return [(f.name, f.read().decode("utf-8", errors="ignore")) for f in files]
    pasted = st.text_area("Paste a Trivy report", height=200,
                          value=(DATA_DIR / "trivy_scan.txt").read_text(encoding="utf-8")
                          if (DATA_DIR / "trivy_scan.txt").exists() else "")
    return [("pasted_report.txt", pasted)] if pasted.strip() else []


def page_rag():
    st.title("🛡️ RAG over Trivy scans")
    st.caption("Load any Trivy report, tune retrieval, and get grounded answers.")

    with st.expander("📂 Documents & retrieval options", expanded=True):
        source = st.radio(
            "Source", ["Bundled scan(s)", "Upload report(s)", "Paste text"],
            horizontal=True,
        )
        docs = _load_documents(source)

        if docs:
            c1, c2, c3, c4 = st.columns(4)
            chunk_size = c1.slider("Chunk size (words)", 50, 1000, 400, 50)
            overlap = c2.slider("Overlap (words)", 0, 300, 80, 10)
            top_k = c3.slider("Top-K chunks", 1, 10, 3)
            min_score = c4.slider("Min similarity", 0.0, 1.0, 0.0, 0.05)
            line_mode = st.checkbox(
                "One chunk per line (best for Trivy reports)", value=True,
            )
            chosen_sevs = st.multiselect(
                "Severity filter (limits what is indexed)",
                SEVERITIES, default=SEVERITIES,
            )

    if not docs:
        st.info("Add at least one document above to build the index.")
        return

    vulns = []
    for name, raw in docs:
        for v in parse_trivy(raw):
            v.source = name
            vulns.append(v)

    if vulns:
        st.subheader("Detected vulnerabilities")
        counts = {s: sum(1 for v in vulns if v.severity == s) for s in SEVERITIES}
        m = st.columns(len(SEVERITIES))
        sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        for col, (sev, n) in zip(m, counts.items()):
            col.metric(f"{sev_icon.get(sev, '')} {sev}", n)

        st.dataframe(
            pd.DataFrame([
                {"CVE": v.cve, "Package": v.package, "Severity": v.severity,
                 "Fixed version": v.fixed_version, "Source": v.source}
                for v in vulns
            ]),
            width="stretch", hide_index=True,
        )

    indexed = tuple(
        (name, filter_by_severity(trivy_json_to_text(raw), set(chosen_sevs)))
        for name, raw in docs
    )
    engine = _build_rag(indexed, chunk_size, overlap, line_mode)
    st.caption(f"Indexed **{len(engine.chunks)}** chunk(s) from {len(docs)} document(s).")

    with st.expander("View source document(s)"):
        for name, raw in docs:
            st.markdown(f"**{name}**")
            st.text(trivy_json_to_text(raw))

    with st.expander("🤖 Answer settings", expanded=False):
        use_llm = st.checkbox("Answer with the LLM", value=True)
        a1, a2, a3 = st.columns(3)
        model = a1.text_input("Model", value=os.getenv("LLM_MODEL", "qwen2.5-coder:1.5b"))
        base_url = a2.text_input("Base URL", value=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
        temperature = a3.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    presets = [
        "Which CVE is critical and what should I update?",
        "List critical and high vulnerabilities with their fixes.",
        "Summarize the risk by severity.",
        "Which packages should I patch first?",
    ]
    if "rag_query" not in st.session_state:
        st.session_state.rag_query = ""
    pcols = st.columns(len(presets))
    for col, preset in zip(pcols, presets):
        if col.button(preset, width="stretch"):
            st.session_state.rag_query = preset

    query = st.text_input("Question", key="rag_query")
    if not query:
        return

    results = engine.retrieve(query, top_k=top_k, min_score=min_score)
    context = "\n\n".join(
        f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in results
    ) or "(no chunks matched the current filters)"

    st.subheader("Retrieved context")
    st.code(context)
    with st.expander("Similarity scores"):
        st.dataframe(
            pd.DataFrame([
                {"source": c.source, "score": round(c.score, 4), "text": c.text}
                for c in results
            ]),
            width="stretch", hide_index=True,
        )

    st.subheader("Answer")
    if not use_llm:
        st.info("LLM answering is disabled — showing retrieved context only.")
        return
    llm = LLMClient(model=model, base_url=base_url or None, temperature=temperature)
    if not llm.is_live:
        st.warning("No LLM endpoint configured. Set a Base URL or API key.")
        return

    if st.button("🤖 Get LLM answer", type="primary"):
        with st.spinner("Thinking..."):
            st.session_state.rag_answer = {
                "query": query, "context": context,
                "text": llm.rag_answer(query, context),
            }
    last = st.session_state.get("rag_answer")
    if last and last["query"] == query and last["context"] == context:
        st.write(last["text"])
    else:
        st.caption("Click **Get LLM answer** to run the model on the retrieved context.")


# --------------------------------------------------------------------------- #
# 13. RAG for summarization
# --------------------------------------------------------------------------- #
SUMMARY_GOALS = {
    "Executive TL;DR": "a short, 3-5 bullet executive summary of the most important points",
    "Action items": "every action item, with its owner and due date where mentioned",
    "Risks & blockers": "all risks, blockers and open questions, with their context",
    "Decisions log": "decisions that were made and the reason behind each",
    "Recurring themes": "the main themes or topics that repeat across the documents",
    "Custom": "",
}


def _load_summary_documents(source: str) -> list[tuple[str, str]]:
    """Return a list of (name, raw_text) documents for the chosen source."""
    if source == "Bundled notes":
        available = sorted(p.name for p in NOTES_DIR.glob("*.md"))
        chosen = st.multiselect("Bundled notes", available, default=available)
        return [(n, (NOTES_DIR / n).read_text(encoding="utf-8", errors="ignore"))
                for n in chosen]
    if source == "Upload file(s)":
        files = st.file_uploader(
            "Documents", type=["txt", "md", "log"], accept_multiple_files=True,
        )
        return [(f.name, f.read().decode("utf-8", errors="ignore")) for f in files]
    pasted = st.text_area("Paste a document", height=200, value="")
    return [("pasted.txt", pasted)] if pasted.strip() else []


def page_rag_summary():
    st.title("📝 RAG for summarization")
    st.caption("Turn a pile of documents into a focused, grounded summary — and learn what actually needs tuning.")
    st.markdown(
        "Summarizing is RAG's most common everyday job. Instead of stuffing a huge document "
        "into the prompt, you **retrieve the most relevant chunks first, then summarize those**. "
        "That matters because of context limits, cost, and *focus* — a summary of everything is "
        "often less useful than a summary of what you asked about. This page is a small, working "
        "**query-focused summarizer** so you can see the knobs that decide whether the summary is good."
    )

    with st.expander("📂 Documents", expanded=True):
        source = st.radio(
            "Source", ["Bundled notes", "Upload file(s)", "Paste text"], horizontal=True,
        )
        docs = _load_summary_documents(source)

    if not docs:
        st.info("Add at least one document above to build the index.")
        return

    total_words = sum(len(t.split()) for _, t in docs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Words", f"{total_words:,}")
    c3.metric("Rough tokens", f"{int(total_words * 1.3):,}")
    with st.expander("Preview sources"):
        for name, txt in docs:
            st.markdown(f"**{name}** — {len(txt.split())} words")
            st.text(txt[:600] + ("..." if len(txt) > 600 else ""))

    with st.expander("🎯 Summary goal & retrieval tuning", expanded=True):
        goal = st.selectbox("What should the summary focus on?", list(SUMMARY_GOALS))
        custom_goal = ""
        if goal == "Custom":
            custom_goal = st.text_input("Describe the summary you want")
        mode = st.radio(
            "Retrieval mode",
            ["Query-focused (RAG)", "Full corpus (map-reduce)"],
            horizontal=True,
            help="Query-focused retrieves only relevant chunks; Full corpus sends everything.",
        )
        t1, t2, t3, t4 = st.columns(4)
        chunk_size = t1.slider("Chunk size (words)", 50, 800, 200, 50)
        overlap = t2.slider("Overlap (words)", 0, 200, 40, 10)
        top_k = t3.slider("Top-K chunks", 1, 20, 5)
        min_score = t4.slider("Min similarity", 0.0, 0.6, 0.05, 0.01)

    engine = _build_rag(tuple(docs), chunk_size, overlap, line_mode=False)
    n_chunks = len(engine.chunks)
    query = custom_goal.strip() or SUMMARY_GOALS[goal] or goal
    full_corpus = mode.startswith("Full")
    results = engine.retrieve(
        query,
        top_k=n_chunks if full_corpus else top_k,
        min_score=0.0 if full_corpus else min_score,
    )
    used = len(results)
    coverage = used / max(n_chunks, 1)

    st.subheader("Retrieval coverage")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks indexed", n_chunks)
    m2.metric("Chunks sent to the model", used)
    m3.metric("Coverage", f"{coverage:.0%}")
    st.progress(min(coverage, 1.0))
    if not full_corpus and coverage < 0.5:
        st.warning(
            "You're sending less than half the corpus. That's fine for a focused summary, "
            "but if the result feels thin, raise **Top-K** or lower **Min similarity**."
        )

    st.subheader("Evidence passed to the model")
    if results:
        st.dataframe(
            pd.DataFrame([
                {"score": round(c.score, 3), "source": c.source,
                 "chunk": c.text[:140] + ("..." if len(c.text) > 140 else "")}
                for c in results
            ]),
            width="stretch", hide_index=True,
        )
    else:
        st.info("Nothing retrieved — lower Min similarity or switch to Full corpus.")

    context = "\n\n".join(
        f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in results
    ) or "(no chunks matched the current settings)"
    with st.expander("Raw context (exactly what the model sees)"):
        st.code(context)

    with st.expander("🤖 Answer settings"):
        use_llm = st.checkbox("Summarize with the LLM", value=True)
        a1, a2, a3 = st.columns(3)
        model = a1.text_input("Model", value=os.getenv("LLM_MODEL", "qwen2.5-coder:1.5b"))
        base_url = a2.text_input("Base URL", value=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
        temperature = a3.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.subheader("Summary")
    if not use_llm:
        st.info("LLM summarization is disabled — showing retrieved evidence only.")
        return
    llm = LLMClient(model=model, base_url=base_url or None, temperature=temperature)
    if not llm.is_live:
        st.warning("No LLM endpoint configured. Set a Base URL or API key.")
        return

    system = (
        "You are a precise analyst. Summarize ONLY from the provided context. "
        "Never invent facts. Cite sources inline like [file]. "
        "If something is missing, say so explicitly."
    )
    prompt = (
        f"Write {goal.lower()} from the context below.\n"
        "Use this structure:\n"
        "- **TL;DR**: 2-3 sentences.\n"
        "- **Key points**: short bullets, each ending with a source citation.\n"
        "- **Action items / risks**: anything owned or unresolved (or say 'none found').\n"
        "- **Gaps**: what the context does NOT cover.\n\n"
        f"Context:\n{context}"
    )

    key = (mode, goal, query, context, model)
    if st.button("📝 Generate summary", type="primary"):
        with st.spinner("Summarizing..."):
            st.session_state["rag_summary"] = {
                "key": key, "text": llm.ask(prompt, system=system),
            }
    last = st.session_state.get("rag_summary")
    if last and last["key"] == key:
        st.markdown(last["text"])
    else:
        st.caption("Click **Generate summary** to run the model on the retrieved context.")

    st.divider()
    st.subheader("⚙️ The tuning that actually matters")
    st.markdown(
        "A summary is only as good as the **chunks you retrieve**. These are the knobs, "
        "roughly in priority order:"
    )
    st.markdown(
        """
| Knob | Too low | Too high | Rule of thumb |
|---|---|---|---|
| Chunk size | fragments lose context | one chunk mixes topics | 1 paragraph ≈ 150-300 words |
| Overlap | a fact splits across chunks | duplicated text, wasted tokens | ~10-20% of chunk size |
| Top-K | misses relevant content | noise + slower, costlier | start 3-5, raise until coverage plateaus |
| Min similarity | irrelevant chunks reach the model | drops valid chunks | 0.05-0.2; check with the score table |
| Prompt | model rambles or hallucinates | over-constrained, robotic | demand citations + "say if missing" |
        """
    )
    st.subheader("🧠 Key lessons")
    st.markdown(
        """
1. **Retrieval, not the model, is usually the bottleneck.** A weaker model with the right
   chunks beats a strong model fed garbage.
2. **Coverage vs. focus is a trade-off.** Raise Top-K for completeness; lower it for a
   sharper, cheaper summary.
3. **Always show the evidence.** If you cannot see the chunks, you cannot trust — or debug —
   the summary.
4. **Ground the prompt.** "Use only the context, cite sources, admit gaps" cuts
   hallucinations dramatically.
5. **Tune against a small eval set.** Write 5-10 question/summary pairs and check whether the
   *right chunks* show up — not just whether the prose reads nicely.
6. **Watch the token budget.** More context means slower and pricier. Retrieve what you need,
   not everything.
7. **Chunk on structure, not character counts.** Split on headings and paragraphs so each
   chunk stays self-contained.
        """
    )
    st.caption(
        "👉 Try the experiment: change **Top-K** or **Min similarity**, regenerate, and compare. "
        "That single loop is 80% of real RAG tuning."
    )


PAGES = {
    "Home": page_home,
    "Python basics": page_python,
    "NumPy": page_numpy,
    "Pandas": page_pandas,
    "Visualization": page_visualization,
    "Regression": page_regression,
    "Classification": page_classification,
    "Pipeline + Pickle": page_pipeline_pickle,
    "ML Workflow": page_ml_workflow,
    "LLM Chat": page_llm,
    "LLM Playground": page_playground,
    "RAG (Trivy)": page_rag,
    "RAG (summary)": page_rag_summary,
}


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("Local, offline-friendly demo. LLM sections auto-enable with an endpoint.")
    PAGES[choice]()


if __name__ == "__main__":
    main()

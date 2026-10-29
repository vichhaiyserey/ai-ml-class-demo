"""Exercise 01 — Python foundations.

Run:
    .venv\\Scripts\\python exercises\\01_basic_python.py   (Windows)

These are the small, high-leverage ideas you will reuse in every AI/ML task.
Each function prints its own result so you can see exactly what happens.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# --------------------------------------------------------------------------- #
# Types, f-strings, truthiness
# --------------------------------------------------------------------------- #
def basic_types_demo():
    x = 10
    y = 3.14
    name = "AI/ML"
    print(f"Types: {type(x).__name__}, {type(y).__name__}, {type(name).__name__}")


def f_string_demo():
    age = 25
    print(f"Hello, you are {age} years old")


def truthiness_demo():
    # Empty containers, 0, "" and None are "falsy".
    values = [[], 0, "", None, [1], -3, "hi"]
    for v in values:
        print(f"{v!r:<8} -> {bool(v)}")


def numbers_demo():
    print("7 / 2  =", 7 / 2)        # true division -> float
    print("7 // 2 =", 7 // 2)       # floor division -> int
    print("7 % 2  =", 7 % 2)        # remainder
    print("2 ** 10 =", 2 ** 10)     # power
    print("round(3.14159, 2) =", round(3.14159, 2))


def is_vs_equals_demo():
    result = None
    print("result is None   ->", result is None)     # identity: use with None
    print("[1, 2] == [1, 2] ->", [1, 2] == [1, 2])   # equality: compares values


# --------------------------------------------------------------------------- #
# Logic and loops
# --------------------------------------------------------------------------- #
def conditional_demo(score=82):
    if score >= 90:
        label = "excellent"
    elif score >= 75:
        label = "good"
    elif score >= 50:
        label = "needs work"
    else:
        label = "poor"
    print(f"score {score} -> {label}")


def boolean_ops_demo():
    print("True and False ->", True and False)
    print("True or False  ->", True or False)
    print("not True       ->", not True)


def for_loop_demo():
    models = ["logreg", "knn", "svc"]
    for i, m in enumerate(models, start=1):     # position + value
        print(f"  {i}. {m}")
    pairs = zip(["age", "income"], [34, 52000])  # walk two lists together
    print("  zip ->", dict(pairs))


def while_break_demo():
    total, i = 0, 0
    while total < 100:            # repeat until the total is big enough
        i += 1
        total += i
    print(f"while stopped at i={i}, sum={total}")

    odds = []
    for n in range(10):
        if n % 2 == 0:
            continue              # skip even numbers
        if n > 7:
            break                 # stop once n passes 7
        odds.append(n)
    print("odd numbers before 8:", odds)


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def mean(values):
    return sum(values) / len(values)


def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"


def total(*nums):          # any number of positional arguments
    return sum(nums)


def describe(**info):      # any number of named arguments
    return info


def fahrenheit_to_celsius(f: float) -> float:
    """Convert a temperature from Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def scope_demo():
    outer = "I'm outer"

    def inner():
        print("  inner sees:", outer)   # reads the name from the outer scope

    inner()
    print("  outer still:", outer)


def function_demo():
    print("mean([2, 4, 6])      ->", mean([2, 4, 6]))
    print("greet('Ada')         ->", greet("Ada"))
    print("total(1, 2, 3)       ->", total(1, 2, 3))
    print("describe(model='knn')->", describe(model="knn", k=5))
    print("F->C(212)            ->", fahrenheit_to_celsius(212))


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
def slicing_demo():
    nums = list(range(10))
    print("nums[2:5] ->", nums[2:5])
    print("nums[:3]  ->", nums[:3])
    print("nums[-2:] ->", nums[-2:])
    print("nums[::2] ->", nums[::2])
    print("nums[::-1]->", nums[::-1])


def containers_demo():
    print("list   ->", [1, 2, 2, 3])
    print("set    ->", sorted({1, 2, 2, 3, 3}))         # removes duplicates
    print("set &  ->", sorted({1, 2, 3} & {2, 3, 4}))   # overlap
    print("dict   ->", {"a": 1, "b": 2}["b"])           # lookup by key


def methods_demo():
    from collections import Counter

    counts = {"high": 3, "low": 5}
    print("get default ->", counts.get("medium", 0))    # safe, no KeyError
    print("items       ->", list(counts.items()))
    print("Counter     ->", Counter("mississippi").most_common(3))


def sort_and_unpack_demo():
    scores = {"knn": 0.91, "logreg": 0.88, "svc": 0.93}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print("ranked ->", ranked)

    first, *rest = [1, 2, 3, 4]
    print("first, rest ->", first, rest)

    a, b = 1, 2
    a, b = b, a                    # swap in one line
    print("swapped ->", a, b)


# --------------------------------------------------------------------------- #
# Comprehensions
# --------------------------------------------------------------------------- #
def list_comprehension():
    squares = [x ** 2 for x in range(10)]
    print(f"Squares: {squares}")


def comprehension_demo():
    print("with condition ->", [n for n in range(20) if n % 3 == 0])
    print("if/else label  ->", ["even" if n % 2 == 0 else "odd" for n in range(5)])
    print("dict comp      ->", {n: n ** 2 for n in range(5)})
    print("set comp       ->", sorted({abs(v) for v in [-3, -1, 2, 2, 3]}))


def lambda_demo():
    double = lambda x: x * 2
    print(f"Double of 5: {double(5)}")
    print("map ->", list(map(double, range(5))))


# --------------------------------------------------------------------------- #
# Errors and files
# --------------------------------------------------------------------------- #
def error_handling_demo():
    values = ["10", "7", "oops", "3"]
    cleaned, skipped = [], []
    for v in values:
        try:
            cleaned.append(int(v))
        except ValueError:
            skipped.append(v)     # one bad value won't crash the loop
    print("cleaned ->", cleaned)
    print("skipped ->", skipped)


def file_read_demo():
    path = DATA / "trivy_scan.txt"
    if not path.exists():
        print(f"(no file at {path})")
        return
    with open(path, encoding="utf-8") as f:   # 'with' closes the file for us
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"read {len(lines)} non-empty lines from {path.name}")
    for ln in lines:
        print("  ", ln)


if __name__ == "__main__":
    print("== Types, f-strings, truthiness ==")
    basic_types_demo()
    f_string_demo()
    truthiness_demo()
    numbers_demo()
    is_vs_equals_demo()

    print("\n== Logic and loops ==")
    conditional_demo()
    boolean_ops_demo()
    for_loop_demo()
    while_break_demo()

    print("\n== Functions ==")
    function_demo()
    scope_demo()

    print("\n== Collections ==")
    slicing_demo()
    containers_demo()
    methods_demo()
    sort_and_unpack_demo()

    print("\n== Comprehensions ==")
    list_comprehension()
    comprehension_demo()
    lambda_demo()

    print("\n== Errors and files ==")
    error_handling_demo()
    file_read_demo()

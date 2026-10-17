import pandas as pd


def series_demo():
    s = pd.Series([1, 3, 5, 7, 9])
    print(f"Series:\n{s}")


def dataframe_demo():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85, 90, 95]
    }
    df = pd.DataFrame(data)
    print(f"DataFrame:\n{df}")


def indexing():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85, 90, 95]
    }
    df = pd.DataFrame(data)
    print(f"Row 0: {df.iloc[0]}")


def statistics():
    data = {
        'Math': [85, 90, 78, 92, 88],
        'Science': [80, 85, 79, 90, 83]
    }
    df = pd.DataFrame(data)
    print(f"Mean per column:\n{df.mean()}")


def trivia_scan_example():
    """Example: Parse Trivy-like scan data into DataFrame"""
    text = """Found vulnerabilities:
- CVE-2023-12345 in python:high
- CVE-2023-98765 in alpine:medium
- CVE-2022-54321 in node:critical"""
    
    lines = text.strip().split('\n')
    vulns = []
    for line in lines:
        if 'CVE' in line:
            parts = line.split(' in ')
            cve = parts[0].strip().lstrip('- ').strip()
            rest = parts[1].strip().split(':')
            tool = rest[0]
            severity = rest[1]
            vulns.append({'CVE': cve, 'Tool': tool, 'Severity': severity})
    df = pd.DataFrame(vulns)
    print(f"Trivy scan DataFrame:\n{df}")


if __name__ == "__main__":
    series_demo()
    dataframe_demo()
    indexing()
    statistics()
    trivia_scan_example()
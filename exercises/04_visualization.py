import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def matplotlib_basic():
    data = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
    plt.hist(data, bins=5)
    plt.title("Matplotlib Histogram")
    plt.show()


def seaborn_pairplot():
    df = sns.load_dataset('iris')
    sns.pairplot(df, hue='species')
    plt.show()


def plotly_bar():
    fig = px.bar(x=['A', 'B', 'C', 'D'], y=[10, 15, 7, 12],
                 title="Plotly Bar Chart")
    fig.show()


def plotly_scatter():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    fig = go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color='blue'))
    fig.update_layout(title="Plotly Scatter")
    fig.show()


def trivy_visualization():
    """Visualize Trivy scan results"""
    data = {'Severity': ['critical', 'high', 'medium', 'low'], 'Count': [1, 3, 2, 4]}
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Severity', y='Count', title="Trivy Vulnerabilities by Severity")
    fig.show()


if __name__ == "__main__":
    matplotlib_basic()
    seaborn_pairplot()
    plotly_bar()
    plotly_scatter()
    trivy_visualization()
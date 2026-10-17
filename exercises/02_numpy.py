import numpy as np


def array_creation():
    arr = np.array([1, 2, 3, 4, 5])
    print(f"1D array: {arr}")


def matrix_demo():
    matrix = np.array([[1, 2], [3, 4]])
    print(f"Matrix:\n{matrix}")


def operations():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"Addition: {a + b}")
    print(f"Dot product: {np.dot(a, b)}")


def reshaping():
    flat = np.arange(12)
    reshaped = flat.reshape(3, 4)
    print(f"Reshaped (3,4):\n{reshaped}")


def statistics():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Mean: {data.mean()}, Std: {data.std()}")


if __name__ == "__main__":
    array_creation()
    matrix_demo()
    operations()
    reshaping()
    statistics()
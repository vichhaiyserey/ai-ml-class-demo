def basic_types_demo():
    x = 10
    y = 3.14
    name = "AI/ML"
    print(f"Types: {type(x)}, {type(y)}, {type(name)}")


def f_string_demo():
    age = 25
    print(f"Hello, you are {age} years old")


def list_comprehension():
    squares = [x**2 for x in range(10)]
    print(f"Squares: {squares}")


def lambda_demo():
    double = lambda x: x * 2
    print(f"Double of 5: {double(5)}")


def scope_demo():
    outer = "I'm outer"
    def inner():
        print(outer)
    inner()


if __name__ == "__main__":
    basic_types_demo()
    f_string_demo()
    list_comprehension()
    lambda_demo()
    scope_demo()
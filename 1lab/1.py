import numpy as np
import matplotlib.pyplot as plt


class Function:
    def __init__(self, func, deriv, name):
        self.func = func
        self.deriv = deriv
        self.name = name


class Num_Deriv:
    def __init__(self, func, name):
        self.func = func
        self.name = name


def f1(x):
    return np.sin(x**2)


def f2(x):
    return np.cos(np.sin(x))


def f3(x):
    return np.exp(np.sin(np.cos(x)))


def f4(x):
    return np.log(x + 3)


def f5(x):
    return np.sqrt(x + 3)


def adf1(x):  # adf - analytical derivative of function
    return 2 * x * np.cos(x**2)


def adf2(x):
    return -np.sin(np.sin(x)) * np.cos(x)


def adf3(x):
    return -np.sin(x) * np.cos(np.cos(x)) * np.exp(np.sin(np.cos(x)))


def adf4(x):
    return 1 / (x + 3)


def adf5(x):
    return 1 / (2 * np.sqrt(x + 3))


def ndf1(f, x, h):  # ndf - numerical derivative of function
    return (f(x + h) - f(x)) / h


def ndf2(f, x, h):
    return (f(x) - f(x - h)) / h


def ndf3(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def ndf4(f, x, h):
    return (4 / 3) * ((f(x + h) - f(x - h)) / (2 * h)) - (1 / 3) * (
        (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    )


def ndf5(f, x, h):
    return (
        (3 / 2) * ((f(x + h) - f(x - h)) / (2 * h))
        - (3 / 5) * ((f(x + 2 * h) - f(x - 2 * h)) / (4 * h))
        + (1 / 10) * ((f(x + 3 * h) - f(x - 3 * h)) / (6 * h))
    )


x = 4 # для всех функций эта точка хороша, т.к. нет особенностей

h = [(2 / 2**i) for i in range(1, 22)]

functions = [
    Function(f1, adf1, "$\\sin{(x^2)}$"),
    Function(f2, adf2, "$\\cos{(\\sin{(x)})}$"),
    Function(f3, adf3, "$\\exp{(\\sin{(\\cos{(x)})})}$"),
    Function(f4, adf4, "$\\ln{(x + 3)}$"),
    Function(f5, adf5, "$(x + 3)^{0.5}$"),
]

num_derivs = [
    Num_Deriv(ndf1, "$\\frac{df}{dx}(x) = \\frac{f(x+h) - f(x)}{h}$"),
    Num_Deriv(ndf2, "$\\frac{df}{dx}(x) = \\frac{f(x) - f(x-h)}{h}$"),
    Num_Deriv(ndf3, "$\\frac{df}{dx}(x) = \\frac{f(x+h) - f(x-h)}{2h}$"),
    Num_Deriv(
        ndf4,
        "$\\frac{df}{dx}(x) = \\frac{4}{3}\\frac{f(x+h) - f(x-h)}{2h} - \\frac{1}{3}\\frac{f(x+2h) - f(x-2h)}{4h}$",
    ),
    Num_Deriv(
        ndf5,
        "$\\frac{df}{dx}(x) = \\frac{3}{2}\\frac{f(x+h) - f(x-h)}{2h} - \\frac{3}{5}\\frac{f(x+2h) - f(x-2h)}{4h} + \\frac{1}{10}\\frac{f(x+3h) - f(x-3h)}{6h}$",
    ),
]

func_1 = [[], [], [], [], []]
func_2 = [[], [], [], [], []]
func_3 = [[], [], [], [], []]
func_4 = [[], [], [], [], []]
func_5 = [[], [], [], [], []]

data = [func_1, func_2, func_3, func_4, func_5]

for arr, f in zip(data, functions):
    for func_arr, num_deriv in zip(arr, num_derivs):
        for step in h:
            func_arr.append(np.fabs(num_deriv.func(f.func, x, step) - f.deriv(x)))


for arr, f in zip(data, functions):
    plt.figure(figsize=[14, 7])

    plt.title(f.name, fontsize=30)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("$\\log{h}$", fontsize=20)

    plt.ylabel("$\\log{(\\text{error})}$", fontsize=20)

    plt.grid(which="both")
    for func_arr, num_deriv in zip(arr, num_derivs):
        plt.scatter(np.array(h), np.array(func_arr), label=num_deriv.name)
        plt.plot(np.array(h), np.array(func_arr))
        plt.legend(loc="best")
        plt.tight_layout()

plt.show()

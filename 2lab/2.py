import numpy as np
import math
import matplotlib.pyplot as plt

e = 1e-6


def dot(u, v):
    return float(np.sum(u * v))


def matvec(A, x):
    """Поэлементное умножение матрицы A на вектор x"""
    n = len(A)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        y[i] = s
    return y


def norm_1(x):
    return np.linalg.norm(x, ord=np.inf)


def norm_2(x):
    return np.linalg.norm(x, ord=1)


def norm_3(x):
    return np.linalg.norm(x, ord=2)


norms = [norm_1, norm_2, norm_3]


def residual(A, x, b):
    return b - matvec(A, x)


# -----------------------
# 1. МЕТОД ГАУССА
# -----------------------
def gauss(A_in, b_in):
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    n = len(A)
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(A[i][k]))
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[k], b[pivot] = b[pivot], b[k]
        for i in range(k + 1, n):
            if abs(A[k][k]) < e:
                continue
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


# -----------------------
# 2. LU-РАЗЛОЖЕНИЕ
# -----------------------
def lu_decompose(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - s
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                s = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - s) / U[i][i]
    return L, U


def lu_solve(A, b):
    L, U = lu_decompose(A)
    n = len(A)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


# -----------------------
# 3. МЕТОД ЯКОБИ
# -----------------------
def jacobi(A, b, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    res = [[], [], []]
    for _ in range(maxiter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        r = residual(A, x_new, b)
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if res[0][-1] < e and res[1][-1] < e and res[2][-1] < e:
            break
        x = x_new
    return x, res


# -----------------------
# 4. МЕТОД ЗЕЙДЕЛЯ
# -----------------------
def seidel(A, b, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    res = [[], [], []]
    for _ in range(maxiter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        r = residual(A, x, b)
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if res[0][-1] < e and res[1][-1] < e and res[2][-1] < e:
            break
    return x, res


# -----------------------
# 5. МЕТОД ВЕРХНЕЙ РЕЛАКСАЦИИ
# -----------------------
def sor(A, b, omega=1.2, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    res = [[], [], []]
    for _ in range(maxiter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] += omega * ((b[i] - s) / A[i][i] - x[i])
        r = residual(A, x, b)
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if res[0][-1] < e and res[1][-1] < e and res[2][-1] < e:
            break
    return x, res


# -----------------------
# 6. ГРАДИЕНТНЫЙ СПУСК
# -----------------------
def grad_descent(A, b, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    r = residual(A, x, b)
    res = [[norm_1(r)], [norm_2(r)], [norm_3(r)]]
    for _ in range(maxiter):
        Ar = matvec(A, r)
        alpha = dot(r, r) / dot(r, Ar)
        x = x + alpha * r
        r = residual(A, x, b)
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if res[0][-1] < e and res[1][-1] < e and res[2][-1] < e:
            break
    return x, res


# -----------------------
# 7. МИНИМАЛЬНЫХ НЕВЯЗОК
# -----------------------
def min_residual(A, b, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    r = residual(A, x, b)
    res = [[norm_1(r)], [norm_2(r)], [norm_3(r)]]
    for _ in range(maxiter):
        Ar = matvec(A, r)
        alpha = dot(Ar, r) / dot(Ar, Ar)
        x = x + alpha * r
        r = residual(A, x, b)
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if min(res[0][-1], res[1][-1], res[2][-1]) < e:
            break
    return x, res

# -----------------------
# 8. СТАБИЛИЗИРОВАННЫЙ МЕТОД БИСОПРЯЖЕННЫХ ГРАДИЕНТОВ
# -----------------------
def bicgstab(A, b, maxiter=1000):
    n = len(A)
    x = np.zeros(n)
    r = residual(A, x, b)
    r_hat = r.copy()
    rho_old = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)
    res = [[norm_1(r)], [norm_2(r)], [norm_3(r)]]
    for _ in range(maxiter):
        rho = dot(r_hat, r)
        if abs(rho) < e:
            break
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = matvec(A, p)
        alpha = rho / (dot(r_hat, v) + e)
        s = r - alpha * v
        t = matvec(A, s)
        omega = dot(t, s) / (dot(t, t) + e)
        x = x + alpha * p + omega * s
        r = s - omega * t
        for i in range(len(norms)):
            res[i].append(norms[i](r))
        if res[0][-1] < e and res[1][-1] < e and res[2][-1] < e:
            break
        rho_old = rho
    return x, res


# -----------------------
# Генерация матрицы (вариант е)
# -----------------------
n = 100
a = 10.0

A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            A[i][j] = a
        else:
            A[i][j] = 1.0 / (i + 1)

b = np.arange(1, n + 1, dtype=float)

print("Решение прямыми методами:")
x_gauss = gauss(A, b)
x_lu = lu_solve(A, b)
print("Гаусс:", x_gauss)
print("LU:", x_lu)

print("\nИтерационные методы:")

methods = {
    "Якоби": jacobi,
    "Зейдель": seidel,
    "Метод верхней релаксации": lambda A, b: sor(A, b, omega=1.25),
    "Градиентный спуск": grad_descent,
    "Мин. невязок": min_residual,
    "Стабилизированный метод бисопряженных градиентов": bicgstab,
}

for name, func in methods.items():
    x, res = func(A, b)
    norm1, norm2, norm3 = res[0], res[1], res[2]
    print(
        f"{name}: norm1 = {res[0][-1]:.2e}, norm2 = {res[1][-1]:.2e}, norm3 = {res[2][-1]:.2e}"
    )
    print(x)

    plt.figure(figsize=(10, 6))

    plt.yscale("log")

    plt.scatter(np.arange(1, len(norm1) + 1), np.array(norm1), label="norm1")
    plt.plot(np.arange(1, len(norm1) + 1), np.array(norm1))

    plt.scatter(np.arange(1, len(norm2) + 1), np.array(norm2), label="norm2")
    plt.plot(np.arange(1, len(norm2) + 1), np.array(norm2))

    plt.scatter(np.arange(1, len(norm3) + 1), np.array(norm3), label="norm3")
    plt.plot(np.arange(1, len(norm3) + 1), np.array(norm3))

    plt.xlabel("Номер итерации")
    plt.ylabel("log(невязка)")
    plt.title(name)
    plt.legend()
    plt.grid(True)

plt.show()

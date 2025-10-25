import numpy as np


def partial_pivoting_gauss(A: np.ndarray, b: np.ndarray):
    n = len(A)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for i in range(n):
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        if abs(Ab[i, i]) < 1e-12:
            raise ValueError("Matrix is singular")
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]
    return x


def lu(A: np.ndarray):
    n = len(A)
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row], k:] = U[[max_row, k], k:]
            # swap only the columns < k of L
            L[[k, max_row], :k] = L[[max_row, k], :k]
            P[[k, max_row]] = P[[max_row, k]]

        if abs(U[k, k]) < 1e-12:
            raise ValueError("Matrix is singular")

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def solve_by_lu(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray):
    n = len(b)
    Pb = P.dot(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x

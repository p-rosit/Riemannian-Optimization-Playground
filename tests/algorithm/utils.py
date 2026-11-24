import numpy as np
from riemopt.manifold.base import Array
from riemopt.algorithm.function import Function


def rayleigh_quotient(a: Array) -> Function:
    m, n = a.shape
    assert m == n, f'expected square matrix, got {(m, n)}'

    def value(x: Array) -> float:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'
        x = np.reshape(x, (n, 1))

        return (x.T @ a @ x) / (x.T @ x)

    def gradient(x: Array) -> Array:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'
        x = np.reshape(x, (n, 1))

        g = np.reshape(2 * (a @ x - value(x) * x) / (x.T @ x), (n, ))
        return g

    return Function(function=value, gradient=gradient)

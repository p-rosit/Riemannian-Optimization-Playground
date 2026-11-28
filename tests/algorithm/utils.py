import numpy as np
from riemopt.typing import Array
import riemopt.function as rf


def rayleigh_quotient(a: Array) -> rf.Function:
    m, n = a.shape
    assert m == n, f'expected square matrix, got {(m, n)}'

    def value(x: Array) -> float:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'
        return -float((x.T @ a @ x) / (x.T @ x))

    def gradient(x: Array) -> Array:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'

        g = np.reshape(2 * (a @ x - value(x) * x) / (x.T @ x), (n, ))
        return -g

    return rf.Function(function=value, gradient=gradient)

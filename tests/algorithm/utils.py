import numpy as np
import riemopt.manifold as rm
import riemopt.algorithm as ra


def rayleigh_quotient(a: rm.Array) -> ra.Function:
    m, n = a.shape
    assert m == n, f'expected square matrix, got {(m, n)}'

    def value(x: rm.Array) -> float:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'
        return -float((x.T @ a @ x) / (x.T @ x))

    def gradient(x: rm.Array) -> rm.Array:
        (length, ) = x.shape
        assert length == n, f'expected vector of length {n}, got {length}'

        g = np.reshape(2 * (a @ x - value(x) * x) / (x.T @ x), (n, ))
        return -g

    return ra.Function(function=value, gradient=gradient)

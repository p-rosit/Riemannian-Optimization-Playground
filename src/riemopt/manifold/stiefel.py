import numpy as np
from riemopt.manifold.base import Manifold, Array


class Stiefel(Manifold):
    def __init__(self, n: int, k: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        if k <= 0:
            raise ValueError(f"expected positive amount of vectors, got {k}")

        self.dim = n * k - int(k * (k + 1) / 2)
        self.shape = (n, k)

    def _point_project(self, point: Array) -> Array:
        u, _, vt = np.linalg.svd(point, full_matrices=False)
        return u @ vt

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        product = point.T @ vector
        symmetric_part = (product + product.T) / 2
        return vector - point @ symmetric_part

    def _retract(self, point: Array, tangent: Array) -> Array:
        q, _ = np.linalg.qr(point + tangent, 'reduced')
        return q

import numpy as np
from riemopt.typing import Array
from riemopt.manifold.base import Manifold, NotProjectableError


class Stiefel(Manifold):
    def __init__(self, n: int, k: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        if k <= 0:
            raise ValueError(f"expected positive amount of vectors, got {k}")
        if k > n:
            raise ValueError(f"expected less vectors than dimensions, got ({n}, {k})")

        self.dim = n * k - int(k * (k + 1) / 2)
        self.shape = (n, k)

    def _tangent_dot(self, point: Array, v1: Array, v2: Array) -> float:
        _ = point
        return np.trace(v1.T @ v2)

    def _point_project(self, point: Array) -> Array:
        u, s, vt = np.linalg.svd(point, full_matrices=False)
        # s sorted descending
        if s[-1] <= s[0] * max(self.shape) * np.finfo(s.dtype).eps:
            raise NotProjectableError
        return u @ vt

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        product = point.T @ vector
        symmetric_part = (product + product.T) / 2
        return vector - point @ symmetric_part

    def _retract(self, point: Array, tangent: Array) -> Array:
        q, r = np.linalg.qr(point + tangent, 'reduced')
        return q * np.sign(np.diag(r))

    def _riemannian_gradient(self, point: Array, euclidean_gradient: Array) -> Array:
        return self._tangent_project(point, euclidean_gradient)

    def _riemannian_hessian(
        self,
        point: Array,
        euclidean_gradient: Array,
        euclidean_hessian_v: Array,
        tangent: Array,
    ) -> Array:
        product = point.T @ euclidean_gradient
        symmetric_part = (product + product.T) / 2
        return self._tangent_project(point, euclidean_hessian_v - tangent @ symmetric_part)

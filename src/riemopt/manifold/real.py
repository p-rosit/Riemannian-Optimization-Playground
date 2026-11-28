import numpy as np
from riemopt.typing import Array
from riemopt.manifold.base import Manifold


class Real(Manifold):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        self.dim = n
        self.shape = (n, )

    def _tangent_dot(self, point: Array, v1: Array, v2: Array) -> float:
        _ = point
        return np.dot(v1, v2)

    def _point_random(self, rng: np.random.Generator) -> Array:
        return rng.normal(size=self.shape)

    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        _ = point
        return rng.normal(size=self.shape)

    def _point_project(self, point: Array) -> Array:
        return point

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        _ = point
        return vector

    def _riemannian_gradient(self, point: Array, euclidean_gradient: Array) -> Array:
        _ = point
        return euclidean_gradient

    def _riemannian_hessian(
        self,
        point: Array,
        euclidean_gradient: Array,
        euclidean_hessian_v: Array,
        tangent: Array,
    ) -> Array:
        _ = point, euclidean_gradient, tangent
        return euclidean_hessian_v

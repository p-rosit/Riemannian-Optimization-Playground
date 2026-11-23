import numpy as np
from riemopt.manifold.base import Manifold, Array


class Sphere(Manifold):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        self.dim = n
        self.shape = (n + 1, )

    def _point_random(self, rng: np.random.Generator) -> Array:
        p, norm = None, np.inf
        while norm == 0 or norm > 1.0:
            p = rng.random(size=self.shape)
            norm = np.linalg.norm(p)
        assert p is not None, "unreachable error"
        return p / norm

    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        return self._tangent_project(point, rng.normal(size=self.shape))

    def _point_project(self, point: Array) -> Array:
        if (norm := np.linalg.norm(point)) <= 0.0:
            raise ZeroDivisionError
        return point / norm

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        return vector - np.dot(point, vector) * point
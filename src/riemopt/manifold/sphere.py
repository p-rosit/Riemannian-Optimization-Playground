import numpy as np
from riemopt.manifold.base import Manifold, Array


class Sphere(Manifold):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        self.dim = n
        self.shape = (n + 1, )

    def _tangent_dot(self, point: Array, v1: Array, v2: Array) -> float:
        _ = point
        return np.dot(v1, v2)

    def _point_random(self, rng: np.random.Generator) -> Array:
        p, norm = None, 0.0
        while norm == 0.0:
            p = rng.normal(size=self.shape)
            norm = np.linalg.norm(p)
        assert p is not None, "unreachable error"
        return p / norm

    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        return self._tangent_project(point, rng.normal(size=self.shape))

    def _point_project(self, point: Array) -> Array:
        if (scale := np.max(np.abs(point))) == 0:
            raise ZeroDivisionError
        point = point / scale
        return point / np.linalg.norm(point)

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        return vector - np.dot(point, vector) * point

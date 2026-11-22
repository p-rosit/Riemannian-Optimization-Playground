import numpy as np
from riemopt.manifold.base import Manifold, Array


class Real(Manifold):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"expected positive dimension, got {n}")
        self.dim = n
        self.shape = (n, )

    def _point_random(self, rng: np.random.Generator) -> Array:
        return self._shaped(rng.normal(size=self.shape))

    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        _ = point
        return self._shaped(rng.normal(size=self.shape))

    def _point_project(self, point: Array) -> Array:
        return point

    def _tangent_project(self, point: Array, vector: Array) -> Array:
        _ = point
        return vector
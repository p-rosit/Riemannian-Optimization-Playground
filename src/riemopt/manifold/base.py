from abc import ABC, abstractmethod
import numpy as np

Array = np.typing.NDArray[np.float64]


class Manifold(ABC):
    dim: int
    shape: tuple[int, ...]

    def point_random(self, rng: np.random.Generator | int | None = None) -> Array:
        return self._point_random(np.random.default_rng(rng))

    @abstractmethod
    def _point_random(self, rng: np.random.Generator) -> Array:
        pass

    def tangent_random(self, point: Array, rng: np.random.Generator | int | None = None) -> Array:
        return self._tangent_random(point, np.random.default_rng(rng))

    @abstractmethod
    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        pass

    def retract(self, point: Array, tangent: Array) -> Array:
        return self.point_project(self._shaped(point + tangent))

    @abstractmethod
    def point_project(self, point: Array) -> Array:
        pass

    @abstractmethod
    def tangent_project(self, point: Array, vector: Array) -> Array:
        pass

    def _shaped(self, x: np.ndarray) -> Array:
        if x.shape != self.shape:
            raise ValueError(f"expected {self.shape}, got {x.shape}")
        return x
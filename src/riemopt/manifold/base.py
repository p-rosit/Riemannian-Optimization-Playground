from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar, cast
import numpy as np

S = TypeVar('S', bound=tuple[int, ...])
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)

Array = np.ndarray[S, np.dtype[np.float64]]


class Manifold(ABC, Generic[S]):
    dim: int
    shape: tuple[int, ...]

    def random(self, rng: np.random.Generator | int | None = None) -> Array[S]:
        return self._random(np.random.default_rng(rng))

    @abstractmethod
    def _random(self, rng: np.random.Generator) -> Array[S]:
        pass

    def retract(self, point: Array[S], tangent: Array[S]) -> Array[S]:
        return self.point_project(self._shaped(point + tangent))

    @abstractmethod
    def point_project(self, point: Array[S]) -> Array[S]:
        pass

    @abstractmethod
    def tangent_project(self, point: Array[S], vector: Array[S]) -> Array[S]:
        pass

    def _shaped(self, x: np.ndarray) -> Array[S]:
        assert x.shape == self.shape, f"expected {self.shape}, got {x.shape}"
        return cast('Array[S]', x)
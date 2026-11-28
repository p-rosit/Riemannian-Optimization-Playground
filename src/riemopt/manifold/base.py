from typing import final
from abc import ABC, abstractmethod
import numpy as np
from riemopt.typing import Array


class NotProjectableError(Exception):
    pass


class Manifold(ABC):
    dim: int
    shape: tuple[int, ...]

    @final
    def tangent_dot(self, point: Array, v1: Array, v2: Array) -> float:
        return self._tangent_dot(self._shaped(point), self._shaped(v1), self._shaped(v2))

    @final
    def tangent_norm(self, point: Array, vector: Array) -> float:
        return self._tangent_norm(self._shaped(point), self._shaped(vector))

    @final
    def point_random(self, rng: np.random.Generator | int | None = None) -> Array:
        return self._shaped(self._point_random(np.random.default_rng(rng)))

    @final
    def tangent_random(self, point: Array, rng: np.random.Generator | int | None = None) -> Array:
        return self._shaped(self._tangent_random(point, np.random.default_rng(rng)))

    @final
    def retract(self, point: Array, tangent: Array) -> Array:
        return self._shaped(self._retract(point, tangent))

    @final
    def point_project(self, point: Array) -> Array:
        return self._shaped(self._point_project(self._shaped(point)))

    @final
    def tangent_project(self, point: Array, vector: Array) -> Array:
        return self._shaped(self._tangent_project(self._shaped(point), self._shaped(vector)))

    @final
    def riemannian_gradient(self, point: Array, euclidean_gradient: Array) -> Array:
        return self._shaped(self._riemannian_gradient(self._shaped(point), self._shaped(euclidean_gradient)))

    @final
    def riemannian_hessian(
        self,
        point: Array,
        euclidean_gradient: Array,
        euclidean_hessian_v: Array,
        tangent: Array,
    ) -> Array:
        return self._shaped(self._riemannian_hessian(
            self._shaped(point),
            self._shaped(euclidean_gradient),
            self._shaped(euclidean_hessian_v),
            self._shaped(tangent),
        ))

    @abstractmethod
    def _tangent_dot(self, point: Array, v1: Array, v2: Array) -> float:
        pass

    def _tangent_norm(self, point: Array, vector: Array) -> float:
        return float(np.sqrt(self._tangent_dot(point, vector, vector)))

    def _point_random(self, rng: np.random.Generator) -> Array:
        N = 100
        for _ in range(N):
            point = 2 * rng.random(size=self.shape) - 1
            try:
                return self.point_project(point)
            except NotProjectableError:
                continue
        raise TimeoutError(f'could not sample manifold within {N} tries')

    def _tangent_random(self, point: Array, rng: np.random.Generator) -> Array:
        N = 100
        for _ in range(N):
            vector = 2 * rng.random(size=self.shape) - 1
            try:
                return self.tangent_project(point, vector)
            except NotProjectableError:
                continue
        raise TimeoutError(f'could not sample manifold within {N} tries')

    def _retract(self, point: Array, tangent: Array) -> Array:
        return self.point_project(self._shaped(point + tangent))

    @abstractmethod
    def _point_project(self, point: Array) -> Array:
        pass

    @abstractmethod
    def _tangent_project(self, point: Array, vector: Array) -> Array:
        pass

    @abstractmethod
    def _riemannian_gradient(self, point: Array, euclidean_gradient: Array) -> Array:
        pass

    @abstractmethod
    def _riemannian_hessian(
        self,
        point: Array,
        euclidean_gradient: Array,
        euclidean_hessian_v: Array,
        tangent: Array,
    ) -> Array:
        pass

    def _shaped(self, x: np.ndarray) -> Array:
        if x.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {x.shape}")
        return x

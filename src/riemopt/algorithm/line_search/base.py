from abc import ABC, abstractmethod
from riemopt.manifold.base import Array, Manifold
from riemopt.algorithm.function import Function


class LineSearch(ABC):
    @abstractmethod
    def search(self, manifold: Manifold, function: Function, point: Array, gradient: Array, step_size: float) -> tuple[float, Array]:
        pass


class NoSearch(LineSearch):
    def search(self, manifold: Manifold, function: Function, point: Array, gradient: Array, step_size: float) -> tuple[float, Array]:
        _ = function
        return step_size, manifold.retract(point, -step_size * gradient)

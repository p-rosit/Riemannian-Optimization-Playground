from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from riemopt.manifold.base import Array, Manifold
from riemopt.algorithm.function import Function

if TYPE_CHECKING:
    from riemopt.algorithm.base import OptimizationEvals


class LineSearch(ABC):
    @abstractmethod
    def search(self, manifold: Manifold, function: Function, point: Array, gradient: Array, step_size: float, evals: 'OptimizationEvals | None' = None) -> tuple[float, Array]:
        pass


class NoSearch(LineSearch):
    def search(self, manifold: Manifold, function: Function, point: Array, gradient: Array, step_size: float, evals: 'OptimizationEvals | None' = None) -> tuple[float, Array]:
        _ = function, evals
        return step_size, manifold.retract(point, -step_size * gradient)

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from riemopt.manifold.base import Array, Manifold
from riemopt.function import Function, Evals


class LineSearch(ABC):
    @abstractmethod
    def search(self, manifold: Manifold, function: Function, point: Array, descent_direction: Array, step_size: float, evals: Evals | None = None) -> tuple[float, Array]:
        pass


class NoSearch(LineSearch):
    def search(self, manifold: Manifold, function: Function, point: Array, descent_direction: Array, step_size: float, evals: Evals | None = None) -> tuple[float, Array]:
        _ = function, evals
        return step_size, manifold.retract(point, step_size * descent_direction)

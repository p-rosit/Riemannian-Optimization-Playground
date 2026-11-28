from abc import ABC, abstractmethod
from riemopt.typing import Array
from riemopt.function.base import Function, Evals
from riemopt.manifold.base import Manifold


class LineSearch(ABC):
    @abstractmethod
    def search(self, manifold: Manifold, function: Function, point: Array, descent_direction: Array, step_size: float, evals: Evals | None = None) -> tuple[float, Array]:
        pass


class NoLineSearch(LineSearch):
    def search(self, manifold: Manifold, function: Function, point: Array, descent_direction: Array, step_size: float, evals: Evals | None = None) -> tuple[float, Array]:
        _ = function, evals
        return step_size, manifold.retract(point, step_size * descent_direction)

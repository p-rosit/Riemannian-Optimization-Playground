from typing import Callable
import dataclasses
from riemopt.typing import Array
from riemopt.manifold.base import Manifold


@dataclasses.dataclass
class Evals:
    function_evals: int = 0
    gradient_evals: int = 0

    def add(self, evals: 'Evals') -> None:
        self.function_evals += evals.function_evals
        self.gradient_evals += evals.gradient_evals


@dataclasses.dataclass
class Function:
    function: Callable[[Array], float]
    gradient: Callable[[Array], Array] | None = None

    def val(self, x: Array, evals: Evals | None = None) -> float:
        if evals is not None:
            evals.function_evals += 1
        return self.function(x)

    def grad(self, manifold: Manifold, x: Array, evals: Evals | None = None) -> Array:
        return manifold.tangent_project(x, self._grad(x, evals))

    def _grad(self, x: Array, evals: Evals | None = None) -> Array:
        if self.gradient is not None:
            if evals is not None:
                evals.gradient_evals += 1
            return self.gradient(x)
        raise NotImplementedError('Numerical approximation')
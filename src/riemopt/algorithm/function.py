from typing import Callable, TYPE_CHECKING
import dataclasses
from riemopt.manifold.base import Array, Manifold

if TYPE_CHECKING:
    from riemopt.algorithm.base import OptimizationEvals


@dataclasses.dataclass
class Function:
    function: Callable[[Array], float]
    gradient: Callable[[Array], Array] | None = None

    def val(self, x: Array, evals: 'OptimizationEvals | None' = None) -> float:
        if evals is not None:
            evals.function_evals += 1
        return self.function(x)

    def grad(self, manifold: Manifold, x: Array, evals: 'OptimizationEvals | None' = None) -> Array:
        return manifold.tangent_project(x, self._grad(x, evals))

    def _grad(self, x: Array, evals: 'OptimizationEvals | None' = None) -> Array:
        if self.gradient is not None:
            if evals is not None:
                evals.gradient_evals += 1
            return self.gradient(x)
        raise NotImplementedError('Numerical approximation')


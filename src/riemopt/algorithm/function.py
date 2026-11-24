from typing import Callable, TYPE_CHECKING
import dataclasses
from riemopt.manifold.base import Array

if TYPE_CHECKING:
    from riemopt.algorithm.base import OptimizationEvals


@dataclasses.dataclass
class Function:
    function: Callable[[Array], float]
    gradient: Callable[[Array], Array] | None = None

    def val(self, x: Array, evals: 'OptimizationEvals | None') -> float:
        if evals is not None:
            evals.function_evals += 1
        return self.function(x)

    def grad(self, x: Array, evals: 'OptimizationEvals | None') -> Array:
        if self.gradient is not None:
            if evals is not None:
                evals.gradient_evals += 1
            return self.gradient(x)
        raise NotImplementedError('Numerical approximation')


from typing import Callable
import dataclasses
from riemopt.manifold.base import Array


@dataclasses.dataclass
class Function:
    function: Callable[[Array], float]
    gradient: Callable[[Array], Array] | None = None

    def val(self, x: Array) -> float:
        return self.function(x)

    def grad(self, x: Array) -> Array:
        if self.gradient is not None:
            return self.gradient(x)
        raise NotImplementedError('Numerical approximation')
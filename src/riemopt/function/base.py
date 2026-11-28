from typing import Callable, final
from abc import ABC, abstractmethod
import dataclasses
from riemopt.typing import Array
from riemopt.manifold.base import Manifold


@dataclasses.dataclass
class Evals:
    function_evals: int = 0
    gradient_evals: int = 0
    hessian_evals: int = 0

    def add(self, evals: 'Evals') -> None:
        self.function_evals += evals.function_evals
        self.gradient_evals += evals.gradient_evals
        self.hessian_evals += evals.hessian_evals


@dataclasses.dataclass
class Function(ABC):
    function: Callable[[Array], float]

    @final
    def val(self, x: Array, evals: Evals | None = None) -> float:
        if evals is not None:
            evals.function_evals += 1
        return self.function(x)

    @final
    def grad(self, manifold: Manifold, x: Array, evals: Evals | None = None) -> Array:
        return manifold.riemannian_gradient(x, self.euclidean_grad(x, evals))

    @final
    def hess(self, manifold: Manifold, x: Array, evals: Evals | None = None) -> Callable[[Array], Array]:
        euclidean_grad = self.euclidean_grad(x, evals)
        return lambda v: manifold.riemannian_hessian(
            x,
            euclidean_grad,
            self.euclidean_hess(x, v, evals),
            v,
        )

    @abstractmethod
    def euclidean_grad(self, x: Array, evals: Evals | None) -> Array:
        pass

    @abstractmethod
    def euclidean_hess(self, x: Array, v: Array, evals: Evals | None) -> Array:
        pass


@dataclasses.dataclass
class FunctionNumerical(Function):
    gradient: Callable[[Array], Array] | None = None
    hessian: Callable[[Array, Array], Array] | None = None

    def euclidean_grad(self, x: Array, evals: Evals | None) -> Array:
        if self.gradient is not None:
            if evals is not None:
                evals.gradient_evals += 1
            return self.gradient(x)
        raise NotImplementedError('Numerical approximation')

    def euclidean_hess(self, x: Array, v: Array, evals: Evals | None) -> Array:
        if self.hessian is not None:
            if evals is not None:
                evals.hessian_evals += 1
            return self.hessian(x, v)
        raise NotImplementedError('Numerical approximation')
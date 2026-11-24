from abc import ABC, abstractmethod
from riemopt.manifold.base import Array, Manifold
from riemopt.algorithm.function import Function


class LineSearch(ABC):
    @abstractmethod
    def search(self, manifold: Manifold, function: Function) -> tuple[float, Array]:
        pass


class NoSearch(LineSearch):
    pass
    # def search(self) -> Array:
    #     pass

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from riemopt.algorithm.function import Function
from riemopt.manifold.base import Manifold, Array


@dataclasses.dataclass
class OptimizationEvals:
    function_evals: int = 0
    gradient_evals: int = 0

    def add(self, evals: OptimizationEvals) -> None:
        self.function_evals += evals.function_evals
        self.gradient_evals += evals.gradient_evals


@dataclasses.dataclass
class OptimizationStep:
    step: int
    step_size: float
    point: Array = dataclasses.field(repr=False)
    gradient: Array = dataclasses.field(init=False, repr=False)
    function_value: float = dataclasses.field(init=False)
    gradient_norm: float = dataclasses.field(init=False)
    evals: OptimizationEvals

    function: dataclasses.InitVar[Function]

    def __post_init__(self, function: Function):
        self.function_value = function.val(self.point)
        self.gradient = function.grad(self.point)
        self.gradient_norm = float(np.linalg.norm(self.gradient))


@dataclasses.dataclass
class OptimizationResult:
    converged: bool | None = dataclasses.field(default=None, init=False)
    value_opt: float | None = dataclasses.field(default=None, init=False)
    point_opt: Array | None = dataclasses.field(default=None, init=False)
    evals: OptimizationEvals = dataclasses.field(default_factory=lambda: OptimizationEvals(function_evals=0, gradient_evals=0), init=False)
    history: list[OptimizationStep] = dataclasses.field(default_factory=list, init=False, repr=False)

    function: dataclasses.InitVar[Function]
    point_start: dataclasses.InitVar[Array]

    def __post_init__(self, function: Function, point_start: Array):
        self.history = [OptimizationStep(
            function=function,
            step=0,
            step_size=0.0,
            point=point_start,
            evals=OptimizationEvals(function_evals=0, gradient_evals=0),
        )]

    def append(
        self,
        function: Function,
        step_size: float,
        point: Array,
        evals: OptimizationEvals,
    ) -> None:
        last = self.history[-1]
        self.history.append(OptimizationStep(
            step=last.step + 1,
            step_size=step_size,
            point=point,
            function=function,
            evals=evals,
        ))
        self.evals.add(evals)

    def set_converged(self, function: Function, point_opt: Array, converged: bool) -> None:
        assert self.converged is None, 'result has already been marked converged/did not converge'
        self.converged = converged
        if converged:
            self.point_opt = point_opt
            self.value_opt = function.val(point_opt)


@dataclasses.dataclass(kw_only=True)
class Optimization(ABC):
    manifold: Manifold
    gen: np.random.Generator = dataclasses.field(init=False)

    rng: dataclasses.InitVar[np.random.Generator | int | None] = None

    def __post_init__(self, rng: np.random.Generator | int | None = None):
        self.gen = np.random.default_rng(rng)

    @abstractmethod
    def minimize(self, function: Function, point_start: Array | None = None) -> OptimizationResult:
        pass

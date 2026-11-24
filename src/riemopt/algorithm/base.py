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
    gradient: Array = dataclasses.field(repr=False)
    function_value: float
    gradient_norm: float
    evals: OptimizationEvals


@dataclasses.dataclass
class OptimizationResult:
    converged: bool = dataclasses.field(default=False, init=False)
    value_opt: float | None = dataclasses.field(default=None, init=False)
    point_opt: Array | None = dataclasses.field(default=None, init=False)
    evals: OptimizationEvals = dataclasses.field(default_factory=lambda: OptimizationEvals(function_evals=0, gradient_evals=0), init=False)
    history: list[OptimizationStep] = dataclasses.field(default_factory=list, init=False, repr=False)

    function: dataclasses.InitVar[Function]
    point_start: dataclasses.InitVar[Array]

    def __post_init__(self, function: Function, point_start: Array):
        self.history = [OptimizationStep(
            step=0,
            step_size=0.0,
            point=point_start,
            gradient=(gradient_start := function.grad(point_start)),
            function_value=function.val(point_start),
            gradient_norm=float(np.linalg.norm(gradient_start)),
            evals=OptimizationEvals(function_evals=1, gradient_evals=1),
        )]

    def last(self) -> OptimizationStep:
        return self.history[-1]

    def append(
        self,
        function: Function,
        step_size: float,
        point: Array,
        evals: OptimizationEvals,
    ) -> None:
        last = self.last()
        self.history.append(OptimizationStep(
            step=last.step + 1,
            step_size=step_size,
            point=point,
            gradient=(g := function.grad(point)),
            function_value=function.val(p),
            gradient_norm=float(np.linalg.norm(g)),
            evals=evals,
        ))
        self.evals.add(evals)



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

import dataclasses
import numpy as np
from riemopt.typing import Array
from riemopt.function import Function, Evals
from riemopt.algorithm.base import OptimizationWithLineSearch, OptimizationResult


@dataclasses.dataclass
class GradientDescent(OptimizationWithLineSearch):
    step_size: float
    max_steps: int
    gradient_threshold: float

    def minimize(self, function: Function, point_start: Array | None = None) -> OptimizationResult:
        if point_start is None:
            point = self.manifold.point_random(self.gen)
        else:
            point = point_start

        step_size = self.step_size
        gradient_norm = np.inf
        result = OptimizationResult(manifold=self.manifold, function=function, point_start=point)

        while len(result.history) < self.max_steps + 1:
            # current step evals
            e = Evals()

            # current gradient
            gradient = function.grad(self.manifold, point, evals=e)

            if (gradient_norm := self.manifold.tangent_norm(point, gradient)) <= self.gradient_threshold:
                result.evals.add(e)
                break

            # step along gradient
            step_size, p_next = self.line_search.search(self.manifold, function, point, -gradient, step_size, e)

            point = p_next
            result.append(
                manifold=self.manifold,
                function=function,
                step_size=step_size,
                point=p_next,
                evals=e
            )

        result.set_converged(
            function=function,
            point_opt=point,
            converged=gradient_norm <= self.gradient_threshold,
        )
        return result

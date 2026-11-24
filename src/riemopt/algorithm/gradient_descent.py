import dataclasses
import numpy as np
from riemopt.algorithm.base import OptimizationWithLineSearch, OptimizationResult, OptimizationEvals
from riemopt.algorithm.function import Function
from riemopt.manifold.base import Array


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

        point = point_start
        gradient_norm = np.inf
        result = OptimizationResult(function=function, point_start=point)

        while len(result.history) < self.max_steps + 1:
            # current step evals
            e = OptimizationEvals()

            # current gradient
            gradient = function.grad(self.manifold, point, evals=e)

            if (gradient_norm := float(np.linalg.norm(gradient))) <= self.gradient_threshold:
                break

            # step along gradient
            p_next, step_size = self.line_search.search(function, point, gradient, step_size)
            # p_next = self.manifold.retract(point, self.step_size * g_projected)

            point = p_next
            result.append(
                function=function,
                step_size=self.step_size,
                point=p_next,
                evals=e
            )

        result.set_converged(
            function=function,
            point_opt=point,
            converged=gradient_norm <= self.gradient_threshold,
        )
        return result

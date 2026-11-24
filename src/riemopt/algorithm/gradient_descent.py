import dataclasses
import numpy as np
from riemopt.algorithm.base import Optimization, OptimizationResult, OptimizationEvals
from riemopt.algorithm.function import Function
from riemopt.manifold.base import Array


@dataclasses.dataclass
class GradientDescent(Optimization):
    step_size: float
    max_steps: int
    gradient_threshold: float

    def minimize(self, function: Function, point_start: Array | None = None) -> OptimizationResult:
        if point_start is None:
            point = self.manifold.point_random(self.gen)
        else:
            point = point_start

        point = point_start
        result = OptimizationResult(function=function, point_start=point)

        while len(result.history) < self.max_steps + 1:
            # current step evals
            e = OptimizationEvals()

            # current gradient
            g = function.grad(point, evals=e)

            if float(np.linalg.norm(g)) <= self.gradient_threshold:
                break

            # project to T_pM
            g_projected = self.manifold.tangent_project(point, g)

            # step along gradient
            p_next = self.manifold.retract(point, self.step_size * g_projected)

            point = p_next
            result.append(
                function=function,
                step_size=self.step_size,
                point=p_next,
                evals=e
            )

        last_step = result.last()
        result.set_converged(
            function=function,
            point_opt=last_step.point,
            converged=last_step.gradient_norm <= self.gradient_threshold,
        )
        return result

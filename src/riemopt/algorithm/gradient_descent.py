import dataclasses
import numpy as np
from riemopt.algorithm.base import Optimization, OptimizationResult, OptimizationStep, OptimizationEvals
from riemopt.algorithm.function import Function
from riemopt.manifold.base import Manifold, Array


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

        result = OptimizationResult(function=function, point_start=point)

        while len(result.history) < self.max_steps + 1:
            previous_step = result.last()
            if previous_step.gradient_norm <= self.gradient_threshold:
                break

            # current step
            e = OptimizationEvals(function_evals=0, gradient_evals=0)

            # current point
            p = previous_step.point

            # current gradient
            g = function.grad(p, evals=e)

            # project to T_pM
            g_projected = self.manifold.tangent_project(p, g)

            # step along gradient
            p_next = self.manifold.retract(p, self.step_size * g_projected)

            result.append(
                function=function,
                step_size=self.step_size,
                point=p_next,
                evals=e
            )

        last_step = history[-1]
        return OptimizationResult(
            point_opt=last_step.point,
            value_opt=last_step.function_value,
            converged=last_step.gradient_norm <= self.gradient_threshold,
            evals=OptimizationEvals(
                function_evals=sum(h.evals.function_evals for h in history),
                gradient_evals=sum(h.evals.gradient_evals for h in history),
            ),
            history=history
        )
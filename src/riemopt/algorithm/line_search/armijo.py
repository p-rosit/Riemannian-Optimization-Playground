from typing import TYPE_CHECKING
import dataclasses
import numpy as np
from riemopt.manifold.base import Manifold, Array
from riemopt.algorithm.function import Function
from riemopt.algorithm.line_search.base import LineSearch

if TYPE_CHECKING:
    from riemopt.algorithm.base import OptimizationEvals


@dataclasses.dataclass
class ArmijoBacktracking(LineSearch):
    rho: float = 0.5
    c1: float = 1e-4
    max_iters: int = 50

    def search(
        self,
        manifold: Manifold,
        function: Function,
        point: Array,
        descent_direction: Array,
        step_size: float,
        evals: 'OptimizationEvals | None' = None,
    ) -> tuple[float, Array]:
        value = function.val(point, evals)
        gradient = function.grad(manifold, point, evals)

        projected_direction = manifold.tangent_project(point, descent_direction)
        slope = np.dot(gradient, projected_direction)
        if slope >= 0:
            raise ValueError(f'did not get a descent direction: grad.T @ direction = {slope}')

        next_point: Array | None = None
        next_step = step_size / self.rho  # Increase initial step size to ensure it isn't just decreasing

        for _ in range(self.max_iters):
            next_point = manifold.retract(point, next_step * descent_direction)
            if function.val(next_point, evals) <= value + self.c1 * next_step * slope:
                return next_step, next_point

            next_step = self.rho * next_step

        # Ran out of iterations, give up and accept small step
        assert next_point is not None
        return next_step, next_point

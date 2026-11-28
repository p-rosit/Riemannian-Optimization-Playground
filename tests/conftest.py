from typing import Generator
import inspect
import numpy as np
from hypothesis import assume, strategies as st
from hypothesis.extra.numpy import arrays
from riemopt.typing import Array
import riemopt.manifold as rm


def _yield_concrete_subclasses(cls: type[rm.Manifold]) -> Generator[type[rm.Manifold], None, None]:
    if not inspect.isabstract(cls):
        yield cls
    for sub in cls.__subclasses__():
        yield from _yield_concrete_subclasses(sub)


def pytest_generate_tests(metafunc):
    if "manifold" in metafunc.fixturenames:
        metafunc.parametrize("manifold", list(_yield_concrete_subclasses(rm.Manifold)))


@st.composite
def _arrays(draw, shape: tuple[int, ...], k: int):
    floats = st.floats(-1e6, 1e6)
    return tuple(draw(arrays(np.float64, shape, elements=floats)) for _ in range(k))


def draw_arrays(data, manifold: rm.Manifold, amount: int) -> tuple[Array, ...]:
    return data.draw(_arrays(manifold.shape, amount))


def try_point_project(manifold: rm.Manifold, point: Array) -> Array:
    try:
        return manifold.point_project(point)
    except rm.NotProjectableError:
        assume(False)


@st.composite
def stiefels(draw, max_n=20):
    n = draw(st.integers(min_value=1, max_value=max_n))
    k = draw(st.integers(min_value=1, max_value=n))  # k <= n
    return rm.Stiefel(n, k)


builds_manifold = {
    rm.Real: st.builds(rm.Real, st.integers(min_value=1, max_value=20)),
    rm.Sphere: st.builds(rm.Sphere, st.integers(min_value=1, max_value=20)),
    rm.Stiefel: stiefels(),
}


def draw_manifold(manifold_type: type[rm.Manifold], data) -> rm.Manifold:
    if manifold_type not in builds_manifold:
        raise NotImplementedError(f'missing sampling method for {manifold_type}')
    return data.draw(builds_manifold[manifold_type])

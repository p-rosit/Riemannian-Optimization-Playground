import pytest
import inspect
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from riemopt.manifold import (
    Manifold,
    Real,
    Sphere,
    Stiefel,
)


@pytest.fixture
def manifold():
    def yield_concrete_classes(cls: type[Manifold]):
        if not inspect.isabstract(cls):
            yield cls
        yield from iter(cls.__subclasses__())
    yield yield_concrete_classes(Manifold)


@st.composite
def _arrays(draw, shape: tuple[int, ...], k: int):
    floats = st.floats(-1e6, 1e6)
    arr = [draw(arrays(np.float64, shape, elements=floats)) for _ in range(k)]
    return *arr


def draw_arrays(data, manifold: Manifold, amount: int) -> tuple[Array, ...]:
    return data.draw(_arrays(manifold.shape, amount))


@st.composite
def stiefels(draw, max_n=20):
    n = draw(st.integers(min_value=1, max_value=max_n))
    k = draw(st.integers(min_value=1, max_value=n))  # k <= n
    return Stiefel(n, k)


builds_manifold = {
    Real: st.builds(Real, st.integers(min_value=1, max_value=20)),
    Sphere: st.builds(Sphere, st.integers(min_value=1, max_value=20)),
    Stiefel: stiefels(),
}


def draw_manifold(manifold_type: type[Manifold], data) -> Manifold:
    if manifold_type not in builds_manifold:
        raise NotImplementedError(f'missing sampling method for {manifold_type}')
    return data.draw(builds_manifold[manifold_type])

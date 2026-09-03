"""The contract that every optimizer implementation must satisfy.

These tests pin down `BaseOptimizer` and `OptimizeResult` themselves, so
that the concrete first- and second-order methods have a stable target to
be written against.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from og_optimizers import BaseOptimizer, FloatArray, OptimizeResult


class _StubOptimizer(BaseOptimizer):
    """Minimal concrete subclass: takes a single steepest-descent step.

    Deliberately not a useful algorithm. Its job is to prove the abstract
    interface can be implemented and that the inherited configuration
    reaches a subclass unchanged.
    """

    def minimize(
        self,
        fun: Callable[[FloatArray], float],
        x0: FloatArray,
        jac: Callable[[FloatArray], FloatArray] | None = None,
        hess: Callable[[FloatArray], FloatArray] | None = None,
    ) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64)
        step = x if jac is None else x - 1e-3 * jac(x)
        trajectory = np.stack([x, step])
        return OptimizeResult(
            x=step,
            fun=fun(step),
            nit=1,
            converged=False,
            trajectory=trajectory,
        )


def test_base_optimizer_cannot_be_instantiated() -> None:
    """`minimize` is abstract, so the base class is not constructible."""
    with pytest.raises(TypeError, match="abstract"):
        BaseOptimizer()  # type: ignore[abstract]


def test_subclass_without_minimize_cannot_be_instantiated() -> None:
    """Inheriting is not enough; the abstract method must be implemented."""

    class _Incomplete(BaseOptimizer):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # type: ignore[abstract]


def test_default_configuration() -> None:
    """Documented defaults, asserted so they cannot drift silently."""
    optimizer = _StubOptimizer()
    assert optimizer.max_iter == 1000
    assert optimizer.tol == 1e-6


def test_configuration_is_forwarded_to_subclass() -> None:
    """A subclass inherits __init__, so overrides must reach it intact."""
    optimizer = _StubOptimizer(max_iter=25, tol=1e-10)
    assert optimizer.max_iter == 25
    assert optimizer.tol == 1e-10


def test_minimize_returns_a_populated_result() -> None:
    """The concrete implementation satisfies the declared return type."""
    from og_optimizers import rosenbrock, rosenbrock_grad

    x0 = np.array([-1.2, 1.0])
    result = _StubOptimizer().minimize(rosenbrock, x0, jac=rosenbrock_grad)

    assert isinstance(result, OptimizeResult)
    assert result.x.shape == x0.shape
    assert isinstance(result.fun, float)
    assert result.nit == 1
    assert result.converged is False
    # One row per visited point, each with the dimension of the problem.
    assert result.trajectory.shape == (2, x0.shape[0])


def test_result_is_immutable() -> None:
    """`OptimizeResult` is a frozen dataclass: callers cannot rewrite the
    outcome of a solve after the fact."""
    result = OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        nit=0,
        converged=True,
        trajectory=np.zeros((1, 2)),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.fun = 1.0  # type: ignore[misc]


def test_result_exposes_the_documented_fields() -> None:
    """Guards against a field being renamed out from under consumers."""
    fields = {f.name for f in dataclasses.fields(OptimizeResult)}
    assert fields == {"x", "fun", "nit", "converged", "trajectory"}

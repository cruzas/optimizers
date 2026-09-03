"""Correctness of the Rosenbrock function and its derivatives.

The analytical gradient and Hessian are hand-derived and vectorized with
overlapping slice updates, which is exactly the kind of code where an
off-by-one in the overlap is easy to write and hard to see. Every
analytical derivative is therefore cross-checked against a finite
-difference approximation, which shares none of that implementation.
"""

from collections.abc import Callable

import numpy as np
import pytest

from og_optimizers import (
    FloatArray,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)
from tests.conftest import DIMENSIONS

# Step size for central differences. Balances truncation error (order
# h**2) against cancellation error (order eps/h); ~1e-5 is near the
# optimum for double precision on a well-scaled problem.
FD_STEP = 1e-5


def _finite_difference_jacobian(
    func: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    step: float = FD_STEP,
) -> FloatArray:
    """Central-difference Jacobian of a vector-valued ``func`` at ``x``.

    Used for the Hessian check by passing the analytical gradient as
    ``func``; the scalar case is handled by wrapping the result.
    """
    n = x.shape[0]
    columns = []
    for i in range(n):
        offset = np.zeros(n, dtype=np.float64)
        offset[i] = step
        forward = np.atleast_1d(func(x + offset))
        backward = np.atleast_1d(func(x - offset))
        columns.append((forward - backward) / (2.0 * step))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_minimum_is_zero_at_all_ones(n: int) -> None:
    """With a = 1 the minimizer is the all-ones vector, where f = 0."""
    assert rosenbrock(np.ones(n)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("a", [0.0, 1.0])
def test_zero_minimum_at_the_fixed_points_of_a(n: int, a: float) -> None:
    """Both squared terms vanish at ``x_i = a`` only if ``a == a**2``.

    That leaves a = 0 and a = 1 as the only parameters for which the
    minimum is exactly zero, and the constant vector as the minimizer.
    """
    x_star = np.full(n, a, dtype=np.float64)
    assert rosenbrock(x_star, a=a) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        rosenbrock_grad(x_star, a=a), np.zeros(n), atol=1e-9
    )


@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("a", [0.5, 2.0])
def test_minimum_is_strictly_positive_off_the_fixed_points(
    n: int, a: float
) -> None:
    """For any other ``a`` the two terms cannot vanish simultaneously,
    so no point drives f to zero -- in particular not the constant
    vector that would be the minimizer if they could."""
    assert rosenbrock(np.full(n, a), a=a) > 0.0


@pytest.mark.parametrize("n", DIMENSIONS)
def test_function_is_non_negative(n: int, rng: np.random.Generator) -> None:
    """Both terms are squares, so f can never go below zero."""
    for _ in range(20):
        assert rosenbrock(rng.uniform(-3.0, 3.0, size=n)) >= 0.0


def test_gradient_matches_finite_differences(
    sample_points: list[FloatArray],
) -> None:
    """The analytical gradient agrees with central differences."""
    for x in sample_points:
        expected = _finite_difference_jacobian(
            lambda p: np.atleast_1d(rosenbrock(p)), x
        ).ravel()
        np.testing.assert_allclose(
            rosenbrock_grad(x), expected, rtol=1e-6, atol=1e-6
        )


def test_hessian_matches_finite_differences(
    sample_points: list[FloatArray],
) -> None:
    """The analytical Hessian agrees with differences of the gradient."""
    for x in sample_points:
        expected = _finite_difference_jacobian(rosenbrock_grad, x)
        np.testing.assert_allclose(
            rosenbrock_hess(x), expected, rtol=1e-6, atol=1e-6
        )


def test_hessian_is_symmetric(sample_points: list[FloatArray]) -> None:
    """Second derivatives commute, so the Hessian must be symmetric."""
    for x in sample_points:
        hessian = rosenbrock_hess(x)
        np.testing.assert_allclose(hessian, hessian.T, rtol=0.0, atol=0.0)


def test_hessian_is_tridiagonal(sample_points: list[FloatArray]) -> None:
    """Only neighbouring coordinates are coupled, so bands beyond the
    first off-diagonal must be exactly zero."""
    for x in sample_points:
        hessian = rosenbrock_hess(x)
        n = hessian.shape[0]
        rows, cols = np.indices((n, n))
        beyond_band = np.abs(rows - cols) > 1
        np.testing.assert_array_equal(hessian[beyond_band], 0.0)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_hessian_positive_definite_at_minimum(n: int) -> None:
    """A strict local minimum requires a positive-definite Hessian."""
    eigenvalues = np.linalg.eigvalsh(rosenbrock_hess(np.ones(n)))
    assert eigenvalues.min() > 0.0


@pytest.mark.parametrize("n", DIMENSIONS)
def test_hessian_ignores_parameter_a(n: int, rng: np.random.Generator) -> None:
    """``a`` enters f only through ``(a - x)**2``, whose second
    derivative is the constant 2, so the Hessian cannot depend on it."""
    x = rng.uniform(-2.0, 2.0, size=n)
    np.testing.assert_array_equal(
        rosenbrock_hess(x, a=1.0), rosenbrock_hess(x, a=7.0)
    )


@pytest.mark.parametrize("n", DIMENSIONS)
def test_shapes_and_dtypes(n: int, rng: np.random.Generator) -> None:
    """Callers rely on these shapes; the abstract interface types them."""
    x = rng.uniform(-2.0, 2.0, size=n)
    assert isinstance(rosenbrock(x), float)
    assert rosenbrock_grad(x).shape == (n,)
    assert rosenbrock_hess(x).shape == (n, n)
    assert rosenbrock_grad(x).dtype == np.float64
    assert rosenbrock_hess(x).dtype == np.float64


def test_larger_b_steepens_the_valley_walls() -> None:
    """``b`` scales the coupling term, which is what makes the classic
    b = 100 problem hard: off-valley points are penalised far harder."""
    x = np.array([-1.2, 1.0])
    assert rosenbrock(x, b=100.0) > rosenbrock(x, b=1.0)

"""Fixtures shared across the test suite."""

import numpy as np
import pytest

from og_optimizers import FloatArray

# Dimensions exercised by the parametrized tests. Includes the smallest
# valid case (n = 2) because the Rosenbrock sum is empty below it.
DIMENSIONS = [2, 3, 5, 10]


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failure reproduces exactly.

    Seeding matters more than usual here: the finite-difference checks
    compare floating-point quantities, and an unseeded generator would
    make a borderline failure impossible to reproduce.
    """
    return np.random.default_rng(20260903)


@pytest.fixture
def sample_points(rng: np.random.Generator) -> list[FloatArray]:
    """Assorted evaluation points away from the minimizer.

    Kept modest in magnitude: the Rosenbrock Hessian grows like ``x**2``,
    so far-flung points would make the finite-difference tolerances a
    test of numerical noise rather than of the analytical derivatives.
    """
    return [rng.uniform(-2.0, 2.0, size=n) for n in DIMENSIONS]

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rosenbrock(x: FloatArray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Multidimensional Rosenbrock function.

    f(x) = sum_{i=0}^{n-2} [ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 ]
    """
    x_curr = x[:-1]
    x_next = x[1:]

    return float(np.sum((a - x_curr) ** 2 + b * (x_next - x_curr**2) ** 2))


def rosenbrock_grad(
    x: FloatArray, a: float = 1.0, b: float = 100.0
) -> FloatArray:
    """
    Compute the analytical gradient vector of the multidimensional Rosenbrock function.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    x_curr = x[:-1]
    x_next = x[1:]

    # Elements 0 to n-2 contribute from terms where they act as x_curr
    grad[:-1] += -2.0 * (a - x_curr) - 4.0 * x_curr * b * (x_next - x_curr**2)

    # Elements 1 to n-1 contribute from terms where they act as x_next
    grad[1:] += 2.0 * b * (x_next - x_curr**2)

    return grad


def rosenbrock_hess(
    x: FloatArray, a: float = 1.0, b: float = 100.0
) -> FloatArray:
    """
    Compute the analytical Hessian matrix of the multidimensional Rosenbrock function.
    """
    n = x.shape[0]
    hess = np.zeros((n, n), dtype=np.float64)

    for i in range(n - 1):
        # Diagonal term d^2 f / d x_i^2
        hess[i, i] += 2.0 + 4.0 * b * (3.0 * x[i] ** 2 - x[i + 1])

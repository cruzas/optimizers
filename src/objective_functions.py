from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rosenbrock(x: FloatArray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Multidimensional Rosenbrock function.

    f(x) = sum_{i=0}^{n-2} [(a - x[i])^2 + b * (x[i+1] - x[i]**2)**2], \quad x \in \mathbb{R}^n
    """
    x_curr = x[:-1]
    x_next = x[1:]

    return float(np.sum((a - x_curr) ** 2 + b * (x_next - x_curr**2) ** 2))


def rosenbrock_grad(
    x: FloatArray, a: float = 1.0, b: float = 100.0
) -> FloatArray:
    """
    Compute the analytical gradient vector of the multidimensional Rosenbrock function.

    \frac{\partial{f}}{\partial{x[0]}} = -2(a - x[0]) - 4b x[0] (x[1] - x[0]^2)
    \frac{\partial{f}}{\partial{x[i]}} = -2(a - x[i]) - 4b x[i] (x[i+1] - x[i]^2) + 2b (x[i] - x[i-1]^2), \quad i = 1, \ldots, n-2
    \frac{\partial{f}}{\partial{x[n-1]}} = 2b (x[n-1] - x[n-2]^2)
    """
    grad = np.zeros_like(x, dtype=np.float64)
    x_curr = x[:-1]
    x_next = x[1:]

    # From i=0...n-2 add contributions from terms that act as x_curr
    grad[:-1] += -2.0 * (a - x_curr) - 4.0 * x_curr * b * (x_next - x_curr**2)

    # From i=1...n-1 add contributions from terms that act as x_next
    grad[1:] += 2.0 * b * (x_next - x_curr**2)

    return grad


def rosenbrock_hess(
    x: FloatArray, a: float = 1.0, b: float = 100.0
) -> FloatArray:
    """
    Compute the analytical Hessian matrix of the multidimensional Rosenbrock function.

    \frac{\partial^2 f}{\partial x[0]^2} = 2 - 4bx[1] + 12bx[0]^2
    \frac{\partial^2 f}{\partial x[n-1]^2} = 2b
    \frac{\partial^2 f}{\partial x[i]^2} = 2 - 4bx[i+1] + 12bx[i]^2 + 2b, \quad i = 1, \ldots, n-2
    \frac{\partial^2 f}{\partial x[i]\partial x[i+1]} = -4bx[i], \quad i = 0, \ldots, n-2
    \frac{\partial^2 f}{\partial x[i+1]\partial x[i]} = -4bx[i], \quad i = 0, \ldots, n-2
    """
    n = x.shape[0]
    x_curr = x[:-1]
    x_next = x[1:]

    # Construct the main diagonal using overlapping contributions
    diag = np.zeros(n, dtype=np.float64)

    # Apply 2 - 4bx[i+1] + 12bx[i]^2 for i=0...n-2
    diag[:-1] += 2.0 - 4.0 * b * x_next + 12.0 * b * (x_curr**2)

    # Add the 2b term to indices i=1...n-1
    diag[1:] += 2.0 * b

    # Construct the off-diagonal
    off_diag = -4.0 * b * x_curr

    # Assemble the tridiagonal matrix
    hessian = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    return hessian

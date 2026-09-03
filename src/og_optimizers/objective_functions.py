from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rosenbrock(x: FloatArray, a: float = 1.0, b: float = 100.0) -> float:
    r"""
    Multidimensional Rosenbrock function.

    .. math::

        f(x) = \sum_{i=0}^{n-2}
               \left[ (a - x_i)^2 + b (x_{i+1} - x_i^2)^2 \right],
        \quad x \in \mathbb{R}^n

    Both terms vanish together only when :math:`x_i = a` and
    :math:`x_{i+1} = x_i^2` hold at once, which requires
    :math:`a = a^2`. So :math:`f = 0` is attainable only for
    :math:`a \in \{0, 1\}`, at the constant vector :math:`x_i = a`; the
    standard choice :math:`a = 1` puts the minimizer at the all-ones
    vector. For any other ``a`` the minimum is strictly positive.

    Args:
        x: Point of evaluation, shape ``(n,)`` with ``n >= 2``.
        a: Location parameter; the minimizer sits at ``x_i == a``.
        b: Curvature of the parabolic valley. Larger values narrow the
            valley and make the problem harder to minimize.

    Returns:
        The function value at ``x``.
    """
    x_curr = x[:-1]
    x_next = x[1:]

    return float(np.sum((a - x_curr) ** 2 + b * (x_next - x_curr**2) ** 2))


def rosenbrock_grad(
    x: FloatArray, a: float = 1.0, b: float = 100.0
) -> FloatArray:
    r"""
    Analytical gradient of the multidimensional Rosenbrock function.

    .. math::

        \frac{\partial f}{\partial x_0} &=
            -2 (a - x_0) - 4 b x_0 (x_1 - x_0^2) \\
        \frac{\partial f}{\partial x_i} &=
            -2 (a - x_i) - 4 b x_i (x_{i+1} - x_i^2)
            + 2 b (x_i - x_{i-1}^2),
            \quad i = 1, \ldots, n-2 \\
        \frac{\partial f}{\partial x_{n-1}} &=
            2 b (x_{n-1} - x_{n-2}^2)

    Args:
        x: Point of evaluation, shape ``(n,)`` with ``n >= 2``.
        a: Location parameter, matching :func:`rosenbrock`.
        b: Curvature parameter, matching :func:`rosenbrock`.

    Returns:
        The gradient at ``x``, shape ``(n,)``.
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
    r"""
    Analytical Hessian of the multidimensional Rosenbrock function.

    The matrix is symmetric and tridiagonal: each term of the sum
    couples only the neighbouring coordinates :math:`x_i` and
    :math:`x_{i+1}`, so second derivatives vanish beyond the first
    off-diagonal.

    .. math::

        \frac{\partial^2 f}{\partial x_0^2} &=
            2 - 4 b x_1 + 12 b x_0^2 \\
        \frac{\partial^2 f}{\partial x_i^2} &=
            2 - 4 b x_{i+1} + 12 b x_i^2 + 2 b,
            \quad i = 1, \ldots, n-2 \\
        \frac{\partial^2 f}{\partial x_{n-1}^2} &= 2 b \\
        \frac{\partial^2 f}{\partial x_i \partial x_{i+1}} &=
            -4 b x_i, \quad i = 0, \ldots, n-2

    Args:
        x: Point of evaluation, shape ``(n,)`` with ``n >= 2``.
        a: Accepted for signature symmetry with :func:`rosenbrock`, but
            unused: differentiating ``(a - x_i)**2`` twice removes it.
        b: Curvature parameter, matching :func:`rosenbrock`.

    Returns:
        The Hessian at ``x``, shape ``(n, n)``.
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

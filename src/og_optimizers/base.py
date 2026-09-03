from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class OptimizeResult:
    x: npt.NDArray[np.float64]  # result
    fun: float  # function
    nit: int  # number of iterations
    converged: bool  # whether the method converged
    trajectory: npt.NDArray[np.float64]  # trajectory of the optimization


class BaseOptimizer(ABC):
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def minimize(
        self,
        fun: Callable[[npt.NDArray[np.float64]], float],
        x0: npt.NDArray[np.float64],
        jac: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
        hess: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
    ) -> OptimizeResult:
        pass

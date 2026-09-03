# og-optimizers

[![CI](https://github.com/cruzas/optimizers/actions/workflows/ci.yml/badge.svg)](https://github.com/cruzas/optimizers/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/github/license/cruzas/optimizers)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![codecov](https://codecov.io/gh/cruzas/optimizers/branch/main/graph/badge.svg)](https://codecov.io/gh/cruzas/optimizers)

Classical optimization algorithms implemented from scratch in Python, with
analytical derivatives and a test suite that checks them against
finite differences rather than against themselves.

> **Status: early.** The objective functions and the optimizer interface
> are implemented and tested. The first- and second-order methods
> themselves are not written yet — `first_order.py` and `second_order.py`
> are deliberately empty placeholders.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.14 or newer.

## Usage

```python
import numpy as np

from og_optimizers import rosenbrock, rosenbrock_grad, rosenbrock_hess

x = np.array([-1.2, 1.0])

rosenbrock(x)  # 24.2, the classic hard starting point
rosenbrock_grad(x)  # analytical gradient, shape (n,)
rosenbrock_hess(x)  # analytical Hessian, shape (n, n), tridiagonal
```

Every optimizer implements `BaseOptimizer.minimize`, which returns a frozen
`OptimizeResult` carrying the solution, the objective value, the iteration
count, a convergence flag, and the full trajectory — so runs can be compared
and plotted after the fact.

## Development

The four checks below are exactly what CI runs, so a clean local run means a
green pull request.

```bash
ruff check .                                    # lint
ruff format --check .                           # formatting
mypy src tests                                  # strict type checking
pytest --cov=og_optimizers --cov-fail-under=90  # tests + coverage gate
```

### Coverage reporting

The coverage gate above runs entirely locally. CI additionally uploads
`coverage.xml` to [Codecov](https://codecov.io/) for the badge and the
per-pull-request diff view, which needs a one-time setup:

1. Sign in at [codecov.io](https://codecov.io/) with GitHub and add this
   repository.
2. Copy the repository upload token.
3. Add it under **Settings → Secrets and variables → Actions** as a new
   repository secret named `CODECOV_TOKEN`.

Until that secret exists the upload step fails, which is deliberate: a
silent failure would leave the badge reading "unknown" with no clue why.

## Licence

[GPL-3.0-only](LICENSE).

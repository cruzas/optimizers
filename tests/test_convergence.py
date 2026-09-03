"""Convergence behaviour of the concrete optimizers.

Reserved for `og_optimizers.first_order` and `og_optimizers.second_order`,
which are not implemented yet. The tests here will assert the properties
that distinguish the methods from one another -- that a second-order
method reaches the Rosenbrock minimizer in far fewer iterations than a
first-order one, that reported `converged` results really satisfy the
gradient tolerance, and that `nit` never exceeds `max_iter`.
"""

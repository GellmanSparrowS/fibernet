# SPDX-License-Identifier: MIT
"""Exercise the bounded retry without an optional NGSolve installation."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from fibernet.sim.tet4_open_stokes import _minres_with_continuity_retry


@pytest.mark.parametrize("path", ["initial", "improves", "not_converged", "raises", "worse"])
def test_pressure_constant_retry_keeps_the_native_system(path):
    # One forced velocity direction and a separate velocity direction
    # constrained by the single pressure row. The pressure-row residual is
    # small relative to the RHS but large relative to flow work.
    stiffness = csr_matrix([[1e6, 0., 0.], [0., 4., -1.], [0., -1., 0.]])
    rhs = np.array([1., 0., 0.])
    initial = np.array([1e-6, 0. if path == "initial" else 1e-11, 0.])
    recovery = np.array([1e-6, 2e-11 if path == "worse" else 0., 0.])
    calls = []
    pre = aslinearoperator(np.eye(3))

    def scripted(k, b, M, check, maxiter, callback, rtol):
        assert k is stiffness and b is rhs and check and maxiter == 600
        calls.append(rtol)
        if len(calls) == 1:
            assert M is pre
            return initial, 0
        np.testing.assert_allclose(M @ [1., 2., 3.], [1., 2., 51.])
        if path == "raises":
            raise ValueError("MINRES retry preconditioner failed its check")
        return recovery, 600 if path == "not_converged" else 0

    result = _minres_with_continuity_retry(stiffness, rhs, pre, np.array([2.]),
                                            2, lambda *_: None, scripted)
    assert calls == ([1e-13] if path == "initial" else [1e-13, 1e-15])
    np.testing.assert_array_equal(result, recovery if path == "improves" else initial)

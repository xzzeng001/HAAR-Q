"""Check the public solver's energy and CI coefficients against PySCF."""

import numpy as np
from pyscf.fci import direct_spin1  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.plugins.pyscf import Solver, _from_pyscf_ci, _to_pyscf_ci


def test_interleaved_ci_order_includes_fermion_exchange_phase() -> None:
    # PySCF stores alpha before beta. Only alpha-1/beta-0 crosses in this basis.
    ci = np.array([[1j, 2], [3, 4j]])
    data = _from_pyscf_ci(ci, 2, (1, 1))
    configs, coeffs = data["haar"]["pool"]
    assert configs.flatten().tolist() == [3, 9, 6, 12]
    assert coeffs.tolist() == [1j, 2, -3, 4j]
    np.testing.assert_array_equal(_to_pyscf_ci(data, "haar", 2, 2), ci)


def test_solver_returns_signed_ritz_state_and_core_offset() -> None:
    h1 = np.array([[-1.7, 0.4], [0.4, -0.8]])
    eri = np.zeros((2, 2, 2, 2))
    offset = 0.65
    expected, ci_ref = direct_spin1.kernel(h1, eri, 2, (1, 1), ecore=offset)
    solver = Solver(
        {
            "action": {
                "name": "haar",
                "params": {
                    "network_seed_count": 4,
                    "previous_seed_count": 4,
                    "optimizer_steps": 1,
                    "krylov_extend_count": 0,
                    "krylov_iteration": 3,
                },
            },
            "network": {"name": "mlp/u1u1", "params": {"hidden": [4]}},
            "optimizer": {"name": "Adam", "params": {"lr": 0.001}},
            "common": {"device": "cpu", "dtype": "float64", "max_relative_step": 1, "random_seed": 123},
        }
    )
    for initial in (ci_ref, None):
        grad_mode = torch.is_grad_enabled()
        energy, ci = solver.kernel(h1, eri, 2, (1, 1), ci0=initial, ecore=offset)
        assert torch.is_grad_enabled() == grad_mode
        assert energy == pytest.approx(expected, rel=0, abs=1e-10)
        assert isinstance(ci, np.ndarray)
        assert abs(np.vdot(ci_ref, ci)) == pytest.approx(1.0, abs=1e-9)

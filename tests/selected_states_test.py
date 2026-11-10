"""Check multi-root SCI on complex, disconnected and degenerate selected spaces."""

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest
import torch

from haar_q.algorithms.haar import _DynamicLanczos
from haar_q.algorithms.selected_states import _block_rayleigh_ritz, solve_selected_states
from haar_q.utility.reporting import CycleResult, calculation_report


def dense_model(matrix: torch.Tensor) -> Mock:
    model = Mock()

    def apply(source: torch.Tensor, psi: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        return matrix[dest.flatten().long()][:, source.flatten().long()] @ psi

    model.apply_within.side_effect = apply
    return model


def test_complete_projected_spectrum_including_degenerate_states() -> None:
    matrix = torch.tensor([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 4]], dtype=torch.complex128)
    configs = torch.arange(4, dtype=torch.uint8).reshape(-1, 1)
    results, residuals = solve_selected_states(dense_model(matrix), configs, roots=4)
    assert [e.item() for e, _, _ in results] == pytest.approx([-1, -1, 1, 4])
    vectors = torch.stack([vector for _, _, vector in results], dim=1)
    torch.testing.assert_close(vectors.mH @ vectors, torch.eye(4, dtype=matrix.dtype))
    assert max(residuals) < 1e-12
    with pytest.raises(ValueError, match="selected basis of 4"):
        solve_selected_states(dense_model(matrix), configs, roots=5)


def test_matrix_free_solver_recovers_repeated_lowest_root() -> None:
    diagonal = torch.arange(132, dtype=torch.float64)
    diagonal[:2] = -1
    configs = torch.arange(132, dtype=torch.uint8).reshape(-1, 1)
    model = Mock()
    model.apply_within.side_effect = lambda src, psi, dst: diagonal * psi
    results, residuals = solve_selected_states(model, configs, roots=3, residual_tolerance=1e-8)
    assert [e.item() for e, _, _ in results] == pytest.approx([-1, -1, 2])
    assert max(residuals) < 1e-8
    with pytest.raises(ValueError, match="one fifth"):
        solve_selected_states(model, configs, roots=27)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_device_block_iteration_with_disconnected_complex_spectrum(device: str) -> None:
    # Two identical coupled blocks have repeated low eigenvalues. The long positive diagonal
    # exercises Krylov restarts while a two-by-two reference supplies the exact roots.
    dimension = 140
    configs = torch.arange(dimension, dtype=torch.uint8, device=device).reshape(-1, 1)
    diagonal = torch.arange(dimension, dtype=torch.float64, device=device)
    diagonal[:4] = torch.tensor([-2, 1, -2, 1], dtype=diagonal.dtype, device=device)
    coupling = 0.8j

    def multiply(v: torch.Tensor) -> torch.Tensor:
        assert v.device == configs.device
        result = diagonal * v
        result[:4:2] += coupling * v[1:4:2]
        result[1:4:2] += coupling.conjugate() * v[:4:2]
        return result

    model = Mock()
    model.apply_within.side_effect = lambda src, v, dst: multiply(v)
    if device == "cpu":
        energies, vectors = _block_rayleigh_ritz(model, configs, 3, 5e-9)
    else:
        states, _ = solve_selected_states(model, configs, 3)
        energies = torch.stack([e for e, _, _ in states])
        vectors = torch.stack([v for _, _, v in states], dim=1)
    pair = torch.linalg.eigvalsh(torch.tensor([[-2, coupling], [coupling.conjugate(), 1]], dtype=vectors.dtype))
    torch.testing.assert_close(energies.cpu(), pair[[0, 0, 1]], atol=1e-9, rtol=0)
    torch.testing.assert_close(vectors.mH @ vectors, torch.eye(3, dtype=vectors.dtype, device=device))
    for energy, column in zip(energies, vectors.T):
        assert torch.linalg.vector_norm(multiply(column) - energy * column).item() < 1e-8


def test_lanczos_breakdown_can_reach_disconnected_determinant() -> None:
    matrix = torch.diag(torch.tensor([3, -2], dtype=torch.complex128))
    configs = torch.arange(2, dtype=torch.uint8).reshape(-1, 1)
    solver = _DynamicLanczos(
        model=dense_model(matrix),
        configs=configs,
        psi=torch.tensor([1.0, 0.0], dtype=torch.complex128),
        step=7,
        threshold=1e-10,
        count_extend=0,
        batch_count_apply_within=1,
        single_extend=False,
        first_extend=False,
        eigen_count=2,
    )
    iterations = list(solver.run())
    assert len(iterations) == 2
    assert [e.item() for e, _, _ in iterations[-1]] == pytest.approx([-2, 3])


def test_excitation_gaps_compare_roots_within_the_same_cycle(tmp_path: Path) -> None:
    config = OmegaConf.create({"action": {"name": "excited"}, "model": {"name": "fcidump"}})
    with calculation_report(tmp_path, config, "cpu") as report:
        for cycle, ground in enumerate((-1.0, -2.0), start=1):
            report.checkpoint(cycle, CycleResult(2, 2, -0.5, ground, ((ground, 0.0), (-0.25, 1e-14))))
    records = [row.split() for row in (tmp_path / "ROOTS").read_text().splitlines() if not row.startswith("#")]
    assert [float(row[4]) for row in records] == [0, 0.75, 0, 1.75]

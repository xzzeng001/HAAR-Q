"""Numerical regressions for SCI fitting, postprocessing and natural orbitals."""

import logging
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from haar_q.algorithms.haar import (
    _DynamicLanczos,
    _fit_amplitudes,
    _merge_pool_from_neural_network_and_pool_from_last_iteration,
)
from haar_q.algorithms.orbit import NaturalOrbitCalculator
from haar_q.algorithms.pert import PerturbationConfig


def test_sci_fit_skips_empty_chunks_and_reports_without_backprop() -> None:
    configs = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
    parameter = torch.nn.Parameter(torch.tensor([0.9, 0.2, -0.4], dtype=torch.float64))
    target = torch.tensor([1.0, 0.35, -0.55], dtype=torch.float64)
    active = []

    def network(batch: torch.Tensor) -> torch.Tensor:
        active.append(torch.is_grad_enabled())
        return parameter[batch.flatten().long()]

    def mse(actual: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return (actual - desired).square().mean()

    reference_loss = mse(parameter / parameter[0], target)
    (expected,) = torch.autograd.grad(reference_loss, parameter)
    loss, amplitudes = _fit_amplitudes(network, configs, target, configs[:1], 5, mse, backward=True)
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(parameter.grad, expected)
    assert not amplitudes and active == [True] * 3
    active.clear()
    reported, amplitudes = _fit_amplitudes(network, configs, target, configs[:1], 2, mse, backward=False)
    assert active == [False] * 2 and not reported.requires_grad
    torch.testing.assert_close(reported, reference_loss)
    torch.testing.assert_close(torch.cat(amplitudes), parameter / parameter[0])
    torch.testing.assert_close(parameter.grad, expected)


def test_periodic_lanczos_refresh_preserves_projected_hamiltonian() -> None:
    generator = torch.Generator().manual_seed(47)
    a = torch.randn(6, 6, generator=generator, dtype=torch.complex128)
    matrix = a + a.mH
    configs = torch.arange(6, dtype=torch.uint8).reshape(-1, 1)
    model = Mock()
    model.apply_within.side_effect = lambda src, x, dst: matrix[:, src.flatten().long()] @ x
    solver = _DynamicLanczos(
        model=model,
        configs=configs,
        psi=matrix[0],
        step=5,
        threshold=1e-12,
        count_extend=0,
        batch_count_apply_within=9,
        single_extend=False,
        first_extend=False,
        eigen_count=6,
        period=1,
    )
    results = list(solver.run())[-1]
    assert model.apply_within.call_count == 36
    for (energy, _, vector), exact in zip(results, torch.linalg.eigvalsh(matrix), strict=True):
        torch.testing.assert_close(energy, exact)
        torch.testing.assert_close(matrix @ vector, energy * vector)


def test_sci_pool_prefers_last_cycle_amplitude_on_overlap() -> None:
    configs = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
    selected, merged = _merge_pool_from_neural_network_and_pool_from_last_iteration(
        configs[:2], torch.tensor([1j, 2j]), configs[1:], torch.tensor([-3j, 4j], dtype=torch.complex128)
    )
    torch.testing.assert_close(selected, configs)
    torch.testing.assert_close(merged, torch.tensor([1j, -3j, 4j], dtype=torch.complex128))


def test_natural_orbital_density_of_complex_one_electron_state() -> None:
    # One alpha electron occupies spatial orbital (|0> + i|1>)/sqrt(2).
    configs = torch.tensor([[1], [4]], dtype=torch.uint8)
    psi = torch.tensor([2.0, 2j], dtype=torch.complex128)
    density = NaturalOrbitCalculator(2).calculate_rdm(configs, psi)
    expected = torch.outer(psi, psi.conj()) / (2 * torch.vdot(psi, psi))
    torch.testing.assert_close(density, expected)
    torch.testing.assert_close(density @ psi, 0.5 * psi)


@pytest.mark.parametrize("degenerate", [False, True])
def test_perturbation_uses_signed_coefficients_and_total_energy(
    caplog: pytest.LogCaptureFixture, degenerate: bool
) -> None:
    # The signed internal vector is the lower root of [[0,1],[1,0]]. Its
    # coupling to the external state [1,-1] is sqrt(2); using magnitudes gives zero.
    configs = torch.tensor([[0], [1]], dtype=torch.uint8)
    psi = torch.tensor([1.0, -1.0], dtype=torch.complex128)
    checkpoint = {
        "haar": {
            "global": 1,
            "excited": {0: [(torch.tensor(-1.0), configs, psi)]},
            "pool": (configs, psi.abs().to(psi.dtype)),
        }
    }
    matrix = torch.tensor([[0, 1, 1], [1, 0, -1], [1, -1, 1]], dtype=psi.dtype)
    model = Mock(ref_energy=-2.0)
    model.apply_within.side_effect = lambda src, x, dst: matrix[dst.flatten().long()][:, src.flatten().long()] @ x
    model.find_relative.return_value = torch.tensor([[2]], dtype=torch.uint8)
    model.diagonal_term.return_value = torch.tensor([-1.0 if degenerate else 1.0], dtype=psi.dtype)
    context = Mock(device=torch.device("cpu"))
    context.create_model.return_value = model
    runtime = OmegaConf.create({"model": {"name": "fcidump"}})
    with caplog.at_level(logging.INFO):
        if degenerate:
            with pytest.raises(ValueError, match="zero energy denominator"):
                PerturbationConfig().main(context, runtime, checkpoint)
        else:
            PerturbationConfig().main(context, runtime, checkpoint)
            assert "Corrected total energy is -2.00000000" in caplog.text
            assert "before and after PT2: 1.00000000, 0.00000000" in caplog.text

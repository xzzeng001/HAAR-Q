"""Check repeated LBFGS evaluations in the two guided VMC optimization stages."""

from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest
import torch

from haar_q.algorithms.guide import GuideConfig
from haar_q.utility.context import RuntimeContext


class _Wavefunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.amplitudes = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.amplitudes[configurations].to(torch.complex128)

    def generate_unique(self, count: int) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        configurations = torch.arange(count)
        return configurations, self(configurations).detach(), None, None


def test_guide_lbfgs_recomputes_each_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    wavefunction, sampler = _Wavefunction(), _Wavefunction()
    model = Mock(ref_energy=0.0)
    levels = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    model.apply_within.side_effect = lambda source, psi, target: levels * psi
    context = Mock(spec=RuntimeContext)
    context.create_model.return_value = model
    context.create_network.side_effect = [wavefunction, sampler]

    # A zero step length makes LBFGS evaluate twice at the same amplitudes.
    context.create_optimizer.side_effect = lambda config, params, state: torch.optim.LBFGS(params, lr=0.0, max_iter=2)
    context.save.side_effect = SystemExit(0)
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", Mock())

    def record_gradients(parameter: torch.Tensor) -> list[torch.Tensor]:
        history: list[torch.Tensor] = []

        def record(p: torch.Tensor) -> None:
            assert p.grad is not None
            history.append(p.grad.clone())

        parameter.register_post_accumulate_grad_hook(record)  # type: ignore[no-untyped-call]
        return history

    gradients = [record_gradients(network.amplitudes) for network in (wavefunction, sampler)]
    config = OmegaConf.create({"model": {}, "network": {}, "sampling": {}, "optimizer": {}})
    with torch.no_grad(), pytest.raises(SystemExit) as finished:
        GuideConfig(sampling_count=2, relative_count=2, local_step=1, dist_step=1).main(context, config, {})
    assert finished.value.code == 0
    context.save.assert_called_once()
    for first, second in gradients:
        assert torch.count_nonzero(first) > 0
        torch.testing.assert_close(second, first, rtol=1e-12, atol=1e-14)

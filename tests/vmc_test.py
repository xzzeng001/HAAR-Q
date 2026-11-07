"""VMC energy/gradient and sampling-loop regressions using a small dense Hamiltonian."""

import copy
import pathlib
import typing

import omegaconf
import pytest
import torch

from haar_q.algorithms import vmc


def _hamiltonian() -> torch.Tensor:
    """A Hermitian matrix with complex couplings and a nontrivial expanded space."""
    return torch.tensor(
        [
            [0.3, -0.7 + 0.2j, 0.4j, 0.1],
            [-0.7 - 0.2j, 1.2, -0.6, 0.2j],
            [-0.4j, -0.6, -0.8, 0.5],
            [0.1, -0.2j, 0.5, 1.7],
        ],
        dtype=torch.complex128,
    )


def _amplitudes() -> torch.Tensor:
    return torch.tensor([0.7 + 0.2j, -0.3 + 0.6j, 0.4 - 0.5j, 0.8 + 0.1j], dtype=torch.complex128)


def _configs(indices: list[int]) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.uint8).reshape(-1, 1)


@pytest.mark.parametrize("real_parameters", [False, True])
def test_projected_gradient_matches_dense_rayleigh_quotient(real_parameters: bool) -> None:
    """The detached Hamiltonian product must still yield the full complex-state gradient."""
    initial = torch.view_as_real(_amplitudes()) if real_parameters else _amplitudes()
    parameters = initial.clone().requires_grad_()
    reference_parameters = initial.clone().requires_grad_()
    psi = torch.view_as_complex(parameters) if real_parameters else parameters
    reference_psi = torch.view_as_complex(reference_parameters) if real_parameters else reference_parameters

    hpsi = _hamiltonian() @ psi.detach()
    energy, loss = vmc._projected_energy_loss(psi, hpsi)
    loss.backward()  # type: ignore[no-untyped-call]

    reference_energy = (reference_psi.conj() @ _hamiltonian() @ reference_psi).real
    reference_energy = reference_energy / reference_psi.abs().square().sum()
    reference_energy.backward()  # type: ignore[no-untyped-call]

    torch.testing.assert_close(energy, reference_energy.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(parameters.grad, reference_parameters.grad, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("scale", [0.4, 3.0])
def test_projected_energy_is_invariant_to_normalization_and_global_phase(scale: float) -> None:
    reference = _amplitudes()
    psi = (reference * scale * complex(0.6, 0.8)).requires_grad_()
    energy, _ = vmc._projected_energy_loss(psi, _hamiltonian() @ psi.detach())
    expected = (reference.conj() @ _hamiltonian() @ reference).real / reference.abs().square().sum()
    torch.testing.assert_close(energy, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad_value", [0.0, float("nan"), float("inf")])
def test_projected_energy_rejects_invalid_norm(bad_value: float) -> None:
    psi = torch.full((2,), bad_value, dtype=torch.complex128, requires_grad=True)
    with pytest.raises(ValueError):
        vmc._projected_energy_loss(psi, torch.ones_like(psi))


class _Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.amplitude = torch.nn.Parameter(_amplitudes())
        self.samples = _configs([0, 1])
        self.sample_counts = torch.tensor([8, 1], dtype=torch.int64)
        self.generation_calls = 0

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        return self.amplitude[configs[:, 0].long()]

    def generate_unique(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        self.generation_calls += 1
        return self.samples, self(self.samples).detach(), None, None

    def generate(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        self.generation_calls += 1
        return self.samples, self(self.samples).detach(), self.sample_counts, None


class _Model:
    ref_energy = -1.0

    def __init__(self, neighbors: list[int]) -> None:
        self.matrix = _hamiltonian()
        self.neighbors = _configs(neighbors)
        self.expansion_calls: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]] = []
        self.apply_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        excluded = None if configs_exclude is None else configs_exclude.clone()
        self.expansion_calls.append((configs_i.clone(), psi_i.clone(), count_selected, excluded))
        return self.neighbors

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        self.apply_calls.append((configs_i.clone(), configs_j.clone()))
        source, target = configs_i[:, 0].long(), configs_j[:, 0].long()
        return self.matrix[target][:, source] @ psi_i


class _Writer:
    def __init__(self, **kwargs: typing.Any) -> None:
        self.values: dict[str, list[float]] = {}

    def add_scalar(self, tag: str, value: typing.Any, step: int) -> None:
        self.values.setdefault(tag, []).append(float(value))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Finished(Exception):
    """Stop the real algorithm after the requested checkpoint boundary."""


class _Context:
    def __init__(self, model: _Model, network: _Network, cycles: int = 1) -> None:
        self.model = model
        self.network = network
        self.cycles = cycles
        self.saved: list[tuple[dict[str, typing.Any], int]] = []
        self.summaries: list[vmc.CycleResult] = []
        self.optimizer: torch.optim.Optimizer | None = None

    def create_model(self, config: typing.Any) -> _Model:
        return self.model

    def create_network(self, config: typing.Any, model: typing.Any, state: dict[str, typing.Any] | None) -> _Network:
        if state is not None:
            self.network.load_state_dict(state)
        return self.network

    def create_optimizer(
        self, config: typing.Any, params: typing.Iterable[torch.Tensor], state: dict[str, typing.Any] | None
    ) -> torch.optim.Optimizer:
        self.optimizer = torch.optim.SGD(params, lr=0.05)
        if state is not None:
            self.optimizer.load_state_dict(state)
        return self.optimizer

    def folder(self) -> pathlib.Path:
        return pathlib.Path("unused-test-log-directory")

    def save(self, data: dict[str, typing.Any], step: int, *, result: vmc.CycleResult) -> None:
        self.saved.append((copy.deepcopy(data), step))
        self.summaries.append(result)
        if len(self.saved) == self.cycles:
            raise _Finished


def _run(
    monkeypatch: pytest.MonkeyPatch,
    config: vmc.VmcConfig,
    neighbors: list[int],
    cycles: int = 1,
    checkpoint: dict[str, typing.Any] | None = None,
) -> tuple[_Context, _Writer]:
    context = _Context(_Model(neighbors), _Network(), cycles)
    writer = _Writer()
    monkeypatch.setattr(vmc.torch.utils.tensorboard, "SummaryWriter", lambda **kwargs: writer)
    runtime_config = omegaconf.OmegaConf.create({"model": {}, "network": {}, "optimizer": {}})
    # RuntimeContext.setup globally disables gradients; the optimizer re-enables its closure.
    with torch.no_grad(), pytest.raises(_Finished):
        config.main(typing.cast(vmc.RuntimeContext, context), runtime_config, checkpoint or {})
    return context, writer


def _expected_sgd_step(support: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    amplitudes = _amplitudes().requires_grad_()
    psi = amplitudes[support]
    matrix = _hamiltonian()[support][:, support]
    energy = (psi.conj() @ matrix @ psi).real / psi.abs().square().sum()
    energy.backward()  # type: ignore[no-untyped-call]
    assert amplitudes.grad is not None
    return amplitudes.detach() - 0.05 * amplitudes.grad, energy.detach()


@pytest.mark.parametrize("unique", [True, False])
def test_guided_step_optimizes_expanded_deduplicated_space(monkeypatch: pytest.MonkeyPatch, unique: bool) -> None:
    """New states receive gradients, and draw frequencies never become arbitrary CI weights."""
    config = vmc.VmcConfig(sampling_count=9, relative_count=5, local_step=1, unique=unique)
    context, writer = _run(monkeypatch, config, neighbors=[2, 1, 2])
    expected_parameters, expected_energy = _expected_sgd_step([0, 1, 2])

    torch.testing.assert_close(context.network.amplitude, expected_parameters, rtol=1e-11, atol=1e-12)
    assert context.network.amplitude[2] != _amplitudes()[2]
    assert context.network.amplitude[3] == _amplitudes()[3]
    assert writer.values["vmc/energy"][0] == pytest.approx(expected_energy.item(), rel=1e-12)
    assert context.summaries == [vmc.CycleResult(2, 3, writer.values["vmc/energy"][0])]
    assert len(context.model.expansion_calls) == 1
    sampled, _, requested, excluded = context.model.expansion_calls[0]
    assert requested == 3
    torch.testing.assert_close(sampled, _configs([0, 1]))
    torch.testing.assert_close(excluded, sampled)
    assert context.model.apply_calls
    for source, target in context.model.apply_calls:
        torch.testing.assert_close(source, _configs([0, 1, 2]))
        torch.testing.assert_close(target, source)


@pytest.mark.parametrize("relative_count", [0, 1, 2])
def test_guided_budget_does_not_discard_sampled_configurations(
    monkeypatch: pytest.MonkeyPatch, relative_count: int
) -> None:
    context, _ = _run(monkeypatch, vmc.VmcConfig(relative_count=relative_count, local_step=1), neighbors=[2])
    assert not context.model.expansion_calls
    expected, _ = _expected_sgd_step([0, 1])
    torch.testing.assert_close(context.network.amplitude, expected, rtol=1e-11, atol=1e-12)


def test_guided_step_accepts_no_new_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = _run(monkeypatch, vmc.VmcConfig(relative_count=5, local_step=1), neighbors=[])
    expected, _ = _expected_sgd_step([0, 1])
    torch.testing.assert_close(context.network.amplitude, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("unique", [True, False])
def test_legacy_mode_preserves_asymmetric_weighted_update(monkeypatch: pytest.MonkeyPatch, unique: bool) -> None:
    config = vmc.VmcConfig(relative_count=3, local_step=1, unique=unique, hamiltonian_guided=False)
    context, writer = _run(monkeypatch, config, neighbors=[2])

    amplitudes = _amplitudes().requires_grad_()
    sampled = amplitudes[:2]
    weight = torch.ones(2, dtype=torch.float64) if unique else torch.tensor([8, 1]) / sampled.detach().abs().square()
    hpsi = _hamiltonian()[:2, :3] @ amplitudes[:3].detach()
    energy = ((sampled.conj() * weight) @ hpsi) / ((sampled.conj() * weight) @ sampled.detach())
    energy.real.backward()  # type: ignore[no-untyped-call]
    assert amplitudes.grad is not None
    expected = amplitudes.detach() - 0.05 * amplitudes.grad
    torch.testing.assert_close(context.network.amplitude, expected, rtol=1e-11, atol=1e-12)
    assert writer.values["vmc/energy"][0] == pytest.approx(energy.real.item(), rel=1e-12)
    assert context.network.amplitude[2] == _amplitudes()[2]
    for source, target in context.model.apply_calls:
        torch.testing.assert_close(source, _configs([0, 1, 2]))
        torch.testing.assert_close(target, _configs([0, 1]))


def test_guided_loop_resamples_and_preserves_checkpoint_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    config = vmc.VmcConfig(relative_count=3, local_step=2)
    context, writer = _run(
        monkeypatch,
        config,
        neighbors=[2],
        cycles=2,
        checkpoint={"vmc": {"global": 7, "local": 11}},
    )
    assert context.network.generation_calls == 2
    assert len(context.model.expansion_calls) == 2
    assert len(writer.values["vmc/energy"]) == 4
    assert [step for _, step in context.saved] == [8, 9]
    final, _ = context.saved[-1]
    assert final["vmc"] == {"global": 9, "local": 15}
    torch.testing.assert_close(final["network"]["amplitude"], context.network.amplitude)
    assert final["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.05)

"""Exact two-pass VMC gradients, optimizer compatibility and saved-activation bounds."""

import copy
import typing

import pytest
import torch

from haar_q.algorithms import vmc
from haar_q.networks.mlp import WaveFunctionNormal as MLP
from haar_q.networks.mps import WaveFunction as MPS
from haar_q.networks.transformers import WaveFunctionNormal as Transformer
from haar_q.utility.parallel_wavefunction import ParallelWavefunction
from vmc_test import _expected_sgd_step, _run


class _DenseModel:
    def __init__(self, size: int) -> None:
        generator = torch.Generator().manual_seed(42)
        matrix = torch.randn(size, size, generator=generator, dtype=torch.complex128)
        self.matrix = matrix + matrix.mH
        self.calls = 0

    def apply_within(self, source: torch.Tensor, psi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        assert len(source) == len(target) == self.matrix.shape[0]
        assert not psi.requires_grad
        return self.matrix @ psi


def _network(family: str) -> torch.nn.Module:
    torch.manual_seed(251)
    if family == "mlp":
        return MLP(sites=4, physical_dim=2, is_complex=True, hidden_size=(16,), ordering=1).double()
    if family == "mps":
        return MPS(sites=4, physical_dim=2, virtual_dim=4).double()
    return Transformer(
        sites=4,
        physical_dim=2,
        is_complex=True,
        embedding_dim=8,
        heads_num=2,
        feed_forward_dim=8,
        shared_num=1,
        routed_num=2,
        selected_num=1,
        depth=1,
        ordering=1,
    ).double()


@pytest.mark.parametrize("family", ["mlp", "mps", "transformer"])
@pytest.mark.parametrize("batch_size", [1, 3, 20])
@pytest.mark.parametrize("scripted", [False, True])
def test_chunked_gradient_matches_full_rayleigh_quotient(family: str, batch_size: int, scripted: bool) -> None:
    reference = _network(family)
    chunked = copy.deepcopy(reference)
    if scripted:
        reference, chunked = torch.jit.script(reference), torch.jit.script(chunked)
    configs = torch.tensor([[0], [1], [3], [5], [9], [12], [15]], dtype=torch.uint8)
    model = _DenseModel(len(configs))
    psi = reference(configs)
    expected = torch.vdot(psi, model.matrix @ psi).real / torch.vdot(psi, psi).real
    expected.backward()  # type: ignore[no-untyped-call]
    energy = vmc._batched_projected_backward(chunked, model, configs, batch_size)  # type: ignore[arg-type]
    torch.testing.assert_close(energy, expected.detach(), rtol=1e-11, atol=1e-12)
    assert model.calls == 1
    for (name, param), (_, other) in zip(reference.named_parameters(), chunked.named_parameters(), strict=True):
        if param.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(other.grad, param.grad, rtol=1e-9, atol=1e-11, msg=name)
    # The extra no-grad pass must not double the persisted routing statistics.
    chunked_buffers = dict(chunked.named_buffers())
    for name, buffer in reference.named_buffers():
        torch.testing.assert_close(chunked_buffers[name], buffer, rtol=1e-11, atol=1e-12, msg=name)


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.LBFGS])
def test_chunked_optimizer_update_matches_full_batch(
    optimizer_type: typing.Callable[..., torch.optim.Optimizer],
) -> None:
    reference = _network("mlp")
    chunked = copy.deepcopy(reference)
    configs = torch.tensor([[0], [1], [3], [5], [9], [12], [15]], dtype=torch.uint8)
    model = _DenseModel(len(configs))
    energies = []
    for network, batch_size in [(reference, 0), (chunked, 3)]:
        kwargs = {"max_iter": 3, "line_search_fn": "strong_wolfe"} if optimizer_type is torch.optim.LBFGS else {}
        optimizer = optimizer_type(network.parameters(), lr=0.01, **kwargs)

        @torch.enable_grad()  # type: ignore[no-untyped-call]
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            if batch_size:
                return vmc._batched_projected_backward(network, model, configs, batch_size)  # type: ignore[arg-type]
            psi = network(configs)
            energy, loss = vmc._projected_energy_loss(psi, model.matrix @ psi.detach())
            loss.backward()  # type: ignore[no-untyped-call]
            return energy

        with torch.no_grad():
            energies.append(optimizer.step(closure))  # type: ignore[arg-type]
    torch.testing.assert_close(energies[0], energies[1], rtol=1e-11, atol=1e-12)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(chunked.state_dict()[name], value, rtol=1e-9, atol=1e-11, msg=name)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_chunked_vmc_matches_single_gpu_update_and_router_statistics() -> None:
    reference = _network("transformer").to("cuda:1")
    master = copy.deepcopy(reference)
    parallel = ParallelWavefunction(typing.cast(vmc.NetworkProto, master), (1, 0), transformer=True)
    configs = torch.tensor([[0], [1], [3], [5], [9], [12], [15]], dtype=torch.uint8, device="cuda:1")
    model = _DenseModel(len(configs))
    model.matrix = model.matrix.to("cuda:1")
    energies = []
    networks: list[tuple[typing.Any, int]] = [(reference, 0), (parallel, 3)]
    for network, batch_size in networks:
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        if batch_size:
            energies.append(vmc._batched_projected_backward(network, model, configs, batch_size))  # type: ignore[arg-type]
        else:
            psi = network(configs)
            energy, loss = vmc._projected_energy_loss(psi, model.matrix @ psi.detach())
            loss.backward()  # type: ignore[no-untyped-call]
            energies.append(energy)
        optimizer.step()
    torch.testing.assert_close(energies[0], energies[1], rtol=1e-9, atol=1e-11)
    for (name, param), (_, other) in zip(reference.named_parameters(), master.named_parameters(), strict=True):
        if param.grad is not None:
            torch.testing.assert_close(other.grad, param.grad, rtol=1e-8, atol=1e-10, msg=name)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(master.state_dict()[name], value, rtol=1e-8, atol=1e-10, msg=name)


def test_guided_loop_uses_chunked_update_and_same_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    context, writer = _run(
        monkeypatch, vmc.VmcConfig(relative_count=5, local_step=1, amplitude_batch_size=2), neighbors=[2, 1, 2]
    )
    expected, energy = _expected_sgd_step([0, 1, 2])
    torch.testing.assert_close(context.network.amplitude, expected, rtol=1e-11, atol=1e-12)
    assert writer.values["vmc/energy"][0] == pytest.approx(energy.item(), rel=1e-12)
    assert len(context.model.apply_calls) == 1
    assert context.saved[0][0]["vmc"] == {"global": 1, "local": 1}
    assert set(context.saved[0][0]) == {"vmc", "network", "optimizer"}


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_invalid_amplitude_batch_size_is_rejected(value: typing.Any) -> None:
    with pytest.raises(ValueError, match="amplitude_batch_size"):
        vmc.VmcConfig(amplitude_batch_size=value)


def test_amplitude_batching_requires_guided_objective() -> None:
    with pytest.raises(ValueError, match="hamiltonian_guided"):
        vmc.VmcConfig(amplitude_batch_size=2, hamiltonian_guided=False)
    vmc.VmcConfig(amplitude_batch_size=0, hamiltonian_guided=False)


class _SavedTensor:
    def __init__(self, tensor: torch.Tensor, tracker: "_SavedBytes") -> None:
        self.tensor = tensor.detach()
        self.tracker = tracker
        self.size = tensor.numel() * tensor.element_size()
        tracker.live += self.size
        tracker.peak = max(tracker.peak, tracker.live)

    def __del__(self) -> None:
        self.tracker.live -= self.size


class _SavedBytes:
    """Count live saved tensor bytes, not allocator/process RSS or unique storage."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    def pack(self, tensor: torch.Tensor) -> _SavedTensor:
        return _SavedTensor(tensor, self)

    @staticmethod
    def unpack(saved: _SavedTensor) -> torch.Tensor:
        return saved.tensor


def test_chunking_bounds_live_saved_autograd_bytes() -> None:
    torch.manual_seed(103)
    network = MLP(sites=8, physical_dim=2, is_complex=True, hidden_size=(64, 64), ordering=1).double()
    configs = torch.arange(256, dtype=torch.uint8).reshape(-1, 1)
    model = _DenseModel(len(configs))
    measurements = []
    gradients = []
    for batch_size in [0, 16]:
        network.zero_grad(set_to_none=True)
        tracker = _SavedBytes()
        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
            if batch_size:
                vmc._batched_projected_backward(network, model, configs, batch_size)  # type: ignore[arg-type]
            else:
                psi = network(configs)
                _, loss = vmc._projected_energy_loss(psi, model.matrix @ psi.detach())
                loss.backward()  # type: ignore[no-untyped-call]
        assert tracker.live == 0
        measurements.append(tracker.peak)
        gradients.append([p.grad.clone() for p in network.parameters() if p.grad is not None])
    for full, chunked in zip(*gradients, strict=True):
        torch.testing.assert_close(full, chunked, rtol=1e-9, atol=1e-11)
    assert measurements[1] < measurements[0] / 4
    print(f"CPU live saved-autograd tensor bytes: full={measurements[0]}, batch16={measurements[1]}")

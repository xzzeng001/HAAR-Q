"""Check complex amplitudes, router statistics, gradients and portable optimizer updates."""

import copy

import pytest
import torch

from haar_q.networks.transformers import WaveFunctionNormal
from haar_q.utility.parallel_wavefunction import AmplitudeBatch, ParallelWavefunction


def transformer() -> WaveFunctionNormal:
    torch.manual_seed(2026)
    return WaveFunctionNormal(
        sites=4,
        physical_dim=2,
        is_complex=True,
        embedding_dim=8,
        heads_num=2,
        feed_forward_dim=8,
        shared_num=1,
        routed_num=2,
        selected_num=1,
        depth=2,
        ordering=1,
    ).double()


def test_shard_amplitudes_and_routing_statistics() -> None:
    reference = transformer()
    master = AmplitudeBatch(copy.deepcopy(reference), transformer=True)
    configs = torch.tensor([[3], [5], [9], [12], [15]], dtype=torch.uint8)
    for _ in range(2):
        with torch.no_grad():
            expected = reference(configs)
        shards = [copy.deepcopy(master)(part) for part in configs.tensor_split(2)]
        pairs = torch.cat([pair for pair, _ in shards])
        master.update_routing(tuple(torch.cat(values) for values in zip(*(updates for _, updates in shards))))
        torch.testing.assert_close(torch.view_as_complex(pairs), expected)
        for key, value in reference.named_buffers():
            torch.testing.assert_close(master.network.get_buffer(key), value)


def test_singleton_routing_statistics_release_backward_graph() -> None:
    network = transformer()
    for _ in range(2):
        network(torch.tensor([[3]], dtype=torch.uint8)).real.sum().backward()  # type: ignore[no-untyped-call]
        for name, buffer in network.named_buffers():
            if name.endswith(".accumulater"):
                assert buffer.grad_fn is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_update_and_checkpoint_match_single_gpu() -> None:
    reference = transformer().to("cuda:1")
    master = copy.deepcopy(reference)
    parallel = ParallelWavefunction(master, (1, 0), transformer=True)
    configs = torch.tensor([[3], [5], [9], [12], [15]], dtype=torch.uint8, device="cuda:1")
    weights = torch.arange(1, 6, device=configs.device)
    for network in (parallel, reference):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        psi = network(configs)
        loss = (weights * (psi.real - 0.75 * psi.imag)).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    for (name, param), (_, other) in zip(reference.named_parameters(), master.named_parameters(), strict=True):
        if param.grad is not None:
            torch.testing.assert_close(other.grad, param.grad, msg=name)
    state = parallel.state_dict()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(state[key], value, msg=key)
    restored = transformer()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.eval()(configs.cpu()), reference.eval()(configs).cpu())

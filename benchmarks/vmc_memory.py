"""CPU VMC activation-memory benchmark with exact-update checks.

Run from the repository root: python benchmarks/vmc_memory.py
Reports unique saved storage excluding model parameters, buffers and input
configurations, plus logical saved bytes including repeated views/parameters.
Neither measures process RSS, CUDA memory, or predicts Pcluster performance.
Wall times include the saved-tensor measurement hooks and both forward passes.
The Hamiltonian is a deterministic Hermitian cyclic operator with O(N) storage.
"""

import argparse
import copy
from collections.abc import Iterable
import json
from pathlib import Path
import sys
import time
import typing

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haar_q.algorithms.vmc import _batched_projected_backward, _projected_energy_loss  # noqa: E402
from haar_q.networks.mlp import WaveFunctionNormal  # noqa: E402
from haar_q.utility.bitspack import pack_int  # noqa: E402
from haar_q.utility.model_dict import ModelProto, NetworkProto  # noqa: E402


class SavedTensor:
    def __init__(self, tensor: torch.Tensor, tracker: "SavedBytes") -> None:
        self.tensor = tensor.detach()
        self.tracker = tracker
        self.size = tensor.numel() * tensor.element_size()
        tracker.live += self.size
        tracker.peak = max(tracker.peak, tracker.live)
        self.key = (tensor.device, tensor.untyped_storage().data_ptr())
        self.storage_bytes = tensor.untyped_storage().nbytes()
        if self.key not in tracker.persistent:
            if self.key not in tracker.references:
                tracker.unique_live += self.storage_bytes
                tracker.unique_peak = max(tracker.unique_peak, tracker.unique_live)
            tracker.references[self.key] = tracker.references.get(self.key, 0) + 1

    def __del__(self) -> None:
        self.tracker.live -= self.size
        if self.key not in self.tracker.persistent:
            self.tracker.references[self.key] -= 1
            if self.tracker.references[self.key] == 0:
                del self.tracker.references[self.key]
                self.tracker.unique_live -= self.storage_bytes


class SavedBytes:
    def __init__(self, persistent: Iterable[torch.Tensor]) -> None:
        self.live = 0
        self.peak = 0
        self.unique_live = 0
        self.unique_peak = 0
        self.references: dict[tuple[torch.device, int], int] = {}
        self.persistent = {(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in persistent}

    def pack(self, tensor: torch.Tensor) -> SavedTensor:
        return SavedTensor(tensor, self)

    @staticmethod
    def unpack(saved: SavedTensor) -> torch.Tensor:
        return saved.tensor


class CyclicHamiltonian:
    def __init__(self, size: int) -> None:
        self.diagonal = torch.linspace(-1, 1, size, dtype=torch.float64)

    def apply_within(self, source: torch.Tensor, psi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.diagonal * psi + 0.1j * psi.roll(1) - 0.1j * psi.roll(-1)


def run(
    initial: torch.nn.Module, configs: torch.Tensor, chunk_size: int, repeats: int
) -> tuple[dict[str, typing.Any], torch.nn.Module]:
    network = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    model = CyclicHamiltonian(len(configs))
    tracker = SavedBytes([*network.parameters(), *network.buffers(), configs])
    energies = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        if chunk_size:
            return _batched_projected_backward(
                typing.cast(NetworkProto, network), typing.cast(ModelProto, model), configs, chunk_size
            )
        psi = network(configs)
        with torch.no_grad():
            hpsi = model.apply_within(configs, psi.detach(), configs)
        energy, loss = _projected_energy_loss(psi, hpsi)
        loss.backward()  # type: ignore[no-untyped-call]
        return energy

    start = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        for _ in range(repeats):
            energy = typing.cast(torch.Tensor, optimizer.step(closure))  # type: ignore[arg-type]
            energies.append(float(energy))
    elapsed = time.perf_counter() - start
    assert tracker.live == 0, "Backward retained saved tensors after completing all steps."
    assert tracker.unique_live == 0 and not tracker.references, "Saved activation storage was not released."
    return {
        "chunk_size": chunk_size,
        "peak_saved_autograd_logical_bytes": tracker.peak,
        "peak_unique_saved_activation_storage_bytes": tracker.unique_peak,
        "total_wall_seconds": elapsed,
        "wall_seconds_per_step": elapsed / repeats,
        "energies": energies,
    }, network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=255)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.batch_size > 2**args.sites:
        parser.error("batch-size cannot exceed the number of configurations, 2**sites")
    torch.set_num_threads(args.threads)
    torch.manual_seed(103)
    initial = WaveFunctionNormal(
        sites=args.sites,
        physical_dim=2,
        is_complex=True,
        hidden_size=(args.network_width, args.network_width),
        ordering=1,
    ).double()
    indices = torch.arange(args.batch_size, dtype=torch.int64)
    bits = ((indices[:, None] >> torch.arange(args.sites)) & 1).to(torch.uint8)
    configs = pack_int(bits, size=1)
    # Warm both paths before timing; each measured run starts from identical parameters.
    run(initial, configs, 0, 1)
    run(initial, configs, args.chunk_size, 1)
    full, reference = run(initial, configs, 0, args.repeats)
    chunked, candidate = run(initial, configs, args.chunk_size, args.repeats)
    max_update_difference = 0.0
    max_gradient_difference = 0.0
    for p, q in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-11)
        if p.numel():
            max_update_difference = max(max_update_difference, (p - q).abs().max().item())
        if p.grad is not None:
            assert q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-11)
            max_gradient_difference = max(max_gradient_difference, (p.grad - q.grad).abs().max().item())
    torch.testing.assert_close(
        torch.tensor(full["energies"], dtype=torch.float64),
        torch.tensor(chunked["energies"], dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )
    print(
        json.dumps(
            {
                "device": "cpu",
                "torch_version": torch.__version__,
                "dtype": "float64/complex128",
                "settings": vars(args),
                "full": full,
                "chunked": chunked,
                "saved_tensor_peak_reduction": full["peak_saved_autograd_logical_bytes"]
                / chunked["peak_saved_autograd_logical_bytes"],
                "unique_saved_activation_peak_reduction": full["peak_unique_saved_activation_storage_bytes"]
                / chunked["peak_unique_saved_activation_storage_bytes"],
                "max_parameter_difference": max_update_difference,
                "max_gradient_difference": max_gradient_difference,
                "equality_checked": True,
                "memory_metric": "unique saved storage excludes parameters/buffers/inputs; logical bytes include duplicates; not RSS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

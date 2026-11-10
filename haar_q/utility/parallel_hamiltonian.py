"""Distribute native Hamiltonian products and guided selection over CUDA devices."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch

from ..hamiltonian import Hamiltonian


class ParallelHamiltonian(Hamiltonian):
    """Split source determinants for projected matvecs and Hamiltonian-guided expansion."""

    def __init__(self, source: Hamiltonian, device_ids: tuple[int, ...]) -> None:
        kind = "fermi" if source.particle_cut == 1 else "bose2"
        terms = (source.site, source.kind, source.coef)
        super().__init__(terms, kind=kind)
        self.devices = tuple(torch.device("cuda", i) for i in device_ids)
        # The native Hamiltonian moves its term tensors in place. Give each worker
        # a separate instance so concurrent launches cannot move another worker's terms.
        self._workers = [Hamiltonian(terms, kind=kind) for _ in self.devices[1:]]

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        if configs_i.device != self.devices[0] or len(configs_i) < 2 or not len(configs_j) or not len(self.site):
            return super().apply_within(configs_i, psi_i, configs_j)
        partials = self._source_shards(Hamiltonian.apply_within, configs_i, psi_i, configs_j)
        result = partials[0]
        for contribution in partials[1:]:
            result.add_(contribution.to(self.devices[0], non_blocking=True))
        return result

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if configs_i.device != self.devices[0] or len(configs_i) < 2 or count_selected <= 0 or not len(self.site):
            return super().find_relative(configs_i, psi_i, count_selected, configs_exclude)
        excluded = configs_i if configs_exclude is None else configs_exclude
        operation = partial(Hamiltonian._relative_candidate_pool, count_selected=count_selected)
        partials = self._source_shards(operation, configs_i, psi_i, excluded)
        pool = torch.cat([rows.to(self.devices[0], non_blocking=True) for rows in partials])
        # Each path has an eight-byte squared matrix-element weight. One
        # determinant may occupy multiple top-K slots, even on different GPUs.
        # Heap padding has zero weight and is discarded before deduplication.
        weights = pool[:, :8].reshape(-1).view(torch.float64)
        top = weights.topk(min(len(weights), count_selected), sorted=False)
        return torch.unique(pool[:, 8:][top.indices[top.values > 0]], dim=0)

    def _source_shards(
        self,
        operation: Callable[[Hamiltonian, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        configs: torch.Tensor,
        psi: torch.Tensor,
        shared_configs: torch.Tensor,
    ) -> list[torch.Tensor]:
        if len(psi) != len(configs) or psi.device != configs.device or shared_configs.device != configs.device:
            raise ValueError("Hamiltonian inputs must share the primary GPU and have matching source lengths.")
        # JIT compilation/registration is process-wide. Complete it once before
        # native workers start. The native kernels synchronize their calling
        # streams, so the host must launch them from distinct threads.
        self._load_module(configs.device.type, configs.size(1), self.particle_cut)
        count = min(len(self.devices), len(configs))
        workers = [self, *self._workers[: count - 1]]
        sources = configs.tensor_split(count)
        amplitudes = psi.tensor_split(count)
        streams = [torch.cuda.current_stream(device) for device in self.devices[:count]]
        # Stage cross-device copies on the caller's streams so they observe input
        # tensors produced on non-default streams. Workers reuse those same streams.
        inputs = [
            (
                source.to(device, non_blocking=True).contiguous(),
                amp.to(device, non_blocking=True).contiguous(),
                shared_configs.to(device, non_blocking=True).contiguous(),
            )
            for device, source, amp in zip(self.devices, sources, amplitudes)
        ]

        def apply_shard(i: int) -> torch.Tensor:
            with torch.cuda.stream(streams[i]), torch.no_grad():
                return operation(workers[i], *inputs[i])

        with ThreadPoolExecutor(max_workers=count) as pool:
            return list(pool.map(apply_shard, range(count)))

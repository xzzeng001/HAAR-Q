"""Distribute or stream packed Hamiltonian terms while retaining the full selected basis."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import TypeVar

import torch

from ..hamiltonian import Hamiltonian

_T = TypeVar("_T")
_Inputs = tuple[torch.Tensor, ...]


def _top_paths(rows: torch.Tensor, count: int) -> torch.Tensor:
    """Trim the path heap without deduplicating destinations across operator tiles."""
    weights = rows[:, :8].reshape(-1).view(torch.float64)
    top = weights.topk(min(len(rows), count), sorted=False)
    return rows[top.indices[top.values > 0]]


def _merge_amplitudes(
    left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    configs, inverse = torch.unique(torch.cat((left[0], right[0])), dim=0, return_inverse=True)
    values = left[1].new_zeros(len(configs))
    values.index_add_(0, inverse, torch.cat((left[1], right[1])))
    return configs, values


class TermHamiltonian(Hamiltonian):
    """Keep one CPU backing operator; assign contiguous term ranges to devices.

    A range fitting the term budget stays on its device. A larger range is
    copied from CPU storage one block at a time for each operation. Source and
    target determinants are replicated so each path is visited exactly once.
    """

    def __init__(self, source: Hamiltonian, devices: tuple[torch.device, ...], batch_size: int) -> None:
        if not devices or batch_size <= 0:
            raise ValueError("Term partitioning requires a device and a positive term batch size.")
        self._kind = "fermi" if source.particle_cut == 1 else "bose2"
        super().__init__((source.site.cpu(), source.kind.cpu(), source.coef.cpu()), kind=self._kind)
        self.batch_size = batch_size
        # Empty devices would duplicate inputs without doing any work.
        self.devices = devices[: max(1, len(self.site))]
        self._resident: dict[int, Hamiltonian] = {}
        for i, device in enumerate(self.devices):
            start, stop = self._range(i, len(self.devices))
            if 0 < stop - start <= self.batch_size:
                block = self._block(start, stop)
                block._prepare_data(device)
                self._resident[i] = block

    def _range(self, i: int, workers: int) -> tuple[int, int]:
        return i * len(self.site) // workers, (i + 1) * len(self.site) // workers

    def _block(self, start: int, stop: int) -> Hamiltonian:
        return Hamiltonian((self.site[start:stop], self.kind[start:stop], self.coef[start:stop]), kind=self._kind)

    def _blocks(self, i: int, workers: int, configured: bool) -> Iterator[Hamiltonian]:
        if configured and i in self._resident:
            yield self._resident[i]
            return
        start, stop = self._range(i, workers)
        for offset in range(start, stop, self.batch_size):
            yield self._block(offset, min(offset + self.batch_size, stop))

    def _on_shards(self, inputs: _Inputs, run: Callable[[Iterator[Hamiltonian], _Inputs], _T]) -> list[_T]:
        primary = inputs[0].device
        if any(x.device != primary for x in inputs):
            raise ValueError("Hamiltonian inputs must be on the same device.")
        # Programmatic callers may request a CPU evaluation on a model used for CUDA training.
        configured = primary in self.devices
        devices = self.devices if configured else (primary,)
        self._load_module(primary.type, inputs[0].size(1), self.particle_cut)
        # Stage copies on the caller's current streams, including nondefault streams.
        staged = [tuple(x.to(dev, non_blocking=True).contiguous() for x in inputs) for dev in devices]
        streams = [torch.cuda.current_stream(dev) if dev.type == "cuda" else None for dev in devices]

        def worker(i: int) -> _T:
            stream = streams[i]
            with torch.cuda.stream(stream) if stream is not None else nullcontext(), torch.no_grad():
                return run(self._blocks(i, len(devices), configured), staged[i])

        if len(devices) == 1:
            return [worker(0)]
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(executor.map(worker, range(len(devices))))
        # Put the caller's result first so reductions land on the input device.
        i = devices.index(primary)
        results[0], results[i] = results[i], results[0]
        return results

    def _sum_terms(self, inputs: _Inputs, evaluate: Callable[[Hamiltonian, _Inputs], torch.Tensor]) -> torch.Tensor:
        def run(blocks: Iterator[Hamiltonian], local: _Inputs) -> torch.Tensor:
            total = evaluate(next(blocks), local)
            for block in blocks:
                total.add_(evaluate(block, local))
            return total

        first, *rest = self._on_shards(inputs, run)
        for item in rest:
            first.add_(item.to(first.device))
        return first

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        if not len(configs_i) or not len(configs_j) or not len(self.site):
            return psi_i.new_zeros(len(configs_j))
        return self._sum_terms((configs_i, psi_i, configs_j), lambda block, x: block.apply_within(x[0], x[1], x[2]))

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        if not len(configs) or not len(self.site):
            return torch.zeros(len(configs), device=configs.device, dtype=torch.complex128)
        return self._sum_terms((configs,), lambda block, x: block.diagonal_term(x[0]))

    def _relative_candidate_pool(
        self,
        configs: torch.Tensor,
        psi: torch.Tensor,
        excluded: torch.Tensor,
        *,
        count_selected: int,
    ) -> torch.Tensor:
        if count_selected < 0:
            raise ValueError("count_selected must be nonnegative.")
        if not count_selected or not len(configs) or not len(self.site):
            return configs.new_empty((0, configs.size(1) + 8))

        def run(blocks: Iterator[Hamiltonian], x: _Inputs) -> torch.Tensor:
            best = x[0].new_empty((0, 8 + x[0].size(1)))
            for block in blocks:
                rows = block._relative_candidate_pool(x[0], x[1], x[2], count_selected=count_selected)
                best = _top_paths(torch.cat((best, rows)), count_selected)
            return best

        pools = self._on_shards((configs, psi, excluded), run)
        if len(pools) == 1:
            return pools[0]
        candidates = torch.cat([rows.to(configs.device) for rows in pools])
        return _top_paths(candidates, count_selected)

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        excluded = configs_i if configs_exclude is None else configs_exclude
        paths = self._relative_candidate_pool(configs_i, psi_i, excluded, count_selected=count_selected)
        return torch.unique(paths[:, 8:], dim=0)

    def list_relative(
        self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_exclude: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not len(configs_i) or not len(self.site):
            return configs_i[:0], psi_i[:0]
        excluded = configs_i if configs_exclude is None else configs_exclude

        def run(blocks: Iterator[Hamiltonian], x: _Inputs) -> tuple[torch.Tensor, torch.Tensor]:
            result = next(blocks).list_relative(x[0], x[1], x[2])
            for block in blocks:
                result = _merge_amplitudes(result, block.list_relative(x[0], x[1], x[2]))
            return result

        first, *rest = self._on_shards((configs_i, psi_i, excluded), run)
        for rows, values in rest:
            first = _merge_amplitudes(first, (rows.to(configs_i.device), values.to(configs_i.device)))
        return first

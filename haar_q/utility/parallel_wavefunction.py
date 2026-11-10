"""Batch wavefunction evaluations across GPUs while keeping one sampling and optimizer state."""

from typing import Any

import torch
from torch import nn

from ..networks.transformers import DecoderUnit
from .model_dict import NetworkProto


class AmplitudeBatch(nn.Module):
    def __init__(self, network: nn.Module, transformer: bool) -> None:
        super().__init__()
        self.network = network
        self.transformer = transformer

    def routing_layers(self) -> list[DecoderUnit]:
        if not self.transformer:
            return []
        return [layer for layer in self.network.modules() if isinstance(layer, DecoderUnit) and layer.training]

    def forward(self, configs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        layers = self.routing_layers()
        for layer in layers:
            # Each replica returns its own increments. Rebinding keeps the master buffers intact.
            layer.accumulater = torch.zeros_like(layer.accumulater)
            layer.count = torch.zeros_like(layer.count)
        amplitudes = self.network(configs)
        increments = tuple(
            torch.cat((layer.accumulater.detach(), layer.count.detach().reshape(1)))[None] for layer in layers
        )
        return torch.view_as_real(amplitudes), increments

    @torch.no_grad()
    def update_routing(self, increments: tuple[torch.Tensor, ...]) -> None:
        for layer, shards in zip(self.routing_layers(), increments, strict=True):
            total = shards.sum(dim=0)
            layer.accumulater = layer.accumulater.detach() + total[:-1]
            layer.count = layer.count + total[-1]


class ParallelWavefunction:
    """Evaluate amplitudes in parallel and delegate sampling and checkpoints to the master."""

    def __init__(self, network: NetworkProto, device_ids: tuple[int, ...], *, transformer: bool) -> None:
        if not isinstance(network, nn.Module) or len(device_ids) < 2:
            raise ValueError("Parallel wavefunction evaluation requires a PyTorch module and at least two GPUs.")
        self.network = network
        self.batch = AmplitudeBatch(network, transformer)
        self.parallel = nn.DataParallel(self.batch, device_ids=list(device_ids), output_device=device_ids[0])

    def __call__(self, configs: torch.Tensor) -> torch.Tensor:
        if len(configs) < 2:
            return self.network(configs)
        pairs, increments = self.parallel(configs)
        self.batch.update_routing(increments)
        return torch.view_as_complex(pairs.contiguous())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.network, name)

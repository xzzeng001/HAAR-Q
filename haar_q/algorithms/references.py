"""Explicit molecular determinants for the first selected-CI cycle."""

from dataclasses import dataclass
from collections.abc import Callable
import math
from itertools import chain

import torch


@dataclass
class ReferenceDeterminant:
    """Spatial occupations in integral order, with a signed trial coefficient."""

    occupation: str
    coefficient: float = 1.0


def pack_references(
    references: list[ReferenceDeterminant], norb: int, nelec: int, ms2: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the references with alpha in the low bit of each spatial orbital."""
    rows: list[list[int]] = []
    seen: set[str] = set()
    coefficients = []
    for index, ref in enumerate(references, 1):
        occupation = ref.occupation
        if len(occupation) != norb or any(c not in "0ab2" for c in occupation):
            raise ValueError(f"Initial reference {index} must contain {norb} spatial occupations from 0, a, b, 2.")
        if occupation in seen:
            raise ValueError(f"Duplicate determinant in initial reference {index}.")
        seen.add(occupation)
        alpha = occupation.count("a") + occupation.count("2")
        beta = occupation.count("b") + occupation.count("2")
        if alpha + beta != nelec or alpha - beta != ms2:
            raise ValueError(
                f"Initial reference {index} has {alpha} alpha and {beta} beta electrons; "
                f"the active space requires NELEC={nelec}, MS2={ms2}."
            )
        if isinstance(ref.coefficient, bool) or not math.isfinite(ref.coefficient):
            raise ValueError(f"Initial reference {index} must have a finite real coefficient.")
        row = [0] * ((2 * norb + 7) // 8)
        for orbital, char in enumerate(occupation):
            row[orbital // 4] |= "0ab2".index(char) << (2 * (orbital % 4))
        rows.append(row)
        coefficients.append(ref.coefficient)
    scale = max((abs(c) for c in coefficients), default=0.0)
    if scale == 0:
        raise ValueError("At least one initial reference coefficient must be nonzero.")
    # Rescaling on the host also handles coefficients far below sqrt(float64.tiny).
    return torch.tensor(rows, dtype=torch.uint8, device=device), torch.tensor(
        [c / scale for c in coefficients], dtype=torch.complex128, device=device
    )


def fit_anchor(
    network: Callable[[torch.Tensor], torch.Tensor], configs: torch.Tensor, target: torch.Tensor, batches: int
) -> torch.Tensor:
    """Choose the largest CI target that admits finite amplitude ratios on both sides."""
    if batches < 1:
        raise ValueError("SCI fitting requires a positive batch count.")
    # Explicit references can occupy a network node. Probe the usual largest
    # coefficient first; search the rest only if that amplitude cannot serve.
    order = target.abs().argsort(descending=True)
    chunk_size = max(1, (len(order) + batches - 1) // batches)
    with torch.no_grad():
        for start in chain((0,), range(1, len(order), chunk_size)):
            indices = order[start : start + (1 if start == 0 else chunk_size)]
            prediction = network(configs[indices]).abs()
            usable = (
                (target[indices].abs() > 0) & prediction.isfinite() & (prediction > torch.finfo(prediction.dtype).tiny)
            )
            if usable.any():
                return indices[usable][0]
    raise ValueError("SCI fitting needs a determinant with nonzero amplitude in both the target and the network.")

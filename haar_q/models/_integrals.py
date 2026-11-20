"""Pack restricted spatial integrals into normal-ordered spin-orbital operators."""

import logging
from collections.abc import Iterator

import numpy as np
from openfermion.config import EQ_TOLERANCE  # type: ignore[import-untyped]
import torch


def _term_blocks(core: float, h1: np.ndarray, eri: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield columns of normal-ordered sites and their real coefficients in bounded blocks."""
    tol = EQ_TOLERANCE
    if abs(core) >= tol:
        yield np.empty((1, 0), np.int16), np.array([core])
    n = len(h1)
    rows, cols = np.nonzero(np.abs(h1) >= tol)
    for spin in (0, 1):
        yield np.column_stack((2 * rows + spin, 2 * cols + spin)), h1[rows, cols]
    # Index each spin orbital pair (p>q). Only annihilator pairs with the same spin counts contribute.
    s, r = np.triu_indices(2 * n, k=1)
    pair_spin = (s % 2) + (r % 2)
    groups = [(r[pair_spin == group], s[pair_spin == group]) for group in (0, 1, 2)]
    a = np.arange(n)
    hi = np.maximum.outer(a, a)
    pair = hi * (hi + 1) // 2 + np.minimum.outer(a, a)

    def half_integral(p: np.integer, s: np.ndarray, q: np.integer, r: np.ndarray) -> np.ndarray:
        if eri.ndim == 4:
            half = 0.5 * eri[p // 2, s // 2, q // 2, r // 2]
        else:
            ps, qr = pair[p // 2, s // 2], pair[q // 2, r // 2]
            major = np.maximum(ps, qr)
            half = 0.5 * eri[major * (major + 1) // 2 + np.minimum(ps, qr)]
        return np.where(((p % 2) == (s % 2)) & (np.abs(half) >= tol), half, 0.0)

    for p, q in zip(r, s):
        ra, sa = groups[(p % 2) + (q % 2)]
        direct = half_integral(p, sa, q, ra)
        exchange = half_integral(p, ra, q, sa)
        # OpenFermion discards small terms after each normal-order addition.
        # Lexicographic raw spin indices add D, -X, -X, D (half integrals).
        if eri.ndim == 4:
            # Full arrays may distinguish (ps|qr) and (qr|ps); retain both.
            value = half_integral(q, ra, p, sa) - half_integral(q, sa, p, ra)
        else:
            value = direct - exchange
        value[np.abs(value) < tol] = 0
        value -= exchange
        value[np.abs(value) < tol] = 0
        value += direct
        active = np.abs(value) >= tol
        if active.any():
            sites = np.empty((np.count_nonzero(active), 4), np.int16)
            sites[:, 0], sites[:, 1] = p, q
            sites[:, 2], sites[:, 3] = ra[active], sa[active]
            yield sites, value[active]


def _packed_terms(integrals: tuple[float, np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Count first so output allocation is exact, even for a nearly dense active space.
    count = sum(len(values) for _, values in _term_blocks(*integrals))
    logging.info(
        "Preparing %d normal-ordered molecular terms (%.2f GiB of packed operator data)", count, count * 28 / 2**30
    )
    sites = np.zeros((count, 4), dtype=np.int16)
    kinds = np.full((count, 4), 2, dtype=np.uint8)
    coefficients = np.zeros((count, 2), dtype=np.float64)
    start = 0
    for block, values in _term_blocks(*integrals):
        end = start + len(values)
        width = block.shape[1]
        sites[start:end, :width] = block
        kinds[start:end, : width // 2] = 1
        kinds[start:end, width // 2 : width] = 0
        coefficients[start:end, 0] = values
        start = end
    return torch.from_numpy(sites), torch.from_numpy(kinds), torch.from_numpy(coefficients)

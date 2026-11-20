"""Fermionic phases between interleaved and spin-separated orbital orderings."""

import numpy as np


def spin_order_sign(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Count occupied beta orbitals crossed when moving each alpha orbital to the front."""
    inversions = (alpha * (beta.cumsum(axis=1) - beta)).sum(axis=1)
    return 1 - 2 * (inversions % 2).astype(np.int64)

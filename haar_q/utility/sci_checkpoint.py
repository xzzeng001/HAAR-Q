"""Read physical CI states from HAAR-SCI training checkpoints."""

from typing import Any

import torch


def latest_ritz(haar: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return (energy, configurations, signed coefficients) for the last completed cycle.

    The sampling pool can combine magnitudes from several roots. Observables
    that require a physical wavefunction should use the stored Ritz vector.
    """
    history = haar.get("excited")
    if not isinstance(history, dict):
        return None
    states = history.get(haar.get("global", 0) - 1)
    if not states:
        return None
    energy, configs, coefficients = states[0]
    return energy, configs, coefficients


def checkpoint_state(haar: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the physical ground state, falling back to a legacy saved pool."""
    root = latest_ritz(haar)
    state = root[1:] if root is not None else haar.get("pool")
    if not isinstance(state, (tuple, list)) or len(state) != 2 or not all(isinstance(x, torch.Tensor) for x in state):
        raise ValueError("The SCI checkpoint has no saved ground-state wavefunction or usable legacy pool.")
    return state[0], state[1]

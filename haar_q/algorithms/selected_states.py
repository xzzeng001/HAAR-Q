"""Lowest eigenpairs of the Hamiltonian projected onto an explicit determinant set."""

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg  # type: ignore[import-untyped]
import torch

from ..utility.model_dict import ModelProto


def solve_selected_states(
    model: ModelProto, configs: torch.Tensor, roots: int, residual_tolerance: float = 1e-8
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[float]]:
    """Return ascending energies, signed normalized CI vectors and absolute projected residual norms."""
    dimension = len(configs)
    if type(roots) is not int or not 1 <= roots <= dimension:
        raise ValueError(
            f"Requested {roots} roots for a selected basis of {dimension} determinants. "
            "Choose a positive root count within the basis or enlarge the selected space."
        )
    if (
        isinstance(residual_tolerance, bool)
        or not isinstance(residual_tolerance, (int, float))
        or not math.isfinite(residual_tolerance)
        or residual_tolerance <= 0
    ):
        raise ValueError("The selected-space residual tolerance must be finite and positive.")

    if dimension <= 128:
        one = torch.ones(1, dtype=torch.complex128, device=configs.device)
        matrix = torch.stack([model.apply_within(configs[i : i + 1], one, configs) for i in range(dimension)], dim=1)
        energies, vectors = torch.linalg.eigh(matrix)
        energies, vectors = energies[:roots], vectors[:, :roots]
    else:
        # SciPy's block method switches to a dense full eigensolve for smaller dimension/block ratios.
        if dimension < 5 * roots:
            raise ValueError(
                "For more than 128 determinants request at most one fifth of the selected dimension "
                "so the block eigensolver can operate without a dense Hamiltonian."
            )

        def apply(vector: np.ndarray) -> np.ndarray:
            v = torch.as_tensor(vector.reshape(-1), device=configs.device, dtype=torch.complex128)
            return model.apply_within(configs, v.contiguous(), configs).cpu().numpy()

        operator = LinearOperator((dimension, dimension), matvec=apply, dtype=np.complex128)
        # Independent trial vectors span degenerate eigenspaces and disconnected Hamiltonian blocks.
        rng = np.random.default_rng(0)
        shape = (dimension, roots)
        guess = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        values, columns = lobpcg(operator, guess, largest=False, tol=residual_tolerance * 0.1, maxiter=300)
        order = np.argsort(values)
        energies = torch.as_tensor(values[order], device=configs.device)
        vectors = torch.as_tensor(columns[:, order], device=configs.device)

    if vectors.shape != (dimension, roots) or energies.shape != (roots,):
        raise RuntimeError(f"Selected CI solver did not return all {roots} requested roots.")
    overlap = vectors.mH @ vectors
    if not torch.allclose(overlap, torch.eye(roots, dtype=overlap.dtype, device=overlap.device), rtol=0, atol=1e-9):
        raise RuntimeError("Selected CI solver returned nonorthogonal eigenvectors.")
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    residuals = []
    for energy, column in zip(energies, vectors.T):
        vector = column.contiguous()
        residual = torch.linalg.vector_norm(model.apply_within(configs, vector, configs) - energy * vector).item()
        if not math.isfinite(residual) or residual > residual_tolerance:
            raise RuntimeError(
                f"Selected CI root {len(results)} has projected residual {residual:.3e}, "
                f"exceeding {residual_tolerance:.3e}."
            )
        results.append((energy, configs, vector))
        residuals.append(residual)
    return results, residuals

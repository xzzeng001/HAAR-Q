"""Check fixed-determinant forces against differences of independently projected PySCF energies."""

import math

import numpy as np
from pyscf import ao2mo, fci, gto, lib, scf  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.algorithms.forces import _gradient, _state_rdms, selected_forces
from haar_q.algorithms.selected_states import solve_selected_states
from haar_q.models.geometry import Model, ModelConfig
from haar_q.utility.parallel_hamiltonian import ParallelHamiltonian


def projected_energies(
    atoms: list[tuple[str, list[float]]], basis: str, charge: int, occupations: list[tuple[int, int]]
) -> np.ndarray:
    """Reoptimize RHF at this geometry, then restrict the full FCI matrix to these orbital occupations."""
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    nstrings = math.comb(nmo, mol.nelec[1])
    addresses, matrix = fci.direct_spin1.pspace(
        mo.T @ mf.get_hcore() @ mo, ao2mo.kernel(mol, mo), nmo, mol.nelec, np=nstrings**2
    )
    row = {int(address): i for i, address in enumerate(addresses)}
    selected = [
        row[fci.cistring.str2addr(nmo, mol.nelec[0], alpha) * nstrings + fci.cistring.str2addr(nmo, mol.nelec[1], beta)]
        for alpha, beta in occupations
    ]
    return np.linalg.eigvalsh(matrix[np.ix_(selected, selected)]) + mol.energy_nuc()


@pytest.mark.parametrize(
    ("symbols", "positions", "basis", "charge", "occupations", "axis"),
    [
        (
            ["He", "H"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.57]],
            "6-31g",
            1,
            [(1, 1), (1, 2), (4, 1), (2, 4), (8, 2)],
            2,
        ),
        (
            ["H"] * 4,
            [[0.0, 0.0, 0.0], [0.1, 0.0, 1.3], [2.4, 0.0, 0.4], [2.7, 0.0, 1.9]],
            "sto-3g",
            0,
            [(3, 3), (3, 5), (3, 6), (5, 3), (6, 9), (9, 5), (5, 10)],
            0,
        ),
    ],
)
def test_truncated_ground_and_excited_forces(
    symbols: list[str],
    positions: list[list[float]],
    basis: str,
    charge: int,
    occupations: list[tuple[int, int]],
    axis: int,
) -> None:
    # Both systems have four orbitals. H4 also probes rotations within the occupied space.
    coords = np.array(positions)
    atom_id = len(symbols) - 1
    step = 1.0e-4
    displacement = np.zeros_like(coords)
    displacement[atom_id, axis] = step
    atoms = "\n".join(f"{symbol} {x} {y} {z}" for symbol, (x, y, z) in zip(symbols, coords))
    configs = [
        sum((((alpha >> i) & 1) + 2 * ((beta >> i) & 1)) << (2 * i) for i in range(4)) for alpha, beta in occupations
    ]
    packed = torch.tensor(configs, dtype=torch.uint8).reshape(-1, 1)
    with lib.with_omp_threads(1):
        model = Model(ModelConfig(atoms=atoms, unit="Bohr", basis=basis, charge=charge, scf_conv_tol=1e-13))
        minus, reference, plus = [
            projected_energies(list(zip(symbols, shifted.tolist())), basis, charge, occupations)
            for shifted in (coords - displacement, coords, coords + displacement)
        ]
        for root in (0, 1):
            energy, force = selected_forces(model, packed, root)
            assert energy == pytest.approx(reference[root], abs=1e-9, rel=0)
            assert force[atom_id, axis] == pytest.approx((minus[root] - plus[root]) / (2 * step), abs=2e-6, rel=0)
            np.testing.assert_allclose(force.sum(axis=0), 0, atol=2e-7)


@pytest.mark.parametrize(
    "backend",
    [
        "cpu-slabs",
        pytest.param("two-cuda", marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")),
    ],
)
def test_slab_forces_match_independent_h2_energy_differences(backend: str) -> None:
    distance, step = 1.4, 1.0e-4
    with lib.with_omp_threads(1):
        model = Model(ModelConfig(atoms=f"H 0 0 0; H 0 0 {distance}", unit="Bohr", basis="sto-3g", scf_conv_tol=1e-13))
        packed = torch.tensor([[3], [12]], dtype=torch.uint8)
        if backend == "two-cuda":
            model.hamiltonian = ParallelHamiltonian(model.hamiltonian, (0, 1))
            packed = packed.to("cuda:0")
        energies = [
            projected_energies([("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, r])], "sto-3g", 0, [(1, 1), (2, 2)])
            for r in (distance - step, distance, distance + step)
        ]
        for root in (0, 1):
            if backend == "two-cuda":
                energy, forces = selected_forces(model, packed, root, gpu_devices=(0, 1))
            else:
                states, _ = solve_selected_states(model, packed, 2)
                eigenvalue, configs, vector = states[root]
                dm1, dm2 = _state_rdms(configs, vector, 2, 1)
                energy = eigenvalue.item()
                forces = -_gradient(model, dm1, dm2, (torch.device("cpu"), torch.device("cpu")))
            expected = np.zeros((2, 3))
            expected[1, 2] = (energies[0][root] - energies[2][root]) / (2 * step)
            expected[0, 2] = -expected[1, 2]
            assert energy == pytest.approx(energies[1][root], abs=1e-9, rel=0)
            np.testing.assert_allclose(forces, expected, atol=2e-7, rtol=0)

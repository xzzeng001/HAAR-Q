"""Verify molecular Hamiltonians from coordinates against independent PySCF calculations."""

from pathlib import Path

from pyscf import fci, gto, lib, scf  # type: ignore[import-untyped]
from pyscf.lib import param  # type: ignore[import-untyped]
from pyscf.tools import fcidump  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.models import fcidump as dump_model
from haar_q.models import geometry


@pytest.mark.parametrize(("charge", "spin"), [(1, 0), (0, 1)])
def test_geometry_hamiltonian_matches_fcidump_and_exact_energy(tmp_path: Path, charge: int, spin: int) -> None:
    # HeH+ is closed shell. Neutral HeH has three electrons in a doublet.
    atoms = "He 0 0 0\nH 0 0 1.43"
    with lib.with_omp_threads(1):
        mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", charge=charge, spin=spin, verbose=0)
        solver = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
        solver.kernel()
        assert solver.converged
        reference_file = tmp_path / "HeH.FCIDUMP"
        fcidump.from_scf(solver, str(reference_file))
        reference_fci = fci.FCI(solver).kernel()[0]
        model = geometry.Model(geometry.ModelConfig(atoms=atoms, basis="sto-3g", unit="Bohr", charge=charge, spin=spin))
        reference = dump_model.Model(dump_model.ModelConfig(model_path=reference_file, ref_energy=solver.e_tot))

        assert (model.n_qubits, model.n_electrons, model.n_spins) == (4, mol.nelectron, spin)
        assert model.ref_energy == pytest.approx(solver.e_tot, abs=1e-9, rel=0)
        vacuum = torch.zeros((1, 1), dtype=torch.uint8)
        assert model.diagonal_term(vacuum).real.item() == pytest.approx(mol.energy_nuc(), abs=1e-10, rel=0)

        # Alternating bits encode alpha and beta occupations in the two spatial orbitals.
        states = [i for i in range(16) if ((i & 0b0101).bit_count(), (i & 0b1010).bit_count()) == mol.nelec]
        configs = torch.tensor(states, dtype=torch.uint8).reshape(-1, 1)
        unit_vectors = torch.eye(len(states), dtype=torch.complex128)
        matrix = torch.stack([model.apply_within(configs, v, configs) for v in unit_vectors], dim=1)
        expected = torch.stack([reference.apply_within(configs, v, configs) for v in unit_vectors], dim=1)
        torch.testing.assert_close(matrix, expected, rtol=0, atol=1e-9)
        assert torch.linalg.eigvalsh(matrix)[0].item() == pytest.approx(reference_fci, abs=1e-9, rel=0)


def test_xyz_and_inline_coordinates_give_the_same_hamiltonian(tmp_path: Path) -> None:
    xyz = tmp_path / "H2.xyz"
    xyz.write_text("2\nH2 in angstrom\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    with lib.with_omp_threads(1):
        from_xyz = geometry.Model(geometry.ModelConfig(atoms="He 0 0 0", xyz_file=xyz, unit="Angstrom", basis="sto-3g"))
        in_bohr = geometry.Model(
            geometry.ModelConfig(atoms=f"H 0 0 0\nH 0 0 {0.74 / param.BOHR:.15f}", unit="Bohr", basis="sto-3g")
        )
        assert from_xyz.n_qubits == in_bohr.n_qubits == 4
        assert from_xyz.ref_energy == pytest.approx(in_bohr.ref_energy, abs=1e-10, rel=0)
        all_configs = torch.arange(16, dtype=torch.uint8).reshape(-1, 1)
        torch.testing.assert_close(from_xyz.diagonal_term(all_configs), in_bohr.diagonal_term(all_configs))


def test_xyz_atom_count_must_match_its_coordinates(tmp_path: Path) -> None:
    xyz = tmp_path / "truncated.xyz"
    xyz.write_text("2\nMissing the second atom\nH 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="XYZ"):
        geometry.Model(geometry.ModelConfig(xyz_file=xyz))


def test_orbital_identity_allows_phase_but_detects_different_basis_or_geometry() -> None:
    with lib.with_omp_threads(1):
        model = geometry.Model(geometry.ModelConfig(atoms="H 0 0 0; H 0 0 0.74"))
    saved = model.orbital_identity()
    coeff, coords = saved["mo_coeff"], saved["coordinates_bohr"]
    assert isinstance(coeff, torch.Tensor) and isinstance(coords, torch.Tensor)
    assert model.matches_orbital_identity({**saved, "mo_coeff": coeff * torch.tensor([-1, 1])})
    assert not model.matches_orbital_identity({**saved, "mo_coeff": coeff.flip(1)})
    assert not model.matches_orbital_identity({**saved, "coordinates_bohr": coords + 0.01})
    assert not model.matches_orbital_identity(None)


@pytest.mark.parametrize(("charge", "spin", "message"), [(0, 0, "incompatible with spin"), (-2, 1, "capacity")])
def test_requested_spin_sector_must_exist(charge: int, spin: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        geometry.Model(geometry.ModelConfig(atoms="H 0 0 0", basis="sto-3g", charge=charge, spin=spin))


def test_unconverged_scf_does_not_start_the_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scf.hf.RHF, "kernel", lambda self: 0.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        geometry.Model(geometry.ModelConfig(atoms="H 0 0 0\nH 0 0 0.74"))

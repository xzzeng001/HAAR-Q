"""Build a molecular Hamiltonian in restricted Hartree–Fock orbitals from Cartesian coordinates."""

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any

import torch
from pyscf import ao2mo, gto, scf  # type: ignore[import-untyped]

from ..utility.model_dict import model_dict
from .pyscf import Model as IntegralModel
from .pyscf import ModelConfig as Integrals


@dataclasses.dataclass
class ModelConfig:
    """Coordinate source and PySCF settings for an all-electron molecular calculation."""

    atoms: str | None = None
    xyz_file: Path | None = None
    basis: str = "sto-3g"
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0
    scf_conv_tol: float = 1.0e-9
    scf_max_cycle: int = 50
    ref_energy: float | None = None


def _coordinates(config: ModelConfig) -> list[tuple[str, list[float]]]:
    if config.xyz_file is not None:
        lines = config.xyz_file.read_text(encoding="utf-8-sig").rstrip().splitlines()
        if not lines or not lines[0].strip().isdecimal():
            raise ValueError(f"Expected an atom count on the first line of XYZ file {config.xyz_file}.")
        count = int(lines[0])
        if count < 1 or len(lines) != count + 2:
            raise ValueError(f"XYZ file {config.xyz_file} must contain a comment line and exactly {count} atom rows.")
        rows = lines[2:]
    elif config.atoms and config.atoms.strip():
        rows = [row for row in config.atoms.replace(";", "\n").splitlines() if row.strip()]
    else:
        raise ValueError("Provide atoms or xyz_file for the geometry system.")

    atoms = []
    for row in rows:
        fields = row.split()
        if len(fields) != 4:
            raise ValueError(f"Expected a Cartesian row 'element x y z', got {row!r}.")
        try:
            position = [float(field) for field in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid Cartesian coordinate in {row!r}.") from exc
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"Cartesian coordinates must be finite in {row!r}.")
        atoms.append((fields[0], position))
    if not atoms:
        raise ValueError("Provide at least one Cartesian atom row.")
    return atoms


def _compute_integrals(config: ModelConfig) -> tuple[Integrals, Any]:
    unit = config.unit.strip().lower()
    if unit not in {"angstrom", "bohr"}:
        raise ValueError("The coordinate unit must be Angstrom or Bohr.")
    if not math.isfinite(config.scf_conv_tol) or config.scf_conv_tol <= 0 or config.scf_max_cycle < 1:
        raise ValueError("scf_conv_tol and scf_max_cycle must be positive.")
    if not config.basis.strip():
        raise ValueError("Provide a PySCF basis-set name.")
    if config.ref_energy is not None and not math.isfinite(config.ref_energy):
        raise ValueError("ref_energy must be a finite total energy in hartree.")

    atoms = _coordinates(config)
    source = config.xyz_file.resolve() if config.xyz_file is not None else "inline atoms"
    logging.info(
        "Cartesian coordinates (%s) from %s\n%s",
        unit,
        source,
        "\n".join(f"  {symbol:<3} {x: .12g} {y: .12g} {z: .12g}" for symbol, (x, y, z) in atoms),
    )
    nelec = sum(gto.charge(symbol) for symbol, _ in atoms) - config.charge
    if nelec < abs(config.spin) or (nelec - config.spin) % 2:
        raise ValueError(f"Electron count {nelec} is incompatible with spin={config.spin} (N_alpha - N_beta).")
    # The spin-free Hamiltonian is invariant under exchange of alpha and beta.
    # Generate the shared spatial orbitals with alpha-majority ROHF, and retain
    # the requested signed spin sector for all downstream wavefunctions.
    mol = gto.M(atom=atoms, basis=config.basis, unit=unit, charge=config.charge, spin=abs(config.spin), verbose=0)
    if max(mol.nelec) > mol.nao_nr():
        raise ValueError(f"The requested spin populations {mol.nelec} exceed the capacity of the {config.basis} basis.")
    method = "RHF" if config.spin == 0 else "ROHF"
    logging.info(
        "PySCF %s with %d atoms, basis=%s, unit=%s, charge=%d, spin=%d",
        method,
        len(atoms),
        config.basis,
        unit,
        config.charge,
        config.spin,
    )
    mean_field = scf.RHF(mol) if config.spin == 0 else scf.ROHF(mol)
    mean_field.chkfile = None
    mean_field.conv_tol = config.scf_conv_tol
    mean_field.max_cycle = config.scf_max_cycle
    hf_energy = float(mean_field.kernel())
    if not mean_field.converged or not math.isfinite(hf_energy):
        raise RuntimeError(f"PySCF {method} did not converge in {config.scf_max_cycle} cycles.")

    coeff = mean_field.mo_coeff
    norb = coeff.shape[1]
    h1e = coeff.T @ mean_field.get_hcore() @ coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, coeff), norb)
    enuc = float(mol.energy_nuc())
    reference = hf_energy if config.ref_energy is None else config.ref_energy
    logging.info("PySCF %s total energy = %.12f Eh, nuclear repulsion = %.12f Eh", method, hf_energy, enuc)
    logging.info("Reporting reference (%s) = %.12f Eh", method if config.ref_energy is None else "user", reference)
    return Integrals(
        h1e=h1e,
        eri=eri,
        n_orbit=norb,
        n_electron=mol.nelectron,
        n_spin=config.spin,
        nuclear_repulsion=enuc,
        ref_energy=reference,
    ), mean_field


class Model(IntegralModel):
    """Use the molecular integral model and its networks in the orbitals generated by PySCF."""

    config_t = ModelConfig  # type: ignore[assignment]

    def __init__(self, args: ModelConfig) -> None:
        self.integrals, self.mean_field = _compute_integrals(args)
        super().__init__(self.integrals)

    def orbital_identity(self) -> dict[str, object]:
        """Record the ordered nuclei and MO basis in which determinant bits are defined."""
        mol = self.mean_field.mol
        return {
            "atoms": [mol.atom_symbol(i) for i in range(mol.natm)],
            "coordinates_bohr": torch.from_numpy(mol.atom_coords()),
            "basis": mol.basis,
            "charge": mol.charge,
            "spin": self.n_spins,
            "mo_coeff": torch.from_numpy(self.mean_field.mo_coeff),
        }

    def matches_orbital_identity(self, saved: object) -> bool:
        """Allow a column sign change, which preserves the selected determinant space."""
        if not isinstance(saved, dict):
            return False
        current = self.orbital_identity()
        if any(saved.get(key) != current[key] for key in ("atoms", "basis", "charge", "spin")):
            return False
        for key in ("coordinates_bohr", "mo_coeff"):
            left, right = saved.get(key), current[key]
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
                return False
            left = left.to(right)
            if key == "mo_coeff":
                right = right * torch.where((left * right).sum(dim=0) < 0, -1, 1)
            if not torch.allclose(left, right, atol=1e-8, rtol=1e-8):
                return False
        return True


model_dict["geometry"] = Model

"""State-specific nuclear forces for a fixed determinant space in canonical RHF orbitals."""

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig
from pyscf.fci import selected_ci  # type: ignore[import-untyped]
from pyscf.hessian import rhf as rhf_hessian  # type: ignore[import-untyped]
from pyscf.scf import cphf  # type: ignore[import-untyped]
import torch

from ..models.geometry import Model
from ..utility.action_dict import action_dict
from ..utility.context import RuntimeContext
from .selected_states import solve_selected_states


def _state_rdms(
    configs: torch.Tensor, wavefunction: torch.Tensor, norb: int, nocc: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the interleaved-spin state into PySCF's selected alpha/beta string representation."""
    bits = np.unpackbits(configs.detach().cpu().numpy(), axis=1, bitorder="little")
    alpha, beta = bits[:, : 2 * norb : 2], bits[:, 1 : 2 * norb : 2]
    if np.any(bits[:, 2 * norb :]) or np.any(alpha.sum(axis=1) != nocc) or np.any(beta.sum(axis=1) != nocc):
        raise ValueError("The force calculation requires determinants in the specified RHF alpha/beta sector.")
    powers = 1 << np.arange(norb, dtype=np.int64)
    astr, ia = np.unique(alpha @ powers, return_inverse=True)
    bstr, ib = np.unique(beta @ powers, return_inverse=True)
    if len(astr) * len(bstr) > 1_000_000:
        raise ValueError("The force RDM implementation supports at most 1,000,000 selected alpha/beta string pairs.")
    # Moving all beta creation operators behind all alpha operators introduces
    # one transposition per occupied beta orbital below an occupied alpha orbital.
    inversions = (alpha * (beta.cumsum(axis=1) - beta)).sum(axis=1)
    parity = 1 - 2 * (inversions % 2).astype(np.int64)
    vec = wavefunction.detach().cpu().numpy()
    vec = vec * np.exp(-1j * np.angle(vec[np.argmax(np.abs(vec))]))
    if np.max(np.abs(vec.imag)) > 1e-7:
        raise ValueError("The isolated root must admit a real wavefunction in the canonical RHF basis.")
    ci = np.zeros((len(astr), len(bstr)))
    ci[ia, ib] = vec.real * parity
    state = selected_ci._as_SCIvector(ci, (astr, bstr))
    dm1, dm2 = selected_ci.SelectedCI().make_rdm12(state, norb, (nocc, nocc))
    # The real Coulomb tensor and all of its total derivatives have pair symmetry.
    dm2 = (dm2 + dm2.swapaxes(0, 1)) * 0.5
    dm2 = (dm2 + dm2.swapaxes(2, 3)) * 0.5
    return dm1, (dm2 + dm2.transpose(2, 3, 0, 1)) * 0.5


def _orbital_response(mf: Any, fock_ao: np.ndarray, overlap_ao: np.ndarray) -> np.ndarray:
    """Compute dC=C U including rotations between occupied or between virtual canonical orbitals."""
    c, eps = mf.mo_coeff, mf.mo_energy
    occ = mf.mo_occ > 0
    co = c[:, occ]
    s1 = c.T @ overlap_ao @ c
    f1 = c.T @ fock_ao @ c
    if (~occ).any():
        response = rhf_hessian.gen_vind(mf, c, mf.mo_occ)
        uo, _ = cphf.solve(response, eps, mf.mo_occ, f1[:, :, occ], s1[:, :, occ], max_cycle=100, tol=1e-11)
    else:
        uo = -0.5 * s1
    # Occupied-occupied gauge terms cancel in the density response.
    dm_half = (c @ uo) @ co.T * 2
    dm1 = dm_half + dm_half.transpose(0, 2, 1)
    f1 += c.T @ mf.get_veff(mf.mol, dm1) @ c
    # Off-diagonal differentiated generalized Fock eigenvalue equation.
    gaps = eps[None, :] - eps[:, None]
    diag = np.diag_indices(len(eps))
    gaps[diag] = 1
    u = (f1 - s1 * eps) / gaps
    u[:, diag[0], diag[1]] = -0.5 * s1[:, diag[0], diag[1]]
    if not np.allclose(u[:, ~occ][:, :, occ], uo[:, ~occ], rtol=1e-7, atol=1e-8):
        raise RuntimeError("The RHF orbital response did not converge.")
    return u


def _gradient(model: Model, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    mf = model.mean_field
    mol, c = mf.mol, mf.mo_coeff
    h1, eri = model.integrals.h1e, model.integrals.eri
    dm1_ao = c @ dm1 @ c.T
    dm2_ao = np.einsum("ap,bq,cr,ds,pqrs->abcd", c, c, c, c, dm2, optimize=True)
    # Differentiating the four MO coefficients in (pq|rs) leaves four equal contributions.
    orbital_derivative = 2 * (h1 @ dm1 + np.einsum("pjkl,qjkl->pq", eri, dm2, optimize=True))
    grad = mf.nuc_grad_method()
    derivative_hcore = grad.hcore_generator(mol)
    bra_overlap = grad.get_ovlp(mol)
    fock_derivatives = rhf_hessian.make_h1(rhf_hessian.Hessian(mf), c, mf.mo_occ)
    result = grad.grad_nuc(mol)
    for atom, (sh0, sh1, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ds = np.zeros_like(bra_overlap)
        ds[:, p0:p1] = bra_overlap[:, p0:p1]
        u = _orbital_response(mf, fock_derivatives[atom], ds + ds.transpose(0, 2, 1))
        result[atom] += np.einsum("pq,xpq->x", orbital_derivative, u)
        result[atom] += np.einsum("pq,xpq->x", dm1_ao, derivative_hcore(atom))
        # int2e_ip1 differentiates the first AO with respect to electron
        # position. Its derivative with respect to the AO center has opposite sign.
        d_eri = mol.intor("int2e_ip1", comp=3, shls_slice=(sh0, sh1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas))
        result[atom] -= 2 * np.einsum("xpqrs,pqrs->x", d_eri, dm2_ao[p0:p1])
    if not np.isfinite(result).all():
        raise RuntimeError("The analytical gradient contains non-finite entries.")
    return result


def selected_forces(model: Model, configs: torch.Tensor, root: int) -> tuple[float, np.ndarray]:
    """Return an isolated projected root's total energy (Eh) and negative nuclear gradient (Eh/bohr)."""
    if model.n_spins != 0 or model.n_electrons == 0:
        raise ValueError("Analytical forces currently require a closed-shell RHF reference with electrons.")
    if model.integrals.n_orbit > 32:
        raise ValueError("The in-memory force implementation supports up to 32 spatial orbitals.")
    if configs.dtype != torch.uint8 or configs.ndim != 2 or configs.shape[1] != (model.n_qubits + 7) // 8:
        raise ValueError("Checkpoint determinants must be packed uint8 rows with the size of the molecular basis.")
    if len(torch.unique(configs, dim=0)) != len(configs):
        raise ValueError("Checkpoint determinants must be unique.")
    if type(root) is not int or not 0 <= root < len(configs):
        raise ValueError(f"Force root {root} is outside the selected space of {len(configs)} determinants.")
    mf = model.mean_field
    if mf.mo_coeff.shape[0] != mf.mo_coeff.shape[1]:
        raise ValueError("Analytical forces require a linearly independent AO basis.")
    if np.any(np.diff(np.sort(mf.mo_energy)) < 1e-7):
        raise ValueError("Canonical orbital energies within 1e-7 Eh do not define an isolated orbital basis.")
    states, _ = solve_selected_states(model, configs, min(root + 2, len(configs)), residual_tolerance=1e-9)
    energy_t, states_configs, vector = states[root]
    energy = float(energy_t.real.item())
    if any(abs(float(other[0].real.item()) - energy) < 1e-7 for i, other in enumerate(states) if i != root):
        raise ValueError("The requested SCI root is degenerate within 1e-7 Eh; a state-specific force is undefined.")
    dm1, dm2 = _state_rdms(states_configs, vector, model.integrals.n_orbit, model.n_electrons // 2)
    mo = model.integrals
    check_energy = mo.nuclear_repulsion + np.sum(mo.h1e * dm1) + 0.5 * np.sum(mo.eri * dm2)
    if abs(check_energy - energy) > 1e-8:
        raise RuntimeError("The state RDM and selected Hamiltonian energies disagree.")
    return energy, -_gradient(model, dm1, dm2)


@dataclass
class ForcesConfig:
    """Post-process a canonical geometry-SCI checkpoint with a zero-based root index."""

    root: int = 0

    def main(self, context: RuntimeContext, runtime_config: DictConfig, checkpoint_data: dict[str, Any]) -> None:
        haar = checkpoint_data.get("haar", {})
        if "model_path" in haar or haar.get("pool") is None:
            raise ValueError("Forces require a canonical geometry SCI checkpoint with an unrotated determinant pool.")
        identity = haar.get("canonical_geometry")
        if not isinstance(identity, dict):
            raise ValueError("The checkpoint lacks canonical geometry provenance. Run geometry SCI to create it.")
        orbitals = identity.get("mo_coeff")
        if isinstance(orbitals, torch.Tensor) and orbitals.ndim == 2 and orbitals.shape[1] > 32:
            raise ValueError("The in-memory force implementation supports up to 32 spatial orbitals.")
        model = context.create_model(runtime_config.model)
        if not isinstance(model, Model) or not model.matches_orbital_identity(identity):
            raise ValueError("The checkpoint must identify the same molecular geometry, basis and canonical orbitals.")
        energy, forces = selected_forces(model, haar["pool"][0].to(context.device), self.root)
        mol = model.mean_field.mol
        lines = [
            f"# Fixed-space SCI root {self.root}   total energy {energy:.14f} Eh",
            "# Atom Element           Fx               Fy               Fz   (Eh/bohr)",
        ]
        for i, (fx, fy, fz) in enumerate(forces):
            lines.append(f"{i + 1:5d} {mol.atom_symbol(i):<5} {fx:16.10f} {fy:16.10f} {fz:16.10f}")
        report = "\n".join(lines) + "\n"
        (context.folder() / "FORCES").write_text(report, encoding="utf-8")
        logging.info("%s", report)


action_dict["forces"] = ForcesConfig

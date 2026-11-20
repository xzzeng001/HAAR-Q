"""
This file implements orbital optimization (Natural Orbitals) for quantum many-body problems.
"""

import logging
import dataclasses
import pathlib
import typing
import torch
import omegaconf
from ..hamiltonian import Hamiltonian
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict
from ..utility.model_dict import ModelProto
from ..utility.sci_checkpoint import checkpoint_state
from ..models.optimized_basis import Model as OptimizedModel, ModelConfig as OptimizedModelConfig


@dataclasses.dataclass
class OrbitConfig:
    """
    Configuration for orbital optimization.
    """

    # Molecular geometry jobs obtain spatial integrals from the model directly.
    src_fcidump: pathlib.Path | None
    # The output file for the optimized basis Hamiltonian
    dst_optimized: pathlib.Path
    # The reference energy, if not provided it will be 0 or read from checkpoint
    ref_energy: float | None = None

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        model = _load_orbital_model(context, runtime_config.model, checkpoint_data)
        state = checkpoint_data.get("haar", checkpoint_data.get("imag", {}))
        configs, psi = checkpoint_state(state)
        configs = configs.to(device=context.device)
        psi = psi.to(device=context.device)

        n_orbit = typing.cast(typing.Any, model).n_qubits // 2
        calculator = NaturalOrbitCalculator(n_orbit)

        rdm_spatial = calculator.calculate_rdm(configs, psi)
        logging.info("Spatial 1-RDM calculated. Diagonalizing...")

        eigvals, U_spatial = torch.linalg.eigh(rdm_spatial)

        output_data = _rotated_basis(model, self.src_fcidump, U_spatial, eigvals, self.ref_energy)
        self.dst_optimized.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_data, self.dst_optimized)
        logging.info("Successfully saved optimized basis to %s", self.dst_optimized)


def _rotated_basis(
    model: ModelProto,
    src_fcidump: pathlib.Path | None,
    U_spatial: torch.Tensor,
    eigvals: torch.Tensor,
    ref_energy: float | None = None,
) -> dict[str, typing.Any]:
    """Rotate one- and two-electron operators and serialize the resulting restricted basis."""
    (norb, nelec, nspin), e0, h1, h2 = _spatial_integrals(model, src_fcidump)

    device = U_spatial.device
    U_spatial = U_spatial.to(dtype=torch.complex128)
    h1 = h1.to(device=device, dtype=torch.complex128)
    h2 = h2.to(device=device, dtype=torch.complex128)

    logging.info("Transforming spatial orbital integrals...")

    # Transform h1: h1' = U^† h1 U
    h1_opt_spatial = U_spatial.conj().T @ h1 @ U_spatial

    # Transform h2 using sequential contractions: h2'_{abcd} = sum U^*_{pa} U^*_{qb} h2_{pqrs} U_{rc} U_{sd}
    # Contract one axis at a time for O(N^5) work.
    tmp = torch.einsum("sd,pqrs->pqrd", U_spatial, h2)
    tmp = torch.einsum("rc,pqrd->pqcd", U_spatial, tmp)
    tmp = torch.einsum("qb,pqcd->pbcd", U_spatial.conj(), tmp)
    h2_opt_spatial = torch.einsum("pa,pbcd->abcd", U_spatial.conj(), tmp)

    # The same spatial rotation applies to both spin channels.
    logging.info("Expanding transformed integrals to spin-orbital basis...")
    h1_so = torch.zeros((2 * norb, 2 * norb), dtype=torch.complex128, device=device)
    h1_so[0::2, 0::2] = h1_opt_spatial  # alpha-alpha
    h1_so[1::2, 1::2] = h1_opt_spatial  # beta-beta

    h2_so = torch.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=torch.complex128, device=device)
    h2_so[0::2, 0::2, 0::2, 0::2] = h2_opt_spatial  # aaaa
    h2_so[0::2, 1::2, 1::2, 0::2] = h2_opt_spatial  # abba
    h2_so[1::2, 0::2, 0::2, 1::2] = h2_opt_spatial  # baab
    h2_so[1::2, 1::2, 1::2, 1::2] = h2_opt_spatial  # bbbb

    # Build spin-orbital unitary for storage (block-diagonal: same U for alpha and beta)
    U_spin = torch.zeros((2 * norb, 2 * norb), dtype=torch.complex128, device=device)
    U_spin[0::2, 0::2] = U_spatial  # alpha block
    U_spin[1::2, 1::2] = U_spatial  # beta block

    logging.info("Building Hamiltonian dictionary...")
    ham_dict: dict[tuple[tuple[int, int], ...], complex] = {}
    ham_dict[()] = complex(e0)

    h1_indices = torch.nonzero(torch.abs(h1_so) > 1e-12)
    for p, q in h1_indices:
        p, q = p.item(), q.item()
        ham_dict[((p, 1), (q, 0))] = h1_so[p, q].item()

    h2_indices = torch.nonzero(torch.abs(h2_so) > 1e-12)
    for p, q, r, s in h2_indices:
        p, q, r, s = p.item(), q.item(), r.item(), s.item()
        key = ((p, 1), (q, 1), (r, 0), (s, 0))
        ham_dict[key] = ham_dict.get(key, 0j) + h2_so[p, q, r, s].item()

    logging.info("Preparing and saving optimized basis model...")
    site, kind, coef = Hamiltonian._prepare(ham_dict)

    output_data: dict[str, typing.Any] = {
        "hamiltonian": (site.cpu(), kind.cpu(), coef.cpu()),
        "n_qubits": 2 * norb,
        "n_electrons": nelec,
        "n_spins": nspin,
        "ref_energy": ref_energy if ref_energy is not None else model.ref_energy,
        "rdm_eigvals": eigvals.cpu(),
        "orbit_unitary": U_spin.cpu(),  # Store the spin-orbital unitary matrix
    }
    # The following rotation must start from these transformed coefficients.
    output_data["spatial_integrals"] = (e0, h1_opt_spatial.cpu(), h2_opt_spatial.cpu())

    return output_data


def _load_orbital_model(
    context: RuntimeContext, config: omegaconf.DictConfig, data: dict[str, typing.Any]
) -> ModelProto:
    """Resume in the saved orbital basis when the pool came from an orbital SCI cycle."""
    saved_path = data.get("haar", data.get("imag", {})).get("model_path")
    if saved_path is not None:
        path = pathlib.Path(saved_path)
        if path.is_file():
            logging.info("Loading the saved orbital basis from %s", path)
            model = OptimizedModel(OptimizedModelConfig(model_path=path))
            context.configure_model_devices(model)
            return model
        # An explicit optimized-basis input can supply a relocated checkpoint model.
        if config.name != "optimized_basis":
            raise FileNotFoundError(f"The checkpoint's orbital basis is required for this calculation: {path}")
    return context.create_model(config)


def _spatial_integrals(
    model: ModelProto, src_fcidump: pathlib.Path | None
) -> tuple[tuple[int, int, int], float, torch.Tensor, torch.Tensor]:
    """Fetch the one- and two-body coefficients in creation-creation-annihilation-annihilation order."""
    if isinstance(model, OptimizedModel) and model.spatial_integrals is not None:
        e0, h1, h2 = model.spatial_integrals
        return (model.n_qubits // 2, model.n_electrons, model.n_spins), e0, h1, h2
    if src_fcidump is not None:
        # Legacy optimized-basis files require an FCIDUMP in the saved pool's basis.
        return _read_fcidump_tensors(src_fcidump)
    if isinstance(model, OptimizedModel):
        raise ValueError("The saved orbital basis lacks spatial integrals. Supply its matching FCIDUMP.")
    from ..models.geometry import Model as GeometryModel

    if isinstance(model, GeometryModel):
        mo = model.integrals
        h2 = torch.as_tensor(mo.eri).permute(0, 2, 3, 1) / 2
        return (mo.n_orbit, mo.n_electron, mo.n_spin), mo.nuclear_repulsion, torch.as_tensor(mo.h1e), h2
    raise ValueError("Orbital optimization requires a geometry model or matching FCIDUMP integrals.")


def _read_fcidump_tensors(
    file_name: pathlib.Path,
) -> tuple[tuple[int, int, int], float, torch.Tensor, torch.Tensor]:
    """Expand restricted FCIDUMP integrals into the orbital rotation convention."""
    from pyscf import ao2mo  # type: ignore[import-untyped]

    from ..models.fcidump import _load_integrals

    metadata, integrals = _load_integrals(file_name)
    assert integrals is not None
    core, h1, packed_eri = integrals
    # Rotations act on all four MO indices. FCIDUMP stores (ij|kl); the
    # rotation code uses 1/2 (ij|kl) with axes in operator order (i,k,l,j).
    eri = torch.from_numpy(ao2mo.restore(1, packed_eri, metadata[0]))
    h2 = eri.permute(0, 2, 3, 1).contiguous().mul_(0.5)
    return metadata, core, torch.from_numpy(h1), h2


class NaturalOrbitCalculator:
    def __init__(self, n_orbit: int):
        self.n_orbit = n_orbit

    def calculate_rdm(self, configs: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Compute the normalized spin-averaged density D[p,q] = <a_q† a_p>.

        The eigenvectors are orbital coefficients in the original spatial basis.
        A fully occupied spatial orbital has spin-averaged occupation one.
        """
        psi = psi.to(dtype=torch.complex128)
        norm = torch.vdot(psi, psi).real
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Natural orbitals require a finite, positive CI norm.")
        density = psi.new_empty((self.n_orbit, self.n_orbit))
        # Only equal-spin operators contribute. Hermiticity supplies the other triangle.
        for p in range(self.n_orbit):
            for q in range(p + 1):
                operators: dict[tuple[tuple[int, int], ...], complex] = {
                    ((2 * q + spin, 1), (2 * p + spin, 0)): 1.0 + 0j for spin in (0, 1)
                }
                summed = Hamiltonian(operators, kind="fermi").apply_within(configs, psi, configs)
                value = torch.vdot(psi, summed) / (2 * norm)
                density[p, q] = value
                density[q, p] = value.conj()
        return density


action_dict["orbit"] = OrbitConfig

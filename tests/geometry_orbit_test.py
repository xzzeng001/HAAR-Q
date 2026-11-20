"""Verify that consecutive natural-orbital rotations use the current geometry or FCIDUMP basis."""

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest
import torch

from haar_q.algorithms.haar_with_orbit import HaarWithOrbitConfig
from haar_q.algorithms.orbit import NaturalOrbitCalculator, OrbitConfig, _load_orbital_model, _read_fcidump_tensors
from haar_q.models import fcidump
from haar_q.models.geometry import Model as GeometryModel, ModelConfig
from haar_q.utility.context import RuntimeContext


@pytest.mark.parametrize("system", ["geometry", "fcidump"])
def test_orbital_restart_rotates_the_current_integrals(tmp_path: Path, system: str) -> None:
    if system == "geometry":
        initial: GeometryModel | fcidump.Model = GeometryModel(
            ModelConfig(atoms="H 0 0 0; H 0 0 0.74", ref_energy=-1.14)
        )
        path = None
        assert isinstance(initial, GeometryModel)
        mo = initial.integrals
        core, one, two = mo.nuclear_repulsion, torch.as_tensor(mo.h1e), torch.as_tensor(mo.eri).permute(0, 2, 3, 1) / 2
    else:
        path = Path(__file__).resolve().parents[1] / "examples" / "hamiltonians" / "H2.FCIDUMP"
        initial = fcidump.Model(fcidump.ModelConfig(path, ref_energy=-1.14))
        _, core, one, two = _read_fcidump_tensors(path)
    context = Mock(spec=RuntimeContext, device=torch.device("cpu"))
    context.folder.return_value = tmp_path
    model_config = OmegaConf.create({"name": system})
    configs = torch.tensor([[3], [6], [9], [12]], dtype=torch.uint8)
    psi = torch.tensor([0.6, 0.3, 0.2, 0.1], dtype=torch.complex128)
    _, saved = HaarWithOrbitConfig(src_fcidump=path)._optimize_orbit(context, initial, configs, psi, 0)
    first = torch.load(saved, weights_only=True)
    checkpoint = {"haar": {"model_path": str(saved), "pool": (configs, psi)}}
    destination = tmp_path / "natural.pt"
    OrbitConfig(src_fcidump=path, dst_optimized=destination).main(
        context, OmegaConf.create({"model": model_config}), checkpoint
    )
    second = torch.load(destination, weights_only=True)
    context.create_model.assert_not_called()
    u1 = first["orbit_unitary"][0::2, 0::2]
    u2 = second["orbit_unitary"][0::2, 0::2]
    density = NaturalOrbitCalculator(2).calculate_rdm(configs, psi)
    torch.testing.assert_close(u1.mH @ density @ u1, torch.diag(first["rdm_eigvals"]).to(u1.dtype))
    assert u1[0, 1].abs() > 0.01
    u = u1 @ u2
    energy_constant, h1, h2 = second["spatial_integrals"]
    assert energy_constant == pytest.approx(core)
    assert second["ref_energy"] == initial.ref_energy
    expected_h1 = u.mH @ one.to(u.dtype) @ u
    eri = two.to(u.dtype)
    expected_h2 = torch.einsum("pa,qb,rc,sd,pqrs->abcd", u.conj(), u.conj(), u, u, eri)
    torch.testing.assert_close(h1, expected_h1)
    torch.testing.assert_close(h2, expected_h2)

    saved.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint's orbital basis"):
        _load_orbital_model(context, model_config, checkpoint)

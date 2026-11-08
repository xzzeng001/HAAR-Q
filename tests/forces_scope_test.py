"""Check that force requests outside the supported state or orbital space fail before differentiation."""

from unittest.mock import Mock

from omegaconf import OmegaConf
from pyscf import lib  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.algorithms.forces import ForcesConfig, selected_forces
from haar_q.models.geometry import Model, ModelConfig
from haar_q.utility.context import RuntimeContext


def test_forces_require_canonical_geometry_provenance() -> None:
    context = Mock(spec=RuntimeContext)
    checkpoint = {"haar": {"pool": (torch.tensor([[3]], dtype=torch.uint8), torch.ones(1))}}
    with pytest.raises(ValueError, match="lacks canonical geometry"):
        ForcesConfig().main(context, OmegaConf.create({}), checkpoint)
    context.create_model.assert_not_called()


def test_fixed_space_force_requires_valid_root_and_isolated_orbitals() -> None:
    with lib.with_omp_threads(1):
        model = Model(ModelConfig(atoms="H 0 0 0; H 0 0 0.74"))
        configs = torch.tensor([[3]], dtype=torch.uint8)
        with pytest.raises(ValueError, match="outside the selected space"):
            selected_forces(model, configs, 1)
        model.mean_field.mo_energy[:] = -1.0
        with pytest.raises(ValueError, match="isolated orbital basis"):
            selected_forces(model, configs, 0)

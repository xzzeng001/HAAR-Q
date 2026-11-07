"""Exercise compatibility checks on resolved jobs from the standard input."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pytest

from haar_q.__main__ import _prepare_input


ROOT = Path(__file__).resolve().parents[1]


def selected_job(*overrides: str) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT)):
        config = hydra.compose(config_name="input", overrides=list(overrides))
    return _prepare_input(config, ROOT, check=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["ansatz=mps"], "mps requires system=hubbard or system=ising"),
        (["system=ising", "symmetry=u1u1"], "symmetry=0 or symmetry=u1"),
        (["method=haar", "system=hubbard", "ansatz=mps"], "requires unique sampling"),
        (["method=guide", "system=hubbard", "sampling={name:mps,params:{virtual_dim:4}}"], "guide sampler mps"),
        (["method=pretrain"], "files.dataset"),
        (["system=openfermion"], "files.molecule"),
        (["method=orbit", "system=hubbard"], "requires a molecular system"),
        (
            ["method=haar_with_orbit", "system=optimized_basis", "common.parent_path=prior.pth", "files.fcidump=null"],
            "files.fcidump",
        ),
    ],
)
def test_incompatible_components_and_missing_inputs(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError) as caught:
        selected_job(*overrides)
    assert message in str(caught.value)


def test_active_options_can_supply_paths_and_shared_controls() -> None:
    config = selected_job(
        "method=pretrain",
        "system=openfermion",
        "options.system.openfermion.model_path=molecule.hdf5",
        "options.method.pretrain.dataset_path=target.pt",
        "training.learning_rate=0.04",
    )
    assert Path(config.model.params.model_path) == ROOT / "molecule.hdf5"
    assert Path(config.action.params.dataset_path) == ROOT / "target.pt"
    assert config.optimizer.params.lr == 0.04


def test_pool_processing_uses_existing_run_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HydraConfig, "initialized", lambda: True)
    monkeypatch.setattr(HydraConfig, "get", lambda: OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}}))
    # Post-processing never instantiates the wavefunction, so its options cannot block this job.
    choices = ["method=pert", "ansatz=mps"]
    with pytest.raises(ValueError, match="common.parent_path"):
        selected_job(*choices)
    (tmp_path / "data.pth").touch()
    assert selected_job(*choices).action.name == "pert"


def test_resumed_orbital_sci_can_take_its_model_from_checkpoint() -> None:
    config = selected_job("method=haar_with_orbit", "system=optimized_basis", "common.parent_path=prior.pth")
    assert config.model.params.model_path is None
    assert Path(config.action.params.src_fcidump).is_file()

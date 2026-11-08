"""Exercise the module selectors through the installed application's input interface."""

from dataclasses import fields, is_dataclass
from pathlib import Path
import sys

from omegaconf import DictConfig, OmegaConf
import pytest

from haar_q import __main__ as cli
from haar_q.utility.action_dict import _PARAMETER_NAMES, create_action

TEMPLATE = Path(__file__).resolve().parents[1] / "input.yaml"


def preview(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], source: Path, *overrides: str
) -> DictConfig:
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", str(source), "--show-input", *overrides])
    cli.main()
    job = OmegaConf.create(capsys.readouterr().out)
    assert isinstance(job, DictConfig)
    return job


@pytest.mark.parametrize(
    "method", ["vmc", "haar", "excited", "forces", "guide", "pretrain", "pert", "chop_imag", "orbit", "haar_with_orbit"]
)
def test_single_method_selector_composes_its_complete_action(
    method: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    overrides = ["files.dataset=study.pt"] if method == "pretrain" else []
    job = preview(monkeypatch, capsys, TEMPLATE, f"method={method}", *overrides)
    assert job.action.name == method
    configured = create_action(job.action)
    assert is_dataclass(configured)
    aliases = _PARAMETER_NAMES.get(method, {})
    normalized = {aliases.get(key, key) for key in job.action.params}
    assert normalized == {field.name for field in fields(configured) if field.init}
    assert ("sampling" in job) == (method == "guide")
    if method == "guide":
        assert job.sampling == job.network
    if method in {"pert", "orbit", "chop_imag", "forces"}:
        assert "network" not in job and "optimizer" not in job
    assert "options" not in job
    assert not list(tmp_path.iterdir())


def test_switch_in_one_input_preserves_shared_and_per_method_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = tmp_path / "study.yaml"
    editable = OmegaConf.load(TEMPLATE)
    OmegaConf.update(editable, "training.optimizer_steps", 3)
    OmegaConf.update(editable, "options.method.haar.krylov_iteration", 2)
    # This expression needs data from a different study. An inactive method must leave it untouched.
    OmegaConf.update(editable, "options.method.pretrain.dataset_path", "${missing_study_dataset}")
    OmegaConf.save(editable, source)

    vmc = preview(monkeypatch, capsys, source)
    assert vmc.action.params.optimizer_steps == 3 and "krylov_iteration" not in vmc.action.params

    # Edit the selector itself, as someone running successive experiments from the same file would.
    method = next(group for group in editable.defaults if isinstance(group, DictConfig) and "method" in group)
    method.method = "haar"
    OmegaConf.save(editable, source)
    sci = preview(monkeypatch, capsys, source)
    assert sci.action.params.optimizer_steps == 3 and sci.action.params.krylov_iteration == 2
    assert "hamiltonian_guided" not in sci.action.params


@pytest.mark.parametrize(
    ("overrides", "model", "network", "optimizer"),
    [
        ([], "fcidump", "mlp/u1u1", "Adam"),
        (["system=geometry"], "geometry", "mlp/u1u1", "Adam"),
        (["system=ising", "ansatz=transformers"], "ising", "transformers/0", "Adam"),
        (["system=hubbard", "ansatz=mps"], "hubbard", "mps", "Adam"),
        (["optim=lbfgs"], "fcidump", "mlp/u1u1", "LBFGS"),
    ],
)
def test_component_selectors(
    overrides: list[str],
    model: str,
    network: str,
    optimizer: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = preview(monkeypatch, capsys, TEMPLATE, *overrides)
    assert (job.model.name, job.network.name, job.optimizer.name) == (model, network, optimizer)
    if network == "mps":
        assert job.action.params.unique_seeds is False
    if optimizer == "LBFGS":
        assert "betas" not in job.optimizer.params


def test_geometry_preview_resolves_xyz_relative_to_yaml_and_omits_integral_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = tmp_path / "molecule.yaml"
    source.write_bytes(TEMPLATE.read_bytes())
    job = preview(monkeypatch, capsys, source, "system=geometry", "method=orbit", "geometry.xyz_file=data/H2.xyz")
    assert Path(job.model.params.xyz_file) == tmp_path / "data/H2.xyz"
    assert job.model.params.atoms is None
    assert job.action.params.src_fcidump is None
    assert job.model.params.ref_energy is None


@pytest.mark.parametrize(
    ("source", "overrides"),
    [(TEMPLATE, ["method=guide", "training.optimizer_steps=2"]), (TEMPLATE.parent / "examples/h2/input.yaml", [])],
)
def test_preview_matches_launched_input(
    source: Path,
    overrides: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    predicted = preview(monkeypatch, capsys, source, *overrides)
    runs: list[DictConfig] = []
    monkeypatch.setattr(cli, "_run", runs.append)
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", str(source), *overrides, "hydra.run.dir=run"])
    cli.main()
    assert runs == [predicted]
    assert Path(predicted.model.params.model_path).is_file()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["method=pretrain"], "files.dataset"),
        (["method=pert"], "common.parent_path"),
        (["ansatz=mps"], "system=hubbard or system=ising"),
    ],
)
def test_invalid_module_combinations_fail_before_running(
    overrides: list[str], message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", str(TEMPLATE), *overrides, "hydra.run.dir=run"])
    monkeypatch.setattr(cli, "_run", lambda config: pytest.fail("unchecked input reached the calculation"))
    with pytest.raises(ValueError) as exc:
        cli.main()
    assert message in str(exc.value)

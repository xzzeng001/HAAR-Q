"""Regression tests for selecting and running a calculation by its YAML file."""

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig

from haar_q import __main__ as cli
from haar_q import __version__


def test_explicit_input_composes_from_its_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = tmp_path / "test case"
    case.mkdir()
    (case / "base.yaml").write_text("common: {device: cpu}\n")
    (case / "H2.vmc.yaml").write_text(
        "defaults: [base, _self_]\nmodel: {params: {model_path: input.FCIDUMP}}\n"
        "hydra: {run: {dir: result}, job: {chdir: true}}\n"
    )
    runs: list[DictConfig] = []
    monkeypatch.setattr(cli, "_run", runs.append)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", "test case/H2.vmc.yaml", "common.device=cuda"])
    cli.main()
    (run,) = runs
    assert run.common.device == "cuda"
    assert Path(run.model.params.model_path) == case / "input.FCIDUMP"
    assert not list((tmp_path / "result").glob("*.log"))


@pytest.mark.parametrize("args", [[], ["--config-name", "config"]])
def test_legacy_launch_preserves_relative_paths(
    args: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("model: {params: {model_path: example.FCIDUMP}}\n")
    runs: list[DictConfig] = []
    monkeypatch.setattr(cli, "_run", runs.append)
    monkeypatch.setattr(sys, "argv", ["haar-q", *args])
    cli.main()
    assert runs[0].model.params.model_path == "example.FCIDUMP"


def test_preview_composes_without_starting_job(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "case.yaml").write_text("common: {device: cpu}\n")
    monkeypatch.setattr(sys, "argv", ["haar-q", "--input=case.yaml", "--cfg", "job", "common.device=cuda"])
    cli.main()
    assert "device: cuda" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()


def test_missing_input(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", str(missing)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert str(missing) in capsys.readouterr().err


@pytest.mark.parametrize("flag, expected", [("--version", f"HAAR-Q {__version__}"), ("--help", "--input")])
def test_information_without_config(
    flag: str, expected: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["haar-q", flag])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert expected in capsys.readouterr().out

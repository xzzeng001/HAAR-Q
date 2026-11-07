import logging
import pathlib
import warnings

from omegaconf import OmegaConf
import pytest

from haar_q.utility.reporting import calculation_report, CycleResult


def test_sci_cycles_and_resumed_history(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = OmegaConf.create({"model": {"name": "fcidump"}, "action": {"name": "haar"}})
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    for first_step in (1, 3):
        with pytest.raises(SystemExit), calculation_report(tmp_path, config, "cpu") as report:
            logging.info("Details of local optimization")
            warnings.warn("Diagnostic from network setup", UserWarning)
            report.checkpoint(first_step, CycleResult(2, 4, -1.25, -1.5))
            report.checkpoint(first_step + 1, CycleResult(4, 4, -1.375, -1.5))
            raise SystemExit(0)
    assert root.handlers == original_handlers
    stdout, stderr = capsys.readouterr()
    assert "Diagnostic from network setup" not in stdout + stderr
    assert "Details of local optimization" not in stdout
    assert stdout.count("Run completed") == 2
    detail = (tmp_path / "OUTCAR").read_text(encoding="utf-8")
    assert "Diagnostic from network setup" in detail and "Details of local optimization" in detail
    history = (tmp_path / "OSZICAR").read_text(encoding="utf-8")
    assert sum(line.endswith("E_RITZ") for line in history.splitlines()) == 2 and "Hartree" in history
    rows = [line.split() for line in history.splitlines() if not line.startswith("#")]
    assert [int(row[0]) for row in rows] == [1, 2, 3, 4]
    assert [row[4] for row in rows] == ["--", "-1.25000e-01", "--", "-1.25000e-01"]
    assert [float(row[5]) for row in rows] == [-1.5] * 4


def test_failed_run_retains_diagnostics(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = OmegaConf.create({"model": {"name": "ising"}, "action": {"name": "vmc"}})
    with pytest.raises(ValueError, match="test error"), calculation_report(tmp_path, config, "cpu"):
        raise ValueError("test error")
    assert "Traceback" in (tmp_path / "OUTCAR").read_text(encoding="utf-8")
    history = (tmp_path / "OSZICAR").read_text(encoding="utf-8")
    assert "Hamiltonian input units" in history and "Run failed" in history
    assert "E_RITZ" not in history
    assert "Run completed" not in capsys.readouterr().out

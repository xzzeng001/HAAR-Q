"""Compact calculation progress and a detailed run log."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import TextIO
import warnings

from omegaconf import DictConfig, OmegaConf

from ..version import __version__


@dataclass
class CycleResult:
    """Observables for one completed outer cycle, in the model's energy units."""

    seeds: int
    basis: int
    energy: float
    ritz: float | None = None
    roots: tuple[tuple[float, float], ...] = ()


class RunReport:
    def __init__(self, history: TextIO | None, roots_file: TextIO | None = None) -> None:
        self.history = history
        self.roots_file = roots_file
        self.previous_energy: float | None = None

    def line(self, text: str, *, comment: bool = True) -> None:
        print(text, flush=True)
        if self.history is not None:
            print(("# " if comment else "") + text, file=self.history, flush=True)
        logging.info("%s", text)

    def checkpoint(self, step: int, result: CycleResult | None) -> None:
        """Emit a row only after the corresponding checkpoint has been saved."""
        if result is None:
            self.line(f"Saved checkpoint for step {step}")
            return
        difference = "--" if self.previous_energy is None else f"{result.energy - self.previous_energy:+.5e}"
        row = f"{step:7d} {result.seeds:8d} {result.basis:8d} {result.energy:19.10f} {difference:>13}"
        if result.ritz is not None:
            row += f" {result.ritz:19.10f}"
        self.line(row, comment=False)
        self.previous_energy = result.energy
        if self.roots_file is not None and result.roots:
            ground = result.roots[0][0]
            for index, (energy, residual) in enumerate(result.roots):
                print(
                    f"{step:7d} {index:5d} {result.basis:8d} {energy:20.12f} {energy - ground:20.12f} {residual:12.4e}",
                    file=self.roots_file,
                    flush=True,
                )
                logging.info(
                    "SCI root %d: E=%.12f, gap=%.12f, projected residual=%.4e", index, energy, energy - ground, residual
                )


@contextmanager
def calculation_report(folder: Path, config: DictConfig, device: str) -> Iterator[RunReport]:
    """Scope files and Python logging to one invocation, including a resumed run."""
    folder.mkdir(parents=True, exist_ok=True)
    action = config.action.name
    is_sci = action in {"haar", "imag", "excited"}
    root = logging.getLogger()
    with ExitStack() as stack:
        history = None
        if is_sci or action == "vmc":
            history = stack.enter_context((folder / "OSZICAR").open("a", encoding="utf-8"))
        roots_file = None
        if action == "excited":
            roots_file = stack.enter_context((folder / "ROOTS").open("a", encoding="utf-8"))
            print("# CYCLE  ROOT  BASIS  TOTAL_ENERGY  GAP_FROM_ROOT_0  PROJECTED_RESIDUAL", file=roots_file)
        report = RunReport(history, roots_file)
        detail = logging.FileHandler(folder / "OUTCAR", mode="a", encoding="utf-8")
        stack.callback(detail.close)
        detail.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        old_handlers, old_level = root.handlers, root.level
        # Keep explicitly configured file and remote handlers; progress owns the terminal.
        root.handlers = [
            handler
            for handler in old_handlers
            if not isinstance(handler, logging.StreamHandler)
            or (isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) != folder / "OUTCAR")
        ] + [detail]
        root.setLevel(logging.INFO)
        try:
            stack.enter_context(warnings.catch_warnings())

            def record_warning(
                message: Warning | str,
                category: type[Warning],
                filename: str,
                lineno: int,
                file: TextIO | None = None,
                line: str | None = None,
            ) -> None:
                logging.warning("%s", warnings.formatwarning(message, category, filename, lineno, line).rstrip())

            warnings.showwarning = record_warning
            title = "HAAR-SCI" if is_sci else action.upper()
            if action == "excited":
                title = "HAAR-SCI (excited states)"
            if action == "vmc":
                guided = OmegaConf.select(config, "action.params.hamiltonian_guided", default=True)
                title += " (Hamiltonian-guided)" if guided else " (legacy estimator)"
            started = datetime.now().astimezone().isoformat(timespec="seconds")
            report.line(f"HAAR-Q {__version__}  |  {title}  |  {started}")
            network = OmegaConf.select(config, "network.name", default="--")
            report.line(f"Model {config.model.name}  |  Network {network}  |  Device {device}")
            report.line(f"Output directory  {folder}")
            if history is not None:
                unit = (
                    "Hartree (Eh)"
                    if config.model.name in {"fcidump", "geometry", "pyscf"}
                    else "Hamiltonian input units"
                )
                report.line(f"Energies in {unit}")
                if roots_file is not None:
                    print(
                        f"# Energies, gaps and residuals in {unit}; zero-based roots ordered within this cycle.",
                        file=roots_file,
                    )
                energy_note = (
                    "E is the fitted network energy in the selected basis. E_RITZ is the lowest Lanczos target."
                    if is_sci
                    else "E is the energy returned by the last optimizer step of this cycle."
                )
                if action == "excited":
                    energy_note = "E is the sampling-network energy. E_RITZ is the lowest selected CI energy. See ROOTS for excitation gaps."
                for note in (energy_note, "dE is E minus the preceding cycle's E in this invocation."):
                    print(f"# {note}", file=history)
                    logging.info("%s", note)
                heading = f"{'CYCLE':>7} {'SEED':>8} {'BASIS':>8} {'E':>19} {'dE':>13}"
                if is_sci:
                    heading += f" {'E_RITZ':>19}"
                report.line(heading)
            logging.info("Run configuration\n%s", OmegaConf.to_yaml(config, resolve=True))
            yield report
        except SystemExit as exc:
            if exc.code in (0, None):
                report.line("Run completed")
            else:
                report.line(f"Run failed (exit {exc.code})")
            raise
        except KeyboardInterrupt:
            report.line("Run interrupted")
            raise
        except Exception as exc:
            logging.exception("Calculation failed")
            report.line(f"Run failed ({type(exc).__name__}). See OUTCAR for details.")
            raise
        else:
            report.line("Run completed")
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)

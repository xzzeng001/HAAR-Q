# HAAR-Q changelog

## 0.2.0

0.1.0 → 0.2.0 (MINOR). The input interface adds compatible names for sampling
and optimization controls across the VMC and SCI workflows.

- Start a CPU H₂ calculation with the annotated [standard input](input.yaml).
  It includes model, network and action alternatives, their prerequisites,
  restart settings and optimizer controls.
- The [software overview](README.md#calculations-and-interfaces) lists the
  available workflows, compatible networks and Python interfaces.
- Pool pruning accepts both current HAAR-SCI checkpoints and legacy
  `imag` checkpoints.
- Optimizer options supplied as YAML lists, including Adam's `betas`, can
  be saved and restored through the weights-only checkpoint loader.

## 0.1.0

The previous checkout at `6d83bf5` followed the inherited `v0.0.58` tag.
This is the first release numbered independently for HAAR-Q. The increment
from the 0.0.58 baseline to 0.1.0 is MINOR because users can now select an
input file explicitly while existing Hydra invocations remain supported.

- Launch a calculation with `python -m haar_q -i input.yaml` or
  `haar-q -i input.yaml`.
- Follow VMC and HAAR-SCI runs in the terminal and inspect the energy history
  in `OSZICAR` and the detailed execution log in `OUTCAR`.
- Save numbered checkpoints and resume calculations on Windows-mounted drives
  where WSL cannot create symbolic links.
- Report the release version through the CLI, Python, package metadata and
  the calculation report. The [version policy](docs/versioning.md) describes
  how future deliveries are numbered and published.

# HAAR-Q changelog

## 0.1.0

The previous checkout at `6d83bf5` followed the inherited `v0.0.58` tag.
This is the first release numbered independently for HAAR-Q. The increment
from the 0.0.58 baseline to 0.1.0 is MINOR because users can now select an
input file explicitly while existing Hydra invocations remain supported.

- Launch a calculation with `python -m haar_q -i input.yaml` or
  `haar-q -i input.yaml`.
- Follow VMC and HAAR-SCI runs in the terminal and inspect the energy history
  in `OSZICAR` and the detailed execution log in `OUTCAR`.
- Report the release version through the CLI, Python, package metadata and
  the calculation report. The [version policy](docs/versioning.md) describes
  how future deliveries are numbered and published.

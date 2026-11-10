# HAAR-Q changelog

## 0.6.0

0.5.0 → 0.6.0 (MINOR). HAAR-Q gains a single-process, single-node multi-GPU
backend for selected-space Hamiltonian products and neural wavefunction fitting.

- Set `common.device: cuda` and `common.gpu_devices: [0, 1]` to share work
  across the two visible GPUs. The first GPU owns sampling and optimizer state.
- On CUDA, multi-root SCI uses device-resident block iteration above 128
  determinants. The projected residual and orthogonality checks also apply here.
- Hamiltonian-guided expansion now ranks candidates from all participating
  GPUs before deduplication. Native CUDA kernels tile large term lists.
- Fixed incorrect connected-determinant enumeration for multi-byte
  configurations on CPU and replaced unsafe CUDA trie traversal with tiled
  reduction. Perturbative corrections computed with affected versions can
  change when external determinants or their accumulated amplitudes differ.
- On supported closed-shell geometries, the two-electron density transformation
  and analytical force contractions can use all selected GPUs.
- The FCIDUMP model packs restricted spatial integrals directly into native
  normal-ordered terms, reducing host memory during preparation. Imported
  integrals use the existing OpenFermion coefficient tolerance.
- Added a [nitrogenase P-cluster example](examples/pcluster/input.yaml) for a
  73-orbital, 114-electron FCIDUMP and single-/two-GPU numerical checks.

## 0.5.0

0.4.0 → 0.5.0 (MINOR). The input gains a multi-state SCI workflow and
analytical nuclear forces for an isolated root of a fixed determinant space.

- Select `method=excited` and `states.roots` to reuse HAAR selection and
  train on the combined state probabilities. Each cycle reports energy
  ranks, excitation gaps and projected residuals in `ROOTS`.
- Select `method=forces` with a matching closed-shell PySCF geometry,
  a canonical SCI checkpoint from this version, and a zero-based `forces.root`.
  `FORCES` gives Cartesian forces in hartree/bohr, including orbital response
  and nuclear repulsion. See the [supported cases and H₂ example](README.md#excited-states-and-analytical-nuclear-forces).
- The shared Lanczos routine stops at the basis dimension and can continue
  in the orthogonal complement after a true breakdown. This can change
  Ritz values in older SCI calculations that exhausted their Krylov space.

## 0.4.0

0.3.0 → 0.4.0 (MINOR). Molecular calculations gain a coordinate-based input.

- Select `system=geometry` to supply Cartesian coordinates directly in YAML
  or read a single-frame XYZ file. Specify the orbital basis, units, total
  charge and spin in the same input.
- PySCF computes RHF or ROHF molecular orbitals and the one- and two-electron
  integrals for the HAAR-Q Hamiltonian. Reported total energies include
  nuclear repulsion. The default reference is the converged SCF total energy.
- VMC, HAAR-SCI and orbital rotation can use the generated integrals. See the
  [coordinate input example](README.md#molecular-coordinates-with-pyscf).

## 0.3.0

0.2.0 → 0.3.0 (MINOR). The standard input gains independent selectors for
calculation method, Hamiltonian, network architecture and optimizer.

- Edit one selector in [input.yaml](input.yaml) or override it for one run,
  for example with `python -m haar_q -i input.yaml method=haar`.
- Keep shared sampling and training controls alongside per-module settings.
  The program loads the selected modules with their English-annotated defaults
  and checks their compatibility before constructing the calculation.
- Use `--show-input` to print the effective calculation and data paths.
  Existing YAML files with explicit component sections remain supported.
- In `method=guide optim=lbfgs`, both training closures now clear gradients
  at each evaluation. This corrects the accumulated gradients that could
  affect the sampling network, wavefunction and their reported energies.

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

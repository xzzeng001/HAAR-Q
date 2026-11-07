# HAAR-Q

**Current author: xiongzhi zeng**

**Contact: [xzzeng@ustc.edu.cn](mailto:xzzeng@ustc.edu.cn)**

HAAR-Q is a [Python][python-url] toolkit for neural quantum states and
Hamiltonian-guided calculations in quantum chemistry and strongly correlated
many-body systems. It brings HAAR-SCI and Hamiltonian-guided variational Monte
Carlo (VMC) into one framework, with shared physical models, autoregressive
networks, and CPU/CUDA Hamiltonian operations.

## Research Background

Many-electron calculations face a rapidly growing configuration space. HAAR-Q
combines autoregressive neural-wavefunction sampling with Hamiltonian connectivity
to identify configurations for a tractable calculation. Its model interfaces
support molecular inputs such as FCIDUMP and [OpenFermion][openfermion-url] data,
as well as lattice models such as Hubbard and Ising systems.

The project builds on the HAAR-SCI research reported in the *Journal of Chemical
Theory and Computation*:

> **Hamiltonian-Guided Autoregressive Selected-Configuration Interaction Achieves Chemical Accuracy in Strongly Correlated Systems**
>
> DOI: [10.1021/acs.jctc.5c01415][haar-url]

The two main algorithm paths share the sampling and Hamiltonian infrastructure:

- **HAAR-SCI** (`method=haar`) combines network sampling and a previous
  configuration pool, expands the selected space using Hamiltonian connections,
  solves in that space with Lanczos, and trains the network for subsequent
  sampling.
- **Hamiltonian-guided VMC** (`method=vmc`) samples, expands, and deduplicates
  configurations, then directly optimizes the neural wavefunction's energy in
  the selected space. It does not use Lanczos or another eigensolver. This is a
  subsequent development of the toolkit; the cited paper is the reference for
  HAAR-SCI, not validation of this newer VMC implementation.

**HAAR-Q** is the umbrella project name. **HAAR-SCI** remains the name of the
published algorithm and is retained in scientific descriptions and citations.

## Calculations and interfaces

The [standard input](input.yaml) runs molecular VMC as supplied. Set `method`
to choose a calculation. Each method loads its own parameters and uses the
model, network and optimizer it needs.

| Action | What it computes | Starting data or additional input |
| --- | --- | --- |
| `vmc` | Optimizes the neural wavefunction on sampled and Hamiltonian-connected configurations. The legacy sampled estimator is also available. | A model, a sampling-capable network and an optimizer |
| `haar` | Expands a selected CI space, computes Ritz states with Lanczos, and fits the network for the next cycle. | A model and a network with unique sampling; the checkpoint stores the pool and requested Ritz states |
| `guide` | Trains a separate sampling network from local-energy fluctuations alongside the wavefunction. | The method creates a second network using the selected ansatz. |
| `pretrain` | Fits the network to known wavefunction amplitudes. | A Torch file containing packed configurations and their matching amplitudes as a pair of tensors |
| `orbit` | Diagonalizes the spatial one-particle density matrix and writes a Hamiltonian in the natural-orbital basis. | A HAAR-SCI pool and its matching FCIDUMP |
| `haar_with_orbit` | Alternates orbital rotation with selected-space Lanczos and neural fitting. | A matching molecular FCIDUMP; each cycle saves an optimized-basis file |
| `pert` | Estimates a second-order contribution from a single batch of external configurations connected to a saved pool. | A HAAR-SCI pool; the logged second-order contribution is an energy correction |
| `chop_imag` | Ranks and removes configurations in batches and records energy against retained pool size. | A current HAAR-SCI or legacy `imag` pool in `common.parent_path` |

The saved SCI pool carries the square root of the summed probabilities of the
requested Ritz states. It supplies the amplitudes used by pool post-processing.
For signed or complex Ritz vectors, read the per-cycle entries in `haar.excited`
from the checkpoint.

The native CPU and CUDA kernels apply the Hamiltonian between selected
configurations, select or enumerate connected configurations and evaluate
diagonal elements. The following choices are available for the sampling network.

| Model input | Compatible network selectors | Physical constraint |
| --- | --- | --- |
| Molecular `fcidump`, `openfermion`, `optimized_basis` | `mlp/u1u1`, `transformers/u1u1`, `mlp/u1`, `transformers/u1` | `u1u1` fixes the two spin populations separately; `u1` fixes the total electron count. The OpenFermion spin-resolved implementation assumes equal spin populations. |
| Open-boundary `hubbard` lattice | The four molecular selectors, plus `mps` | Hopping, on-site interaction and chemical potential; the MPS samples without an electron-number constraint. |
| Open-boundary `ising` spin lattice | `mlp/0`, `transformers/0`, `mlp/u1`, `transformers/u1`, `mps` | `/0` allows all spin configurations. The `/u1` options select the fixed half-filled sector. Choose a sector preserved by your Hamiltonian. |

The `system` selector supplies the default symmetry and `ansatz` supplies
the sampling mode. For example, `system=ising ansatz=transformers` selects
`transformers/0`, while `system=hubbard ansatz=mps` selects an MPS with
replacement sampling. Both SCI methods require a network with unique generation.

**Python interfaces.** `haar_q.plugins.pyscf.Solver.kernel` accepts one- and
two-electron integral arrays and electron counts for a HAAR-SCI calculation.
It accepts a resolved configuration with explicit component sections, as
printed by `python -m haar_q -i input.yaml method=haar --show-input`.
It can return a PySCF-shaped CI array with warm-start data. The underlying
`pyscf` model receives NumPy arrays through Python. For a molecular YAML run,
use an FCIDUMP, an OpenFermion MolecularData file, or a saved optimized-basis
file. `haar_q.plugins.hamiltonians` exposes wavefunction text I/O and the
Hamiltonian operations for standalone analysis.

## Authorship and Provenance

The current author and maintainer is **xiongzhi zeng**
([xzzeng001](https://github.com/xzzeng001)). For questions, bug reports, or
collaboration, please contact [xzzeng@ustc.edu.cn](mailto:xzzeng@ustc.edu.cn).

The project repository is [xzzeng001/HAAR-Q](https://github.com/xzzeng001/HAAR-Q).

HAAR-Q continues the original project developed collaboratively by
[Hao Zhang (hzhangxyz)](https://github.com/hzhangxyz) and
[xzzeng001](https://github.com/xzzeng001). Hao Zhang was the principal original
developer. The historical source is
[USTC-KnowledgeComputingLab/qmp-kit](https://github.com/USTC-KnowledgeComputingLab/qmp-kit).
Original author credit, contributor history, and the GPL license are preserved
as development continues independently under the HAAR-Q name.

When reporting results based on HAAR-SCI, cite the paper above and record the
HAAR-Q commit or version and calculation configuration used.

## Naming and Migration

| Component | Name |
|-----------|------|
| Project | **HAAR-Q** |
| Python distribution | `haar-q` |
| Command-line entry point | `haar-q` |
| Python import package | `haar_q` |
| Module entry point | `python -m haar_q` |

Update downstream imports, launch scripts, and environment installations to
these names. Algorithm selectors such as `haar` and `vmc` remain unchanged.
Existing tensor/state-dictionary checkpoints keep their format; custom serialized
Python objects may require migration if they embed earlier module paths.

## Getting Started

### Installation

HAAR-Q requires Python >= 3.12. Hamiltonian operations compile native extensions
at runtime and need a compatible C++20 toolchain; GPU execution also requires
a compatible PyTorch/CUDA setup. CPU and CUDA implementations are included.

Install from the root of your local HAAR-Q checkout:

```bash
python -m pip install -e .
```

These instructions use the local source tree and do not assume that `haar-q`
has been published to a package index. Installation creates the `haar-q`
command-line launcher in the active Python environment (`haar-q.exe` on Windows).

### Usage

Run the [annotated standard input](input.yaml) from the project root. Edit it
to test parameters, or copy it beside your own data and update the relative
paths. Wheel installations also include the input and its H₂ integrals under
`share/haar-q` in the Python installation prefix.

```bash
python -m haar_q -i input.yaml
```

This runs two Hamiltonian-guided VMC cycles. Each cycle starts with two sampled
configurations, targets four after expansion, and performs ten Adam updates.
The reference energy for the bundled integrals is −1.427525196849 hartree,
including the constant term. A short run is sufficient to inspect the workflow.

```bash
python -m haar_q -i input.yaml method=haar                              # Run H2 HAAR-SCI
python -m haar_q -i input.yaml system=ising ansatz=transformers         # Run spin-lattice VMC
python -m haar_q -i input.yaml method=haar training.optimizer_steps=3 --show-input
python -m haar_q --version
```

`haar-q -i input.yaml` uses the same entry point. Paths to the FCIDUMP in
explicit inputs are relative to the YAML file, so you can launch from another
directory, for example with `python -m haar_q -i examples/h2/input.yaml`.
Existing jobs using a working-directory `config.yaml` or Hydra's
`--config-name` option still work.

#### Running from Windows PowerShell with WSL

If HAAR-Q is installed in the Ubuntu virtual environment `~/.venvs/haar-q`,
open PowerShell in the checkout. The following function selects that Linux
Python for the current PowerShell session. WSL inherits the working directory
when the checkout is on a mounted Windows drive.

```powershell
$HaarPython = (wsl.exe -d Ubuntu --exec printenv HOME).Trim() + '/.venvs/haar-q/bin/python'
function python { wsl.exe -d Ubuntu --exec $HaarPython @args }
python -m haar_q -i input.yaml
```

### Reading a calculation

The terminal prints the output directory and one progress row per completed
outer cycle. The bundled examples write under `outputs/vmc/` or `outputs/haar/`
in the launch directory. Each invocation uses its own timestamped directory.

| Output | Contents |
| --- | --- |
| `OSZICAR` | Compact convergence history for VMC and HAAR-SCI. Use this file to follow cycle energies and configuration counts. |
| `OUTCAR` | Human-readable run information, settings, detailed iteration logs and diagnostics. |
| `data.pth` | Latest checkpoint with the network, optimizer, and progress. |
| `events.out.tfevents.*` | TensorBoard metrics, including VMC energies for individual local optimization steps. |

For guided VMC, the cycle energy is the last energy returned by the local
optimizer on that cycle's expanded basis. With Adam, this is evaluated just
before the last update. For HAAR-SCI, the fitted network energy and the lowest
Lanczos target energy are reported separately. The energy change compares
successive cycles within the same invocation. Both molecular examples use
hartree. The two text files follow the short-history and detailed-report
conventions of [VASP OSZICAR](https://vasp.at/wiki/OSZICAR) and
[VASP OUTCAR](https://vasp.at/wiki/OUTCAR).

### Editing the standard input

Edit the four selectors at the top of [input.yaml](input.yaml). For example,
change `- method: vmc` to `- method: haar` and rerun the same command for
HAAR-SCI on the same Hamiltonian. To change a choice for one run, append
`method=haar` to the command. Each selector loads a packaged YAML with
English comments for its parameters.

| Selector | Choices | Parameter reference |
| --- | --- | --- |
| `method` | `vmc`, `haar`, `guide`, `pretrain`, `pert`, `chop_imag`, `orbit`, `haar_with_orbit` | [Calculation methods](haar_q/conf/method) |
| `system` | `fcidump`, `hubbard`, `ising`, `openfermion`, `optimized_basis` | [Hamiltonians and lattice couplings](haar_q/conf/system) |
| `ansatz` | `mlp`, `transformers`, `mps` | [Network architectures](haar_q/conf/ansatz) |
| `optim` | `adam`, `lbfgs` | [Optimizers](haar_q/conf/optim) |

Use `common` for runtime settings and `files` for data paths. Set
`sampling_control.seed_count` once for the VMC, guide and SCI methods.
`sampling_control.target_basis_size` applies to VMC and guide. SCI controls
expansion separately through its Krylov parameters. `training.optimizer_steps`
controls network updates for VMC, guide and both SCI methods.
`training.learning_rate` sets the initial step size for either optimizer.

Use `options.<group>.<choice>` for a setting specific to one module. The
following entries can coexist under the same `options` key in your input.
Switching methods keeps each method's settings available for the next run.

```yaml
options:
  method:
    vmc: {hamiltonian_guided: true}
    haar: {krylov_iteration: 1}
  system:
    hubbard: {m: 4, n: 2, u: 8.0}
  ansatz:
    mlp: {hidden: [32, 16]}
```

A value set for a specific module takes precedence over its default, including
a default inherited from a shared setting. For example, setting
`options.method.haar.network_seed_count` changes only the SCI network sample
count. The previous-pool sample count still follows `sampling_control.seed_count`.
Append `--show-input` to print the selected calculation, resolved settings
and data paths without starting a run. Pool post-processing loads the model
and checkpoint and omits the network and optimizer.

For example, run a short SCI calculation and use its pool for a second-order
correction from the project root. Choose a fresh output directory for the
first run. Both invocations read the same input.

```powershell
python -m haar_q -i input.yaml method=haar hydra.run.dir=outputs/sci-example
python -m haar_q -i input.yaml method=pert common.parent_path=outputs/sci-example/data.pth
```

For supervised training, select `method=pretrain` and supply `files.dataset`.
For natural orbitals, select `method=orbit`, supply the SCI checkpoint and
matching `files.fcidump`, and choose a destination in `files.orbital_output`.
To use the rotated Hamiltonian, select `system=optimized_basis` and set
`files.optimized_basis` to that output file.

Existing explicit YAML jobs with `action`, `model`, `network` and `optimizer`
sections still run, including the [compact SCI example](examples/h2/sci.yaml).
Use the selectors with the new standard input. Use existing parameter names
when overriding keys in older files.

#### Hamiltonian-guided VMC

`method=vmc` defaults to Hamiltonian-guided optimization on an
expanded configuration space, without Lanczos or any eigensolver:

1. Sample configurations from the current neural wavefunction.
2. Use `model.find_relative` to select Hamiltonian-connected configurations.
3. Merge and deduplicate the samples and selected neighbors into a fixed set `S`.
4. Optimize the projected Rayleigh quotient
   `E_S = real(psi(S)^dagger H_SS psi(S)) / (psi(S)^dagger psi(S))`.
   Every configuration in `S`, including newly selected neighbors, participates
   in the gradient. Both amplitudes and `H_SS psi(S)` are recomputed each step.
5. Save the checkpoint, then resample and expand for the next outer cycle.

The native Hamiltonian operation runs without autograd. A detached-residual
surrogate supplies the full Rayleigh-quotient gradient for a Hermitian
Hamiltonian, including normalization and the factor of two.

Set `sampling_control.target_basis_size` to the target **total** number of
configurations. Expansion makes one `find_relative`
call per outer cycle. Fewer states may be returned after selection and
deduplication. If this target is no larger than the sampled set, expansion
is skipped and all sampled states are retained. The set remains fixed during
the `optimizer_steps` optimizer updates. Use `common.max_relative_step` or
`common.max_absolute_step` to limit the number of outer cycles.

The standard H₂ input requests two seeds and an expanded basis of four.
The expanded set participates in network backpropagation. Size larger runs
according to the available memory.

With `hamiltonian_guided: true`, `unique_seeds: false` changes how seed states are
sampled, but sample multiplicities are not used as energy weights: each state
in `S` occurs once in the projected basis. The reported energy is that of
the neural wavefunction projected onto `S`, not a Monte Carlo estimate of
the untruncated neural wavefunction's energy. Expanding the set does not by
itself guarantee an energy decrease before optimization.

To reproduce the original sampled VMC objective, set
`options.method.vmc.hamiltonian_guided=false`. That mode uses only the sampled states on the
left side of the energy estimator, includes neighbors on the Hamiltonian
application side, and retains the original sampling weights and gradient.
Existing checkpoints keep their format, but resuming with the new default
changes the optimization objective. The HAAR-SCI algorithm is unchanged.

TensorBoard also records `vmc/configs/sampled` and `vmc/configs/expanded`.
Focused tests (no native Hamiltonian compilation required) can be run with:

```bash
python -m pytest tests/vmc_test.py
```

## Contributing

See the [contribution guide](CONTRIBUTING.md) for development and validation.
The [version policy](docs/versioning.md) defines how each delivery is numbered
and pushed to GitHub. The [changelog](CHANGELOG.md) records the changes in each release.

## License

HAAR-Q is distributed under **GPL-3.0-or-later**. See [LICENSE.md](LICENSE.md)
for the GPLv3 license text. The project rename does not change the license or
remove original author and contributor attribution.

[python-url]: https://www.python.org/
[openfermion-url]: https://quantumai.google/openfermion
[cuda-url]: https://docs.nvidia.com/cuda/
[anaconda-url]: https://www.anaconda.com/
[miniconda-url]: https://docs.anaconda.com/miniconda/
[venv-url]: https://docs.python.org/3/library/venv.html
[pyenv-url]: https://github.com/pyenv/pyenv
[pytorch-install-url]: https://pytorch.org/get-started/locally/
[hydra-url]: https://hydra.cc/
[haar-url]: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415

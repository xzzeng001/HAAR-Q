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

- **HAAR-SCI** (`action.name: haar`) combines network sampling and a previous
  configuration pool, expands the selected space using Hamiltonian connections,
  solves in that space with Lanczos, and trains the network for subsequent
  sampling.
- **Hamiltonian-guided VMC** (`action.name: vmc`) samples, expands, and deduplicates
  configurations, then directly optimizes the neural wavefunction's energy in
  the selected space. It does not use Lanczos or another eigensolver. This is a
  subsequent development of the toolkit; the cited paper is the reference for
  HAAR-SCI, not validation of this newer VMC implementation.

**HAAR-Q** is the umbrella project name. **HAAR-SCI** remains the name of the
published algorithm and is retained in scientific descriptions and citations.

## Calculations and interfaces

The [standard input](input.yaml) runs molecular VMC as supplied. Its English
comments cover the controls for each workflow. Choose an action according to
the input data and the result you need.

| Action | What it computes | Starting data or additional input |
| --- | --- | --- |
| `vmc` | Optimizes the neural wavefunction on sampled and Hamiltonian-connected configurations. The legacy sampled estimator is also available. | A model, a sampling-capable network and an optimizer |
| `haar` | Expands a selected CI space, computes Ritz states with Lanczos, and fits the network for the next cycle. | A model and a network with unique sampling; the checkpoint stores the pool and requested Ritz states |
| `guide` | Trains a separate sampling network from local-energy fluctuations alongside the wavefunction. | A second top-level `sampling` network configuration |
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

The lattice MPS supports sampling with replacement. For VMC set
`unique_seeds: false`. HAAR-SCI requires unique generation. Transformer
dimensions and sector-specific settings are listed in the standard input.

**Python interfaces.** `haar_q.plugins.pyscf.Solver.kernel` accepts one- and
two-electron integral arrays and electron counts for a HAAR-SCI calculation.
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
python -m haar_q -i examples/h2/sci.yaml                  # Run the compact HAAR-SCI example
python -m haar_q -i input.yaml common.max_relative_step=20 # Change the number of VMC cycles
python -m haar_q -i input.yaml --cfg job                  # Inspect the configuration
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

The active `common`, `model`, `network`, `optimizer` and `action` sections
form one H₂ job. The values are deliberately small. The `hydra.run.dir`
setting gives each invocation its own output directory. To switch workflows,
replace the whole relevant section with one of the commented alternatives
and supply any checkpoint, integral file or dataset named there. This keeps
parameters for one algorithm out of another algorithm's input.

| If you want to change… | Edit |
| --- | --- |
| Hamiltonian, molecule, lattice size or couplings | `model.name` and its `model.params` |
| Network family, symmetry, width or depth | `network.name` and its `network.params` |
| Device, precision, seed, number of cycles or restart | `common`; set `random_seed: null` to restore the checkpoint's RNG state |
| Optimizer or learning rate | `optimizer.name` and the keyword arguments in `optimizer.params` for that PyTorch optimizer |
| Sample size, expansion, Lanczos or network updates | The selected `action.params` block |

For a one-off test, override an active key on the command line, for example
`python -m haar_q -i input.yaml action.params.optimizer_steps=3`.
`--cfg job` previews the composed YAML. Previous inputs may keep their
original parameter names. Use one spelling per control and use the spelling
in your file for a Hydra override.

| Workflow | Preferred input key | Original key |
| --- | --- | --- |
| VMC, guide | `seed_count` | `sampling_count` |
| VMC, guide | `target_basis_size` | `relative_count` |
| VMC, guide | `unique_seeds` | `unique` |
| VMC, guide and both SCI actions | `optimizer_steps` | `local_step` |
| Guide | `sampler_steps_per_update` | `dist_step` |
| Both SCI actions | `network_seed_count` | `sampling_count_from_neural_network` |
| Both SCI actions | `previous_seed_count` | `sampling_count_from_last_iteration` |

#### Hamiltonian-guided VMC

`action.name: vmc` now defaults to Hamiltonian-guided optimization on an
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

For example, replace the `action` section of your existing configuration with:

```yaml
action:
  name: vmc
  params:
    hamiltonian_guided: true
    seed_count: 1024
    target_basis_size: 4096
    optimizer_steps: 1000
    unique_seeds: true
```

`target_basis_size` is the target **total** number of configurations, not the
number of neighbors per sampled state. Expansion makes one `find_relative`
call per outer cycle. Fewer states may be returned after selection and
deduplication. If this target is no larger than the sampled set, expansion
is skipped and all sampled states are retained. The set remains fixed during
the `optimizer_steps` optimizer updates. Use `common.max_relative_step` or
`common.max_absolute_step` to limit the number of outer cycles.

The example uses smaller sets than the defaults (4000 sampled, 40000 total).
The expanded set now participates in network backpropagation, increasing
memory use compared with the legacy objective; scale these counts to fit
your system.

With `hamiltonian_guided: true`, `unique_seeds: false` changes how seed states are
sampled, but sample multiplicities are not used as energy weights: each state
in `S` occurs once in the projected basis. The reported energy is that of
the neural wavefunction projected onto `S`, not a Monte Carlo estimate of
the untruncated neural wavefunction's energy. Expanding the set does not by
itself guarantee an energy decrease before optimization.

**Compatibility:** to reproduce the previous VMC objective, explicitly set
`hamiltonian_guided: false`. That mode uses only the sampled states on the
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

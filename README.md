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

Pass a YAML input file to the Python module or the installed command.
For a first CPU calculation, run the two-orbital H₂ example from the project root.

```bash
cd examples/h2
python -m haar_q -i input.yaml
```

This runs two Hamiltonian-guided VMC cycles. Each cycle starts with two sampled
configurations, targets four after expansion, and performs ten Adam updates.
The reference energy for the bundled integrals is −1.427525196849 hartree,
including the constant term. A short run is sufficient to inspect the workflow.

```bash
python -m haar_q -i sci.yaml                              # Run HAAR-SCI
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
Set-Location examples/h2
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

#### Configuration Structure

A typical `config.yaml` includes several sections: `common`, `model`, `network`, `optimizer`, and `action`.

```yaml
# Example config.yaml
model:
  name: fcidump
  params:
    model_path: /path/to/molecule.fcidump
    ref_energy: -7.44606892

network:
  name: mlp/u1u1
  params:
    hidden: [512, 512]

optimizer:
  name: Adam
  params:
    lr: 0.001

common:
  random_seed: 2333
  device: cuda
  dtype: bfloat16

action:
  name: haar
  params:
    sampling_count_from_neural_network: 1024
    krylov_iteration: 32
```

#### Configuration Options

##### Common Settings (`common`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | Computing device (`cuda`, `cpu`, `cuda:0`, etc.) | `cuda:0` |
| `dtype` | Data type (`bfloat16`, `float16`, `float32`, `float64`) | `None` |
| `random_seed` | Manual random seed for reproducibility | `None` |
| `checkpoint_interval` | Numbered checkpoint snapshot interval (outer steps) | `5` |
| `parent_path` | Checkpoint file to resume from | `None` |
| `max_absolute_step` | Maximum absolute step for the process | `None` |
| `max_relative_step` | Maximum relative step for the process | `None` |

##### Model Settings (`model`)
- **fcidump**: Interface for FCIDUMP files.
  - `model_path`: Path to the FCIDUMP file.
  - `ref_energy`: Reference energy (optional).
- **hubbard**: 2D Hubbard model.
  - `m`, `n`: Lattice dimensions.
  - `t`, `u`: Model coefficients.
  - `electron_number`: Number of electrons.
- **ising**: 2D Ising-like model on a lattice.
  - `m`, `n`: Lattice dimensions.
  - `x`, `y`, `z`: Coefficients for the transverse and longitudinal fields.
  - `xh`, `yh`, `zh`: Coefficients for horizontal bond interactions (XX, YY, ZZ).
  - `xv`, `yv`, `zv`: Coefficients for vertical bond interactions.
  - `xd`, `yd`, `zd`: Coefficients for diagonal bond interactions.
  - `xa`, `ya`, `za`: Coefficients for anti-diagonal bond interactions.

##### Network Settings (`network`)
The available network architectures and their naming conventions depend on the chosen model, as they implement specific symmetry and conservation laws. **Explicit naming is recommended** to clarify the constraints being used (e.g., total electron number `u1` or spin-resolved `u1u1`).

For models like **fcidump** and **hubbard**, recommended options include:
- **mlp/u1u1**: Multi-Layer Perceptron with $U(1) \times U(1)$ symmetry (conserves spin-up and spin-down electrons separately).
- **mlp/u1**: MLP with $U(1)$ symmetry (conserves total electron number).
- **transformers/u1u1**: Transformer architecture with $U(1) \times U(1)$ symmetry.
- **transformers/u1**: Transformer architecture with $U(1)$ symmetry.

Parameters for these networks:
- **mlp** variants:
  - `hidden`: Tuple of hidden layer widths (e.g., `[512, 512]`).
- **transformers** variants:
  - `embedding_dim`, `heads_num`, `depth`, etc.

##### Action Settings (`action`)
The `action` section determines the algorithm to run. The primary algorithms are **haar** and **vmc**. Each action has its own set of parameters defined in its corresponding configuration class.

Example for `haar` action:
```yaml
action:
  name: haar
  params:
    sampling_count_from_neural_network: 1024
    krylov_iteration: 32
    # ... other haar specific parameters
```

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
    sampling_count: 1024
    relative_count: 4096
    local_step: 1000
    unique: true
```

`relative_count` is the target **total** number of configurations, not the
number of neighbors per sampled state. Expansion makes one `find_relative`
call per outer cycle. Fewer states may be returned after selection and
deduplication. If this target is no larger than the sampled set, expansion
is skipped and all sampled states are retained. The set remains fixed during
the `local_step` optimizer updates. Use `common.max_relative_step` or
`common.max_absolute_step` to limit the number of outer cycles.

The example uses smaller sets than the defaults (4000 sampled, 40000 total).
The expanded set now participates in network backpropagation, increasing
memory use compared with the legacy objective; scale these counts to fit
your system.

With `hamiltonian_guided: true`, `unique: false` changes how seed states are
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

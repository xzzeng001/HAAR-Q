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
support molecular coordinates, FCIDUMP and [OpenFermion][openfermion-url] data,
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
| `excited` | Solves for the lowest states on each HAAR-selected space and fits the network to their combined probability. | Set `states.roots`, counting the ground state. The `ROOTS` file lists total energies, excitation gaps and projected residuals. |
| `forces` | Computes the analytical nuclear force on a chosen fixed-space CI energy surface. | A matching closed-shell geometry and a canonical `haar` or `excited` checkpoint from v0.5.0 or later. |
| `guide` | Trains a separate sampling network from local-energy fluctuations alongside the wavefunction. | The method creates a second network using the selected ansatz. |
| `pretrain` | Fits the network to known wavefunction amplitudes. | A Torch file containing packed configurations and their matching amplitudes as a pair of tensors |
| `orbit` | Diagonalizes the spatial one-particle density matrix and writes a Hamiltonian in the natural-orbital basis. | A HAAR-SCI pool and the corresponding geometry or matching FCIDUMP |
| `haar_with_orbit` | Alternates orbital rotation with selected-space Lanczos and neural fitting. | A molecular geometry or matching FCIDUMP. Each cycle saves an optimized-basis file. |
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
| Molecular `fcidump`, `geometry`, `openfermion`, `optimized_basis` | `mlp/u1u1`, `transformers/u1u1`, `mlp/u1`, `transformers/u1` | `u1u1` fixes the two spin populations separately; `u1` fixes the total electron count. The OpenFermion spin-resolved implementation assumes equal spin populations. |
| Open-boundary `hubbard` lattice | The four molecular selectors, plus `mps` | Hopping, on-site interaction and chemical potential; the MPS samples without an electron-number constraint. |
| Open-boundary `ising` spin lattice | `mlp/0`, `transformers/0`, `mlp/u1`, `transformers/u1`, `mps` | `/0` allows all spin configurations. The `/u1` options select the fixed half-filled sector. Choose a sector preserved by your Hamiltonian. |

The `system` selector supplies the default symmetry and `ansatz` supplies
the sampling mode. For example, `system=ising ansatz=transformers` selects
`transformers/0`, while `system=hubbard ansatz=mps` selects an MPS with
replacement sampling. The SCI sampling methods require a network with unique generation.

**Python interfaces.** `haar_q.plugins.pyscf.Solver.kernel` accepts one- and
two-electron integral arrays and electron counts for a HAAR-SCI calculation.
It accepts a resolved configuration with explicit component sections, as
printed by `python -m haar_q -i input.yaml method=haar --show-input`.
It can return a PySCF-shaped CI array with warm-start data. The underlying
`pyscf` model receives NumPy arrays through Python. For a molecular YAML run,
select `system=geometry` to compute integrals from coordinates, or load an
FCIDUMP, OpenFermion MolecularData or optimized-basis file.
`haar_q.plugins.hamiltonians` exposes wavefunction text I/O and the
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

#### Single-node CUDA and the nitrogenase P-cluster

Use one HAAR-Q process per job. The ordinals in `common.gpu_devices` are relative to the
devices visible to that process. The first GPU owns the network parameters, optimizer,
selected basis and checkpoints. An empty list uses `common.device`. For example, from
an environment with a CUDA build of PyTorch, a matching CUDA toolkit (`nvcc`) and a C++20 compiler:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
nvcc --version
python -m haar_q -i input.yaml common.device=cuda:0
python -m haar_q -i input.yaml common.device=cuda 'common.gpu_devices=[0,1]'
```

| Stage | Where it runs |
| --- | --- |
| Autoregressive generation and unique sampling | Primary GPU, preserving one sampling distribution and one RNG stream for generation |
| Hamiltonian-guided expansion | GPUs search disjoint source determinants. The primary ranks candidate paths globally and deduplicates the selected determinants |
| Projected Hamiltonian products | GPUs process disjoint source determinants and sum their contributions to each target on the primary |
| Network amplitudes and fitting | MLP, Transformer and MPS split determinant batches across GPUs. The primary combines gradients before one optimizer update |
| Excited-state SCI | Small selected spaces use dense diagonalization on the primary. Above 128 determinants the CUDA solver keeps its block Krylov vectors on the primary and distributes its Hamiltonian products |
| Analytical forces on supported geometry inputs | GPUs transform two-body density slabs and contract two-electron integral derivatives. PySCF constructs the RDM, derivative integrals and orbital response on the CPU |
| Molecular preparation and ground-state Lanczos | FCIDUMP parsing and PySCF SCF/integrals use the CPU. Ground-state HAAR also uses it for older Lanczos vectors and its small tridiagonal solve |

Each participating GPU must hold the Hamiltonian terms and full target determinant list.
Network replicas, temporary tensors and the primary solver workspace need additional memory.
The matvec cost still grows with the product of the term count and the selected source count.
Adding cards distributes work within one node. Multi-node MPI/DDP launches are unsupported.

The [P-cluster input](examples/pcluster/input.yaml) is a one-cycle Transformer HAAR-SCI
smoke calculation. For the supplied nitrogenase [8Fe–7S] integral file, `NORB=73`,
`NELEC=114` and `MS2=0` select 57 alpha and 57 beta electrons. The determinant
uses 19 packed bytes. Packed Hamiltonian terms alone occupy about 1.08 GiB per GPU.
The value of `MS2` fixes a spin projection. Assigning a total spin or identifying an
oxidation state requires additional information. The FCIDUMP supplies energies and
electronic integrals. Analytical nuclear forces require integral derivatives and
the supported closed-shell geometry workflow, which has a 32-spatial-orbital limit.

For this file, a CPU one-cycle HAAR-SCI run expanded four network seeds to
12 determinants, and an expanded-space VMC run used eight determinants.
A standalone CUDA 12.6 kernel check streamed all 41,528,584 operator terms
through a GTX 1650 for a two-determinant projected matrix and its diagonal.
The maximum difference against the native CPU result was below
1.5e-11 hartree. Full two-GPU Python runs and P-cluster scaling have yet to
be measured on a multi-GPU node.

From the checkout on a Slurm cluster, set the path to the copy visible on the node:

```bash
srun --nodes=1 --ntasks=1 --gpus-per-node=2 python -m haar_q -i examples/pcluster/input.yaml files.fcidump=/path/to/models/FCIDUMP
```

In PowerShell with the WSL Python function below, use the WSL path to the file:

```powershell
$fcidump = '/mnt/j/paper/HAAR-SCI/data2/Pcluster/models/FCIDUMP'
python -m haar_q -i examples/pcluster/input.yaml "files.fcidump=$fcidump" --show-input
python -m haar_q -i examples/pcluster/input.yaml "files.fcidump=$fcidump"
```

Set `method=vmc` for the expanded-space energy objective or `method=excited states.roots=2`
for the two lowest selected-space energies. To run on one GPU, pass `'common.gpu_devices=[0]'`.
The example's four seeds and two Krylov steps exercise the calculation path. To study
energy convergence, increase the seed count, expansion count, selected-space size and
cycle count, and compare energies between runs. For excited states, inspect both the
energy gaps and the projected residuals in `ROOTS`. The repository does not distribute
the P-cluster integrals.

On a two-GPU node with pytest installed, run the single-/two-GPU numerical checks
from the checkout:

```bash
python -m pytest -q tests/parallel_hamiltonian_test.py tests/parallel_wavefunction_test.py tests/parallel_forces_test.py tests/selected_states_test.py tests/gpu_workflow_test.py
```

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
| `ROOTS` | In an `excited` run, one row per cycle and state with energy rank, basis size, total energy, gap above root 0 and projected residual norm. |
| `FORCES` | In a `forces` run, the selected root's total energy and one Cartesian force vector per atom, in input atom order and hartree/bohr. |
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
| `method` | `vmc`, `haar`, `excited`, `forces`, `guide`, `pretrain`, `pert`, `chop_imag`, `orbit`, `haar_with_orbit` | [Calculation methods](haar_q/conf/method) |
| `system` | `fcidump`, `geometry`, `hubbard`, `ising`, `openfermion`, `optimized_basis` | [Hamiltonians and lattice couplings](haar_q/conf/system) |
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
its corresponding geometry or matching `files.fcidump`, and choose a
destination in `files.orbital_output`.
To use the rotated Hamiltonian, select `system=optimized_basis` and set
`files.optimized_basis` to that output file.

Existing explicit YAML jobs with `action`, `model`, `network` and `optimizer`
sections still run, including the [compact SCI example](examples/h2/sci.yaml).
Use the selectors with the new standard input. Use existing parameter names
when overriding keys in older files.

#### Molecular coordinates with PySCF

Choose `- system: geometry` in the standard input to compute molecular
integrals directly from Cartesian coordinates. The supplied geometry is
neutral H₂ at 0.74 Å in STO-3G. Edit the `geometry` section for your molecule.

```yaml
geometry:
  atoms: |
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
  xyz_file: null
  unit: Angstrom
  basis: sto-3g
  charge: 0
  spin: 0
  ref_energy: null
```

The same coordinates can be used for either calculation from PowerShell.

```powershell
python -m haar_q -i input.yaml system=geometry method=vmc
python -m haar_q -i input.yaml system=geometry method=haar
```

For an XYZ file, set `geometry.xyz_file` to its path relative to the YAML
file, for example `geometry.xyz_file=molecules/H2.xyz`. The file supplies
one frame with an atom count, a comment line (which may be blank), and one
element and three coordinates per atom. The file takes precedence over
`geometry.atoms`. Set `geometry.unit` to `Angstrom` or `Bohr` for either
input. `geometry.charge` is the total charge and `geometry.spin` is
`N_alpha - N_beta` (0 for a singlet, 1 for a doublet, 2 for a triplet).
[PySCF molecular inputs](https://pyscf.org/user/gto.html)

HAAR-Q runs restricted Hartree–Fock (RHF for spin zero, ROHF for open shells).
It transforms the one-electron core Hamiltonian and two-electron Coulomb
integrals to the converged spatial molecular orbitals and includes the
nuclear repulsion once in the total Hamiltonian. All molecular orbitals
are included. Use an all-electron basis for the atoms in the calculation.
The SCF total energy supplies the default reporting reference.
Set `geometry.ref_energy` to compare against a known total energy for the
same geometry and basis. `OUTCAR` records the SCF and nuclear repulsion
energies in hartree. The SCF iteration limit and tolerance are listed in
the [geometry parameter reference](haar_q/conf/system/geometry.yaml).
[PySCF integral conventions](https://pyscf.org/quickstart.html)

For the H₂ coordinates above, RHF gives −1.1167593074 hartree. To check the
integrals against exact diagonalization in STO-3G, include all four
determinants with one alpha and one beta electron and run three
Lanczos steps beyond the initial vector.

```powershell
python -m haar_q -i input.yaml system=geometry method=haar sampling_control.seed_count=4 options.method.haar.krylov_extend_count=0 options.method.haar.krylov_iteration=3 common.max_relative_step=1
```

The `E_RITZ` column in `OSZICAR` gives −1.1372838345 hartree, including
nuclear repulsion. The separate `E` column reports the fitted network energy.

Start a fresh run when changing the molecule, basis, charge or spin.
When resuming a calculation or using its SCI pool, use the same orbital
basis and retain any optimized-basis files referenced by the checkpoint.

#### Excited states and analytical nuclear forces

Select `method=excited` and set `states.roots` to the **total number of
states**, including root 0. Each cycle samples and expands the determinant
space using the usual HAAR settings. The solver then diagonalizes the
projected Hamiltonian for the lowest requested states. The network learns
from the sum of their probabilities. Each state's signed or complex
coefficients are saved separately in `data.pth` under `haar.excited`.

The `GAP_FROM_ROOT_0` column in `ROOTS` gives the vertical excitation energy
in the model's energy units (hartree for molecular inputs). Roots are ordered
by energy within the chosen symmetry sector at each cycle. A fixed spin
projection can contain states of different total spin. Energy rank alone
does not track a state's physical character across a crossing. The residual
tests convergence **within the selected space**. To assess configuration
selection error, repeat with a larger space.

The following PowerShell example uses H₂ at 0.74 Å in STO-3G. All four
determinants with one alpha and one beta electron fit in the example's
selected space. Use the same molecular settings in both commands.

```powershell
$sciRun = 'outputs/h2-states-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$molecule = @('system=geometry', 'geometry.xyz_file=examples/h2/H2.xyz', 'geometry.unit=Angstrom', 'geometry.basis=sto-3g', 'geometry.charge=0', 'geometry.spin=0')
python -m haar_q -i input.yaml @molecule method=excited states.roots=4 sampling_control.seed_count=4 common.max_relative_step=1 "hydra.run.dir=$sciRun"
python -m haar_q -i input.yaml @molecule method=forces forces.root=1 "common.parent_path=$sciRun/data.pth" "hydra.run.dir=$sciRun/force-1"
Get-Content "$sciRun/ROOTS", "$sciRun/force-1/FORCES"
```

For this full-space H₂ example the first gap is 0.6065104775 hartree.
That root is the zero-spin-projection component of a triplet. The next
root is a singlet at 0.9689314015 hartree above the ground state. These
spin assignments were checked independently for this example.

Use `forces.root=0` for the ground state. The force action reads the last
selected determinant set, solves for the chosen eigenstate and contracts
its reduced density matrices with analytical derivatives. It includes
nuclear repulsion, AO integral and overlap derivatives, and the response of
the canonical RHF orbitals. Each component is `F = -dE/dR` in hartree/bohr.
The derivative holds the determinant labels fixed. A fresh stochastic
selection at a displaced geometry can change those labels and the resulting
energy surface. [Selected-CI gradient theory](https://doi.org/10.1021/acs.jctc.2c01062)

The force implementation supports isolated states in closed-shell,
all-electron canonical RHF orbitals. The checkpoint must match the input
nuclei and orbital basis. FCIDUMP alone lacks nuclear derivatives.
Rotated-orbital and open-shell checkpoints are unsupported. Canonical orbital
or target-state energy separations below `1e-7` hartree are rejected,
including the degenerate π orbitals of linear N₂. The in-memory force
implementation supports up to 32 spatial orbitals and requires the number
of distinct alpha strings times the number of distinct beta strings to be
at most 1,000,000. For selected spaces above 128 determinants, the shared
matrix-free eigensolver accepts up to one fifth of the dimension as its
root count. Force calculations also request the next root when available
to test whether the target is isolated.

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

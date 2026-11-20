# Contributing to HAAR-Q

Thanks for your interest in contributing to HAAR-Q!
Any kinds of contributions are welcome.

## How to contribute to HAAR-Q

If you have/implemented an idea or find/fixed a bug, use GitHub issues or pull requests.
Providing a minimum working example for bugs would be helpful.
If you have any questions, please use the GitHub discussions.

## Code structure

Start at [the CLI](haar_q/__main__.py). It composes the selected groups in
`haar_q/conf`, resolves paths against the input YAML and validates the active
method. [RuntimeContext](haar_q/utility/context.py) restores the checkpoint,
builds the model, network and optimizer, and saves each completed cycle.

| Area | Execution path and responsibilities |
| --- | --- |
| Molecular models | `models/geometry.py` obtains integrals from PySCF. `models/pyscf.py` and `models/fcidump.py` delegate spin-orbital packing to `models/_integrals.py`. |
| Hamiltonian | `hamiltonian/hamiltonian.py` dispatches projection, expansion and diagonals to the native CPU/CUDA implementations. `utility/parallel_hamiltonian.py` distributes source determinants across GPUs. |
| Sampling and VMC | `networks/mlp.py` and `networks/transformers.py` call the same unique path selector. `algorithms/vmc.py` differentiates a projected Rayleigh quotient. `algorithms/guide.py` also fits a separate sampler. |
| SCI | `algorithms/haar.py` selects a basis, constructs Ritz targets and fits the network. `algorithms/selected_states.py` solves for multiple roots. Orbital SCI reuses the Lanczos and fitting routines. |
| Physical observables | `utility/sci_checkpoint.py` reads the signed state for post-processing. The PySCF adapter and analytic forces account for fermionic phase changes on conversion to separate alpha/beta strings. |

Use **HAAR-Q** in prose, `haar-q` for the distribution and command-line entry
point, and `haar_q` in Python imports. HAAR-SCI is the published algorithm's
name and should remain unchanged in scientific references.

## How to get involved

Please learn the basic Git usage to contribute to HAAR-Q.
Please provide the essential information with proper formatting in the git commit message and the pull request description.

Please make sure that your code is properly formatted, linted and typed when you submit a pull request.
In Python code conventions, use double quotes (") for string literals and single quotes (') for character literals.
The comments in the code are expected to be enough for other developers to understand your code.
Please add docstrings to the code in NumPy style.
If necessary, please update documentations for your changes.
Address warnings introduced by your changes before submitting a pull request.
At last, please check carefully on the correctness and the robustness of the code when submitting a pull request.

## Development and Validation

Install development dependencies from your local checkout:

```bash
python -m pip install -e ".[dev]"
```

Run the CPU suite in the checkout with the development dependencies, a C++20
compiler and Ninja. GitHub Actions uses the same command. GPU cases run
when the required CUDA devices are visible.

```bash
python -m pytest -q
```

For a targeted run, choose the suite covering the physical quantity you changed.

```bash
python -m pytest -q tests/vmc_batch_test.py tests/unique_sampling_test.py
python -m pytest -q tests/integral_conversion_test.py tests/pyscf_solver_test.py
python -m pytest -q tests/sci_workflow_test.py tests/selected_force_reference_test.py
```

For algorithm changes, verify energies and gradients against a small,
independently evaluated reference. Describe the tested device, input, and
configuration. Record the timed operation, thread count and repeat count
alongside performance results. See [the benchmark notes](docs/performance.md).

## Release Packaging

Follow the [version and delivery policy](docs/versioning.md) for every
user-facing update. Record the version transition in [CHANGELOG.md](CHANGELOG.md)
and push the verified commit and release tag to the project's GitHub repository.

The wheel workflow builds HAAR-Q packages. PyPI publication requires configuring
Trusted Publishing for the new distribution and explicitly setting the repository
variable `HAAR_Q_PUBLISH_PYPI` to `true`. A local rename or wheel build does not
publish a release.

## Maintainer and Attribution

Current author and maintainer: **xiongzhi zeng**.

For questions, bug reports, or collaboration, please contact
[xzzeng@ustc.edu.cn](mailto:xzzeng@ustc.edu.cn).

Preserve the GPL license, original author credit, and contributor history.
See the README's authorship and provenance section for the project's origins
and the HAAR-SCI publication.

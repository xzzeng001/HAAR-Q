# Contributing to HAAR-Q

Thanks for your interest in contributing to HAAR-Q!
Any kinds of contributions are welcome.

## How to contribute to HAAR-Q

If you have/implemented an idea or find/fixed a bug, use GitHub issues or pull requests.
Providing a minimum working example for bugs would be helpful.
If you have any questions, please use the GitHub discussions.

## Code structure

HAAR-Q follows the standard Python package structure.
We organize the package code into the folder named `haar_q`.
The file `pyproject.toml` is used to define the package metadata.
There are also some other files such as `.clang-format`, `.pre-commit-config.yaml` used to format and lint the code.

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

Run the Python tests that do not require native Hamiltonian compilation:

```bash
python -m pytest tests/bitspack_test.py tests/vmc_test.py
```

The Hamiltonian tests require a working native build toolchain:

```bash
python -m pytest tests/hamiltonian_test.py
```

For algorithm changes, verify energies and gradients against a small,
independently evaluated reference. Describe the tested device, input, and
configuration; distinguish these checks from larger-system benchmarks.

## Release Packaging

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

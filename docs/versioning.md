# Versioning and delivery

HAAR-Q uses `MAJOR.MINOR.PATCH` for every update delivered to users, including
the 0.x series. Classify the whole batch since the previous delivery and bump
the highest applicable component once. Reset all lower components to zero.
Multiple commits may share a delivery. Reinstalling, rebuilding or rerunning
the same source keeps its version.

| Level | When to increment | Example |
| --- | --- | --- |
| MAJOR | Existing inputs, public calls, documented machine-readable fields or scientific interpretations require migration. | Remove a YAML key or change the units or energy reference of an existing observable. |
| MINOR | Add a compatible input form, algorithm, physical model, observable or backend. This also covers a change to the solver principle that preserves the public meaning of the result. | Accept a named YAML file while retaining existing input files and invocations. |
| PATCH | Correct implementation within the documented scientific definition, improve performance, adjust report layout, or update documentation or examples. | Fix a sign error so the code evaluates the published expression. |

If a numerical correction changes previous results, describe the affected
observables and configurations in the changelog even for a patch update.
Include migration instructions for a breaking change.

## Deliver an update

1. Set `__version__` in [haar_q/version.py](../haar_q/version.py). The Python
   API, build metadata, CLI and calculation report read this value.
2. Add a [changelog entry](../CHANGELOG.md) with the previous and new version,
   impact level, reason for the bump and user-visible changes.
3. Check the affected calculations and build the wheel. Install it into a
   clean environment and compare `python -m haar_q --version`, `haar-q --version`,
   `haar_q.__version__`, `importlib.metadata.version("haar-q")` and the version
   in a sample run's `OUTCAR`. Refresh editable installations with
   `python -m pip install --no-deps -e .` to update installed metadata.
4. Commit the verified delivery to `main`. Annotate that commit with
   `vMAJOR.MINOR.PATCH`, publish the branch and its new tag to
   [GitHub](https://github.com/xzzeng001/HAAR-Q), and verify both remote
   references. Each tag identifies a single released commit.

For example, with the version set to `0.1.0` and the delivery committed:

```bash
git tag -a v0.1.0 -m "HAAR-Q 0.1.0"
git push --atomic origin main refs/tags/v0.1.0
git ls-remote origin refs/heads/main "refs/tags/v0.1.0^{}"
```

The existing wheel workflow controls optional PyPI publication. See
[CONTRIBUTING.md](../CONTRIBUTING.md#release-packaging) for that configuration.

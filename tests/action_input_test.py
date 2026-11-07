"""Compatibility checks for parameters accepted through both action entry points."""

from collections.abc import Mapping
from pathlib import Path
import sys

import pytest
from omegaconf import DictConfig, OmegaConf

from haar_q import __main__ as cli
from haar_q.utility.action_dict import ActionProto, create_action


def action(name: str, params: Mapping[str, object]) -> ActionProto:
    return create_action(OmegaConf.create({"action": {"name": name, "params": dict(params)}}).action)


@pytest.mark.parametrize(
    ("name", "preferred", "legacy"),
    [
        (
            "vmc",
            {"seed_count": 7, "target_basis_size": 16, "unique_seeds": False, "optimizer_steps": 9},
            {"sampling_count": 7, "relative_count": 16, "unique": False, "local_step": 9},
        ),
        ("guide", {"seed_count": 4, "sampler_steps_per_update": 7}, {"sampling_count": 4, "dist_step": 7}),
        (
            "haar",
            {"network_seed_count": 4, "previous_seed_count": 2, "optimizer_steps": 6},
            {"sampling_count_from_neural_network": 4, "sampling_count_from_last_iteration": 2, "local_step": 6},
        ),
        (
            "haar_with_orbit",
            {"src_fcidump": "h2.FCIDUMP", "previous_seed_count": 4},
            {"src_fcidump": "h2.FCIDUMP", "sampling_count_from_last_iteration": 4},
        ),
    ],
)
def test_action_parameter_aliases(name: str, preferred: dict[str, object], legacy: dict[str, object]) -> None:
    configured = action(name, preferred)
    assert configured == action(name, legacy)
    for key, expected in legacy.items():
        assert getattr(configured, key) == (
            Path(expected) if key == "src_fcidump" and isinstance(expected, str) else expected
        )


def test_legacy_selector_constructs_haar() -> None:
    assert action("imag", {"network_seed_count": 4}) == action("haar", {"sampling_count_from_neural_network": 4})


@pytest.mark.parametrize("name", ["vmc", "haar", "guide", "haar_with_orbit"])
def test_conflicting_parameter_names(name: str) -> None:
    with pytest.raises(ValueError, match="both 'optimizer_steps' and its alias 'local_step'"):
        action(name, {"local_step": 2, "optimizer_steps": 2})


@pytest.mark.parametrize("name", ["vmc", "haar"])
def test_unknown_parameter_names(name: str) -> None:
    with pytest.raises(ValueError, match=rf"Unknown {name} action.params.*optimzer_steps"):
        action(name, {"optimzer_steps": 1})


@pytest.mark.parametrize("key", ["seed_count", "sampling_count"])
def test_hydra_override_uses_input_spelling(key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "input.yaml").write_text(f"action:\n  name: vmc\n  params: {{{key}: 2}}\n", encoding="utf-8")
    runs: list[ActionProto] = []

    def check(config: DictConfig) -> None:
        runs.append(create_action(config.action))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_run", check)
    monkeypatch.setattr(sys, "argv", ["haar-q", "-i", "input.yaml", f"action.params.{key}=3"])
    cli.main()
    assert runs == [action("vmc", {"sampling_count": 3})]

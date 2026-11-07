"""Keep the editable examples in the standard input in sync with the action schemas."""

from dataclasses import fields, is_dataclass
from pathlib import Path

from omegaconf import OmegaConf

from haar_q.utility.action_dict import _PARAMETER_NAMES, create_action


def test_standard_input_covers_every_action_parameter() -> None:
    template = Path(__file__).resolve().parents[1] / "input.yaml"
    document = template.read_text(encoding="utf-8")
    active = OmegaConf.create(document)
    assert (template.parent / active.model.params.model_path).is_file()
    actions = [active.action]
    # Alternatives are copyable top-level YAML blocks separated by a blank line.
    for paragraph in document.split("\n\n"):
        lines = paragraph.splitlines()
        if "# action:" in lines:
            snippet = "\n".join(line.removeprefix("# ") for line in lines[lines.index("# action:") :])
            actions.append(OmegaConf.create(snippet).action)

    assert {action.name for action in actions} == {
        "vmc",
        "haar",
        "guide",
        "pretrain",
        "pert",
        "chop_imag",
        "orbit",
        "haar_with_orbit",
    }
    for action in actions:
        configured = create_action(action)
        assert is_dataclass(configured)
        aliases = _PARAMETER_NAMES.get(action.name, {})
        documented = {aliases.get(key, key) for key in action.params}
        assert documented == {field.name for field in fields(configured) if field.init}, action.name

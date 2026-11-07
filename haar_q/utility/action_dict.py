"""
This module is used to store a dictionary that maps action names to their corresponding dataclass types.

Other packages or subpackages can register their actions by adding entries to this dictionary, such as
```
from haar_q.utility.action_dict import action_dict
action_dict["my_action"] = MyAction
```
"""

import importlib
import typing

import dacite
import omegaconf
from .context import DACITE_CAST, RuntimeContext


class ActionProto(typing.Protocol):
    """
    This protocol defines a dataclass with a `main` method, which will be called when the action is executed.
    """

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main method to be called when the action is executed.
        """


action_dict: dict[str, type[ActionProto]] = {}

# Preferred YAML names map to existing dataclass fields, which are also accepted in inputs.
_VMC_PARAMETERS = {
    "seed_count": "sampling_count",
    "target_basis_size": "relative_count",
    "optimizer_steps": "local_step",
    "unique_seeds": "unique",
}
_HAAR_PARAMETERS = {
    "network_seed_count": "sampling_count_from_neural_network",
    "previous_seed_count": "sampling_count_from_last_iteration",
    "optimizer_steps": "local_step",
}
_PARAMETER_NAMES = {
    "vmc": _VMC_PARAMETERS,
    "guide": {**_VMC_PARAMETERS, "sampler_steps_per_update": "dist_step"},
    "haar": _HAAR_PARAMETERS,
    "haar_with_orbit": _HAAR_PARAMETERS,
    "imag": _HAAR_PARAMETERS,
}


def create_action(action: omegaconf.DictConfig) -> ActionProto:
    """Construct the selected action from composed YAML or a PySCF solver configuration."""
    name = action.name
    if name not in action_dict:
        module = "haar" if name == "imag" else name
        importlib.import_module(f"haar_q.algorithms.{module}")
    data = omegaconf.OmegaConf.to_container(action.params, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping for {name} action.params.")
    for preferred, original in _PARAMETER_NAMES.get(name, {}).items():
        if preferred in data:
            if original in data:
                raise ValueError(f"{name} action.params specifies both {preferred!r} and its alias {original!r}.")
            data[original] = data.pop(preferred)
    try:
        return dacite.from_dict(
            data_class=action_dict[name],
            data=data,
            config=dacite.Config(cast=DACITE_CAST, strict=name in {"vmc", "haar", "imag"}),
        )
    except dacite.UnexpectedDataError as exc:
        keys = ", ".join(sorted(exc.keys))
        raise ValueError(f"Unknown {name} action.params field(s): {keys}. See the project's input.yaml.") from exc

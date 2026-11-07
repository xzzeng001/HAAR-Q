"""Check selected components before constructing scientific runtime objects."""

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


_MODEL_FILES = {"fcidump": "fcidump", "openfermion": "molecule", "optimized_basis": "optimized_basis"}
_POOL_METHODS = {"pert", "chop_imag", "orbit"}


def _has_checkpoint_source(config: DictConfig) -> bool:
    if config.common.get("parent_path"):
        return True
    return HydraConfig.initialized() and (Path(HydraConfig.get().runtime.output_dir) / "data.pth").is_file()


def _check_network(name: str, role: str, system: str, method: str, unique: bool) -> None:
    if name == "mps":
        if system not in {"hubbard", "ising"}:
            raise ValueError(f"The {role} mps requires system=hubbard or system=ising.")
        if method in {"haar", "imag", "haar_with_orbit"}:
            raise ValueError(f"method={method} requires unique sampling. Choose ansatz=mlp or ansatz=transformers.")
        if method in {"vmc", "guide"} and unique:
            raise ValueError(f"The {role} mps requires sampling_control.unique_seeds=false for method={method}.")
        return

    family, _, sector = name.partition("/")
    sectors = ("0", "u1") if system == "ising" else ("u1u1", "u1")
    if family not in {"mlp", "transformers"} or sector not in sectors:
        raise ValueError(
            f"Unsupported {role} {name!r} for system={system}. "
            f"For MLP and Transformer networks, choose symmetry={sectors[0]} or symmetry={sectors[1]}."
        )


def validate_selection(config: DictConfig) -> None:
    """Validate the effective job after active modules and input paths have been resolved."""
    system = config.model.name
    method = config.action.name
    params = config.action.params
    if system in _MODEL_FILES and not config.model.params.get("model_path"):
        # Orbital SCI can restore its rotated model from the checkpoint's saved model_path.
        if not (method == "haar_with_orbit" and _has_checkpoint_source(config)):
            raise ValueError(f"system={system} requires files.{_MODEL_FILES[system]} to point to a model file.")

    if method in {"orbit", "haar_with_orbit"}:
        if system not in _MODEL_FILES:
            raise ValueError(f"method={method} requires a molecular system and matching FCIDUMP integrals.")
        if not params.get("src_fcidump"):
            raise ValueError(f"method={method} requires files.fcidump to point to the matching molecular integrals.")
    if method == "orbit" and not params.get("dst_optimized"):
        raise ValueError("method=orbit requires files.orbital_output for the rotated Hamiltonian.")
    if method == "pretrain" and not params.get("dataset_path"):
        raise ValueError("method=pretrain requires files.dataset to point to saved configurations and amplitudes.")

    if method in _POOL_METHODS:
        if not _has_checkpoint_source(config):
            raise ValueError(f"method={method} requires a HAAR-SCI pool. Set common.parent_path to its checkpoint.")
        return

    unique = params.get("unique_seeds", params.get("unique", True))
    _check_network(config.network.name, "wavefunction", system, method, unique)
    if method == "guide":
        _check_network(config.sampling.name, "guide sampler", system, method, unique)

"""Command-line entry point for HAAR-Q calculations."""

import argparse
import copy
import importlib
import sys
from pathlib import Path

import dacite
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .version import __version__


def _resolve_input_paths(runtime_config: DictConfig, input_dir: Path) -> None:
    """Anchor user-supplied files at the explicit YAML, even if Hydra changes directory."""
    for key in (
        "model.params.model_path",
        "common.parent_path",
        "action.params.dataset_path",
        "action.params.src_fcidump",
        "action.params.dst_optimized",
    ):
        value = OmegaConf.select(runtime_config, key)
        if value is not None:
            path = input_dir / Path(value).expanduser()
            OmegaConf.update(runtime_config, key, str(path.resolve()))


def _run(runtime_config: DictConfig) -> None:
    from .utility.action_dict import action_dict
    from .utility.context import DACITE_CAST, RuntimeContext

    context = dacite.from_dict(
        data_class=RuntimeContext,
        data=OmegaConf.to_container(runtime_config.common, resolve=True),  # type: ignore[arg-type]
        config=dacite.Config(cast=DACITE_CAST),
    )
    with context.reporting(runtime_config):
        importlib.import_module(f".algorithms.{runtime_config.action.name}", package=__package__)
        importlib.import_module(f".models.{runtime_config.model.name}", package=__package__)
        checkpoint_data = context.setup()
        run = dacite.from_dict(
            data_class=action_dict[runtime_config.action.name],  # type: ignore[arg-type]
            data=OmegaConf.to_container(runtime_config.action.params, resolve=True),  # type: ignore[arg-type]
            config=dacite.Config(cast=DACITE_CAST),
        )
        run.main(context=context, runtime_config=runtime_config, checkpoint_data=checkpoint_data)


def main(runtime_config: DictConfig | None = None) -> None:
    """Select the primary input file and let Hydra compose and launch the job."""
    if runtime_config is not None:
        _run(runtime_config)
        return
    parser = argparse.ArgumentParser(
        prog="haar-q",
        allow_abbrev=False,
        usage="%(prog)s [-i INPUT.yaml] [--cfg job] [key=value ...]",
        description="Run HAAR-Q using a YAML input file. The default input is ./config.yaml.",
        epilog="Use --cfg job to preview the input or --hydra-help for advanced Hydra options.",
    )
    parser.add_argument("-i", "--input", type=Path, metavar="FILE.yaml", help="path to the calculation input")
    parser.add_argument("--version", action="version", version=f"HAAR-Q {__version__}")
    opts, hydra_args = parser.parse_known_args()

    config_dir = Path.cwd()
    config_name = "config"
    if opts.input is not None:
        path = opts.input.expanduser().resolve()
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
        if path.suffix != ".yaml":
            parser.error(f"Hydra input files require the .yaml extension: {path}")
        if any(arg.split("=", 1)[0] in {"--config-path", "-cp", "--config-name", "-cn"} for arg in hydra_args):
            parser.error("choose either --input or the Hydra --config-path/--config-name options")
        config_dir, config_name = path.parent, path.name

    # Keep the input directory as Hydra's main search path for sibling defaults.
    @hydra.main(version_base=None, config_path=str(config_dir), config_name=config_name)
    def launch(runtime_config: DictConfig) -> None:
        if opts.input is not None:
            _resolve_input_paths(runtime_config, config_dir)
        _run(runtime_config)

    # The calculation reporter owns logging. A YAML defaults list or a Hydra
    # group override can still explicitly select a different job logger.
    store = ConfigStore.instance()
    original = store.load("hydra/config.yaml")
    settings = copy.deepcopy(original.node)
    for group in settings.defaults:
        if isinstance(group, DictConfig) and "job_logging" in group:
            group["job_logging"] = "none"
    store.store(group="hydra", name="config", node=settings, provider="haar_q")

    # Hydra's public launcher reads sys.argv directly.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *hydra_args]
        launch()
    finally:
        sys.argv = original_argv
        store.store(group="hydra", name="config", node=original.node, provider=original.provider)


if __name__ == "__main__":
    main()

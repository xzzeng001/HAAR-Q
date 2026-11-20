import errno
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from haar_q.utility.context import RuntimeContext
from haar_q.models import ising


def test_restoring_double_network_preserves_weights_and_next_random_draws() -> None:
    model = ising.Model(ising.ModelConfig(2, 1, x=1))
    context = RuntimeContext(device=torch.device("cpu"), dtype="float64")
    config = OmegaConf.create({"name": "mlp/0", "params": {"hidden": [3]}})
    network = context.create_network(config, model)
    state = network.state_dict()
    # Add a value below the float32 rounding threshold to a double parameter.
    next(iter(network.parameters())).detach().fill_(1 + 2.0**-35)
    rng = torch.get_rng_state()
    restored = context.create_network(config, model, state)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, state[key])


@pytest.mark.parametrize("field", ["checkpoint_interval", "max_relative_step", "max_absolute_step"])
def test_zero_step_settings_fail_before_calculation(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RuntimeContext(**{field: 0})  # type: ignore[arg-type]


def test_absolute_cycle_limit_uses_only_active_checkpoint_counter() -> None:
    context = RuntimeContext(max_absolute_step=4)
    data = {"imag": {"global": 4}, "guide": {"global": 2}, "vmc": {"global": 6}}
    for action in ("haar", "excited", "haar_with_orbit", "vmc"):
        assert context.cycle_limit_reached(action, data)
    assert not context.cycle_limit_reached("guide", data)
    assert not context.cycle_limit_reached("forces", data)


def test_optimizer_yaml_sequences_survive_checkpoint(tmp_path: Path) -> None:
    ctx = RuntimeContext(device=torch.device("cpu"))
    config = OmegaConf.create({"name": "Adam", "params": {"betas": [0.8, 0.98]}})
    optimizer = ctx.create_optimizer(config, [torch.nn.Parameter(torch.ones(1))])
    checkpoint = tmp_path / "optimizer.pth"
    torch.save(optimizer.state_dict(), checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    resumed = ctx.create_optimizer(config, [torch.nn.Parameter(torch.ones(1))], state)
    assert tuple(resumed.param_groups[0]["betas"]) == (0.8, 0.98)


def test_checkpoint_on_a_filesystem_without_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = RuntimeContext(device=torch.device("cpu"))
    monkeypatch.setattr(context, "folder", lambda: tmp_path)
    monkeypatch.setattr(Path, "symlink_to", Mock(side_effect=OSError(errno.EPERM, "symlinks disabled")))
    context.save({"step": 4}, 4)
    context.save({"step": 5}, 5)
    snapshot = tmp_path / "data.5.pth"
    latest = tmp_path / "data.pth"
    assert not latest.is_symlink() and latest.read_bytes() == snapshot.read_bytes()
    context.save({"step": 6}, 6)
    assert torch.load(snapshot, weights_only=True)["step"] == 5
    assert torch.load(latest, weights_only=True)["step"] == 6


@pytest.mark.parametrize("unavailable", [False, True])
def test_failed_snapshot_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    context = RuntimeContext(device=torch.device("cpu"))
    monkeypatch.setattr(context, "folder", lambda: tmp_path)
    context.save({"step": 4}, 4)
    previous = (tmp_path / "data.pth").read_bytes()
    io_error = OSError(errno.EIO, "disk error")
    symlink_error = OSError(errno.EPERM, "symlinks disabled") if unavailable else io_error
    monkeypatch.setattr(Path, "symlink_to", Mock(side_effect=symlink_error))

    def partial_copy(source: Path, destination: Path) -> None:
        destination.write_bytes(b"partial")
        raise io_error

    if unavailable:
        monkeypatch.setattr("haar_q.utility.context.shutil.copyfile", partial_copy)
    with pytest.raises(OSError) as caught:
        context.save({"step": 5}, 5)
    assert caught.value is io_error
    assert (tmp_path / "data.pth").read_bytes() == previous
    assert set(tmp_path.iterdir()) == {tmp_path / "data.pth"}

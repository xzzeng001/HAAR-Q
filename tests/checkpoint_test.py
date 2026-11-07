import errno
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from haar_q.utility.context import RuntimeContext


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

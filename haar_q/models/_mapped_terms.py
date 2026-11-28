"""Build a private file-backed operator without materializing its full spin expansion in RAM."""

from contextlib import ExitStack
import errno
from pathlib import Path
import tempfile

import numpy as np
import torch

from ._integrals import _term_blocks


def write_mapped_terms(directory: Path, integrals: tuple[float, np.ndarray, np.ndarray]) -> None:
    """Write the three native columns in expansion order, then publish the directory atomically."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".packing-") as tmp:
        staged = Path(tmp) / "operator"
        staged.mkdir()
        count = 0
        with ExitStack() as stack:
            sites, kinds, coefs = [
                stack.enter_context((staged / name).open("wb")) for name in ("sites", "kinds", "coefs")
            ]
            for indices, values in _term_blocks(*integrals):
                size, width = indices.shape
                if not size:
                    continue
                site = np.zeros((size, 4), dtype=np.int16)
                site[:, :width] = indices
                kind = np.full((size, 4), 2, dtype=np.uint8)
                kind[:, : width // 2] = 1
                kind[:, width // 2 : width] = 0
                coef = np.zeros((size, 2), dtype=np.float64)
                coef[:, 0] = values
                for stream, column in ((sites, site), (kinds, kind), (coefs, coef)):
                    stream.write(memoryview(column).cast("B"))
                count += size
        (staged / "count").write_text(str(count), encoding="ascii")
        try:
            staged.rename(directory)
        except OSError as exc:
            # Another process can finish the same immutable cache first.
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not directory.is_dir():
                raise


def map_terms(directory: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate byte lengths before privately mapping each native column."""
    count = int((directory / "count").read_text(encoding="ascii"))

    def column(name: str, width: int, dtype: torch.dtype) -> torch.Tensor:
        path = directory / name
        size = count * width
        if count < 0 or path.stat().st_size != size * dtype.itemsize:
            raise ValueError(f"Invalid mapped FCIDUMP cache column: {path}")
        return torch.from_file(str(path), shared=False, size=size, dtype=dtype).reshape(count, width)

    return column("sites", 4, torch.int16), column("kinds", 4, torch.uint8), column("coefs", 2, torch.float64)

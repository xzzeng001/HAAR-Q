"""Exercise the installed CLI on the same visible GPUs that will run a cluster job."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices and nvcc")
@pytest.mark.parametrize("method", ["vmc", "excited"])
def test_two_gpu_h2_matches_single_gpu(tmp_path: Path, method: str) -> None:
    template = Path(__file__).resolve().parents[1] / "input.yaml"
    rows = []
    for label, devices in [("single", "[0]"), ("pair", "[0,1]")]:
        output = tmp_path / label
        subprocess.run(
            [
                sys.executable,
                "-m",
                "haar_q",
                "-i",
                str(template),
                f"method={method}",
                "common.device=cuda:0",
                f"common.gpu_devices={devices}",
                "common.max_relative_step=1",
                "sampling_control.seed_count=4",
                "training.optimizer_steps=2",
                "states.roots=4",
                f"hydra.run.dir={output}",
            ],
            check=True,
            timeout=900,
        )
        filename = "ROOTS" if method == "excited" else "OSZICAR"
        rows.append(
            [
                float(line.split()[3])
                for line in (output / filename).read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        )
        assert (output / "data.pth").is_file()
    assert rows[1] == pytest.approx(rows[0], abs=1e-8, rel=0)
    if method == "excited":
        assert len(rows[1]) == 4
        assert rows[1][0] == pytest.approx(-1.4275251968493827, abs=1e-8)
    else:
        assert len(rows[1]) == 1

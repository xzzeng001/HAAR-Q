"""Check multi-device work partitioning and CUDA target selection."""

import pytest
import torch

from haar_q.__main__ import main
from haar_q.hamiltonian import Hamiltonian
from haar_q.utility.context import RuntimeContext
from haar_q.utility.parallel_hamiltonian import ParallelHamiltonian


@pytest.mark.parametrize(
    ("primary", "devices", "message"),
    [
        ("cpu", (0, 1), "requires common.device"),
        ("cuda", (0, 0), "distinct"),
        ("cuda", (-1, 0), "nonnegative"),
        ("cuda:0", (1, 0), "first entry"),
        ("cuda:0", (0, 2), "only 2 GPUs"),
    ],
)
def test_invalid_devices(primary: str, devices: tuple[int, ...], message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        RuntimeContext(device=torch.device(primary), gpu_devices=devices)


def test_multi_process_launcher_rejected_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(RuntimeError, match="manages all GPUs in one process"):
        main()


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_partitioned_matvec_matches_serial(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if backend == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices and a CUDA build toolchain.")
    device = torch.device("cpu" if backend == "cpu" else "cuda:1")
    source = Hamiltonian({(): 0.5, ((1, 1), (0, 0)): 2j, ((0, 1), (1, 0)): -2j}, kind="fermi")
    parallel = ParallelHamiltonian(source, (1, 0))
    if backend == "cpu":
        # Exercise the real native matvec, split, threaded dispatch and reduction
        # on CPU-only CI. The CUDA case also checks actual device/stream transfers.
        parallel.devices = (device, device)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    configs = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.uint8, device=device)
    targets = configs[[4, 2, 1]]
    psi = torch.tensor([0.2, -1j, 2 + 3j, 0.1, 1 + 1j], dtype=torch.complex128, device=device)
    stream = torch.cuda.Stream(device=device) if backend == "cuda" else None  # type: ignore[no-untyped-call]
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        expected = source.apply_within(configs, psi, targets)
        for _ in range(2):
            result = parallel.apply_within(configs, psi, targets)
            assert result.device == device
            torch.testing.assert_close(result, expected)
        torch.testing.assert_close(parallel.apply_within(configs[:0], psi[:0], targets), torch.zeros_like(result))
        assert parallel.apply_within(configs, psi, targets[:0]).numel() == 0


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_global_selection_preserves_path_multiplicities(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if backend == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices and a CUDA build toolchain.")
    device = torch.device("cpu" if backend == "cpu" else "cuda:0")
    source = Hamiltonian({((0, 0),): 3.0, ((1, 0),): 2j, ((2, 0),): 1.0, ((3, 1),): 0.1}, kind="fermi")
    parallel = ParallelHamiltonian(source, (0, 1))
    if backend == "cpu":
        parallel.devices = (device, device)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    configs = torch.tensor([[1], [2], [4]], dtype=torch.uint8, device=device)
    psi = torch.tensor([2, 1j, 0.8], dtype=torch.complex128, device=device)
    for count, exclude, expected in [(3, None, [0]), (4, None, [0, 9]), (50, [0], [9, 10, 12]), (0, None, [])]:
        excluded = None if exclude is None else configs.new_tensor(exclude).reshape(-1, 1)
        exact = configs.new_tensor(expected).reshape(-1, 1)
        torch.testing.assert_close(parallel.find_relative(configs, psi, count, excluded), exact)
        torch.testing.assert_close(source.find_relative(configs, psi, count, excluded), exact)
    assert parallel.find_relative(configs, psi * 0, 1).shape == (0, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        parallel.find_relative(configs, psi, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device and a CUDA build toolchain.")
def test_cuda_heap_under_concurrent_updates() -> None:
    # Each source has two distinct allowed excitations. Their weights and final
    # configurations can be ranked directly without invoking another native heap.
    device = torch.device("cuda:0")
    source = Hamiltonian({((12, 1),): 1.0, ((13, 1),): 1.03}, kind="fermi")
    ids = torch.arange(2048, device=device)
    configs = torch.stack((ids % 256, ids // 256), dim=1).to(torch.uint8)
    psi = (1.0 + ids.double() / 2053).to(torch.complex128)
    targets = torch.cat([configs | configs.new_tensor([0, mask]) for mask in (16, 32)])
    weights = torch.cat([psi.abs().square(), (1.03 * psi.abs()).square()])
    for quota in (1, 7, 64, 511):
        expected = torch.unique(targets[weights.topk(quota).indices], dim=0)
        torch.testing.assert_close(source.find_relative(configs, psi, quota), expected)

"""Exact connected configurations across a CUDA enumeration tile boundary."""

import pytest
import torch

from haar_q.hamiltonian import Hamiltonian


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_packed_146_sites_and_complex_path_sums(device: str) -> None:
    count = 513
    # Each copy of the same term creates a particle at spin orbital 145. With
    # 513 input states the Cartesian product spans two 262144-pair CUDA tiles.
    site = torch.full((count, 4), 145, dtype=torch.int16)
    kind = torch.full((count, 4), 2, dtype=torch.uint8)
    kind[:, 0] = 1
    coef = torch.ones((count, 2), dtype=torch.float64)
    coef[1::2] *= -1
    model = Hamiltonian((site, kind, coef), kind="fermi")

    ids = torch.arange(count)
    configs = torch.zeros((count, 19), dtype=torch.uint8)
    configs[:, 0] = ids % 256
    configs[:, 1] = ids // 256
    psi = ids.to(torch.complex128) + 2j
    expected_configs = configs.clone()
    expected_configs[:, -1] = 2
    parity = torch.tensor([(-1) ** value.bit_count() for value in range(count)])
    expected_psi = psi * parity * (1 + 1j)
    # Exclude arbitrary keys and compare the surviving rows in byte order.
    exclude = expected_configs[[257, 31, 0]]
    rows = sorted(set(range(count)) - {257, 31, 0}, key=lambda i: (i % 256, i // 256))
    states, amplitudes = model.list_relative(configs.to(device), psi.to(device), exclude.to(device))
    torch.testing.assert_close(states.cpu(), expected_configs[rows])
    torch.testing.assert_close(amplitudes.cpu(), expected_psi[rows])

    empty_states, empty_amplitudes = model.list_relative(configs[:1].to(device), psi[:1].to(device), exclude.to(device))
    assert empty_states.shape == (0, 19) and empty_amplitudes.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("terms", "batch"), [(777, 1), (777, 4), (17, 8 * 65535 + 5)])
def test_cuda_term_and_batch_grid_tails(terms: int, batch: int) -> None:
    # Repeated site-zero number operators give integer sums at both grid tails.
    site = torch.zeros((terms, 4), dtype=torch.int16)
    kind = torch.full((terms, 4), 2, dtype=torch.uint8)
    kind[:, 0] = 1
    kind[:, 1] = 0
    coef = torch.zeros((terms, 2), dtype=torch.float64)
    coef[:, 0] = 1
    model = Hamiltonian((site, kind, coef), kind="fermi")
    configs = torch.zeros((batch, 19), dtype=torch.uint8, device="cuda")
    configs[::2, 0] = 1
    expected = configs[:, 0].to(torch.complex128) * terms
    torch.testing.assert_close(model.diagonal_term(configs), expected)

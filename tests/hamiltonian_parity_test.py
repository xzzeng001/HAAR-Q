"""Packed parity must preserve operator order across byte and machine-word boundaries."""

import random

import pytest
import torch

from haar_q.hamiltonian import Hamiltonian


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
@pytest.mark.parametrize("particle_kind", ["fermi", "bose2"])
@pytest.mark.parametrize("n_sites", [146, 257])
def test_operator_order_and_packed_prefix_parity(device: str, particle_kind: str, n_sites: int) -> None:
    # Dropping a partial byte, truncating to 64/128 bits, or using the initial
    # occupancy for every operation changes these independently computed sums.
    rng = random.Random(1873)
    boundaries = [0, 7, 8, 31, 32, 63, 64, 127, 128, n_sites - 1]
    terms: dict[tuple[tuple[int, int], ...], complex] = {(): 0.25 + 0.5j}
    for site in boundaries:
        terms[((site, 1),)] = 0.5 - 0.25j
        terms[((site, 0),)] = -0.75 + 0.5j
        terms[((site, 1), (site, 0))] = 1.25 + 0j
        terms[((site, 0), (site, 1))] = -0.25 + 0j
        terms[((site, 1), (site, 1))] = 9.0 + 0j  # Pauli-blocked
        terms[((site, 0), (site, 0))] = 9.0 + 0j
    for _ in range(100):
        term = tuple((rng.choice(boundaries), rng.randrange(2)) for _ in range(rng.randrange(1, 5)))
        terms[term] = complex(rng.randrange(-4, 5), rng.randrange(-4, 5)) / 8
    source = [0, (1 << n_sites) - 1] + [rng.getrandbits(n_sites) for _ in range(22)]
    amplitudes = [complex(rng.randrange(1, 5), rng.randrange(-4, 5)) / 4 for _ in source]
    expected: dict[int, complex] = {}
    diagonal = []
    for initial, amplitude in zip(source, amplitudes):
        diag = 0j
        for term, coefficient in terms.items():
            state = initial
            sign = 1
            for site, kind in reversed(term):
                if ((state >> site) & 1) == kind:
                    break
                if particle_kind == "fermi" and (state & ((1 << site) - 1)).bit_count() % 2:
                    sign = -sign
                state ^= 1 << site
            else:
                expected[state] = expected.get(state, 0j) + sign * coefficient * amplitude
                if state == initial:
                    diag += sign * coefficient
        diagonal.append(diag)

    n_bytes = (n_sites + 7) // 8

    def packed(states: list[int]) -> torch.Tensor:
        return torch.tensor(
            [list(state.to_bytes(n_bytes, "little")) for state in states], dtype=torch.uint8, device=device
        )

    # Exact enumeration retains keys even when several paths cancel.
    targets = sorted(expected, key=lambda state: state.to_bytes(n_bytes, "little"))
    configs = packed(source)
    psi = torch.tensor(amplitudes, dtype=torch.complex128, device=device)
    model = Hamiltonian(terms, kind=particle_kind)
    projected = model.apply_within(configs, psi, packed(targets))
    torch.testing.assert_close(
        projected.cpu(), torch.tensor([expected[state] for state in targets], dtype=torch.complex128)
    )
    torch.testing.assert_close(model.diagonal_term(configs).cpu(), torch.tensor(diagonal, dtype=torch.complex128))
    actual_configs, actual_psi = model.list_relative(configs, psi, configs[:0])
    order = sorted(range(len(actual_configs)), key=lambda row: bytes(actual_configs[row].cpu().tolist()))
    torch.testing.assert_close(actual_configs[order].cpu(), packed(targets).cpu())
    torch.testing.assert_close(
        actual_psi[order].cpu(), torch.tensor([expected[state] for state in targets], dtype=torch.complex128)
    )

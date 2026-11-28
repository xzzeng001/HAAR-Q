"""Validate term partitioning with 216 spin orbitals and cross-shard interference."""

from unittest.mock import Mock

import pytest
import torch

from haar_q.hamiltonian import Hamiltonian
from haar_q.utility.context import RuntimeContext
from haar_q.utility.term_hamiltonian import TermHamiltonian


@pytest.mark.parametrize("batch", [1, 64], ids=["streamed", "resident"])
@pytest.mark.parametrize(
    "devices",
    [
        ("cpu",),
        ("cpu", "cpu", "cpu"),
        pytest.param(("cuda:0",), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
        pytest.param(
            ("cuda:1", "cuda:0"), marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
        ),
    ],
)
def test_108_orbitals_cross_shard_reduction(batch: int, devices: tuple[str, ...]) -> None:
    device = torch.device(devices[0])
    # Opposite-sign paths on separate terms can cancel; the two paths still
    # occupy distinct slots in selection before their destination is deduplicated.
    terms = {
        (): 0.375,
        ((215, 1), (215, 0)): 0.75,
        ((1, 0),): 2.0,
        ((214, 0),): -4.0,
        ((211, 1),): 0.125j,
        ((1, 1), (214, 0)): 1.25j,
        ((214, 1), (1, 0)): -1.25j,
    }
    serial = Hamiltonian(terms, kind="fermi")  # type: ignore[arg-type]
    operator = TermHamiltonian(serial, tuple(map(torch.device, devices)), batch)
    # Three representatives exercise the first and last bytes of a determinant.
    configs = torch.zeros((3, 27), dtype=torch.uint8, device=device)
    configs[0, 0] = 2
    configs[1, -1] = 64
    configs[2, -1] = 128
    psi = torch.tensor([2, 1, -0.875j], dtype=torch.complex128, device=device)
    excluded = configs[:0]
    external, values = operator.list_relative(configs, psi, excluded)
    expected_rows, expected = serial.list_relative(configs, psi, excluded)
    torch.testing.assert_close(external, expected_rows)
    torch.testing.assert_close(values, expected)
    assert values[(external == 0).all(dim=1)].item() == 0
    targets = expected_rows.flip(0)
    torch.testing.assert_close(operator.apply_within(configs, psi, targets), expected.flip(0))
    torch.testing.assert_close(operator.diagonal_term(configs), serial.diagonal_term(configs))
    # Highest two paths create the same vacuum, so only one row survives.
    assert torch.equal(operator.find_relative(configs, psi, 2), torch.zeros_like(configs[:1]))
    for k in (0, 3, 1000):
        torch.testing.assert_close(operator.find_relative(configs, psi, k), serial.find_relative(configs, psi, k))
    assert operator.find_relative(configs, psi * 0, 2).shape == (0, 27)
    assert operator.site.device.type == "cpu"
    if len(set(devices)) > 1:
        # A caller on a secondary GPU must reuse the term shards and receive
        # the reduction on its own stream and device.
        other = torch.device(devices[1])
        stream = torch.cuda.Stream(device=other)  # type: ignore[no-untyped-call]
        with torch.cuda.stream(stream):
            result = operator.apply_within(configs.to(other), psi.to(other), targets.to(other))
            torch.testing.assert_close(result, expected.flip(0).to(other))


def test_empty_operator_and_device_configuration() -> None:
    ctx = RuntimeContext(device=torch.device("cpu"), hamiltonian_term_batch_size=2)
    model = Mock(hamiltonian=Hamiltonian({}, kind="fermi"))
    ctx.configure_model_devices(model)
    assert isinstance(model.hamiltonian, TermHamiltonian)
    states = torch.zeros((1, 27), dtype=torch.uint8)
    psi = torch.ones(1, dtype=torch.complex128)
    assert model.hamiltonian.apply_within(states, psi, states).item() == 0
    assert model.hamiltonian.diagonal_term(states).item() == 0
    assert not len(model.hamiltonian.list_relative(states, psi)[0])
    with pytest.raises(ValueError, match="nonnegative"):
        model.hamiltonian.find_relative(states, psi, -1)
    with pytest.raises(ValueError, match="hamiltonian_term_batch_size"):
        RuntimeContext(hamiltonian_term_batch_size=-1)

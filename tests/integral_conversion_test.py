"""Check operator packing against OpenFermion and orbital imports against known integrals."""

import gzip
from pathlib import Path

import numpy as np
import openfermion
from openfermion.config import EQ_TOLERANCE  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.algorithms.orbit import _read_fcidump_tensors
from haar_q.models.pyscf import Model, ModelConfig


@pytest.mark.parametrize("pair_matrix", [False, True])
@pytest.mark.parametrize("scale", [0.43, EQ_TOLERANCE * 0.43, EQ_TOLERANCE])
def test_pyscf_packing_preserves_array_conventions_and_screening(pair_matrix: bool, scale: float) -> None:
    n = 3
    rng = np.random.default_rng(447)
    h1 = rng.integers(-15, 16, (n, n)) * scale
    if pair_matrix:
        pairs = n * (n + 1) // 2
        array = rng.integers(-15, 16, (pairs, pairs)) * scale
        # The historical reader takes pair-matrix values from the lower triangle.
        i, j, k, ell = np.indices((n,) * 4)
        ij = np.maximum(i, j) * (np.maximum(i, j) + 1) // 2 + np.minimum(i, j)
        kl = np.maximum(k, ell) * (np.maximum(k, ell) + 1) // 2 + np.minimum(k, ell)
        dense = array[np.maximum(ij, kl), np.minimum(ij, kl)]
    else:
        dense = array = rng.integers(-15, 16, (n,) * 4) * scale
    ham = Model(ModelConfig(h1, array, n, 3, -1)).hamiltonian
    actual = {
        tuple((int(s), int(k)) for s, k in zip(row, flags) if k != 2): complex(*coef.tolist())
        for row, flags, coef in zip(ham.site, ham.kind, ham.coef)
    }
    one = np.zeros((2 * n, 2 * n))
    two = np.zeros((2 * n,) * 4)
    for a in (0, 1):
        one[a::2, a::2] = h1
        for b in (0, 1):
            two[a::2, b::2, b::2, a::2] = dense.transpose(0, 2, 3, 1) / 2
    operator = openfermion.InteractionOperator(0.0, one, two)  # type: ignore[no-untyped-call]
    expected = openfermion.normal_ordered(openfermion.get_fermion_operator(operator)).terms  # type: ignore[no-untyped-call]
    assert actual.keys() == expected.keys()
    for term, value in expected.items():
        assert actual[term] == pytest.approx(value, rel=1e-14, abs=1e-20)


def test_orbital_import_uses_same_restricted_fcidump_format(tmp_path: Path) -> None:
    source = tmp_path / "orbit.FCIDUMP.gz"
    with gzip.open(source, "wt") as stream:
        stream.write("&FCI NORB=2, NELEC=1, MS2=-1,\n/\n0.7D0 2 1 1 1\n-0.2 2 1 0 0\n9 1 0 0 0\n2 0 0 0 0\n")
    metadata, core, one, two = _read_fcidump_tensors(source)
    assert metadata == (2, 1, -1) and core == 2.0
    torch.testing.assert_close(one, torch.tensor([[0, -0.2], [-0.2, 0]], dtype=torch.float64))
    assert torch.count_nonzero(two) == 4
    for axes in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
        assert two[axes].item() == pytest.approx(0.35)

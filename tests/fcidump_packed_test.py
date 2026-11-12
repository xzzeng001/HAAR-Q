"""Compare direct FCIDUMP packing with the dense OpenFermion convention."""

from pathlib import Path
import sys

import numpy as np
import openfermion
from openfermion.config import EQ_TOLERANCE  # type: ignore[import-untyped]
import pytest
import torch

from haar_q.models.fcidump import _load_integrals, _packed_terms, read_fcidump
from haar_q.models import fcidump


@pytest.mark.parametrize("scale", [1.0, EQ_TOLERANCE])
def test_direct_packing_preserves_normal_order_and_thresholds(tmp_path: Path, scale: float) -> None:
    n = 3
    rng = np.random.default_rng(81)
    orbital_pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    h1 = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    lines = [f"&FCI NORB={n},NELEC=3,MS2=-1,", "/"]
    for index, (i, j) in enumerate(orbital_pairs):
        for k, ell in orbital_pairs[: index + 1]:
            v = float(rng.integers(-15, 16)) * scale * 0.43
            lines.append(f"{v:.17e} {i + 1} {j + 1} {k + 1} {ell + 1}")
            for a, b in [(i, j), (j, i)]:
                for c, d in [(k, ell), (ell, k)]:
                    eri[a, b, c, d] = eri[c, d, a, b] = v
        v = float(rng.integers(-15, 16)) * scale * 0.43
        lines.append(f"{v:.17e} {i + 1} {j + 1} 0 0")
        h1[i, j] = h1[j, i] = v
    lines.extend(["0.125D0 0 0 0 0", ""])
    path = tmp_path / "FCIDUMP"
    path.write_text("\n".join(lines), encoding="ascii")
    header, terms = read_fcidump(path)
    assert header == (3, 3, -1)

    one = np.zeros((2 * n, 2 * n))
    two = np.zeros((2 * n,) * 4)
    for a in (0, 1):
        one[a::2, a::2] = h1
        for b in (0, 1):
            two[a::2, b::2, b::2, a::2] = eri.transpose(0, 2, 3, 1) / 2
    interaction = openfermion.InteractionOperator(0.125, one, two)  # type: ignore[no-untyped-call]
    expected = openfermion.normal_ordered(openfermion.get_fermion_operator(interaction)).terms  # type: ignore[no-untyped-call]
    assert terms.keys() == expected.keys()
    for key in expected:
        assert terms[key] == pytest.approx(expected[key], abs=1e-20, rel=1e-14)
    _, integrals = _load_integrals(path)
    assert integrals is not None
    site, kind, coeff = _packed_terms(integrals)
    packed = {
        tuple((int(s), int(k)) for s, k in zip(sites, kinds) if k != 2): float(value[0])
        for sites, kinds, value in zip(site, kind, coeff)
    }
    assert packed == terms


def test_cached_storage_is_private_and_preserves_v6_terms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating a loaded tensor must not alter the cache or another model."""
    path = tmp_path / "FCIDUMP"
    path.write_text("&FCI NORB=1,NELEC=2,MS2=0,\n/\n0.5 1 1 0 0\n0.25 1 1 1 1\n1.0 0 0 0 0\n")
    cache = tmp_path / "cache"
    monkeypatch.setattr(fcidump.platformdirs, "user_cache_path", lambda *args: cache)
    cold = fcidump.Model(fcidump.ModelConfig(path))
    (cache_file,) = cache.iterdir()
    assert cache_file.name.endswith("v6")
    expected_bytes = cache_file.read_bytes()
    expected_tensors = torch.load(cache_file, weights_only=True)
    warm = fcidump.Model(fcidump.ModelConfig(path))
    for model in (cold, warm):
        for actual, expected in zip(
            (model.hamiltonian.site, model.hamiltonian.kind, model.hamiltonian.coef), expected_tensors
        ):
            assert torch.equal(actual, expected)
        assert (model.n_qubits, model.n_electrons, model.n_spins) == (2, 2, 0)
    cold.hamiltonian.coef[0, 0] = 100
    warm.hamiltonian.coef[0, 0] = 200
    assert cache_file.read_bytes() == expected_bytes
    reloaded = fcidump.Model(fcidump.ModelConfig(path))
    assert torch.equal(reloaded.hamiltonian.coef, expected_tensors[2])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exposes mapped-file ownership in /proc")
def test_warm_model_uses_file_backed_private_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a cache should avoid keeping a second anonymous copy of it."""
    path = tmp_path / "FCIDUMP"
    path.write_text("&FCI NORB=1,NELEC=2,MS2=0,\n/\n0.5 1 1 0 0\n")
    cache = tmp_path / "cache"
    monkeypatch.setattr(fcidump.platformdirs, "user_cache_path", lambda *args: cache)
    fcidump.Model(fcidump.ModelConfig(path))
    model = fcidump.Model(fcidump.ModelConfig(path))
    (cache_file,) = cache.iterdir()
    address = model.hamiltonian.coef.data_ptr()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        lower, upper = (int(value, 16) for value in fields[0].split("-"))
        if lower <= address < upper:
            assert fields[1] == "rw-p"
            assert fields[-1] == str(cache_file)
            break
    else:
        pytest.fail("Hamiltonian coefficient storage has no file mapping")

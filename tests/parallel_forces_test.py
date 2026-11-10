"""Compare atom-sliced two-electron force contractions against explicit NumPy contractions."""

import numpy as np
import pytest
import torch

from haar_q.algorithms.forces import _TwoElectronGradient


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_two_body_density_and_derivatives_across_device_boundary(backend: str) -> None:
    if backend == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    devices = tuple(torch.device("cuda", i) if backend == "cuda" else torch.device("cpu") for i in range(2))
    rng = np.random.default_rng(2026)
    c, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    eri, dm2 = rng.normal(size=(2, 5, 5, 5, 5))
    split = _TwoElectronGradient(c, eri, dm2, devices)
    np.testing.assert_allclose(split.orbital_derivative, np.einsum("pjkl,qjkl->pq", eri, dm2), atol=1e-12)
    exact = np.einsum("ap,bq,cr,ds,pqrs->abcd", c, c, c, c, dm2, optimize=True)
    for p0, p1 in [(0, 2), (2, 4), (4, 5), (2, 2)]:
        derivative = rng.normal(size=(3, p1 - p0, 5, 5, 5))
        np.testing.assert_allclose(
            split.contract(derivative, p0, p1), np.einsum("xpqrs,pqrs->x", derivative, exact[p0:p1]), atol=1e-12
        )

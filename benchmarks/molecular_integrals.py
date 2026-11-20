"""Time restricted PySCF model construction from synthetic molecular-orbital integrals.

From each checkout, set PYTHONPATH=. and run this script by absolute path.
Timings include the complete model constructor and exclude SCF, integral
generation, extension compilation and the downstream SCI/VMC calculation.
"""

import argparse
import hashlib
import json
import statistics
import time

import numpy as np
from pyscf import ao2mo  # type: ignore[import-untyped]
import torch

import haar_q
from haar_q.hamiltonian import Hamiltonian
from haar_q.models.pyscf import Model, ModelConfig


def term_digest(hamiltonian: Hamiltonian) -> str:
    """Hash canonical rows so coefficient-sort ties and unused padding compare equally."""
    kinds = hamiltonian.kind.numpy()
    sites = np.where(kinds == 2, 0, hamiltonian.site.numpy())
    keys = np.column_stack((kinds, sites))
    order = np.lexsort(keys.T)
    checksum = hashlib.sha256()
    checksum.update(keys[order].astype("<i2", copy=False).tobytes())
    checksum.update(hamiltonian.coef.numpy()[order].astype("<f8", copy=False).tobytes())
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norb", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.norb <= 16384 or min(args.repeats, args.threads) < 1:
        parser.error("Use 1..16384 spatial orbitals and positive repeat/thread counts.")
    n = args.norb
    pairs = n * (n + 1) // 2
    rng = np.random.default_rng(3251)
    one = rng.standard_normal((n, n))
    eri = ao2mo.restore(1, rng.standard_normal(pairs * (pairs + 1) // 2), n)
    config = ModelConfig(one + one.T, eri, n, n, n % 2, 1.2)
    torch.set_num_threads(args.threads)
    Hamiltonian._load_module()
    warmup = Model(config)
    count = len(warmup.hamiltonian.site)
    digest = term_digest(warmup.hamiltonian)
    del warmup
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        model = Model(config)
        times.append(time.perf_counter() - start)
        del model
    print(
        json.dumps(
            {
                "version": haar_q.__version__,
                "source": haar_q.__file__,
                "spatial_orbitals": n,
                "seed": 3251,
                "threads": args.threads,
                "terms": count,
                "canonical_terms_sha256": digest,
                "times_seconds": times,
                "median_seconds": statistics.median(times),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

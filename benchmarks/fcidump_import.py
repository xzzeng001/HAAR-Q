"""Measure a single FCIDUMP import in a fresh process on Linux or WSL."""

import argparse
from pathlib import Path
import resource
from time import perf_counter

from haar_q.models.fcidump import Model, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--layout", choices=["standard", "mapped"], required=True)
    args = parser.parse_args()
    start = perf_counter()
    model = Model(ModelConfig(args.fcidump, cache_layout=args.layout))
    elapsed = perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{args.layout}: {len(model.hamiltonian.site):,} terms, {elapsed:.2f} s import, {rss:.1f} MiB peak RSS")


if __name__ == "__main__":
    main()

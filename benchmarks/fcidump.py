"""Measure a fresh process's FCIDUMP load and fully touched CPU storage.

Use PYTHONPATH to select the source checkout. Each invocation measures one load;
use separate cache roots for cold runs and reuse a root for warm runs. Cold/warm
refers to the application cache, not the operating system's file page cache.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time


def memory() -> dict[str, int]:
    """Return OS memory measurements; these are not allocator counters."""
    result = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.exists():
        keys = {"Rss": "rss_bytes", "Anonymous": "anonymous_bytes", "Pss_File": "file_pss_bytes"}
        for line in rollup.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in keys:
                result[keys[key]] = int(value.split()[0]) * 1024
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expect-cache", choices=("cold", "warm"), required=True)
    parser.add_argument("--prime-cache", action="store_true", help="Read packed-cache bytes before timing a warm load")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run under Linux/WSL for consistent RSS units and anonymous/file memory accounting.")
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ["XDG_CACHE_HOME"] = str(args.cache_root.resolve())
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import platformdirs
    import torch
    from haar_q.models import fcidump

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Reading the source digest outside the timed load gives both revisions the
    # same warm-source-page precondition and identifies the exact input/cache.
    with args.fcidump.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    cache_file = platformdirs.user_cache_path("haar-q", "HAAR-Q") / (source_sha256 + "v6")
    if cache_file.exists() != (args.expect_cache == "warm"):
        parser.error(f"Expected {args.expect_cache} application cache at {cache_file}")
    if args.prime_cache:
        if args.expect_cache != "warm":
            parser.error("--prime-cache requires --expect-cache warm")
        with cache_file.open("rb") as source:
            while source.read(8 * 1024 * 1024):
                pass

    before = memory()
    start = time.perf_counter()
    model = fcidump.Model(fcidump.ModelConfig(args.fcidump))
    load_seconds = time.perf_counter() - start
    after_load = memory()
    tensors = (model.hamiltonian.site, model.hamiltonian.kind, model.hamiltonian.coef)
    start = time.perf_counter()
    digest = hashlib.sha256()
    for tensor in tensors:
        # No clone or bytes conversion: hash every byte and fault in mapped pages.
        digest.update(memoryview(tensor.numpy()).cast("B"))
    touch_seconds = time.perf_counter() - start
    after_touch = memory()
    print(
        json.dumps(
            {
                "source_module": fcidump.__file__,
                "fcidump": str(args.fcidump.resolve()),
                "source_sha256": source_sha256,
                "cache_file": str(cache_file),
                "application_cache": args.expect_cache,
                "os_page_cache": (
                    "source and packed-cache pages primed before timing"
                    if args.prime_cache
                    else "source primed by digest; packed-cache pages uncontrolled"
                ),
                "torch_version": torch.__version__,
                "threads": args.threads,
                "interop_threads": 1,
                "qubits": model.n_qubits,
                "electrons": model.n_electrons,
                "terms": len(model.hamiltonian.site),
                "cpu_tensor_storage_bytes": sum(t.untyped_storage().nbytes() for t in tensors),
                "cpu_allocator_peak_bytes": None,
                "memory_note": "RSS/anonymous/file PSS are OS counters; PyTorch has no CPU allocator peak counter.",
                "load_seconds": load_seconds,
                "touch_and_sha256_seconds": touch_seconds,
                "load_plus_touch_seconds": load_seconds + touch_seconds,
                "packed_sha256": digest.hexdigest(),
                "before_load": before,
                "after_load": after_load,
                "after_touch": after_touch,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

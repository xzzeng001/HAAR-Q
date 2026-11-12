# Performance and memory

HAAR-Q 0.7.0 optimizes fermionic sign evaluation, FCIDUMP loading and the
saved activations used by projected VMC. Scientific calculations still use
the configured network precision and complex128 Hamiltonian amplitudes.

## Packed-bit Hamiltonian evaluation

Determinants already use one occupation bit per spin orbital. The P-cluster
active space of 146 spin orbitals fits into 19 bytes. To apply a fermionic
creation or annihilation operator, the kernel checks occupation, flips the
occupation bit and determines the sign from occupied lower orbitals.
The updated kernels reduce packed bytes with XOR and count set bits.
CPU code uses C++20 `std::popcount`, and CUDA uses `__popc`.
These integer instructions evaluate the parity exactly. NVIDIA documents
the population-count intrinsics in its [CUDA Math API](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html).

## FCIDUMP memory

Existing FCIDUMP caches can be read through a private memory mapping.
The operating system brings pages into physical memory when the code
accesses them. This reduces eager allocation during startup. A complete
CPU Hamiltonian scan still touches its coefficient and operator pages,
and each GPU still holds its own operator tensors.
PyTorch describes this behavior in its [memory-mapped loading recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html).

## Bounded-memory VMC

Set the maximum number of determinants evaluated with a live backward
graph using the unified input:

```yaml
options:
  method:
    vmc:
      hamiltonian_guided: true
      amplitude_batch_size: 256
```

The equivalent command is:

```bash
python -m haar_q -i input.yaml method=vmc options.method.vmc.amplitude_batch_size=256
```

The default value 0 uses the existing full-batch evaluation. With a positive
limit, VMC first evaluates amplitudes in chunks without retaining their
backward graphs. It calculates the full-basis norm, energy and Hamiltonian
product once, then recomputes each chunk to accumulate the exact projected
energy gradient before the optimizer update. The additional forward pass
trades computation for lower activation memory. The complete determinant,
amplitude and Hamiltonian-product vectors still exist.

The mathematical gradient is unchanged, but floating-point reduction order
can differ between full and batched evaluations. Checkpoints need not be
bitwise identical, and Adam can amplify rounding differences in very small
gradients. The regression suite compares energies, gradients and optimizer
updates with numerical tolerances rather than requiring identical bytes.

The batch limit applies to the process. When multiple GPUs are selected,
the neural network divides each chunk among them. Check both time and peak
memory for the chosen batch size on the target node.

## Reproduce the measurements

The repository includes scripts that emit JSON. Use Linux or WSL for the
FCIDUMP process-memory measurements. Run each revision in a separate process
and select its source checkout with `PYTHONPATH`. Reuse the same integral
file, CPU thread count and cache preconditions.

```bash
PYTHONPATH=. python benchmarks/fcidump.py /path/to/FCIDUMP --cache-root /tmp/haar-q-benchmark --expect-cache cold
PYTHONPATH=. python benchmarks/fcidump.py /path/to/FCIDUMP --cache-root /tmp/haar-q-benchmark --expect-cache warm --prime-cache
python benchmarks/vmc_memory.py --batch-size 255 --chunk-size 16 --network-width 64 --repeats 3 --threads 1
```

Use a fresh cache directory for the cold command. The warm command requires
the cache produced by the first run. The SHA-256 of every tensor byte verifies
the imported operator and forces all mapped pages to be read. The JSON
reports memory both before and after this full read.

The VMC benchmark uses an eight-site neural wavefunction and a deterministic
complex Hermitian cyclic Hamiltonian. It checks the energy, gradient and
updated parameters against the full-batch calculation. Its activation metric
counts unique tensor storage retained for backward, excluding persistent
parameters, buffers and input configurations. Timings include measurement
hooks. These are CPU measurements of this model and its memory tradeoff.

## Measured FCIDUMP loading

Measured on 2026-11-13 with an Intel Core i7-13700KF, Ubuntu under WSL,
PyTorch 2.9.1+cpu and one CPU thread. The reference is v0.6.0, commit
`aff1a70`. The real P-cluster input contains 73 spatial orbitals,
114 electrons and 41,528,584 packed operator terms. Its SHA-256 is
`b8f31dfc6429feceb9e7bc1d6ca62c1c0326eb0cd536978d8543bc8040fded94`.

For three paired warm-cache runs on the WSL ext4 filesystem, with packed
cache pages primed before each run:

| Measurement | v0.6.0 | v0.7.0 |
| --- | ---: | ---: |
| Median model construction | 0.993 s | 0.410 s |
| Median construction plus reading and hashing all tensor bytes | 1.504 s | 0.912 s |
| Anonymous process memory after the full read | 1.355 GiB | 0.272 GiB |
| Resident process memory after the full read | About 1.468 GiB | About 1.468 GiB |

The median construction-plus-read time improves by 1.65 times in this test.
The mapped operator replaces about 1.08 GiB of anonymous RAM with reclaimable
file-backed pages. Both revisions produce identical packed tensor bytes.
These timings cover loading and full-byte hashing, excluding Hamiltonian
application and network training.

Filesystem placement matters. A single warm-cache comparison on the mounted
Windows J drive measured 5.73 s and 5.78 s for the same construction-plus-read
operation. First-time conversion also retains the freshly constructed tensors
and needs sorting workspace. The individual cold-conversion runs took
64.51 s for v0.6.0 and 80.95 s for v0.7.0 on that mounted drive, with peak
RSS of 2.727 and 2.700 GiB respectively. These single cold runs do not
demonstrate a first-conversion speedup.

## Measured P-cluster Hamiltonian kernels

Using the same P-cluster integral data, all 41,528,584 terms were applied
to two or eight valid 114-electron determinants, with an eight-determinant
result basis. The benchmark extracts each revision's production CPU or
CUDA kernel verbatim into a standalone harness. Terms are streamed in
chunks of 262,144, alternating the revision evaluated first in each chunk.
The following values are medians from three paired runs:

| Device | Source determinants | v0.6.0 | v0.7.0 | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Intel Core i7-13700KF, one thread | 2 | 5.16749 s | 1.72402 s | 3.00x |
| Intel Core i7-13700KF, one thread | 8 | 19.7691 s | 5.71378 s | 3.46x |
| NVIDIA GTX 1650 | 2 | 0.213649 s | 0.0621288 s | 3.44x |
| NVIDIA GTX 1650 | 8 | 0.747899 s | 0.192070 s | 3.89x |

These timings cover only `apply_within`, excluding compilation, disk reads,
CPU/GPU transfers and basis sorting. They do not measure an entire SCI/VMC
calculation or predict large-basis or multi-GPU scaling. CPU amplitudes
matched exactly. Across the GPU repetitions the maximum absolute amplitude
difference was 1.02e-10, consistent with changing floating-point atomic
summation order. No coefficient or wavefunction precision was reduced.

The CUDA test used NVCC 12.6 and actual GPU execution independently of
the installed CPU-only PyTorch. The complete CUDA translation unit also
compiled successfully. A separate randomized operator check covered
1,572,864 applications at 8, 146 and 257 sites with no parity mismatches.
End-to-end CUDA-enabled PyTorch and multi-GPU execution still require
validation on the target compute node.

To reproduce the kernel comparison on Linux/WSL, create a checkout of
v0.6.0 and use the P-cluster cache generated by the FCIDUMP benchmark:

```bash
python benchmarks/native_parity.py --baseline-checkout /path/to/v060 --cache-file /path/to/cache/haar-q/INPUT_HASHv6 --device cpu --repeats 3 --output-dir /tmp/haar-q-native-cpu
python benchmarks/native_parity.py --baseline-checkout /path/to/v060 --cache-file /path/to/cache/haar-q/INPUT_HASHv6 --device cuda --cuda-arch sm_75 --repeats 3 --output-dir /tmp/haar-q-native-cuda
```

Replace `INPUT_HASH` with the input SHA-256 above. Select the architecture
for the actual GPU; `sm_75` applies to the GTX 1650 used here. A C++20
compiler is required; pass `--compiler /path/to/g++` or `/path/to/nvcc`
if it is not on `PATH`. This benchmark intentionally expects the documented
P-cluster cache dimensions. It checks amplitudes before reporting timings
and retains generated source and raw runs in the output directory.

## Measured VMC activation tradeoff

On the same CPU, using the VMC benchmark defaults of 255 configurations,
two hidden layers of width 64, float64 network parameters and three SGD
updates, the full and batched calculations gave the following measurements:

| Measurement | Full batch | Batch of 16 |
| --- | ---: | ---: |
| Peak unique saved activation storage | 4,653,240 bytes | 295,792 bytes |
| Wall time per update, including measurement hooks | 4.20 ms | 35.10 ms |

The saved activation peak drops by 15.73 times, about 94 percent. The
maximum final parameter difference is 1.39e-17 and gradient difference
is 1.11e-16. Repeated small forward/backward launches make this small
CPU example slower. The option defaults to 0 so each job can opt into
the memory tradeoff explicitly. This experiment measures saved activation
storage, and the complete process also holds weights, optimizer state,
Hamiltonian data and selected-space vectors.

A separate real P-cluster CLI check used eight selected determinants and
one Adam update with chunks of three. The full-batch v0.7.0 parameters
matched v0.6.0 exactly. Full and chunked energies differed by 3.64e-12 Eh,
gradients by at most 4.69e-11 and updated parameters by at most 4.35e-9.
The residual subtracts two large terms, `H psi - E psi`; an independent
fixed-residual control reduced the gradient discrepancy to 1.05e-14.
Adam amplified the remaining tiny-gradient rounding. This is a correctness
check of a small selected space, not a converged P-cluster calculation.

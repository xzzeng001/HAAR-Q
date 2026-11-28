# Large molecular active spaces and multi-reference SCI

[中文说明](large-systems.zh-CN.md)

## Benzene with 30 electrons in 108 spatial orbitals

This example uses the published frozen-core benzene Hamiltonian in optimized
SHCI orbitals. Its fixed spin projection gives 15 alpha and 15 beta electrons.
A determinant uses 216 occupation bits (27 bytes). Its size still permits
directly storing a selected basis, while the full sector contains
`binom(108,15)^2` determinants. The samples, operator and size of the selected
space determine the practical cost. [Benzene benchmark](https://doi.org/10.1021/acs.jpclett.0c02621)
and [integral provenance](https://github.com/seunghoonlee89/SI-benzene-paper-DMRG/tree/1d40f216317563d64971785cd274e08ef3710fe1).

From the repository root in PowerShell, download the 334,144,528-byte input
once. Its SHA-256 is `8ac25808d1cac9ab15edb79c4e69c6cd0aef585d412094541c7cf136d0886b3e`.
The Git LFS pointer in a source-only checkout is 137 bytes.

```powershell
curl.exe -fL https://media.githubusercontent.com/media/seunghoonlee89/SI-benzene-paper-DMRG/1d40f216317563d64971785cd274e08ef3710fe1/FCIDUMP_optSHCIorbs -o examples/benzene108/FCIDUMP_optSHCIorbs
Get-FileHash examples/benzene108/FCIDUMP_optSHCIorbs -Algorithm SHA256
python -m haar_q -i examples/benzene108/input.yaml
```

The input requests one small HAAR-SCI cycle, starting with two network samples
and a closed-shell trial occupation. The trial fills the first 15 orbitals in
file order. The reported Ritz energy describes the selected space. Increase
`sampling_control.seed_count`, `options.method.haar.krylov_extend_count`,
`options.method.haar.krylov_iteration` and the number of cycles for convergence
studies. Record the selected dimension and energy across independent random seeds.
A small residual inside a selected space does not bound its missing correlation energy.

## Host memory during FCIDUMP import

With `options.system.fcidump.cache_layout: mapped`, the reader stores the
restricted spatial integrals in eightfold-packed form. With
`q = n(n+1)/2` spatial pairs, the ERI array requires `8 q(q+1)/2` bytes
(about 132 MiB for 108 orbitals). Spin expansion writes each block once,
in generation order, into three native columns. Each accepted operator
term costs 28 bytes on disk. A completed cache is published atomically
and opened through private file mappings. Later runs reuse it by SHA-256.
Use `XDG_CACHE_HOME` on Linux/WSL to select a fast local disk before launch.
Budget disk for the complete expanded operator, including simultaneous cold
builders if launching multiple jobs. OS page cache, Python, PyTorch and solver
workspace also consume memory.

The default `standard` layout keeps the existing coefficient-sorted cache.
Both layouts apply the same integral threshold. Term order can change the
winner of equal-weight selection ties and the rounding of sums.

## Device memory and parallel reduction

Set `common.hamiltonian_term_batch_size` to a positive term count. On a
single node with `P` selected GPUs and `T` terms, each GPU gets a disjoint
range of at most `ceil(T/P)` terms. Ranges fitting the cap `B` remain
resident. Larger ranges are copied from the host in blocks on each application.
The packed operator columns per GPU occupy at most `28 min(B,ceil(T/P))`
bytes. Source and target determinants, network replicas, temporary reduction
buffers, allocator reservations and the primary solver are additional.
With the default `B=0` multiple GPUs divide the source determinants and
hold a complete operator copy each.

```powershell
python -m haar_q -i examples/benzene108/input.yaml common.device=cuda 'common.gpu_devices=[0,1]' common.hamiltonian_term_batch_size=1000000
```

Each shard computes its contribution to the projected product and diagonal.
For selection, each block keeps its largest `K` individual path weights.
A path outside a block's top `K` cannot outrank the global cutoff. Keep
the global top `K`, then deduplicate the destinations. For external amplitudes
sum signed or complex contributions to the same determinant, including across
devices. This preserves cancellation. Scanning still costs work proportional
to the number of source determinants times the number of terms. Too small
a block repeats transfers, kernel launches and target setup. The full set of
external determinants in PT2 can also become a separate memory bottleneck.

## Multi-reference sampling for transition metals

In `method=haar` and `method=excited`, set `options.method.haar.initial_references`
on a molecular `u1u1` job. Write one character per spatial orbital in FCIDUMP
or active-orbital order. Use `0` for empty, `a` for alpha, `b` for beta and `2` for doubly occupied.
Quote occupation strings in YAML. Each determinant must match both electron
counts. The following example puts two electrons in two orbitals.

```yaml
initial_references:
  - {occupation: 'ab', coefficient: 1.0}
  - {occupation: 'ba', coefficient: -1.0}
```

On the first fresh cycle the solver takes the union of explicit references
and network samples. It rescales each group's amplitudes by its largest
magnitude. A supplied trial coefficient takes precedence on an overlap.
The subsequent Hamiltonian expansion and selected-space solve can change the
relative signs and magnitudes. Later cycles draw from the fitted network
and previous pool. Resuming uses the checkpoint. These references guide the
initial subspace and fix the spin projection. Assigning a total spin requires
additional evidence such as an independent spin expectation.

Use the [Cr₂ case study](../examples/cr2/README.md) to reproduce a four-orbital
FCI control and a 12-orbital valence calculation at the same bond length. Comparing
SCI and FCI only has a physical meaning for the same orbitals, frozen-core
constant, spin sector and geometry. Increasing the selected basis and enlarging
the molecular active space address different sources of error.

## Measurements with the released inputs

| Check | Result |
| --- | --- |
| Benzene, published FCIDUMP | 105,195,501 packed terms, 2,945,474,028 bytes of columns. One CPU cycle with seed 2333 grew 3 initial determinants to 7, gave a projected Ritz energy of −230.7180803044 Eh, and saved the fitted network. This small space tests the workflow. Convergence to the benzene benchmark requires a much larger study. |
| Independent benzene matrix check | One trial doubly occupies orbitals 1–15. The other moves the pair from 15 to 108. The maximum error of their 2×2 projected matrix and of their diagonals was 5.69×10⁻¹⁴ Eh against the original spatial integrals on CPU. The nonzero pair coupling tests a matrix element across the packed determinant. |
| Benzene on one physical GTX 1650 (4 GiB) | The native CUDA matrix-vector kernel streamed all 105,195,501 terms in 106 blocks for three 30-electron determinants. The maximum absolute difference from the unblocked CPU product was 5.72×10⁻¹³ Eh (rounded). This check used a standalone CUDA 12.6 launcher. The full Python SCI cycle above ran on CPU. |
| Nitrogenase P-cluster | For the 41,528,584-term operator and 8 saved determinants, the full and blocked CPU products differed by at most 5.43×10⁻²⁰ Eh in amplitude for the same input vector. Both selected 13 unique destinations from the top 17 paths. One million terms per block reduces the largest operator payload from 1,162,800,352 to 28,000,000 bytes. |
| Cold import, separate 42-orbital Cr₂ FCIDUMP | 525,635 terms. The standard layout used 0.78 s and 417.4 MiB process peak RSS. The mapped layout used 0.59 s and 387.1 MiB. Medians of three fresh processes per layout on Ubuntu/WSL, one CPU thread, alternating layouts and fresh caches on the Linux filesystem. RSS includes the scientific Python runtime. |

From a source checkout with HAAR-Q installed, run the independent matrix check
and import comparison as follows. Use an empty cache for each cold import.
Multi-GPU throughput and total peak GPU memory require measurements on a multi-GPU node.

```bash
python benchmarks/benzene108.py examples/benzene108/FCIDUMP_optSHCIorbs
XDG_CACHE_HOME=$(mktemp -d) OMP_NUM_THREADS=1 python benchmarks/fcidump_import.py /path/to/FCIDUMP --layout standard
XDG_CACHE_HOME=$(mktemp -d) OMP_NUM_THREADS=1 python benchmarks/fcidump_import.py /path/to/FCIDUMP --layout mapped
```

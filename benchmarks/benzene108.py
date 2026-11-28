"""Check selected matrix elements of the published benzene Hamiltonian against spatial integrals."""

import argparse
import gzip
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from haar_q.models.fcidump import Model, ModelConfig
from haar_q.utility.term_hamiltonian import TermHamiltonian


def reference_matrix(path: Path, occupied: list[set[int]]) -> np.ndarray:
    """Closed-shell Slater rules for two determinants differing by a pair excitation.

    The input stores one representative of each permutational integral class.
    Repeated entries overwrite earlier entries as they do in a restricted FCIDUMP.
    Orbital indices here follow the one-based numbering of the text file.
    """
    (old,) = occupied[0] - occupied[1]
    (new,) = occupied[1] - occupied[0]
    pair = {old, new}
    one: dict[int, float] = {}
    coulomb: dict[tuple[int, int], float] = {}
    exchange: dict[tuple[int, int], float] = {}
    core = coupling = 0.0
    with gzip.open(path, "rt", encoding="ascii") if path.suffix == ".gz" else path.open(encoding="ascii") as stream:
        for line in stream:
            if "&END" in line.upper() or "/" in line:
                break
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            i, j, k, ell = map(int, fields[1:])
            if i == j == k == ell == 0:
                core = float(fields[0].upper().replace("D", "E"))
            elif i == j and k == ell == 0:
                one[i] = float(fields[0].upper().replace("D", "E"))
            elif i == j and k == ell and k:
                coulomb[min(i, k), max(i, k)] = float(fields[0].upper().replace("D", "E"))
            elif i != j and ((i == k and j == ell) or (i == ell and j == k)):
                value = float(fields[0].upper().replace("D", "E"))
                exchange[min(i, j), max(i, j)] = value
                if {i, j} == pair:
                    coupling = value

    def diagonal(orbitals: set[int]) -> float:
        energy = core + 2 * sum(one.get(i, 0.0) for i in orbitals)
        for i in orbitals:
            energy += coulomb.get((i, i), 0.0)
            for j in orbitals:
                if i < j:
                    energy += 4 * coulomb.get((i, j), 0.0) - 2 * exchange.get((i, j), 0.0)
        return energy

    return np.array([[diagonal(occupied[0]), coupling], [coupling, diagonal(occupied[1])]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump", type=Path)
    parser.add_argument("--term-batch-size", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    start = perf_counter()
    model = Model(ModelConfig(args.fcidump, cache_layout="mapped"))
    if (model.n_qubits, model.n_electrons, model.n_spins) != (216, 30, 0):
        raise ValueError("Expected the 108-orbital benzene FCIDUMP with 15 alpha and 15 beta electrons.")
    print(f"Model load {perf_counter() - start:.2f} s; {len(model.hamiltonian.site):,} packed terms", flush=True)
    occupied = [set(range(1, 16)), set(range(1, 15)) | {108}]
    expected = reference_matrix(args.fcidump, occupied)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    operator = TermHamiltonian(model.hamiltonian, (device,), args.term_batch_size)
    bits = np.zeros((2, 108, 2), dtype=np.uint8)
    for row, occ in enumerate(occupied):
        bits[row, np.array(sorted(occ)) - 1, :] = 1
    configs = torch.from_numpy(np.packbits(bits.reshape(2, -1), axis=1, bitorder="little")).to(device)
    columns = [operator.apply_within(configs, v, configs) for v in torch.eye(2, dtype=torch.complex128, device=device)]
    actual = torch.stack(columns, dim=1).cpu().numpy()
    diag = operator.diagonal_term(configs).cpu().numpy()
    print("Spatial-integral reference (Eh):\n", expected)
    for name, result, reference in (("matrix", actual, expected), ("diagonal", diag, expected.diagonal())):
        print(f"Max {name} error {np.max(np.abs(result - reference)):.3e} Eh")
        np.testing.assert_allclose(result, reference, rtol=0, atol=1e-8)


if __name__ == "__main__":
    main()

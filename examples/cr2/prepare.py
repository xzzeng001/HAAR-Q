"""Generate the two fixed-orbital chromium dimer FCIDUMPs used by the examples."""

import argparse
import math
from pathlib import Path

import numpy as np
from pyscf import fci, gto, lib, mcscf, scf, symm  # type: ignore[import-untyped]
from pyscf.mcscf import avas  # type: ignore[import-untyped]
from pyscf.tools import fcidump  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", choices=("delta", "valence"), default="delta")
    parser.add_argument("--output", type=Path, help="Output FCIDUMP. By default write beside this script.")
    parser.add_argument("--fci", action="store_true", help="Also diagonalize the 12-orbital model with PySCF.")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    lib.num_threads(args.threads)
    mol = gto.M(atom="Cr 0 0 -0.84; Cr 0 0 0.84", unit="Angstrom", basis="sto-3g", spin=0, symmetry="Dooh", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    if not mf.converged:
        raise RuntimeError("Cr2 RHF failed to converge.")

    if args.space == "delta":
        # Explicit labels keep both components of each degenerate delta pair.
        names = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff))
        delta = ["E2gx", "E2gy", "E2ux", "E2uy"]
        if any(names.count(label) != 1 for label in delta):
            raise ValueError("Expected exactly one MO for each delta symmetry label in STO-3G.")
        indices = [names.index(label) for label in delta]
        if list(mf.mo_occ[indices]) != [2, 2, 0, 0]:
            raise ValueError("The RHF occupations differ from the delta-space reference.")
        n = nel = 4
        cas = mcscf.CASCI(mf, n, nel)
        mo = cas.sort_mo(indices, base=0)
    else:
        n, nel, mo = avas.avas(mf, ["Cr 3d", "Cr 4s"], threshold=0.5, verbose=0)
        if (n, nel) != (12, 12):
            raise ValueError(f"Expected Cr 3d/4s AVAS(12e,12o), got ({nel}e,{n}o). Check the mean-field solution.")
        cas = mcscf.CASCI(mf, n, int(nel))

    # Choose real orbital phases using the first substantial AO coefficient.
    # Equivalent sign choices can change CI coefficient signs, but not eigenvalues.
    mo = np.array(mo, copy=True)
    pivot = np.argmax(np.abs(mo) > 1e-5, axis=0)
    mo *= np.sign(mo[pivot, np.arange(mo.shape[1])])
    h1, ecore = cas.get_h1eff(mo)
    eri = cas.get_h2eff(mo)
    out = args.output or Path(__file__).with_name(f"Cr2-{args.space}.FCIDUMP")
    fcidump.from_integrals(str(out), h1, eri, n, int(nel), nuc=ecore, ms=0, tol=1e-14)
    print(f"{out} | ({int(nel)}e,{n}o), M_S=0 | {math.comb(n, int(nel) // 2) ** 2:,} determinants")
    print(f"RHF = {mf.e_tot:.12f} Eh | frozen-core and nuclear offset = {ecore:.12f} Eh")
    if args.space == "delta" or args.fci:
        solver = fci.direct_spin1.FCI()
        solver.conv_tol = 1e-11
        # Closely spaced Cr2 spin states require a larger Davidson subspace.
        solver.max_space = 48
        solver.max_cycle = 500
        ci0 = None
        if args.space == "valence":
            # Sample all spin and spatial symmetries in the fixed M_S sector.
            # A single initial determinant can converge to a higher symmetry root.
            dim = math.comb(n, int(nel) // 2) ** 2
            ci0 = np.random.default_rng(2333).standard_normal(dim)
        energy, ci = solver.kernel(h1, eri, n, int(nel), ci0=ci0, ecore=ecore)
        if not solver.converged:
            raise RuntimeError("The independent PySCF FCI reference did not converge.")
        print(f"PySCF FCI = {energy:.12f} Eh | <S^2> = {solver.spin_square(ci, n, int(nel))[0]:.8f}")


if __name__ == "__main__":
    main()

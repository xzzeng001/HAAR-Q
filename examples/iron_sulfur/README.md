# Coupled iron sites in the Fe₂S₂ dimer and Fe₄S₄ cubane

These restricted active-space Hamiltonians from Li and Chan connect two formal
Fe(III) sites in the dimer and two Fe(III) plus two Fe(II) sites in the cubane.
Their localized orbitals make the distribution and coupling of the unpaired
electrons part of the calculation. [Original study](https://doi.org/10.1021/acs.jctc.7b00270)

| HAAR-Q input | Active electrons, spatial orbitals | α, β electrons | Determinants in the chosen spin-projection sector | Authors' singlet active-space DMRG energy (Eh) |
| --- | --- | --- | ---: | ---: |
| [Fe2S2.yaml](Fe2S2.yaml) | 30e, 20o | 15, 15 | 240,374,016 | −116.6056091 (bond dimension 8,000) |
| [Fe4S4.yaml](Fe4S4.yaml) | 54e, 36o | 27, 27 | 8,862,957,169,158,400 | −327.2396369 (bond dimension 4,000) |

The authors identify the dimer reference as exact to the reported digits.
For the cubane they explicitly report incomplete convergence with bond dimension.
Both FCIDUMPs have a zero scalar constant. HAAR-Q's `E_RITZ` and these
two literature references use the same active-space energy convention.
To obtain the authors' total-energy convention, add their core offset of
−4976.26532397 Eh for the dimer or −8105.56038966 Eh for the cubane once.
`files.ref_energy` provides a comparison value and does not shift the Hamiltonian.
[Authors' reference data and energy offsets](https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters/blob/9a08b9de0af9a28b70fffaf9c88318a28ef256b8/README.md)

## Running from the checkout

```powershell
python -m haar_q -i examples/iron_sulfur/Fe2S2.yaml
python -m haar_q -i examples/iron_sulfur/Fe4S4.yaml
```

The integrals are included. Each input starts a CPU calculation with six neural
samples and a pair of spin-reversed trial determinants. In one-based FCIDUMP
order, the dimer trial singly occupies orbitals 2–6 with α and 13–17 with β.
The cubane trial uses 3–11 for α and 24–25 plus 27–33 for β. All other spatial
orbitals are doubly occupied. These are initial guesses for opposing local spin
patterns. The calculation fixes `MS2=0`. Determining the total spin of a
selected-space state requires an independent spin analysis. Increase the
network seed count, expansion count and cycles to study convergence against
the table. Repeat with different seeds and record the actual basis size.

The following single-cycle CPU runs used the shipped inputs and seed 2333.
As an independent matrix-element check, we read the original uncompressed
spatial integrals with PySCF and summed the one-body, Coulomb and same-spin
exchange contributions for the first trial determinant. The last column
compares that value with the HAAR-Q diagonal using the bundled gzip file.

| System | Packed terms | Initial → selected determinants | HAAR-Q `E_RITZ` (Eh) | First trial diagonal (Eh) | Absolute diagonal difference (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fe₂S₂ | 232,892 | 8 → 40 | −116.0373194339 | −115.9546700846 | 1.14 × 10⁻¹³ |
| Fe₄S₄ | 2,472,752 | 8 → 40 | −326.4281248294 | −326.2938406288 | 7.96 × 10⁻¹³ |

## 铁硫簇算例

这两份真实积分可直接用于上面的 PowerShell 命令。二核模型包含两个形式上的
Fe(III)，四核立方烷模型包含两个 Fe(III) 和两个 Fe(II)。它们分别保留 20 和
36 个空间轨道，在给定自旋投影下的行列式数分别约为 2.40 × 10⁸ 和 8.86 × 10¹⁵。
两个输入都选取自旋相反的两个试探行列式，再加入六个网络样本，执行一轮扩展、
求解和网络拟合。该初态允许反向局域自旋参与首轮选态。总自旋仍需单独核验。

第一张表给出原作者公开的单重态活性空间 DMRG 参考能量，单位为 Hartree。作者把二核
体系的参考值视为在所列精度内已收敛，四核体系在键维数 4,000 时仍未收敛。
两个积分的常数项均为零，可直接比较 `OSZICAR` 中的 `E_RITZ` 与表中数值。
若需换算到作者采用的总能标度，二核、四核的活性空间能量分别加上
−4976.26532397 和 −8105.56038966 Eh。将参考值写入 `files.ref_energy`
只影响比较输出。扩大采样、扩展数与循环数后，应同时报告实际行列式数和能量。
第二张表是 HAAR-Q 的单周期 CPU 验证。两个模型均从 8 个行列式扩展到 40 个，
对应的 Ritz 能量仍分别高于所列 DMRG 值约 0.5683 和 0.8115 Eh。用 PySCF 独立读取
原始积分并计算首个试探行列式的对角元，与 HAAR-Q 的差均小于 8 × 10⁻¹³ Eh。

## Integral data and attribution

The two gzip files are unmodified copies from the USTC Knowledge Computing Lab
QMB Models dataset, revision `ab1a7002ccbaca0ac43acafaf0f40400d14c3940`,
distributed there under GPL-3.0. The uncompressed bytes match the FCIDUMPs
in Zhendong Li's public archive at revision `9a08b9de0af9a28b70fffaf9c88318a28ef256b8`.
两份压缩数据沿用 QMB Models 标注的 GPL-3.0 许可证，解压后的字节与 Li 和 Chan
公开的原始积分逐一核验相同。[QMB Models data and license](https://huggingface.co/datasets/USTC-KnowledgeComputingLab/qmb-models/tree/ab1a7002ccbaca0ac43acafaf0f40400d14c3940) · [Original Hamiltonians](https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters/tree/9a08b9de0af9a28b70fffaf9c88318a28ef256b8/Fe2S2_and_Fe4S4)

| File | SHA-256, gzip | SHA-256, uncompressed FCIDUMP |
| --- | --- | --- |
| Fe2S2_30_40.FCIDUMP.gz | `821d868434ff42d400394fec066dbde52d7d8baef49324be0bbc28bf96b5044b` | `95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9` |
| Fe4S4_54_72.FCIDUMP.gz | `9a35a23bea86c00dd2ce9b644d75f1c7745b87cda92a029bad3b46f715a5326b` | `0b840102a4894fee4648944f2bf3bb5d85828deb4b7e71b964136bbc4c9e01ce` |

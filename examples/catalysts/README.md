# FeMoco and ruthenium catalyst XVIII · FeMoco 与钌催化剂 XVIII

Each YAML runs one CPU HAAR-SCI cycle on the supplied compressed FCIDUMP. The
files contain one scalar offset each, including nuclear and frozen-orbital contributions.
HAAR-Q includes that constant in its energies. `MS2` specifies the electron-number
difference `Nα − Nβ`.

| Input | Model | Electrons / spatial orbitals | Nα, Nβ | Scalar offset (Eh) |
| --- | --- | --- | --- | ---: |
| [femoco54.yaml](femoco54.yaml) | FeMoco model, charge +3 | 54/54 | 27, 27 | −13212.970326 |
| [femoco57.yaml](femoco57.yaml) | FeMoco model, charge 0 | 65/57 | 34, 31 | −13164.048069 |
| [ru-xviii.yaml](ru-xviii.yaml) | Ru catalytic intermediate XVIII | 64/56 | 32, 32 | −7180.18924584 |

Reiher and co-workers prepared the two nitrogenase cofactor Hamiltonians with different
geometries, electron counts and orbital optimizations. Their energies describe different
models. These inputs are useful for comparing the computational cost of even- and
odd-electron sectors. Assigning a resting-state structure or spin to the biological
cofactor requires further physical evidence. [Reiher et al., PNAS (2017)](https://doi.org/10.1073/pnas.1619152114)

Intermediate XVIII belongs to a ruthenium-catalyzed CO₂-to-methanol cycle. In the
published `XVIII-cas4-fb-64e56o` model, a CASSCF(4,4) calculation supplied the
orbital basis and the Hamiltonian retains 64 electrons in 56 orbitals. A single
intermediate suffices to test electronic energies. A reaction energy requires
additional structures on a consistent footing. [von Burg et al., Phys. Rev. Research (2021)](https://doi.org/10.1103/PhysRevResearch.3.033055)

## Run from the repository root

```powershell
python -m haar_q -i examples/catalysts/femoco54.yaml
python -m haar_q -i examples/catalysts/femoco57.yaml
python -m haar_q -i examples/catalysts/ru-xviii.yaml
```

Each input starts from a determinant that fills the first `Nα` and `Nβ` orbitals
of the file for the two spin species, plus two network samples. This gives a
reproducible initial guess in the supplied orbital order. A small number of
Hamiltonian expansions, Krylov iterations and one fit step exercise the full
workflow. Read `E_RITZ` in `OSZICAR` for the Ritz energy and `E` for the fitted
network energy. A ground-state study must check convergence in the determinant
space and in the Krylov solve.

### CPU checks · CPU 数值核验

| Input | Packed operator terms | Trial determinant diagonal (Eh) | One-cycle `E_RITZ` (Eh), 11 selected determinants |
| --- | ---: | ---: | ---: |
| FeMoco 54/54 | 12,602,031 | −13481.6573881860 | −13481.6573855565 |
| FeMoco 65/57 | 15,655,568 | −13482.6028994449 | −13482.6030074866 |
| Ru XVIII 64/56 | 14,580,401 | −7475.3136525044 | −7475.3136587593 |

For each trial determinant we also read the original spatial integrals with
PySCF and evaluated its Slater–Condon diagonal from the occupied orbitals.
All three HAAR-Q diagonals agreed within 3×10⁻¹¹ Eh. The short Krylov
solve only spans part of the 11-dimensional space, so its Ritz estimate
can lie slightly above the diagonal of a determinant in the selected basis.

## Integral provenance

The gzip archives decompress byte-for-byte to these upstream files. Their
SHA-256 values apply to the **uncompressed** FCIDUMPs. The FeMoco data use
standard chemists' integral ordering. QDK's maintainers document an index-order
error in an older Microsoft 54-orbital download. Use the pinned files when
comparing results. [Upstream FeMoco correction and model details](https://github.com/microsoft/qdk-chemistry-data/blob/eaadc022cc3bf521f1a66ab1f6f9001bfdd7f510/data/hamiltonians/FeMoco-Reiher2017/README.md) · [Upstream Ru model details](https://github.com/microsoft/qdk-chemistry-data/blob/eaadc022cc3bf521f1a66ab1f6f9001bfdd7f510/data/hamiltonians/RuCatalyst-vonBurg2021/README.md) · [Bundled upstream license](LICENSE-qdk.txt)

| Local archive | Upstream FCIDUMP | SHA-256 after decompression |
| --- | --- | --- |
| `femoco54.FCIDUMP.gz` | [`FeMoco_S0_charge+3_orbsRAS24.16_targetCAS54.54_test3.fcidump`](https://github.com/microsoft/qdk-chemistry-data/blob/eaadc022cc3bf521f1a66ab1f6f9001bfdd7f510/data/hamiltonians/FeMoco-Reiher2017/FeMoco_S0_charge%2B3_orbsRAS24.16_targetCAS54.54_test3.fcidump) | `84f3a34d61cb1552797cc2574159ec4b6efa641c380631fb4f66be654a34940e` |
| `femoco57.FCIDUMP.gz` | [`FeMoco_S1.5_charge0_orbsRAS21.12_targetCAS65.57_test7_b.fcidump`](https://github.com/microsoft/qdk-chemistry-data/blob/eaadc022cc3bf521f1a66ab1f6f9001bfdd7f510/data/hamiltonians/FeMoco-Reiher2017/FeMoco_S1.5_charge0_orbsRAS21.12_targetCAS65.57_test7_b.fcidump) | `c3759d0bdc71801dc54f5f6536dd8b2d5fec16c6e0e57ae3dc22e4ef7a996073` |
| `ru-xviii.FCIDUMP.gz` | [`XVIII-cas4-fb-64e-56o.fcidump`](https://github.com/microsoft/qdk-chemistry-data/blob/eaadc022cc3bf521f1a66ab1f6f9001bfdd7f510/data/hamiltonians/RuCatalyst-vonBurg2021/XVIII-cas4-fb-64e-56o.fcidump) | `bdd81245ad09f93031e1fa7f0e7f5328b7f220b9b1c27a8688b7941837584be8` |

## 中文说明

以上三份输入均随附可直接读取的积分。两种 FeMoco 模型分别有 54 个电子、54 个轨道和
65 个电子、57 个轨道。后者含 34 个 α 电子和 31 个 β 电子，适合检验奇电子自旋扇区。
两者的电荷、几何和轨道均有区别，直接相减所得的能量不能解释为同一几何上的自旋能隙。
钌算例采用 CO₂ 转化为甲醇过程中的中间体 XVIII。轨道来自 CASSCF(4,4)，
实际送入 HAAR-Q 的哈密顿量是 64 电子、56 轨道。

在 PowerShell 执行上方命令即可运行。按文件轨道顺序填入 α、β 电子构造试探态，
再加入两个网络样本，经过哈密顿量扩展、Krylov 求解和网络拟合。输出已包含上表中的
常数能量，复算时不要重复相加。比较选中空间结果请读取 `OSZICAR` 的 `E_RITZ`。
这三份输入的默认规模用于核验计算路径。评估基态能量时，还需分别检查选中维数和
Krylov 迭代数的收敛。上表给出的 SHA-256 可用来确认轨道顺序、积分和能量零点。
数值核验表同时列出了单个试探行列式的对角元与一轮 Ritz 能量。用 PySCF 读取原始
积分并按 Slater–Condon 公式独立求得的三个对角元均与 HAAR-Q 相差不到 3×10⁻¹¹ Eh。

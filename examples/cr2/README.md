# Chromium dimer · 铬二聚体

Neutral Cr₂ at 1.68 Å provides a small multireference test before moving to metal clusters. Both supplied Hamiltonians use the same real, nonrelativistic STO-3G RHF orbitals. The FCIDUMPs include the nuclear repulsion and frozen-core offset. They can be used directly without rerunning the PySCF orbital preparation.

| Input | Active orbitals | Electrons (α, β) | Determinants at Mₛ = 0 |
| --- | --- | --- | ---: |
| `input.yaml` | Two bonding and two antibonding 3d δ orbitals | 2, 2 | 36 |
| `valence.yaml` | AVAS projection onto Cr 3d and 4s on both atoms | 6, 6 | 853,776 |

### Running from PowerShell

From the repository root, run the two selected-CI examples or the full-sector check.

```powershell
python -m haar_q -i examples/cr2/input.yaml
python -m haar_q -i examples/cr2/valence.yaml
python -m haar_q -i examples/cr2/input.yaml method=excited sampling_control.seed_count=36 options.method.haar.krylov_extend_count=0 common.max_relative_step=1
```

In `OSZICAR`, compare the selected-space energy `E_RITZ` with FCI. The column `E` is the energy of the network fitted to that cycle. For `method=excited`, `ROOTS` lists the selected-space eigenvalues and residuals. Set `common.device=cuda:0` to use one GPU. For a single node with two visible GPUs use `common.device=cuda 'common.gpu_devices=[0,1]'`. These small calculations default to CPU.

### Reference determinants and measured energies

The δ orbital order is E2gx, E2gy, E2ux, E2uy. The valence orbitals use the order written by the AVAS generator. Each character in a reference string refers to one spatial orbital in FCIDUMP order. The characters `0`, `a`, `b`, and `2` denote empty, α, β, and double occupancy. The signed starting coefficients in the YAML are trial guesses. In the first cycle, HAAR-Q combines these determinants with neural samples, merges duplicates, and expands the basis through Hamiltonian couplings. Revisit the guesses and orbital phases when supplying integrals in a different orbital basis.

For the bundled δ Hamiltonian, independent PySCF FCI gives −2064.147434813173 Eh. The full-sector HAAR-Q check reproduced that value, with a ground-root projected residual of 1.21 × 10⁻¹² Eh. The following CPU results use seed 2333, two SCI cycles and eight nominal initial slots. A neural sample overlapped a reference in the first row.

| First-cycle seeds | Cycle 1 basis / Ritz energy (Eh) | Cycle 2 basis / Ritz energy (Eh) | Final error (mEh) |
| --- | --- | --- | ---: |
| 4 neural + 4 explicit | 11 / −2064.1289626140 | 11 / −2064.1452304928 | 2.2043 |
| 8 neural | 12 / −2064.0900404095 | 17 / −2064.1438003178 | 3.6345 |

Repeating the pair gives final errors of 18.4451 versus 17.9956 mEh at seed 2334, and 15.7249 versus 16.6393 mEh at seed 2335 (explicit references first). With this sampling budget, the ordering and magnitude of the errors vary with seed.

To reproduce the second row, run

```powershell
python -m haar_q -i examples/cr2/input.yaml 'options.method.haar.initial_references=[]' sampling_control.seed_count=8
```

The valence example used 32 neural samples and four explicit references. The second cycle reached −2064.0718487256 Eh with 89 selected determinants. With 36 neural samples and the same seed 2333, the second cycle reached −2064.1450176761 Eh with 104 determinants. Here the neural-only run gave a lower energy at the cost of a larger basis. Independent PySCF FCI gives −2064.382695372985 Eh in this Mₛ = 0 sector. The two short runs therefore miss 310.847 and 237.678 mEh, respectively. The FCI root has ⟨S²⟩ = 2 (triplet) in this STO-3G active-space model. A generic Davidson starting vector reaches this root across spin and spatial symmetries. A binding curve or dissociation energy would also require consistent atomic or dissociation-limit calculations, a larger orbital basis and additional correlation. The Cr₂ benchmark study and the AVAS paper describe the physical problem and the orbital construction. [Larsson et al., JACS (2022)](https://doi.org/10.1021/jacs.2c06357) · [Sayfutyarova et al., JCTC (2017)](https://doi.org/10.1021/acs.jctc.7b00128)

### Regenerating the Hamiltonians

With PySCF installed, write fresh copies to the current directory. The first command always runs an independent FCI check. Adding `--fci` to the second command requests a larger Davidson calculation. Different BLAS builds and thread counts can choose a different ordering within degenerate AVAS orbital pairs. Use the supplied FCIDUMPs to reproduce the determinant-level comparison above.

```powershell
python examples/cr2/prepare.py --space delta --output Cr2-delta.FCIDUMP
python examples/cr2/prepare.py --space valence --threads 2 --output Cr2-valence.FCIDUMP
```

### 中文说明

两个算例采用中性 Cr₂，核间距为 1.68 Å。脚本先做 STO-3G 基组的非相对论 RHF 计算，再固定轨道构造小空间哈密顿量，能量已包含核排斥与冻结电子的贡献。上表第一行只保留两对成键与反键 δ 轨道，用于逐行列式核验。第二行通过 AVAS 投影选出两个 Cr 原子的 3d 和 4s 轨道，在固定的自旋投影下共有 853,776 个行列式。仓库已附两份积分文件，可直接在 PowerShell 运行前述命令。

参考串按 FCIDUMP 的空间轨道排序，`0`、`a`、`b`、`2` 依次表示空轨道、单占 α、单占 β、双占。YAML 中的带符号系数是初始试探值。首轮把显式参考组态与神经网络样本合并去重，再用哈密顿量扩展。换用新轨道时，需根据新轨道的顺序、相位和化学特征重新检查参考组态及系数。

核验时读取 `OSZICAR` 的 `E_RITZ`，`E` 记录拟合后的网络能量。在 δ 空间的全部 36 个行列式上，`ROOTS` 输出的基态能量与独立 PySCF FCI 一致。采用相同的八个首轮名额和随机种子 2333，两轮后，四个显式参考加四个网络样本的误差为 2.2043 mEh，八个网络样本的误差为 3.6345 mEh。换成种子 2334，误差分别为 18.4451 与 17.9956 mEh。种子 2335 下分别为 15.7249 与 16.6393 mEh。较低的能量取决于初始采样。

价空间示例在种子 2333 下，32 个网络样本加四个显式参考在两轮后得到 89 个行列式，Ritz 能量为 −2064.0718487256 Eh。同一随机种子下改用 36 个网络样本，基组扩大至 104 个行列式，能量降至 −2064.1450176761 Eh。独立 PySCF FCI 给出的 Mₛ = 0 扇区最低能量为 −2064.382695372985 Eh，这两次截断计算的误差分别为 310.847 和 237.678 mEh。该 FCI 根的 ⟨S²⟩ = 2，为本 STO-3G 活性空间模型的三重态。生成脚本为 Davidson 算法提供在各自旋和空间对称子空间均有分量的初始向量。研究真实 Cr₂ 的解离能还需要更大的基组、更多电子相关，并对分子与解离极限采用一致的处理。上面的 PySCF 命令可重新生成积分。对价空间增加 `--fci` 可独立对角化。重新生成时 BLAS 和线程数可能改变简并轨道的顺序，复现上述采样结果请使用随仓库提供的固定积分文件。

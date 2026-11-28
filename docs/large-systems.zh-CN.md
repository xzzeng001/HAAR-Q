# 大活性空间与过渡金属多参考 SCI

[English guide](large-systems.md)

## 用苯挑战的 108 个空间轨道检验程序

苯挑战的冻结核模型含 30 个活性电子和 108 个空间轨道。本例采用作者公开的
SHCI 优化轨道积分。在指定自旋投影下，α、β 电子各 15 个，每个行列式占用
216 位（27 字节），完整行列式空间有 `binom(108,15)^2` 个基矢。
HAAR-Q 只保存选中空间的行列式。实际计算量取决于算符项数、采样数和选中空间大小。
[苯挑战论文](https://doi.org/10.1021/acs.jpclett.0c02621) · [积分来源及固定版本](https://github.com/seunghoonlee89/SI-benzene-paper-DMRG/tree/1d40f216317563d64971785cd274e08ef3710fe1)

在仓库根目录的 PowerShell 中下载一次，随后即可使用同一个输入修改计算参数。
完整文件为 334,144,528 字节，SHA-256 应为
`8ac25808d1cac9ab15edb79c4e69c6cd0aef585d412094541c7cf136d0886b3e`。
仅克隆 Git 而未拉取 LFS 时，得到的是 137 字节的指针文件。

```powershell
curl.exe -fL https://media.githubusercontent.com/media/seunghoonlee89/SI-benzene-paper-DMRG/1d40f216317563d64971785cd274e08ef3710fe1/FCIDUMP_optSHCIorbs -o examples/benzene108/FCIDUMP_optSHCIorbs
Get-FileHash examples/benzene108/FCIDUMP_optSHCIorbs -Algorithm SHA256
python -m haar_q -i examples/benzene108/input.yaml
```

例子默认从两个网络样本和一个按文件顺序占据前 15 个轨道的闭壳层试探态出发，
完成一个小规模 HAAR-SCI 周期。该周期给出选中空间的 Ritz 能量。研究能量收敛时，
可逐步增加 `sampling_control.seed_count`、`options.method.haar.krylov_extend_count`、
`options.method.haar.krylov_iteration` 和外循环数，并用多个随机种子记录能量与实际
选中维数。投影空间内的残差只衡量该空间内的求解精度，空间外遗漏的相关能需要另行评估。

## 首次读取 FCIDUMP 时怎样节省内存

设置 `options.system.fcidump.cache_layout: mapped` 后，先按实空间轨道积分的八重
置换对称性存储受限轨道双电子积分。设 `q = n(n+1)/2`，双电子积分数组占
`8 q(q+1)/2` 字节，在 108 个空间轨道时约为 132 MiB。随后将自旋展开的各块
直接写入三个文件，每个保留的算符项占 28 字节。整份缓存成功写完后，通过原子
重命名发布，再以私有内存映射打开。相同文件通过 SHA-256 命中缓存，可跨作业复用。
Linux/WSL 可在启动前设置 `XDG_CACHE_HOME` 到本地快速磁盘。磁盘需容纳完整展开，
并发首次构建时还需考虑各进程的临时文件。系统页缓存、Python、PyTorch 和求解工作区另计。

默认的 `standard` 缓存按系数模排序，两种布局的积分筛选阈值相同。累加顺序会影响
浮点舍入，截断处路径权重相等时也可能选中不同的行列式。

## 如何分配多卡上的算符

将 `common.hamiltonian_term_batch_size` 设置为正整数。假设共有 `T` 项算符和
同一节点的 `P` 张 GPU，每张卡分到最多 `ceil(T/P)` 项。分到的区间能放进上限
`B` 时，常驻该卡。更大的区间在每次作用哈密顿量时分批从主机传输。每张卡上的
算符列占用上限为 `28 min(B,ceil(T/P))` 字节。源与目标行列式、网络副本、归约
临时量、分配器预留和主卡求解工作区需要额外显存。默认 `B=0` 时，多卡按源行列式
分工，每张卡各保存一份完整算符。

```powershell
python -m haar_q -i examples/benzene108/input.yaml common.device=cuda 'common.gpu_devices=[0,1]' common.hamiltonian_term_batch_size=1000000
```

计算矩阵向量积和对角元时，对各块的贡献求和。选态时，各块先保留权重最大的
`K` 条路径，因为块内已排在前 `K` 以外的路径无法超过全局截断值。合并各块的
全局前 `K` 条路径之后，再按目标行列式去重。对 PT2 等需要的外部空间振幅，
按目标相加复振幅，保留不同路径间的相消。扫描开销仍随算符项数与源行列式数的
乘积增长。块太小会增加传输、内核启动及目标索引的重复开销。PT2 中完整外部空间
本身也可能占用较多内存。

## 如何为过渡金属选择多个初始参考态

对采用 `u1u1` 对称性的分子，在 `method=haar` 或 `method=excited` 下填写
`options.method.haar.initial_references`。按积分文件或活性轨道的顺序，每个空间轨道
写一个字符。`0`、`a`、`b`、`2` 分别表示空、仅 α、仅 β 和双占据。请在 YAML
中给占据串加引号，每个参考态须满足体系的两种电子数。以下是两个轨道上两个电子的例子。

```yaml
initial_references:
  - {occupation: 'ab', coefficient: 1.0}
  - {occupation: 'ba', coefficient: -1.0}
```

新作业首轮取显式参考态与网络采样的并集，两组振幅分别除以本组最大模。
两者重合时采用显式给定的试探系数。哈密顿量扩展及选中空间对角化会重新确定相对
权重和符号，后续周期继续从已拟合网络和上轮池中抽样。重启时使用检查点中的状态。
这些占据约束自旋投影，总自旋归属还需要自旋期望值等独立依据。

可按 [Cr₂ 算例的中英文说明](../examples/cr2/README.md) 先运行四轨道 FCI 对照，
再在相同键长考察 12 轨道价层活性空间。比较 SCI 与 FCI 时，应采用相同的轨道、
冻结核常数、自旋扇区和几何。扩大选中空间检验 SCI 截断误差，扩大活性空间则改变
了有效哈密顿量，两项检查回答不同的问题。

## 实测结果

| 检查 | 结果 |
| --- | --- |
| 苯挑战原始积分 | 实际展开为 105,195,501 项，三列共 2,945,474,028 字节。CPU 上用随机种子 2333 运行一轮，从 3 个初始行列式扩展到 7 个，Ritz 能量为 −230.7180803044 Eh，并完成网络更新与检查点保存。该小空间用于检验计算流程。逼近苯挑战基准能量仍需系统扩大选中空间。 |
| 苯的独立矩阵元检查 | 一个行列式双占据第 1 至 15 轨道，另一个将第 15 轨道的一对电子移到第 108 轨道。从原始空间轨道积分直接计算的 2×2 矩阵和对角元，与 CPU 算符的最大误差均低于 5.69×10⁻¹⁴ Eh。非零的轨道对跃迁还能检验跨行列式字节的耦合。 |
| 单张物理 GTX 1650（4 GiB） | 原生 CUDA 矩阵向量积内核对三个 30 电子行列式计算，将全部 105,195,501 项分为 106 块。与不分块 CPU 结果相比，复数分量的最大绝对差低于 5.73×10⁻¹³ Eh。这项检查使用独立的 CUDA 12.6 启动程序，上述完整 Python SCI 周期运行于 CPU。 |
| 固氮酶 P-cluster | 在相同的 8 个已保存行列式及输入振幅上，41,528,584 项整算符与分块 CPU 结果的逐分量最大差为 5.43×10⁻²⁰ Eh 以内。权重最大的 17 条路径均给出相同的 13 个目标。每块上限 100 万项时，单块算符数据上限由 1,162,800,352 降到 28,000,000 字节。 |
| 另一个 42 轨道的 Cr₂ 文件首次导入 | 525,635 项。标准布局耗时 0.78 秒，进程峰值 RSS 为 417.4 MiB。映射布局为 0.59 秒、387.1 MiB。Ubuntu/WSL 单 CPU 线程，交替顺序，每种布局以空的 Linux 文件系统缓存目录启动三个独立进程，表中列中位数。RSS 包含 Python 科学计算环境。 |

在已安装 HAAR-Q 的源码目录中可用以下命令复核。每次测首次导入时应提供新的
空缓存目录。多卡吞吐率和整卡峰值显存还需在实际多卡节点测量。

```bash
python benchmarks/benzene108.py examples/benzene108/FCIDUMP_optSHCIorbs
XDG_CACHE_HOME=$(mktemp -d) OMP_NUM_THREADS=1 python benchmarks/fcidump_import.py /path/to/FCIDUMP --layout standard
XDG_CACHE_HOME=$(mktemp -d) OMP_NUM_THREADS=1 python benchmarks/fcidump_import.py /path/to/FCIDUMP --layout mapped
```

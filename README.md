# 在 Tesla P100 上跑 Qwen3.8-Flash-Next（Strata 引擎）

**一句话**：Strata 的官方引擎包**根本没有 `sm_60` 内核**，所以 Tesla P100 开箱即用会直接报
`this engine has no code for … sm_60`。本仓库记录了从源码编译 Pascal 内核、并把它从
**32 tok/s 调到 68.1 tok/s**（代码场景、256K 上下文）的全过程，以及一批"看起来该有用、实测却没用"的参数。

> English version: [README.en.md](README.en.md)
> 完整调优手册（每一项的前后对比）：[docs/TUNING.md](docs/TUNING.md)

---

## 最大的一条：`--pipeline-windows 4`（+23~26%）

这一项值得单独写在最前面，因为**它在 CUDA 上默认关闭**——帮助里写的是
`on by default for 2+ stages on AMD gfx12`，也就是只有 AMD 白送。我在 NVIDIA 上测了三十多种
配置（线程数、投机、KV、分块、量化）都没碰它，一开就是 +23~26%。

原理：两层以上 layer split 时，stage 0 跑下一个 verify window 的同时，stage 1 验证当前窗口——
**两张卡真正并行**，而不是一级等一级。

| `--pipeline-windows` | 0（默认） | 2 | 3 | **4** | 6 | 8 |
|---|---|---|---|---|---|---|
| 解码 tok/s | 54.7 | 67.3 | 68.8 | **69.3** | 69.1 | 67.6 |
| 读题 tok/s | 191 | — | — | **255** | — | — |

已用 `temperature=0` 对比验证：开与关的输出**逐字一致**，不是拿正确性换速度。

> **通用经验**：读 `--help` 时专门搜 `default on`、`off by default`、`on by default for`、`opt-in`。
> 别的平台默认开启的选项，往往就是本平台最便宜的增量。

---

## 预编译引擎（不想自己编的话）

本文用到的引擎二进制放在 **Releases** 页面：
`strata-0.1.54-sm60-linux-x86_64.tar.gz` —— 243 MB，CUDA 12.9.86 / gcc-13，
目标架构 `60;61;70;86`，sha256 `b2f4ff55b5a76d042bbcb7d04e167768f5ae55f9095087c546704d86af08bbd0`。

它以 **Release 附件**形式发布，**不能提交进 git**：GitHub 单文件上限 100 MB。

安装说明在包内的 `INSTALL.md` 和 [release/README.md](release/README.md)。一句话版本：
把包里的 `strata` 覆盖到官方 CUDA 12 引擎组件目录里（**保留该目录自带的
`libcublas.so.12` / `libcudart.so.12`**），然后用 `strata serve --profile …` 启动——
**不要用 `strata launch`**，它会把引擎按签名 feed 校验后悄悄换回官方版。

---

## 一、测试机

| | |
|---|---|
| 显卡 | 2 × Tesla P100-PCIE-16GB（GP100，`sm_60`），250 W 上限，各接 PCIe 3.0 **x4** |
| CPU | Intel i5-12600KF（6P+4E，16 线程） |
| 内存 | 31.2 GiB |
| 硬盘 | WD_BLACK SN750 SE 500 GB（无 DRAM 缓存） |
| 系统 | Debian 13 (trixie)，内核 6.12，驱动 580.196.07，CUDA 12.9 |

模型：Qwen3.8-Flash-Next **IQ2_XS**（无审查版，24,576 专家，激活 3B）
引擎：Strata 0.2.8 / build 0.1.54（自编译 sm_60）

---

## 二、最终成绩

所有数字都取自引擎自己的日志（`done: N tokens in T s`），temperature 0，同一台机器。

| 项目 | 配置 |
|---|---|
| 上下文 | **262144（256K），单槽** |
| `--pipeline-windows` | **4** |
| `--spec` / `--spec-min-p` / `--suffix-draft` | **4 / 0.65 / 0** |
| `--kv` / `--kv-resident` | **int8 / 32768** |
| `--prefill` | `auto` |
| `--mtp` | Lamina 草稿器 |
| dense 权重 | Q8_0 + `STRATA_Q8_SM60=1` |

**代码生成：68.1 tok/s**（600 token，8 次连续测量的均值，8/8 次都 ≥60）；
**读题：255 tok/s**（随机文本，排除缓存）。
开箱默认配置只有 32–37 tok/s。

配置模板：[`configs/abl-iq2xs-p100.json`](configs/abl-iq2xs-p100.json)

### 速度取决于"你让它写什么"——必须说清场景

| 任务类型 | 关流水线 | 开流水线 | 提升 |
|---|---|---|---|
| 重复型代码（红黑树） | 49.6 | **62.3** | **+26%** |
| 多样型代码（三种数据结构） | 37.6 | **46.3** | +23% |
| 散文解释（B+树 vs LSM） | 36.2 | 36.4 | **+0.6%** |

重叠靠投机验证窗口：内容越可预测，草稿接受率越高、窗口越长，两张卡才有活可叠。
**所以"68 tok/s"是代码场景的数字，写散文仍然是 ~36 tok/s。**
任何不带场景说明的 tok/s 都没有意义。

---

## 三、四项真正有效的优化（按收益排序）

| # | 改动 | 收益 | 原理 |
|---|---|---|---|
| 1 | `--pipeline-windows 4` | **+23~26%** | 两卡流水重叠（CUDA 默认关） |
| 2 | `--spec-min-p 0.65` + `--suffix-draft 0` | +4~5% | 草稿接受率 76%→85%，且不截短窗口 |
| 3 | `--kv-resident 32768` | +2.6% | KV 流到锁定内存，腾 1.35 GiB 给专家缓存（5908→6928 槽） |
| 4 | Q8_0 dense + `STRATA_Q8_SM60=1` | +7% | GP100 是唯一没有 `__dp4a` 的 Pascal 芯片 |

### 3.1 接受率不是越高越好

| `--spec-min-p` | 0.3 | 0.5 | 0.6 | **0.65** | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| 接受率 | 68.9% | 76.3% | 81.5% | **84.1%** | 84.9% | 88.5% | **96.2%** |
| 解码 | 55.0 | 56.0 | 57.5 | **58.7** | 58.1 | 58.1 | **56.9** |

min-p 0.9 时接受率高达 96%，速度反而最低——因为高阈值会把草稿窗口提前截断，
每次产出的 token 变少。**要优化的是"每次产出多少 token"，不是接受率。**

### 3.2 `--kv-resident` 用 KV 延迟换专家缓存

| 配置 | 专家缓存 | 解码 |
|---|---|---|
| `--kv int8` | 5908 槽 / 7.79 GiB | 54.7 |
| `--kv q4_0` | 6442 / 8.50 | 56.5 |
| `--kv k8v4` | 6174 / 8.14 | 56.0 |
| **`--kv int8 --kv-resident 32768`** | **6928 / 9.14** | **56.1–57.4** |

注意边际递减：缓存槽 +17%，解码只 +2.6%——新增的槽装的是**最不常用**的专家。

### 3.3 上下文：256K 不再是陷阱

早期（128K 时代）测出 256K 会掉 12%，因为 KV 挤压专家缓存。加上 `--kv-resident` 之后
KV 大部分搬进锁定内存，256K 只掉约 2%，**可以放心开满 256K**。

---

## 四、试过但**没用**的（都有数字）

| 改动 | 结果 |
|---|---|
| `--spec 6` / `--spec 8` | 56.9 / −11% |
| `--lookup-chain 4` | 持平 |
| `--expert-cache-per-layer` | −15%（命中率 96→90%） |
| 16 线程（含 SMT）替代 10 物理核 | **−41%** |
| `--prefill auto:16384/32768` | 持平（245–255） |
| `--prefill-pipe 1` | 持平（255 vs 255，**流水线开启后复测仍无效**） |
| `--pool-tasks 512/2048` | 持平 |
| `STRATA_PREFILL_CPU_SHARE=0/0.3` | 持平 |
| `--prefill-head-alias` | 启动失败 |
| `--aux-cpus auto` / `--remote-expert-opt` | 54.6 / 54.2 vs 54.7 |
| `--native-qsa` / `--no-fused-gr` / `--sync-every-layer` / 关 vision | 均在噪声内 |
| `--expert-cache-device1 7500`（手动） | 启动失败，`auto` 已填满 |
| CPU governor → performance、禁 C6/C8/C10 | **0%**，见下 |
| 加第三张卡（RTX 3050 8 GB） | −3%（它成了最慢的一级） |
| 加内存 | **0%**（瓶颈是显存不是内存） |

### CPU 已经满频，没有降频可抢

解码时：P 核 **4.4–4.5 GHz**、E 核 **3.4 GHz**（标称 4.9 / 3.6），CPU 封装功耗
**67 W / 150 W 上限**。改成 performance governor、EPP performance、禁用深层 C-state
之后速度 56.0 vs 56.1——**毫无变化**。10 个物理核是在接近满频干真实的活。

---

## 五、剩下的 21% GPU 空闲说明什么

开启流水线后实测：**GPU 70–79%**（之前只有 50–60%），**CPU 10 个物理核全满**。

GPU 那 21% 空闲是在等 CPU 算专家——只有约 45% 的专家装得进 32 GB 显存，
其余每 token 都要在 10 个核上实时算。

| 下一步 | 预期 | 代价 |
|---|---|---|
| **加显存**（如 24G 卡） | **+30~50%**（专家常驻，GPU 与 CPU 同时解放） | 换卡 |
| 换 8P 核 CPU | +15~25% | 换平台 |
| 加内存 | **0%** | — |
| 继续调参 | **0%**（约 45 种配置已穷尽） | — |

---

## 六、测量方法论（两个必须避开的坑）

1. **测读题必须每次换 prompt。** 重复发同一个 prompt 会全命中缓存，
   日志显示 `1667 reused + 5 read in 94 ms`——那是缓存速度不是 prefill 速度。
   要用随机生成的文本，并且只拿 `read` 的 token 数和 `ms` 算速度。
2. **重启后的第一次请求是冷的**，专家缓存与提示缓存都空，会低 15–25%。先预热再测。
3. 单次测量波动 ±2 tok/s，**"提升 3%"这种必须交替多次取均值**才可信。

---

## 七、仓库结构

```
configs/    最终调优的 profile（abl-iq2xs-p100.json）、示例 profile、systemd 单元、DSH provider
scripts/    01 装驱动、02 编译引擎、03 转 Q8 dense、04 下模型
scripts/fans/    P100 风扇曲线守护进程（按显卡温度调主板 PWM）
scripts/stress/  CUDA 烤机 + 整机烤机
docs/TUNING.md   完整调优手册（每一项前后对比、失败清单、方法论）
docs/RESULTS.md  早期（Q2_0 / 128K）的原始测量数据
```

## 八、复现步骤

```bash
# 1. 驱动（580 是最后一个还支持 Pascal 的分支）
scripts/01-install-driver.sh          # 屏蔽 nouveau 后重启
scripts/01-install-driver.sh --install

# 2. 从源码编译引擎（需要 CUDA 12.9 + gcc-13；脚本内含 glibc 2.41 补丁）
scripts/02-build-engine.sh

# 3. 下模型 + 用 Strata 自己的 setup 生成专家包
scripts/04-download-model.sh

# 4. 可选，但值 7%
scripts/03-convert-q8-dense.sh <shard-00001.gguf>

# 5. 跑起来
cp configs/abl-iq2xs-p100.json ~/profiles/abl.json   # 改里面的路径
strata serve --profile ~/profiles/abl.json --host 0.0.0.0 --port 8080
```

## 九、两个最耗时间的坑

**1. `strata launch` 在 Pascal 上不可用。**
它会拿签名 feed 校验引擎，发现你换过就**悄悄重新下载官方包覆盖**（那个包没有 sm_60）。
必须用 `strata serve --profile`，这条路径不校验引擎。

**2. CUDA 12.9 与 glibc 2.41 打架。**
`bits/mathcalls.h` 里 `cospi/sinpi/tanpi` 用了 `__THROW`，cudafe++ 判定与 CUDA 头里的
`noexcept(true)` 不兼容。网上常见的 `-U_GNU_SOURCE` 会连带搞坏 `pthread_*` 和 `fwide`，
正确做法是只删掉那三行声明（脚本里已处理）。

## 十、已知边界

- **单请求延迟的上限由显存带宽决定**，加卡加的是容量和吞吐，不是延迟。
- 本仓库**不包含模型权重**（遵循 Qwen 许可，需自行下载）。

## 十一、许可

本仓库代码 MIT。Strata 与 llama.cpp 均为 MIT。模型权重遵循 Qwen 许可，不在此分发。

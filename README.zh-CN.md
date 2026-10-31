# 在 Tesla P100 上跑 Qwen3.8-Flash-Next（Strata 引擎）

**一句话**：Strata 的官方引擎包**根本没有 `sm_60` 内核**，所以 Tesla P100 开箱即用会直接报
`this engine has no code for … sm_60`。本仓库记录了从源码编译 Pascal 内核、并把它调到
**59.8 tok/s**（代码场景、128K 上下文）的全过程，以及一批"看起来该有用、实测却没用"的参数。

> English version: [README.md](README.md)

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

模型：`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的 **Q2_0**（66.4 GB，24,576 专家，激活 3B）
引擎：Strata 0.2.8 / build 0.1.54（自编译）

---

## 二、最终成绩

所有解码数字都取自引擎自己的日志（`done: N tokens in T s`），temperature 0，同一台机器。

| 项目 | 配置 |
|---|---|
| 上下文 | **131072（128K），单槽** |
| `--kv` | `int8` |
| `--prefill` | `8192` |
| `--spec` | `4`（MTP 投机解码） |
| dense 权重 | **Q8_0** + `STRATA_Q8_SM60=1`（GP100 专用内核） |
| 视觉 | CPU 读图（`--vision`） |

**代码生成：59.8 tok/s**（600 token）；开机默认配置下只有 32–37 tok/s。

### 关键认知：速度取决于"你让它写什么"

| prompt 类型 | 草稿接受率 | 解码速度 |
|---|---|---|
| 代码（600 token） | 420/500 = **84 %** | 53.9 / 56.5 / 56.9 |
| 散文（"什么是 MoE"） | 50/98 = **51 %** | 36.9 / 38.6 |

同一个模型、同一份权重，**代码比散文快 50%**。因为 MTP 投机解码只有在草稿猜得准时才划算——
代码的 token 高度可预测（缩进、括号、变量名重复），散文每个词都是独立的。

**所以任何不带场景说明的 tok/s 都没有意义。**

---

## 三、三条实测曲线

### 3.1 上下文长度 ↔ 专家缓存 ↔ 速度

每一份 KV 缓存都是从一个专家缓存槽里抢来的。同一 prompt 实测：

| 上下文 | 专家缓存 | 解码 | 命中率 |
|---|---|---|---|
| 8K | 8826 槽 | 53.8 | 95.5–99 % |
| 32K | 8431 | 53.9–56.3 | 94.7–98.8 % |
| 64K | 7907 | 53.4–55.7 | 93.5–98.1 % |
| 128K | 6859 | 52.8–54.8 | 90.7–96.9 % |
| **256K** | **4759** | **45.0–49.6** | **69.1–89.6 %** |

约 **711 槽 / GiB**。曲线到 64K 是平的，128K 掉 3%，**256K 断崖式掉 12%**（命中率从 99% 崩到 69%）。

> 32 GB 显存的卡上，256K 上下文是个陷阱。

### 3.2 槽位 × 上下文（双并发 vs 单并发）

| 上下文 | 槽位 | 专家缓存 | 解码 |
|---|---|---|---|
| 64K | 2 | 4406 | 53.0–57.5 |
| **128K** | **2** | 3353 | **48.3** ← 最差 |
| 64K | 1 | 5213 | **60.7** |
| 128K | 1 | 4861 | **59.8** ← 性价比最高 |

**128K + 双槽比单槽慢 24%**：两份 KV 缓存对专家缓存的挤压，远超"多服务一个人"的价值。
自己用的话，单槽 128K 是最优解。

### 3.3 Q8 dense 内核（唯一真正有效的引擎级优化）

GP100 是唯一**没有 `__dp4a`** 的 Pascal 芯片，它读 dense 权重的默认内核只有 **40–80 GB/s**
（显存带宽是 730）。Strata 为此专门写了个 GP100 内核，但要求 dense 矩阵**和输出头**都是 Q8_0——
而 ISTA-DASLab 的量化包把 dense 放在 Q5_K / Q3_K / IQ4_XS。

用 `tools/q8_dense_gguf.py` 转一下（专家偏移不变，包可以复用）。同一 prompt 实测：

| dense 格式 | 多次运行 | 平均 |
|---|---|---|
| 原始（Q5_K/Q3_K） | 50.7 / 52.5 / 52.9 | **52.0** |
| Q8_0 + `STRATA_Q8_SM60=1` | 55.9 / 57.5 / 55.6 | **55.8** |

**+7%**，两轮独立测量都复现。代价：+39 GB 磁盘，输出头打包占 644 MiB，专家缓存从 7907 掉到 4406 槽
（内核提速勉强盖过缓存损失）。

---

## 四、试过但**没用**的（都有数字）

| 改动 | 结果 | 原因 |
|---|---|---|
| 加第三张卡（RTX 3050 8 GB） | **38.9 vs 42.0** | 流水线多一级；8 GB 扣掉 dense 副本 + KV + 4.24 GiB prompt 缓冲后只剩约 4% 的专家，它成了最慢的一级 |
| `--kv q4_0` 替代 `int8` | 39.1 vs 42.0 | 4-bit KV 每步要做 Hadamard 旋转，在 P100 上比省下的显存更贵 |
| `--pipeline-windows 2` | 54.1 vs 55.1 | 双卡 NVIDIA 上无收益 |
| `--pcie-frac 0` | 40.5 vs 41.3 | 引擎自动选的 0.09（对应 3.3 GB/s 链路）已经是对的 |
| `--spec 6` / `--spec 8` | 49.7 / — vs 55.8 | 草稿加深的成本超过多命中的收益 |
| DFlash2 草稿器 | 55.3 / 54.8 / 56.6 vs 53.9 / 56.5 / 56.9 | 一样快；Strata 自己的表也写了 Ampere/Pascal 上是负收益 |
| `STRATA_STAGE_PIN=1` | 39.5 vs 41.8 | 文档里那 +8% 是 IQ3_XXS + 内存限 24 GB 的场景 |
| 256K 上下文 | −12% | 见 3.1 |
| `STRATA_EXPERT_NIBBLE=1` | 无法启动 | 需要全部 24,576 个专家常驻 = **34.2 GiB > 32 GiB 总显存**，不是"差一点" |

---

## 五、仓库结构

```
configs/    Strata profile 模板、systemd 单元、DSH provider 示例
scripts/    01 装驱动、02 编译引擎、03 转 Q8 dense、04 下模型
scripts/fans/    P100 风扇曲线守护进程（按显卡温度调主板 PWM）
scripts/stress/  CUDA 烤机 + 整机烤机
docs/       全部原始测量数据（RESULTS.md）
```

## 六、复现步骤

```bash
# 1. 驱动（580 是最后一个还支持 Pascal 的分支）
scripts/01-install-driver.sh          # 屏蔽 nouveau 后重启
scripts/01-install-driver.sh --install

# 2. 从源码编译引擎（需要 CUDA 12.9 + gcc-13；脚本内含 glibc 2.41 补丁）
scripts/02-build-engine.sh

# 3. 下模型（66.4 GB，走 hf-mirror）
scripts/04-download-model.sh
#   专家包由 Strata 自己的 setup 生成一次：
#   strata launch --family qwen --model q2_0 --model-path <目录> --gpu 1,2
#   （在 P100 上会倒在引擎启动那一步，但包已经写好了）

# 4. 可选，但值 7%
scripts/03-convert-q8-dense.sh <shard-00001.gguf>

# 5. 跑起来
cp configs/strata-profile.example.json ~/profiles/q2_0.json   # 改里面的路径
strata serve --profile ~/profiles/q2_0.json --host 0.0.0.0 --port 8080
```

## 七、两个最耗时间的坑

**1. `strata launch` 在 Pascal 上不可用。**
它会拿签名 feed 校验引擎，发现你换过就**悄悄重新下载官方包覆盖**（那个包没有 sm_60）。
必须用 `strata serve --profile`，这条路径不校验引擎。

**2. CUDA 12.9 与 glibc 2.41 打架。**
`bits/mathcalls.h` 里 `cospi/sinpi/tanpi` 用了 `__THROW`，cudafe++ 判定与 CUDA 头里的
`noexcept(true)` 不兼容。网上常见的 `-U_GNU_SOURCE` 会连带搞坏 `pthread_*` 和 `fwide`，
正确做法是只删掉那三行声明（脚本里已处理）。

## 八、已知边界

- **单请求延迟的上限由显存带宽决定**，加卡加的是容量和吞吐，不是延迟。
  想在单请求上更快，只能换更高带宽的卡（3090 936 GB/s / 4090 1008 GB/s，P100 是 732）。
- **专家 arena 34 GB > 内存 31.2 GB**，所以始终有约 3 GB 要落在页缓存里。
  加内存到 64 GB 能让它全常驻，命中率与读题速度还会再涨一截。
- 本仓库**不包含模型权重**（Qwen 许可，需自行下载）。

## 九、烤机结果（10 分钟满载）

`stress-ng --cpu 16 --vm 6 --vm-bytes 2G --io 4 --hdd 2` + 三卡 CUDA FMA 风暴：

- 三卡 100%，峰值 64 °C（3050）/ 54–57 °C（P100），**降频标志全为 0**
- `stress-ng: failed: 0`，无内核报错、无 OOM
- P100 在烤机时只有 91 W（FMA 循环是带宽型），而**跑模型推理反而能到 100–133 W**——
  对这张卡来说，推理才是更真实的高负载

## 十、许可

本仓库代码 MIT。Strata 与 llama.cpp 均为 MIT。模型权重遵循 Qwen 许可，不在此分发。

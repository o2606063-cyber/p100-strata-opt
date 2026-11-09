# Tuning playbook — 54.7 → 68.1 tok/s on 2× P100

Every number below is measured on the same box (2× Tesla P100 16 GB, i5-12600KF,
31.2 GiB RAM), same model (Qwen3.8-Flash-Next, IQ2_XS expert pack), temperature 0,
engine-reported tokens (`done: N tokens in T s`).

Benchmark prompt: a 600-token Python generation (red-black tree, "output only code").
**Quote the workload with the number** — see [Peak vs average](#peak-vs-average) and the
prose row in the table below.

## TL;DR — the four knobs that mattered

Ranked by effect. Everything else in this document did nothing or hurt.

| # | change | gain | why it works |
|---|---|---|---|
| 1 | `--pipeline-windows 4` | **+23–26 %** | overlaps stage 0's next verify window with stage 1's verification. **Off by default on CUDA** (on by default for AMD gfx12) |
| 2 | `--spec-min-p 0.65` `--suffix-draft 0` | +4–5 % | raises draft acceptance 76 % → 85 % without truncating the window |
| 3 | `--kv-resident 32768` | +2.6 % | streams KV from pinned RAM, freeing ~1.35 GiB VRAM for experts (5908 → 6928 slots) |
| 4 | Q8_0 dense + `STRATA_Q8_SM60=1` | +7 % | GP100 is the only Pascal chip without `__dp4a`; needs Q8_0 dense **and head** |

Combined: **54.7 → 68.1 tok/s** (code), and prefill 191 → 255 tok/s.

## Final configuration

```jsonc
{
  "gpu": [1, 2],                     // the two P100s; the 3050 stays free for MiMo
  "layer_split": "auto",
  "args": [
    "--pack", "/home/yu/packs/iq2_xs",
    "--native", "…-IQ2_XS-00001-of-00002.gguf",
    "--ple-gguf", "…-IQ2_XS-00002-of-00002.gguf",
    "--expert-profile", "…/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4",
    "--spec-min-p", "0.65",
    "--suffix-draft", "0",
    "--mtp", "/home/yu/mtp/lamina",
    "--max-context", "262144",
    "--resident-experts",
    "--kv", "int8",
    "--kv-resident", "32768",
    "--pipeline-windows", "4",
    "--vision"
  ],
  "env": { "STRATA_Q8_SM60": "1", "STRATA_HC_Q8": "1" }
}
```

Ready-to-edit copy: [`configs/abl-iq2xs-p100.json`](../configs/abl-iq2xs-p100.json).

## 1. `--pipeline-windows` — the one everybody misses

`strata serve --help` says:

> `--pipeline-windows N` — with a layer split on two GPUs (**on by default for 2+ stages on
> AMD gfx12**): one conversation's verify windows with the GPUs overlapped — 1 = the prompt's
> short reads, 2 = decode as well (stage 0 runs the next window while stage 1 verifies this one)

On CUDA it is **off**. Turning it on was worth more than the previous thirty knobs combined:

| `--pipeline-windows` | 0 (default) | 2 | 3 | **4** | 6 | 8 |
|---|---|---|---|---|---|---|
| decode (tok/s) | 54.7 | 67.3 | 68.8 | **69.3** | 69.1 | 67.6 |
| prefill (tok/s) | 191 | — | — | **255** | — | — |

Verification that this is not a correctness trade: with `--pipeline-windows 0` and `4`, the same
prompt at `temperature=0` produces **byte-identical output** (math word problem, code, logic puzzle).

> **Lesson for any engine tuning:** grep `--help` for `default on`, `off by default`,
> `on by default for`, `opt-in`. Options that another platform gets for free are the cheapest
> wins on yours.

## 2. Speculation: acceptance is not the goal

`--spec N` is the MTP verify window. Raising it does not help because the drafter starts guessing:

| `--spec` | 3 | **4** | 5 | 6 | 8 |
|---|---|---|---|---|---|
| decode | 55.4 | **68.1** | 57.3 | 56.9 | −11 % vs spec 4 |

What does help is the **acceptance gate**:

| `--spec-min-p` | 0.3 | 0.5 | 0.6 | **0.65** | 0.7 | 0.75 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| acceptance | 68.9 % | 76.3 % | 81.5 % | **84.1 %** | 84.9 % | 87.9 % | 88.5 % | 96.2 % |
| decode | 55.0 | 56.0 | 57.5 | **58.7** | 58.1 | 57.0 | 58.1 | 56.9 |

**The highest acceptance rate is not the fastest setting.** At `0.9` the drafter almost always
agrees — because a high threshold truncates the window, so each pass proposes fewer tokens.
Optimise tokens-per-pass, not the accept ratio.

`--suffix-draft 0` (disable the prompt-lookup drafter) is a small extra: it stops competing with
MTP for the window.

## 3. `--kv-resident` — buy expert cache with KV latency

`--kv int8 --kv-resident 32768` keeps only 32K cells of each QSA layer in VRAM and streams the
rest from pinned RAM. The freed VRAM goes to experts:

| config | expert cache | decode |
|---|---|---|
| `--kv int8` | 5908 slots / 7.79 GiB | 54.7 |
| `--kv q4_0` | 6442 / 8.50 | 56.5 |
| `--kv k8v4` | 6174 / 8.14 | 56.0 |
| **`--kv int8 --kv-resident 32768`** | **6928 / 9.14** | **56.1–57.4** |
| `--kv int8 --kv-resident 20480` | 6986 / 9.21 | 57.2 |

Note the diminishing returns: +17 % more cache slots bought only +2.6 % decode. The extra slots
are the *least* frequently routed experts — most tokens never touch them.

## 4. What did **not** help (all measured)

| change | result |
|---|---|
| `--spec 6` / `--spec 8` | 56.9 / −11 % vs spec 4 |
| `--lookup-chain 4` | flat |
| `--expert-cache-per-layer` | −15 % (hit rate 96 → 90 %) |
| `--mtp-q4 all` | fails to start |
| `--host-core last` | flat |
| 16 threads (SMT) instead of 10 physical cores | **−41 %** |
| `--pool-cores` with the same 9 cores | flat (harmless) |
| `--prefill auto:16384` / `auto:32768` | flat (245–255) |
| `--prefill-pipe 1` | flat (255 vs 255) — **re-tested after pipeline was on** |
| `--pool-tasks 512/2048` | flat |
| `STRATA_PREFILL_CPU_SHARE=0/0.3` | flat |
| `--prefill-head-alias` | fails to start |
| `--aux-cpus auto` | 54.6 vs 54.7 |
| `--remote-expert-opt` | 54.2 vs 54.7 |
| `--native-qsa` + `--native-qsa-indexer` | 57.9 avg (noise) |
| `--no-fused-gr` | 58.0 avg (noise) |
| `--sync-every-layer` | 58.7 avg (noise) |
| `--vision` off | 58.7 avg (noise) |
| `--expert-cache-device1 7500` (manual) | fails to start; `auto` already fills the card |
| CPU governor `powersave` → `performance`, EPP, disabling C6/C8/C10 | **0 %** — see below |

### The CPU is already at full clock

Under decode: P-cores **4.4–4.5 GHz**, E-cores **3.4 GHz** (rated 4.9 / 3.6), package power
**67 W of a 150 W RAPL cap**. Setting the governor to `performance`, EPP to `performance` and
disabling deep C-states changed nothing (56.0 vs 56.1). There is no throttling to recover —
the ten physical cores are simply busy.

## Where the remaining time goes

Measured during decode with `--pipeline-windows 4`:

| | before pipeline | after |
|---|---|---|
| GPU 1 / GPU 2 utilisation | 50–60 % | **70–79 %** |
| CPU (10 physical cores of 16 threads) | 97 % each | 97 % each |
| GPU power / temp | 100–133 W | 98–137 W, 49–51 °C |

The GPUs still idle ~21 % of the time, waiting for the CPU expert pool: only ~45 % of the
11776 expert pairs fit in 32 GB of VRAM, so the rest are computed on ten cores every token.

**That is why further knob-twiddling is pointless and more VRAM is not:** at 45 % residency the
CPU is the wall, and every software knob that could move work off the CPU has been tried.

| next step | expected | cost |
|---|---|---|
| More VRAM (e.g. 24 GB cards) | **+30–50 %** — experts become resident, the CPU goes idle | new hardware |
| 8 P-core CPU | +15–25 % | new platform |
| More system RAM | **0 %** — the bottleneck is VRAM, not RAM | — |

## Peak vs average

Single measurements on this box swing ±2 tok/s, so a ±3 % "improvement" read off one run is noise.
The 68.1 figure is the mean of **8 consecutive runs** after a warm-up, all ≥ 60:

```
67.6  67.4  71.9  69.8  66.5  66.8  69.4  65.8      mean 68.1, median 67.5, max 71.9
```

Interleaved A/B (3 configs × 2 rounds × 4 runs) is what settled `--spec-min-p`:

```
min-p 0.65   n=8   mean 58.3   median 58.3   max 59.6
min-p 0.70   n=8   mean 57.7   median 57.8   max 58.4
min-p 0.70 + suffix-draft 3   n=8   mean 57.3
```

## Two measurement traps

1. **The prompt cache will fool you.** Re-sending the same prompt reports
   `1667 reused + 5 read in 94 ms` — that is cache speed, not prefill. Generate a **fresh random
   prompt for every prefill measurement**, and compute tok/s only from the `read` count and the
   `ms` in `prompt N tokens = X reused + Y read in Z ms`.
2. **The first request after a restart is cold.** Expert cache and prompt cache are empty; it
   measures 15–25 % low. Always warm up, then measure.

## Workload dependence — the honest caveat

The same binary, same flags, three different tasks:

| task | without pipeline | with pipeline | gain |
|---|---|---|---|
| repetitive code (red-black tree) | 49.6 | **62.3** | **+26 %** |
| varied code (three data structures) | 37.6 | **46.3** | +23 % |
| prose explanation (B+ tree vs LSM) | 36.2 | 36.4 | **+0.6 %** |

Pipeline overlap is fed by speculative verify windows. Predictable output ⇒ long windows ⇒
two GPUs have work to overlap. Explaining a concept in prose produces short, low-acceptance
windows and the overlap finds nothing to do.

**So "68 tok/s" is a code-generation number.** Prose stays at ~36 tok/s.

---

## 中文摘要

| 优化 | 收益 | 说明 |
|---|---|---|
| `--pipeline-windows 4` | **+23~26 %** | CUDA 上默认关闭、AMD 默认开启。开了之后 stage 0 跑下一个窗口、stage 1 验证当前窗口，两卡真正并行 |
| `--spec-min-p 0.65` + `--suffix-draft 0` | +4~5 % | 接受率 76 %→85 %。注意 min-p 0.9 时接受率 96 % 反而更慢——高阈值会截短草稿窗口 |
| `--kv-resident 32768` | +2.6 % | KV 流到锁定内存，腾出 1.35 GiB 给专家缓存（5908→6928 槽） |
| Q8_0 dense + `STRATA_Q8_SM60=1` | +7 % | GP100 是唯一没有 `__dp4a` 的 Pascal 芯片 |

**54.7 → 68.1 tok/s（代码类任务）；读题 191 → 255 tok/s。**

三个必须知道的坑：

1. **读题测量必须每次换 prompt**，否则提示缓存全命中（`1667 reused + 5 read`），测出来的是缓存速度。
2. **接受率不是越高越好**，要优化"每次产出多少 token"。
3. **+26 % 只对代码类任务成立**，写散文几乎没变化（+0.6 %）——重叠依赖投机窗口，内容可预测才有窗口。

剩下 21 % 的 GPU 空闲是在等 CPU 算专家（只有 45 % 的专家装得进 32 GB 显存）。
软件层面已穷尽约 45 种配置，再快只能加显存。

# P100 + Strata: running Qwen3.8-Flash-Next on Pascal

**TL;DR** — The official Strata engine packages ship **no `sm_60` kernels**, so a Tesla P100
("this engine has no code for … sm_60") cannot run any model out of the box. This repo documents
how to build the engine from source for Pascal, and a set of tuning results on **2× P100 16 GB +
32 GB RAM**, ending at **68.1 tok/s** on code generation at 256K context (versus **32–37 tok/s**
stock) — plus the optimisations that *look* like they should help and measurably do not.

The single biggest win was **`--pipeline-windows 4`**, which is **off by default on CUDA** and
worth **+23–26 %**. Full playbook: [`docs/TUNING.md`](docs/TUNING.md).

> 中文说明见下方。所有数字都是同一台机器上实测的，不是推算。

> **Note:** the sections below are the earlier write-up (Q2_0 pack, 128K context, 59.8 tok/s).
> The current tuned result is **68.1 tok/s at 256K** on the IQ2_XS pack — see
> [`docs/TUNING.md`](docs/TUNING.md) for the full playbook, and for the two entries that were
> later overturned (`--pipeline-windows` and the 256K penalty). 中文版：[README.md](README.md).

---

## Prebuilt engine (no build required)

The engine built for this article is attached to the **Releases** page:
`strata-0.1.54-sm60-linux-x86_64.tar.gz` — 243 MB, CUDA 12.9.86 / gcc-13,
architectures `60;61;70;86`, sha256 `b2f4ff55b5a76d042bbcb7d04e167768f5ae55f9095087c546704d86af08bbd0`.

It is uploaded as a release asset, not committed: GitHub rejects files over 100 MB.
Install instructions are inside the archive (`INSTALL.md`) and in
[release/README.md](release/README.md). Short version: drop its `strata` into an official
CUDA 12 engine component folder (keep that folder's `libcublas.so.12` / `libcudart.so.12`),
then start it with `strata serve --profile …` — **not** `strata launch`, which re-verifies
the engine against the signed feed and silently replaces it.

---

## Hardware

| | |
|---|---|
| GPU | 2 × Tesla P100-PCIE-16GB (GP100, `sm_60`), 250 W cap, PCIe 3.0 **x4** each |
| CPU | Intel i5-12600KF (6P+4E, 16 threads) |
| RAM | 31.2 GiB |
| Disk | WD_BLACK SN750 SE 500 GB (DRAM-less) |
| OS | Debian 13 (trixie), kernel 6.12, NVIDIA 580.196.07, CUDA 12.9 |

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **Q2_0** (66.4 GB, 24,576 experts,
3B active). Engine: Strata 0.2.8 / build 0.1.54.

---

## Results

All decode numbers are engine-reported (`done: N tokens in T s`), same prompt family, temperature 0.

### Final configuration

| knob | value |
|---|---|
| context | **131072** (1 slot) |
| `--kv` | `int8` |
| `--prefill` | `8192` |
| `--spec` | `4` (MTP) |
| dense weights | **Q8_0** + `STRATA_Q8_SM60=1` |
| vision | CPU reader (`--vision`) |

**59.8 tok/s** on a 600-token code generation at 128K context.

### Why 37 or 57? (draft acceptance, not hardware)

| prompt | drafts accepted | decode |
|---|---|---|
| Python module, 600 tok | 420/500 = **84 %** | 53.9 / 56.5 / 56.9 |
| prose "what is MoE", 100 tok | 50/98 = **51 %** | 36.9 / 38.6 |

The same engine on the same weights is **50 % faster on code than on prose**, because MTP
speculative decoding only pays when the drafter can predict the next tokens. Quote the workload.

### Context length vs expert cache vs speed

Every KV byte is an expert-cache slot. Measured on the same prompt:

| context | expert cache | decode |
|---|---|---|
| 8K | 8826 slots | 53.8 |
| 32K | 8431 | 53.9–56.3 |
| 64K | 7907 | 53.4–55.7 |
| 128K | 6859 | 52.8–54.8 |
| **256K** | **4759** | **45.0–49.6** |

The curve is flat to 64K, −3 % at 128K, then **falls off a cliff at 256K (−12 %)** as the cache
hit rate drops from 99 % to 69 %. 256K is a trap on a 32 GB card.

### Slots × context (the batch/KV trade)

| context | slots | expert cache | decode |
|---|---|---|---|
| 64K | 2 | 4406 | 53.0–57.5 |
| 128K | 2 | 3353 | **48.3** ← worst |
| 64K | 1 | 5213 | **60.7** |
| 128K | 1 | 4861 | **59.8** ← best value |

Two slots at 128K costs **24 %** versus one slot: two KV caches squeeze the expert cache harder
than a second user is worth on this card. One slot at 128K is the sweet spot for a single user.

---

## The optimisation that worked: `STRATA_Q8_SM60`

GP100 is the only Pascal chip **without `__dp4a`**, so its default dense-projection kernel reads
weights at 40–80 GB/s out of ~730. Strata ships a dedicated GP100 kernel, but it needs the dense
matrices **and the head** in Q8_0 — and the ISTA-DASLab packs keep them at Q5_K / Q3_K / IQ4_XS.

Convert them with `tools/q8_dense_gguf.py` (experts keep their offsets, so the pack is reused):

```bash
STRATA_GGUF_PY=<llama.cpp>/gguf-py \
  python tools/q8_dense_gguf.py <shard-00001>.gguf <out-dir>
```

Measured on the identical prompt (red-black tree, 600 tokens):

| dense | runs | mean |
|---|---|---|
| original (Q5_K/Q3_K) | 50.7 / 52.5 / 52.9 | **52.0** |
| Q8_0 + `STRATA_Q8_SM60=1` | 55.9 / 57.5 / 55.6 / 53.0 / 57.3 / 55.7 | **55.8** |

**+7 %**, reproducible across two independent runs. Cost: +39 GB disk, and the packed head
(644 MiB) plus bigger dense matrices cut the expert cache from 7907 to 4406 slots — the kernel
win outweighs the cache loss, but only just.

---

## What did **not** help (measured, in this order)

| change | result | note |
|---|---|---|
| 3rd GPU (RTX 3050 8 GB) in the split | **38.9 vs 42.0 tok/s** | pipeline stage; its 8 GB is eaten by dense copy + KV + 4.24 GiB prompt buffers, so it holds ~4 % of the experts and becomes the slowest stage |
| `--kv q4_0` instead of `int8` | 39.1 vs 42.0 | the Hadamard rotation costs more on P100 than the VRAM it saves |
| `--pipeline-windows 2` | 54.1 vs 55.1 | no gain on a 2-stage NVIDIA split |
| `--pcie-frac 0` | 40.5 vs 41.3 | the engine's auto pick (0.09 on a 3.3 GB/s link) is already right |
| `--spec 6` / `--spec 8` | 49.7 / — vs 55.8 | deeper drafts cost more than the extra accepts |
| DFlash2 drafter | 55.3 / 54.8 / 56.6 vs 53.9 / 56.5 / 56.9 | identical; the drafter's own table predicts this for Ampere/Pascal |
| 256K context | −12 % | see the curve above |
| `STRATA_STAGE_PIN=1` | 39.5 vs 41.8 | docs' +8 % was for IQ3_XXS with RAM capped at 24 GB |
| `STRATA_EXPERT_NIBBLE=1` | not attempted | needs **all** 24,576 experts resident = 34.2 GiB > 32 GiB total VRAM. Impossible, not marginal |

---

## Repository layout

```
configs/    example Strata profile, systemd unit, DSH provider patch
scripts/    01 driver, 02 engine build, 03 Q8 dense conversion, 04 model download
scripts/fans/    P100 fan curve daemon (GPU temp -> motherboard PWM)
scripts/stress/  CUDA burn-in + whole-machine burn-in
docs/       results table, troubleshooting notes
```

## Reproduction

```bash
# 1. driver (580 is the last branch that still runs Pascal)
scripts/01-install-driver.sh

# 2. engine from source, with sm_60 (CUDA 12.9 + gcc-13; see the script for the
#    glibc 2.41 / mathcalls.h patch that CUDA 12.9 needs on Debian 13)
scripts/02-build-engine.sh

# 3. model + pack (pack is built once by Strata's own setup)
scripts/04-download-model.sh

# 4. optional but worth ~7 %
scripts/03-convert-q8-dense.sh

# 5. run
cp configs/strata-profile.example.json ~/profiles/q2_0.json   # edit the paths
strata serve --profile ~/profiles/q2_0.json --host 0.0.0.0 --port 8080
```

### Two gotchas that cost the most time

1. **`strata launch` is unusable for Pascal.** It verifies the engine against the signed feed and
   silently re-installs the release build (which has no `sm_60` code). Use
   `strata serve --profile …` — that path does not check the engine.
2. **CUDA 12.9 vs glibc 2.41.** `bits/mathcalls.h` declares `cospi`/`sinpi`/`tanpi` with `__THROW`
   and cudafe++ rejects the mismatch. `01`/`02` patch those three declarations locally; a blanket
   `-U_GNU_SOURCE` breaks `pthread_*` and `fwide` instead.

## License

MIT for everything in this repo. Strata is MIT; llama.cpp is MIT; the model weights carry
Qwen's license and are **not** included here.

# Measurements — 2× Tesla P100 16 GB, Qwen3.8-Flash-Next Q2_0

All numbers engine-reported (`done: N tokens in T s (X tok/s)`), temperature 0, same box,
same weights. "code" = a 600-token Python generation; "prose" = a short explanation.

## Decode depends on the workload, not the hardware

| prompt | drafts accepted | decode |
|---|---|---|
| code | 420/500 = 84 % | 53.9 / 56.5 / 56.9 |
| prose | 50/98 = 51 % | 36.9 / 38.6 |

MTP speculative decoding only pays when the drafter can predict; code is far more
predictable. Any single "tok/s" number without the workload is meaningless.

## Context vs expert cache (single slot, original dense)

| context | cache | decode | hit |
|---|---|---|---|
| 8K | 8826 | 53.8 / 55.7 | 95.5–99 % |
| 32K | 8431 | 53.9 / 55.9 / 56.3 | 94.7–98.8 % |
| 64K | 7907 | 53.4 / 55.7 | 93.5–98.1 % |
| 128K | 6859 | 52.8 / 54.8 | 90.7–96.9 % |
| 256K | 4759 | 45.0 / 49.4 / 49.6 | 69.1–89.6 % |

~711 cache slots per GiB. Flat to 64K, −3 % at 128K, −12 % at 256K.

## Slots × context (with the Q8 dense kernel)

| context | slots | cache | decode |
|---|---|---|---|
| 64K | 2 | 4406 | 53.0–57.5 |
| 128K | 2 | 3353 | 48.3 |
| 64K | 1 | 5213 | 59.2–60.9 |
| 128K | 1 | 4861 | 59.2–60.1 |

## Q8 dense kernel (same prompt, red-black tree, 600 tok)

| dense | runs | mean |
|---|---|---|
| Q5_K / Q3_K / IQ4_XS (pack) | 50.7 / 52.5 / 52.9 | 52.0 |
| Q8_0 + `STRATA_Q8_SM60=1` | 55.9 / 57.5 / 55.6 | 55.8 |
| Q8_0, second session | 53.0 / 57.3 / 55.7 | 55.3 |

+7 %, at the cost of 39 GB disk and ~3500 cache slots (the packed head alone is 644 MiB).

## Negative results

| change | result | why |
|---|---|---|
| 3rd GPU (RTX 3050 8 GB) | 38.9 vs 42.0 | pipeline stage; 8 GB minus dense+KV+4.24 GiB prompt buffers leaves ~4 % of the experts, so it is the slowest stage |
| `--kv q4_0` | 39.1 vs 42.0 | Hadamard rotation costs more than the VRAM saved |
| `--pipeline-windows 2` | 54.1 vs 55.1 | opt-in, no gain on a 2-stage NVIDIA split |
| `--pcie-frac 0` | 40.5 vs 41.3 | auto (0.09 at 3.3 GB/s) is right |
| `--spec 6` | 49.7 vs 55.8 | deeper drafts cost more than the accepts |
| DFlash2 | 55.3 / 54.8 / 56.6 vs MTP 53.9 / 56.5 / 56.9 | no gain; Strata's own table says Ampere/Pascal lose |
| `STRATA_STAGE_PIN=1` | 39.5 vs 41.8 | docs' +8 % was IQ3_XXS with RAM capped at 24 GB |
| 256K context | −12 % | cache hit falls to 69 % |
| `STRATA_EXPERT_NIBBLE=1` | impossible | needs all 24,576 experts resident (34.2 GiB) > 32 GiB total VRAM |

## Whole-machine burn-in (10 min)

`stress-ng --cpu 16 --vm 6 --vm-bytes 2G --io 4 --hdd 2` + a CUDA FMA storm on all 3 GPUs.

- 3 GPUs at 100 %, peak 64 °C (3050), 54–57 °C (P100), **zero throttle reasons**
- `stress-ng: failed: 0`, no kernel errors, no OOM
- P100 draw under the burn: ~91 W of a 250 W cap (the FMA loop is bandwidth-bound; *inference*
  pulls 100–133 W and is the more realistic load)

## When the machine is idle

GPU 3050 12 W + P100 34 W + 32 W, CPU package 7 W → ~85 W software-visible.

## Final configuration

```
--max-context 131072   --parallel 1
--kv int8   --prefill 8192   --spec 4   --resident-experts
--vision (CPU reader, ~3.4 s per image)
env: STRATA_Q8_SM60=1
dense: Q8_0 (converted shard)
```

**59.8 tok/s on code at 128K context**, 99 %+ cache hit on short contexts.

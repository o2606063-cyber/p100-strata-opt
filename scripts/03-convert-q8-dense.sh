#!/bin/bash
# 03 - convert the dense projections + head to Q8_0 so the GP100 kernel can run.
#
# GP100 is the only Pascal chip without __dp4a: its default dense kernel reads weights at
# 40-80 GB/s of ~730. Strata has a dedicated GP100 Q8_0 kernel (STRATA_Q8_SM60=1), but it
# needs the dense matrices AND the head in Q8_0 - the ISTA-DASLab packs keep them Q5_K/Q3_K/
# IQ4_XS. Numbers on 2x P100: 52.0 -> 55.8 tok/s (+7%), for +39 GB of disk and ~3500 fewer
# expert-cache slots (the kernel win is bigger than the cache loss, but only just).
set -euo pipefail

SRC=${SRC:-$HOME/src/Strata-0.2.8}
LLAMACPP=${LLAMACPP:-$HOME/src/llama.cpp}
PY=${PY:-python3}                      # needs numpy + gguf-py
SHARD=${1:?usage: 03-convert-q8-dense.sh <model-shard-00001.gguf> [out-dir]}
OUT=${2:-$(dirname "$SHARD")-q8dense}

export STRATA_GGUF_PY="$LLAMACPP/gguf-py"
cd "$SRC"
"$PY" tools/q8_dense_gguf.py "$SHARD" "$OUT"

cat <<EOF

converted into $OUT
  - point --native   at $OUT/$(basename "$SHARD")
  - point --ple-gguf at $OUT/<the other shard>   (symlinked)
  - add "STRATA_Q8_SM60": "1" to the profile's "env"
  - the engine logs "q8 sm60: Q8_0 decode GEMVs on the Pascal kernel" when it is on

rerun scripts/01-install-driver.sh --install? no. just restart the server and watch
  grep 'q8 sm60' <profile-dir>/<model>-engine.log
EOF

#!/bin/bash
# 04 - download the Q2_0 model (66.4 GB) and the vision encoder from a mirror.
#
# hf-mirror.com is a full HF mirror; direct HF is ~60 KB/s from here. aria2 caps at
# 16 connections per server, so two files in parallel is the ceiling (~150 MB/s on a 2.5 GbE).
set -euo pipefail

DEST=${DEST:-$HOME/models/qwen38-q2_0}
REPO=https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main
mkdir -p "$DEST" && cd "$DEST"

fetch() {  # $1 = remote path, $2 = local name, $3 = expected bytes
    if [ -f "$2" ] && [ "$(stat -c %s "$2")" = "$3" ]; then echo "have $2"; return; fi
    aria2c -x8 -s8 -k1M -c --file-allocation=none --console-log-level=warn -o "$2" "$REPO/$1"
    [ "$(stat -c %s "$2")" = "$3" ] || { echo "size mismatch on $2" >&2; exit 1; }
}

fetch "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
      "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" 37623740192
fetch "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
      "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" 28800138432
fetch "mmproj-Qwen3.8-Flash-Next-BF16.gguf" \
      "mmproj-Qwen3.8-Flash-Next-BF16.gguf" 907543008

cat <<'EOF'

shard 2 is the 28.8 GB n-gram (PLE) table and is identical across the Q2_0/Coder packs,
so it can be hardlinked instead of downloaded:
  ln -f .../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
        .../coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf

the expert pack and the MTP draft layer are built by Strata itself on first start:
  strata launch --family qwen --model q2_0 --model-path <this dir> --gpu 1,2 ...
  (it will fail at the engine start on Pascal - that is fine, the pack is already written;
   then build the profile from that run config, point "exe" at your own build and use
   `strata serve --profile`)
EOF

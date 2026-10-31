#!/bin/bash
# 02 - build the Strata engine from source with sm_60 code.
#
# Why: the release packages (cuda12 and cuda13) contain no sm_60 kernels at all.
# `cuobjdump --list-elf` on the release binary shows sm_89 only, so a P100 dies with
#   "this engine has no code for Tesla P100-PCIE-16GB (sm_60)"
# even though BUILD.json claims archs 60..89.
set -euo pipefail

SRC=${SRC:-$HOME/src/Strata-0.2.8}       # unpacked source tree
LLAMACPP=${LLAMACPP:-$HOME/src/llama.cpp} # needed for ggml (FetchContent cannot reach github here)
CUDA=${CUDA:-/usr/local/cuda-12.9}
VERSION=${VERSION:-v0.2.8}

echo "== toolchain: CUDA 12.9 + gcc-13 (nvcc refuses GCC 14) =="
apt-get install -y cmake ninja-build gcc-13 g++-13
"$CUDA/bin/nvcc" --version | tail -1

echo "== patch glibc's cospi/sinpi/tanpi for CUDA 12.9 =="
# cudafe++ rejects glibc 2.41's __THROW on these three; a blanket -U_GNU_SOURCE instead
# breaks pthread_* and fwide. Cut only the three declarations.
H=/usr/include/x86_64-linux-gnu/bits/mathcalls.h
[ -f "$H.strata-bak" ] || cp -n "$H" "$H.strata-bak"
sed -i 's|^__MATHCALL_VEC (cospi,.*$|/* strata: removed (CUDA declares cospi) */|;
        s|^__MATHCALL_VEC (sinpi,.*$|/* strata: removed (CUDA declares sinpi) */|;
        s|^__MATHCALL_VEC (tanpi,.*$|/* strata: removed (CUDA declares tanpi) */|' "$H"

cd "$SRC"
if [ ! -d .git ] && [ ! -f CMakeLists.txt ]; then
    echo "unpack $VERSION here first (github.com/Niko1221/Strata/archive/refs/tags/$VERSION.tar.gz)" >&2
    exit 1
fi

echo "== configure =="
cmake -S . -B build-cuda12 -G Ninja \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DSTRATA_GGML_DIR="$LLAMACPP" \
    -DCMAKE_CUDA_ARCHITECTURES="60;61;70;86" \
    -DGGML_CUDA_ARCHITECTURES="60;61;70;86" \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"

echo "== build (173 targets, ~10 min on 16 threads) =="
cmake --build build-cuda12 --target strata -j "$(nproc)"

echo "== verify sm_60 is really in there =="
"$CUDA/bin/cuobjdump" --list-elf build-cuda12/strata | grep -oE 'sm_(60|86)' | sort | uniq -c
echo "expect ~76 sm_60 entries (the release build has 1, and it is the error string)"

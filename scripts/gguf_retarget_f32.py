#!/usr/bin/env python3
"""Retarget specific GGUF tensors to F32 in place.

Strata's qwen35 path needs ssm_alpha/ssm_beta as F32 (or BF16/F16/Q8_0), but
Ornith's GGUF ships them as Q4_K.  Rather than rewriting the whole 21.7 GB file,
append the dequantized F32 blobs to the end and patch the tensor-info records
(type + offset), which are fixed-width fields.
"""
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/yu/src/llama.cpp/gguf-py")
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize

PATH = sys.argv[1]
TARGETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ssm_alpha.weight", "ssm_beta.weight"]
F32 = 0
Q4_K = 12

TYPE_SIZE = {  # bytes per element for the types we can meet here
    0: 4,    # F32
    1: 2,    # F16
    30: 2,   # BF16
    8: 34 / 32,  # Q8_0 (34 bytes per 32 values)
}
BLOCK = {12: 256, 8: 32}   # Q4_K, Q8_0
BLOCK_BYTES = {12: 144, 8: 34}


def read_str(buf, off):
    n, = struct.unpack_from("<Q", buf, off)
    return buf[off + 8: off + 8 + n].decode("utf-8"), off + 8 + n


def skip_value(buf, off, t):
    if t == 8:
        n, = struct.unpack_from("<Q", buf, off)
        return off + 8 + n
    if t == 9:
        et, = struct.unpack_from("<I", buf, off)
        off += 4
        n, = struct.unpack_from("<Q", buf, off)
        off += 8
        for _ in range(n):
            off = skip_value(buf, off, et)
        return off
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    return off + sizes[t]


def main():
    with open(PATH, "rb") as fh:
        head = fh.read(64 * 1024 * 1024)
    magic = head[:4]
    assert magic == b"GGUF", magic
    off = 4
    ver, = struct.unpack_from("<I", head, off); off += 4
    n_tensors, = struct.unpack_from("<Q", head, off); off += 8
    n_kv, = struct.unpack_from("<Q", head, off); off += 8

    alignment = 32
    for _ in range(n_kv):
        k, off = read_str(head, off)
        t, = struct.unpack_from("<I", head, off); off += 4
        if k == "general.alignment":
            alignment, = struct.unpack_from("<I", head, off)
        off = skip_value(head, off, t)

    infos = []
    for _ in range(n_tensors):
        name, off = read_str(head, off)
        n_dims, = struct.unpack_from("<I", head, off); off += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, head, off)
        off += 8 * n_dims
        tpos = off
        ttype, = struct.unpack_from("<I", head, off); off += 4
        opos = off
        toff, = struct.unpack_from("<Q", head, off); off += 8
        infos.append([name, tpos, ttype, opos, toff, dims])

    data_start = (off + alignment - 1) // alignment * alignment
    print(f"  GGUF v{ver} 张量 {n_tensors} data_start {data_start} alignment {alignment}")

    picks = [i for i in infos if any(t in i[0] for t in TARGETS)]
    print(f"  命中 {len(picks)} 个张量")

    fh = open(PATH, "r+b")
    fh.seek(0, 2)
    end = fh.tell()
    end = (end + alignment - 1) // alignment * alignment

    changed = 0
    for name, tpos, ttype, opos, toff, dims in picks:
        n_elem = 1
        for d in dims:
            n_elem *= d
        if ttype == F32:
            continue
        if ttype not in BLOCK:
            print(f"  ! {name} 类型 {ttype} 不在处理范围")
            continue
        blk = BLOCK[ttype]
        raw_bytes = (n_elem // blk) * BLOCK_BYTES[ttype]
        fh.seek(data_start + toff)
        raw = fh.read(raw_bytes)
        vals = dequantize(np.frombuffer(raw, dtype=np.uint8), QT(ttype)).astype(np.float32)
        assert vals.size == n_elem, (name, vals.size, n_elem)
        blob = vals.tobytes()
        fh.seek(end)
        fh.write(blob)
        # patch type -> F32 and offset -> appended position
        fh.seek(tpos)
        fh.write(struct.pack("<I", F32))
        fh.seek(opos)
        fh.write(struct.pack("<Q", end - data_start))
        end += len(blob)
        end = (end + alignment - 1) // alignment * alignment
        changed += 1
        if changed <= 3:
            print(f"  {name}: {QT(ttype).name} -> F32 ({n_elem} 个值) @ +{len(blob)} 字节")

    fh.flush()
    fh.close()
    print(f"  完成：{changed} 个张量已转 F32")


main()

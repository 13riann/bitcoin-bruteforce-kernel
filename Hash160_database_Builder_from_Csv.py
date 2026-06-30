#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import struct
import hashlib
import csv
from datetime import datetime

# ===================== CONFIG =====================
INPUT_CSV        = "Hash_db_Full_ICEMAN_5_jumps.csv"                  # Your CSV file
INPUT_HASH_FILE  = "Hash_db_Merged_full.bin"     # Temporary bin
OUTPUT_FOLDER    = "HASH_DB_FINAL"
META_FILE        = os.path.join(OUTPUT_FOLDER, "Test_hashdb_cuckoo_meta.txt")
PIF_FILE         = os.path.join(OUTPUT_FOLDER, "Hash_db_pif_Final.bin")
OVERSIZE_RATIO   = 1.5
NUM_HASH_FUNCS   = 3
MAX_ITERATIONS   = 200000
SEEDS            = [17, 31, 73]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== UTIL =====================
def now():
    return datetime.now().strftime("%H:%M:%S")

# ===================== CSV -> BIN =====================
def csv_to_bin(csv_file, bin_file):
    with open(csv_file, newline='') as csvfile, open(bin_file, "wb") as binfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            h_hex = row["Hash160"].strip()
            if len(h_hex) != 40:
                raise ValueError(f"Invalid hash160 length: {h_hex}")
            h_bytes = bytes.fromhex(h_hex)
            binfile.write(h_bytes)
    print(f"[{now()}] Stage 1 complete: CSV converted to {bin_file}")

# ===================== MURMUR3 =====================
def murmur3_32(data: bytes, seed: int = 0) -> int:
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    h1 = seed & 0xFFFFFFFF
    length = len(data)
    rounded_end = (length & ~3)

    for i in range(0, rounded_end, 4):
        k1 = struct.unpack_from("<I", data, i)[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF

    tail = data[rounded_end:]
    if tail:
        k1 = 0
        for i in range(len(tail)):
            k1 |= tail[i] << (i * 8)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85ebca6b) & 0xFFFFFFFF
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xc2b2ae35) & 0xFFFFFFFF
    h1 ^= (h1 >> 16)
    return h1

# ===================== READ HASHES =====================
def read_hashes(file):
    hashes = []
    with open(file, "rb") as f:
        while True:
            h = f.read(20)
            if not h:
                break
            if len(h) != 20:
                raise RuntimeError("Corrupted hash160 in input file")
            hashes.append(h)
    print(f"[{now()}] Loaded {len(hashes)} hashes from {file}")
    return hashes

# raw20 <-> (uint64,uint64,uint32)
def pack_tuple(h20):
    return struct.unpack(">QQI", h20)

def unpack_tuple(t):
    return struct.pack(">QQI", *t)

# ===================== CUCKOO BUILD =====================
def build_cuckoo(hashes, table_size, seeds):
    table = [None] * table_size
    total = len(hashes)

    for i, h in enumerate(hashes):
        p = pack_tuple(h)
        choices = [murmur3_32(h, s) % table_size for s in seeds]
        placed = False

        for attempt in range(MAX_ITERATIONS):
            for slot in choices:
                if table[slot] is None:
                    table[slot] = p
                    placed = True
                    break
            if placed:
                break

            victim = random.choice(choices)
            table[victim], p = p, table[victim]
            h_evicted = unpack_tuple(p)
            choices = [murmur3_32(h_evicted, s) % table_size for s in seeds]

        if not placed:
            raise RuntimeError(f"Cuckoo insertion failed at index {i}")

        if i % max(1, total // 20) == 0:
            print(f"[{now()}] {i+1}/{total} placed")

    return table

# ===================== META =====================
def write_meta(table, seeds):
    md5 = hashlib.md5(open(PIF_FILE, "rb").read()).hexdigest()
    lines = [
        f"input_file={INPUT_HASH_FILE}",
        f"table_file={os.path.basename(PIF_FILE)}",
        f"table_size={len(table)}",
        f"num_hash_funcs={NUM_HASH_FUNCS}",
        f"seeds={','.join(map(str,seeds))}",
        "hash_func=murmur3_32(seed||hash160) % table_size",
        "hash160_struct=uint64|uint64|uint32",
        "entry_size=20",
        "format=pif_8_8_4_be",
        f"md5={md5}",
        f"created_at={now()}",
        f"updated_at={now()}",
    ]
    with open(META_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"[{now()}] Meta written. md5={md5}")

# ===================== WRITE PIF =====================
def write_pif(table):
    with open(PIF_FILE, "wb") as f:
        for entry in table:
            if entry is None:
                a = b = c = 0
            else:
                a, b, c = entry
            f.write(struct.pack(">Q", a))
            f.write(struct.pack(">Q", b))
            f.write(struct.pack(">I", c))
    expected = len(table) * 20
    real = os.path.getsize(PIF_FILE)
    print(f"[{now()}] Wrote PIF file {PIF_FILE}: expected={expected}, real={real}")
    if expected != real:
        raise RuntimeError("PIF file size mismatch.")

# ===================== MAIN =====================
def main():
    # Stage 1: CSV → BIN
    csv_to_bin(INPUT_CSV, INPUT_HASH_FILE)

    # Stage 2: BIN → final PIF
    if not os.path.exists(INPUT_HASH_FILE):
        raise RuntimeError(f"Missing input file: {INPUT_HASH_FILE}")

    hashes = read_hashes(INPUT_HASH_FILE)
    table_size = math.ceil(len(hashes) * OVERSIZE_RATIO)
    table = build_cuckoo(hashes, table_size, SEEDS)

    write_pif(table)
    write_meta(table, SEEDS)

    print(f"[{now()}] SUCCESS — final PIF + meta written to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import hashlib
import struct
import ecdsa

DB_FILE = "Hash_db_pif_Final.bin"

# ---------------- HASH160 ----------------

def ripemd160(data):
    h = hashlib.new("ripemd160")
    h.update(data)
    return h.digest()

def hash160(data):
    return ripemd160(hashlib.sha256(data).digest())

# ---------------- SECP256K1 ----------------

curve = ecdsa.SECP256k1
G = curve.generator

# ---------------- Murmur3 ----------------

def murmur3_32(data, seed):
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    h1 = seed

    length = len(data)
    rounded_end = length & 0xfffffffc

    for i in range(0, rounded_end, 4):
        k1 = struct.unpack_from("<I", data, i)[0]

        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff

        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xffffffff
        h1 = (h1 * 5 + 0xe6546b64) & 0xffffffff

    k1 = 0
    tail = length & 3

    if tail == 3:
        k1 ^= data[rounded_end + 2] << 16
    if tail >= 2:
        k1 ^= data[rounded_end + 1] << 8
    if tail >= 1:
        k1 ^= data[rounded_end]

        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff

        h1 ^= k1

    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85ebca6b) & 0xffffffff
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xc2b2ae35) & 0xffffffff
    h1 ^= (h1 >> 16)

    return h1

# ---------------- DB Lookup ----------------

def lookup_hash160(h160, db, slots):
    seeds = [17, 31, 73]

    A = int.from_bytes(h160[0:8], "little")
    B = int.from_bytes(h160[8:16], "little")
    C = int.from_bytes(h160[16:20], "little")

    for seed in seeds:
        hv = murmur3_32(h160, seed)
        slot = hv % slots
        base = slot * 20

        a = int.from_bytes(db[base:base+8], "little")
        if a != A:
            continue

        b = int.from_bytes(db[base+8:base+16], "little")
        c = int.from_bytes(db[base+16:base+20], "little")

        if b == B and c == C:
            return True, slot

    return False, None

# ---------------- Main ----------------

def main():
    print("=== Private Key → HASH160 → DB Lookup Tool ===")

    priv = int(input("Enter private key (decimal): ").strip())

    # Compressed public key
    point = priv * G

    x = point.x()
    y = point.y()

    prefix = b"\x02" if (y % 2 == 0) else b"\x03"
    pubkey = prefix + x.to_bytes(32, "big")

    # HASH160
    h160 = hash160(pubkey)

    print(f"HASH160: {h160.hex()}")

    print("Loading database...")
    with open(DB_FILE, "rb") as f:
        db = f.read()

    if len(db) % 20 != 0:
        print("ERROR: DB size is not a multiple of 20 bytes.")
        return

    slots = len(db) // 20
    print(f"DB loaded: {slots} entries")

    found, slot = lookup_hash160(h160, db, slots)

    if found:
        print("\n=== RESULT: FOUND ===")
        print(f"Slot index: {slot}")
        print(f"HASH160: {h160.hex()}")
    else:
        print("\n=== RESULT: NOT FOUND ===")

if __name__ == "__main__":
    main()

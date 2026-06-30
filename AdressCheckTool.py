import hashlib
import struct

DB_FILE = "Hash_db_pif_Final.bin"

# ---------------- Base58Check decode ----------------
alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
alphabet_map = {c: i for i, c in enumerate(alphabet)}

def b58decode(addr):
    num = 0
    for c in addr:
        num = num * 58 + alphabet_map[c]
    full = num.to_bytes(25, byteorder="big")
    payload, checksum = full[:-4], full[-4:]
    if hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] != checksum:
        raise ValueError("Invalid Base58Check checksum")
    return payload  # version + hash160

# ---------------- Bech32 decode ----------------
# Minimal implementation for P2WPKH (20-byte hash)
bech32_charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
bech32_map = {c: i for i, c in enumerate(bech32_charset)}

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = (chk >> 25)
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_verify_checksum(hrp, data):
    return bech32_polymod(bech32_hrp_expand(hrp) + data) == 1

def bech32_decode(addr):
    addr = addr.lower()
    if "1" not in addr:
        raise ValueError("Invalid Bech32 address")

    hrp, data_part = addr.split("1", 1)
    data = [bech32_map[c] for c in data_part]

    if not bech32_verify_checksum(hrp, data):
        raise ValueError("Invalid Bech32 checksum")

    data = data[:-6]  # remove checksum

    # Convert 5-bit groups to 8-bit bytes
    bits = 0
    value = 0
    out = []
    for d in data[1:]:  # skip witness version
        value = (value << 5) | d
        bits += 5
        if bits >= 8:
            bits -= 8
            out.append((value >> bits) & 0xFF)

    return bytes(out)  # HASH160 for P2WPKH

# ---------------- Murmur3 (same as CUDA version) ----------------
def murmur3_32(data, seed):
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    h1 = seed
    length = len(data)
    rounded_end = (length & 0xfffffffc)

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

# ---------------- DB lookup (same as CUDA PIF20) ----------------
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
    print("=== Address → HASH160 → DB Lookup Tool ===")
    addr = input("Enter Bitcoin address: ").strip()

    # Detect address type
    if addr.startswith("1") or addr.startswith("3"):
        payload = b58decode(addr)
        version = payload[0]
        h160 = payload[1:]
        print(f"Address type: Base58 (version {version})")
    elif addr.lower().startswith("bc1"):
        h160 = bech32_decode(addr)
        print("Address type: Bech32 (P2WPKH)")
    else:
        print("Unknown address format")
        return

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

// hybrid_kernel.cu — Anchor + Linear hybrid engine (frozen grid, direct PRIV, clean heartbeat)
// - Absolute stride grid over [Gs_start_dec, Gs_end_dec]
// - Safe-window anchors (no mod-n), stateless linear walker (+G steps)
// - Chunked kernel launches to avoid TDR
// - Matches printed after kernel: "(N) Match PRIV: <priv> : <hash160>"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <ctime>

#pragma intrinsic(_umul128)

#define DEFINE_SECP256K1_CONSTANTS
#include "secp256k1.cuh"
#include "Sha256.cuh"
#include "Ripmed160.cuh"

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error in %s at line %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#endif

// ---------------- CONFIG ----------------
// S_START = 2^70
static const char* Gs_start_dec = "100160157190278508012948553706761263946222797637435022279920894145857442545664";
// S_END = curve order (same as builder)
static const char* Gs_end_dec = "115792089237316195423570985008687907852837564279074904382605163141518161494337";

static const uint32_t THREADS_PER_BLOCK = 256;
static const char* HASHDB_PATH = "Hash_db_pif_Final.bin";
static const uint32_t MAX_MATCHES_ON_GPU = 1025;

// MUST match builder
static const uint32_t TOTAL_SCALARS = 1000000;
static const uint32_t STEPS_PER_ROUND = 8;

static const char* RESUME_FILE_PATH = "RoundResume.txt";

// chunk size for TDR-safe launches
static const uint32_t TDLR_chunksize = 131072;

// proof interval placeholder (disabled here)
static const uint64_t FULL_PROOF_ROUND = 0;

// 5 forced anchors (must match builder)
static const char* FORCED_ANCHORS[] = {
    "2315841784746323908471419700173758157056751285581498087628491430454197",
    "4631683569492647816942839400347516314113502571162996175256982860908395",
    "6947525354238971725414259100521274471170253856744494262885474291362593",
    "9263367138985295633885678800695032628227005142325992350513965721816791",
    "11000248477545038565239243575825351246019568606512115916235334294657439"
};

static const size_t NUM_FORCED_ANCHORS =
sizeof(FORCED_ANCHORS) / sizeof(FORCED_ANCHORS[0]);

// ------------- STRIDE GRID + ANCHOR -------------
__device__ ECPointJac d_stridePoints[TOTAL_SCALARS];

__device__ BigInt     d_stride;        // stride
__device__ BigInt     d_start_scalar;  // S_START
__device__ BigInt     d_anchor_scalar; // current anchor scalar a
__device__ ECPointJac d_anchor_point;  // A = a*G
__device__ ECPointJac d_roundPoint;    // R = (round * STEPS_PER_ROUND) * G

// ------------- HIT RECORD (direct PRIV) -------------
struct HitRecord {
    BigInt  priv;      // actual scalar k for this hit
    uint8_t h160[20];  // hash160
};

// ------------- HOST BIGINT HELPERS -------------
static BigInt bigint_from_u64(uint64_t v) {
    BigInt r{};
    r.data[0] = (uint32_t)(v & 0xFFFFFFFFULL);
    r.data[1] = (uint32_t)(v >> 32);
    return r;
}

static BigInt bigint_add(const BigInt& a, const BigInt& b) {
    BigInt r{}; uint64_t c = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)a.data[i] + b.data[i] + c;
        r.data[i] = (uint32_t)s;
        c = s >> 32;
    }
    return r;
}

static BigInt bigint_sub(const BigInt& a, const BigInt& b) {
    BigInt r{}; uint64_t br = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = a.data[i], bv = b.data[i];
        uint64_t d = av - bv - br;
        r.data[i] = (uint32_t)d;
        br = (d >> 63) & 1ULL;
    }
    return r;
}

static bool bigint_ge(const BigInt& a, const BigInt& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.data[i] > b.data[i]) return true;
        if (a.data[i] < b.data[i]) return false;
    }
    return true;
}

static BigInt bigint_mul_u64(const BigInt& a, uint64_t m) {
    BigInt r{};
    uint32_t m_lo = (uint32_t)(m & 0xFFFFFFFFULL);
    uint32_t m_hi = (uint32_t)(m >> 32);
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t ai = (uint64_t)a.data[i];
        uint64_t prod_lo = ai * m_lo;
        uint64_t prod_hi = ai * m_hi;
        uint64_t sum = (prod_lo & 0xFFFFFFFFULL) + (carry & 0xFFFFFFFFULL);
        r.data[i] = (uint32_t)sum;
        uint64_t new_carry =
            (prod_lo >> 32) +
            (prod_hi & 0xFFFFFFFFULL) +
            (carry >> 32);
        carry = (prod_hi >> 32) + new_carry;
    }
    return r;
}

static BigInt bigint_div_u64(const BigInt& a, uint64_t d) {
    BigInt q{}; uint64_t rem = 0;
    for (int i = 7; i >= 0; --i) {
        uint64_t cur = (rem << 32) | a.data[i];
        uint64_t qword = cur / d;
        rem = cur % d;
        q.data[i] = (uint32_t)qword;
    }
    return q;
}

static BigInt bigint_mod(const BigInt& a, const BigInt& m) {
    BigInt r = a;
    while (bigint_ge(r, m)) {
        r = bigint_sub(r, m);
    }
    return r;
}

static std::string bigint_to_dec(const BigInt& b) {
    uint32_t t[8];
    for (int i = 0; i < 8; i++) t[i] = b.data[i];
    std::string o; o.reserve(78);
    while (1) {
        uint64_t c = 0; bool z = true;
        for (int i = 7; i >= 0; i--) {
            uint64_t cur = (c << 32) | t[i];
            t[i] = (uint32_t)(cur / 10);
            c = cur % 10;
            if (t[i]) z = false;
        }
        o.push_back((char)('0' + c));
        if (z) break;
    }
    std::reverse(o.begin(), o.end());
    return o;
}

static BigInt bigint_from_dec_string(const char* s) {
    BigInt r{};
    for (const char* p = s; *p; p++) {
        int dig = *p - '0';
        BigInt t{}; uint64_t c = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t v = (uint64_t)r.data[i] * 10ULL + c;
            t.data[i] = (uint32_t)v;
            c = v >> 32;
        }
        r = t;
        uint64_t a = (uint64_t)dig;
        for (int i = 0; i < 8 && a; i++) {
            uint64_t v = (uint64_t)r.data[i] + a;
            r.data[i] = (uint32_t)v;
            a = v >> 32;
        }
    }
    return r;
}

static std::vector<uint8_t> loadFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open " << p << "\n"; exit(1); }
    auto sz = f.tellg();
    if (sz <= 0) { std::cerr << "File empty or invalid: " << p << "\n"; exit(1); }
    f.seekg(0);
    std::vector<uint8_t> b((size_t)sz);
    if (!f.read((char*)b.data(), sz)) { std::cerr << "Read fail\n"; exit(1); }
    return b;
}

static uint64_t load_resume_rounds() {
    std::ifstream f(RESUME_FILE_PATH);
    if (!f) {
        std::ofstream nf(RESUME_FILE_PATH);
        if (nf) nf << "0\n";
        return 0;
    }
    uint64_t v = 0;
    if (!(f >> v)) return 0;
    return v;
}

static void save_resume_rounds(uint64_t rounds) {
    std::ofstream f(RESUME_FILE_PATH, std::ios::trunc);
    if (f) f << rounds << "\n";
}

static std::string h160_to_hex(const uint8_t h[20]) {
    char buf[41];
    for (int i = 0; i < 20; ++i)
        std::sprintf(&buf[i * 2], "%02x", (unsigned int)h[i]);
    buf[40] = '\0';
    return std::string(buf);
}

static void append_match_db(const std::string& s) {
    std::ofstream f("bingobook.txt", std::ios::app);
    if (f) f << s << "\n";
}

// ------------- DEVICE HELPERS -------------
__device__ uint32_t murmur3_32_dev(const uint8_t* d, int len, uint32_t seed) {
    const uint32_t c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
    uint32_t h1 = seed;
    int end = (len / 4) * 4;
    for (int i = 0; i < end; i += 4) {
        uint32_t k1 = (uint32_t)d[i] |
            ((uint32_t)d[i + 1] << 8) |
            ((uint32_t)d[i + 2] << 16) |
            ((uint32_t)d[i + 3] << 24);
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64u;
    }
    uint32_t k1 = 0;
    int t = len & 3;
    int idx = end;
    if (t == 3) k1 ^= (uint32_t)d[idx + 2] << 16;
    if (t >= 2) k1 ^= (uint32_t)d[idx + 1] << 8;
    if (t >= 1) {
        k1 ^= (uint32_t)d[idx];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
    }
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6bu;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
}

__device__ void compress_point33_dev(const ECPoint& A, uint8_t o[33]) {
    o[0] = 0x02 + (A.y.data[0] & 1);
    for (int w = 0; w < BIGINT_WORDS; w++) {
        uint32_t wd = A.x.data[BIGINT_WORDS - 1 - w];
        o[1 + w * 4 + 0] = (uint8_t)((wd >> 24) & 0xFF);
        o[1 + w * 4 + 1] = (uint8_t)((wd >> 16) & 0xFF);
        o[1 + w * 4 + 2] = (uint8_t)((wd >> 8) & 0xFF);
        o[1 + w * 4 + 3] = (uint8_t)(wd & 0xFF);
    }
}

__device__ BigInt bigint_add_dev(const BigInt& a, const BigInt& b) {
    BigInt r{}; uint64_t c = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)a.data[i] + b.data[i] + c;
        r.data[i] = (uint32_t)s;
        c = s >> 32;
    }
    return r;
}

__device__ BigInt bigint_mul_u64_dev(const BigInt& a, uint64_t m) {
    BigInt r{};
    uint32_t m_lo = (uint32_t)(m & 0xFFFFFFFFULL);
    uint32_t m_hi = (uint32_t)(m >> 32);
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t ai = (uint64_t)a.data[i];
        uint64_t prod_lo = ai * m_lo;
        uint64_t prod_hi = ai * m_hi;
        uint64_t sum = (prod_lo & 0xFFFFFFFFULL) + (carry & 0xFFFFFFFFULL);
        r.data[i] = (uint32_t)sum;
        uint64_t new_carry =
            (prod_lo >> 32) +
            (prod_hi & 0xFFFFFFFFULL) +
            (carry >> 32);
        carry = (prod_hi >> 32) + new_carry;
    }
    return r;
}

__device__ BigInt bigint_from_u64_dev(uint64_t v) {
    BigInt r{};
    r.data[0] = (uint32_t)(v & 0xFFFFFFFFULL);
    r.data[1] = (uint32_t)(v >> 32);
    return r;
}

__device__ bool match_hash160_pif20_dev(const uint8_t* h, const uint8_t* t, uint64_t slots) {
    uint64_t A = 0, B = 0;
    uint32_t C = 0;

    for (int i = 0; i < 8; ++i) A = (A << 8) | (uint64_t)h[i];
    for (int i = 8; i < 16; ++i) B = (B << 8) | (uint64_t)h[i];
    for (int i = 16; i < 20; ++i) C = (C << 8) | (uint32_t)h[i];

    const uint32_t seeds[3] = { 17, 31, 73 };

    for (int i = 0; i < 3; i++) {
        uint32_t hv = murmur3_32_dev(h, 20, seeds[i]);
        uint64_t slot = hv % slots;
        const uint8_t* base = t + slot * 20;

        uint64_t a = 0, b = 0;
        uint32_t c = 0;

        for (int j = 0; j < 8; ++j) a = (a << 8) | (uint64_t)base[j];
        if (a != A) continue;

        for (int j = 8; j < 16; ++j) b = (b << 8) | (uint64_t)base[j];
        for (int j = 16; j < 20; ++j) c = (c << 8) | (uint32_t)base[j];

        if (b == B && c == C) return true;
    }
    return false;
}

// ------------- KERNELS -------------

// build P[i] = (S_START + stride * i) * G once at startup (frozen grid)
__global__ void init_stride_points_kernel(uint32_t totalScalars)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalScalars) return;

    BigInt stride = d_stride;
    BigInt start = d_start_scalar;

    uint64_t eff_idx = (uint64_t)idx;

    BigInt bi = bigint_mul_u64_dev(stride, eff_idx);
    bi = bigint_add_dev(bi, start);

    ECPointJac P;
    scalar_multiply_jac_device(&P, &const_G_jacobian, &bi);
    d_stridePoints[idx] = P;
}

// scalar multiply wrapper
__global__ void scalar_mul_kernel(ECPointJac* out, BigInt k)
{
    scalar_multiply_jac_device(out, &const_G_jacobian, &k);
}

// HYBRID KERNEL (chunked):
// 1) Linear sweep: P = stridePoints[idx] + roundPoint, then STEPS_PER_ROUND adds
//    When a match is found, compute k = s_grid + round*STEPS_PER_ROUND + step
// 2) Anchor sweep: P = anchor_point + stridePoints[idx]
//    When a match is found, compute k = s_grid + anchor_scalar
__global__ void hybrid_kernel(
    const uint8_t* db,
    uint64_t slots,
    uint32_t totalScalars,
    HitRecord* hits,
    uint32_t* hit_count,
    uint32_t max_hits,
    uint64_t round_id,
    uint32_t base,
    uint32_t count)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stepThreads = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

    uint64_t start_idx = (uint64_t)base;
    uint64_t end_idx = (uint64_t)base + (uint64_t)count;

    BigInt stride = d_stride;
    BigInt start = d_start_scalar;

    // ---- 1) LINEAR SWEEP ----
    for (uint64_t idx = start_idx + tid; idx < end_idx; idx += stepThreads) {
        ECPointJac P = d_stridePoints[idx];
        add_point_jac(&P, &P, &d_roundPoint); // base for this round

        for (uint32_t step = 0; step < STEPS_PER_ROUND; ++step) {
            ECPoint A;
            jacobian_to_affine(&A, &P);

            uint8_t pk[33], sha[32], h160[20];
            compress_point33_dev(A, pk);
            sha256_gpu(pk, 33, sha);   // 33 bytes (compressed pubkey)
            ripemd160_gpu(sha, 32, h160);

            if (match_hash160_pif20_dev(h160, db, slots)) {
                uint32_t pos = atomicAdd(hit_count, 1u);
                if (pos < max_hits) {
                    BigInt off = bigint_mul_u64_dev(stride, (uint64_t)idx);
                    BigInt s_grid = bigint_add_dev(start, off);

                    BigInt roundBI = bigint_from_u64_dev(round_id);
                    BigInt off_round = bigint_mul_u64_dev(roundBI, (uint64_t)STEPS_PER_ROUND);
                    BigInt k = bigint_add_dev(s_grid, off_round);
                    BigInt stepBI = bigint_from_u64_dev(step);
                    k = bigint_add_dev(k, stepBI);

                    hits[pos].priv = k;
                    for (int i = 0; i < 20; ++i)
                        hits[pos].h160[i] = h160[i];
                }
            }

            add_point_jac(&P, &P, &const_G_jacobian); // next linear step
        }
    }

    __syncthreads();

    // ---- 2) ANCHOR SWEEP ----
    ECPointJac A0 = d_anchor_point;

    for (uint64_t idx = start_idx + tid; idx < end_idx; idx += stepThreads) {
        ECPointJac P = A0;
        add_point_jac(&P, &P, &d_stridePoints[idx]);

        ECPoint A;
        jacobian_to_affine(&A, &P);

        uint8_t pk[33], sha[32], h160[20];
        compress_point33_dev(A, pk);
        sha256_gpu(pk, 33, sha);   // 33 bytes here too
        ripemd160_gpu(sha, 32, h160);

        if (!match_hash160_pif20_dev(h160, db, slots)) continue;

        uint32_t pos = atomicAdd(hit_count, 1u);
        if (pos < max_hits) {
            BigInt off = bigint_mul_u64_dev(stride, (uint64_t)idx);
            BigInt s_grid = bigint_add_dev(start, off);

            BigInt k = bigint_add_dev(s_grid, d_anchor_scalar);

            hits[pos].priv = k;
            for (int i = 0; i < 20; ++i)
                hits[pos].h160[i] = h160[i];
        }
    }
}

// ------------- MAIN ENGINE -------------
int main(int argc, char** argv) {
    CHECK_CUDA(cudaSetDevice(0));
    srand((unsigned int)time(nullptr));

    std::vector<uint8_t> h_db = loadFile(HASHDB_PATH);
    if (h_db.empty() || (h_db.size() % 20) != 0) {
        std::cerr << "DB invalid\n";
        return 1;
    }
    uint64_t slots = h_db.size() / 20;
    double   mb = (double)h_db.size() / (1024.0 * 1024.0);

    BigInt p_host = { {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,
                       0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF} };
    ECPointJac G_host = {
        {{0x16F81798,0x59F2815B,0x2DCE28D9,0x029BFCDB,
          0xCE870B07,0x55A06295,0xF9DCBBAC,0x79BE667E}},
        {{0xFB10D4B8,0x9C47D08F,0xA6855419,0xFD17B448,
          0x0E1108A8,0x5DA4FBFC,0x26A3C465,0x483ADA77}},
        {{1,0,0,0,0,0,0,0}},
        false
    };
    BigInt n_host = { {0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,
                       0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF} };

    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_G_jacobian, &G_host, sizeof(ECPointJac)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt)));

    uint8_t* d_db = nullptr;
    CHECK_CUDA(cudaMalloc(&d_db, h_db.size()));
    CHECK_CUDA(cudaMemcpy(d_db, h_db.data(), h_db.size(), cudaMemcpyHostToDevice));

    uint64_t resume_rounds = load_resume_rounds();
    if (resume_rounds == 0) {
        printf("Program starting fresh (no resume).\n");
    }
    else {
        uint64_t offset = resume_rounds * (uint64_t)STEPS_PER_ROUND;
        printf("Program resuming from round %llu (linear offset = %llu steps).\n",
            (unsigned long long)resume_rounds,
            (unsigned long long)offset);
    }

    uint64_t rounds = resume_rounds;
    uint64_t anchors = 0;
    uint64_t visited = 0;

    // If resuming, skip forced anchors (they were already processed)
    if (resume_rounds > 0) {
        anchors = NUM_FORCED_ANCHORS;
    }

    std::random_device rd;
    std::mt19937_64    gen(rd());

    const uint64_t EFFECTIVE_POINTS = (uint64_t)TOTAL_SCALARS;

    BigInt start_host = bigint_from_dec_string(Gs_start_dec);
    BigInt end_host = bigint_from_dec_string(Gs_end_dec);

    BigInt range_host = bigint_sub(end_host, start_host);
    range_host = bigint_add(range_host, bigint_from_u64(1ULL));

    BigInt stride_host = bigint_div_u64(range_host, EFFECTIVE_POINTS);
    if (bigint_ge(bigint_from_u64(0), stride_host)) {
        fprintf(stderr, "Stride is zero\n");
        return 1;
    }

    CHECK_CUDA(cudaMemcpyToSymbol(d_stride, &stride_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_start_scalar, &start_host, sizeof(BigInt)));

    BigInt max_offset = bigint_mul_u64(stride_host, EFFECTIVE_POINTS - 1ULL);
    BigInt grid_max = bigint_add(start_host, max_offset);
    BigInt a_max = bigint_sub(end_host, grid_max);
    BigInt a_max_plus_one = bigint_add(a_max, bigint_from_u64(1ULL));

    int threads = THREADS_PER_BLOCK;
    int blocks_init = (TOTAL_SCALARS + threads - 1) / threads;
    if (blocks_init == 0) blocks_init = 1;

    init_stride_points_kernel << <blocks_init, threads >> > (TOTAL_SCALARS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    HitRecord* d_hits = nullptr;
    uint32_t* d_hit_count = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hits, sizeof(HitRecord) * MAX_MATCHES_ON_GPU));
    CHECK_CUDA(cudaMalloc(&d_hit_count, sizeof(uint32_t)));

    uint32_t zero = 0;

    using clk = std::chrono::steady_clock;
    auto t_run_start = clk::now(); // timing starts after init + resume

    printf("=== Hybrid_AnchorStride_Linear (safe-window anchors + stateless linear, chunked, frozen grid) ===\n");
    printf("Hashdb: %s : %.0f MB : %llu entries\n",
        HASHDB_PATH, mb, (unsigned long long)slots);
    printf("Full search range (anchor): %s -> %s\n", Gs_start_dec, Gs_end_dec);
    printf("TOTAL_SCALARS = %u\n", TOTAL_SCALARS);
    printf("STEPS_PER_ROUND = %u\n", STEPS_PER_ROUND);
    printf("THREADS_PER_BLOCK = %u\n", THREADS_PER_BLOCK);
    printf("MAX_MATCHES_ON_GPU = %u\n", MAX_MATCHES_ON_GPU);
    printf("TDLR_chunksize = %u\n", TDLR_chunksize);
    printf("Safe anchor window a_max = %s\n\n", bigint_to_dec(a_max).c_str());

    for (;;) {
        BigInt anchor_scalar;

        if (anchors < NUM_FORCED_ANCHORS) {
            BigInt forced_anchor = bigint_from_dec_string(FORCED_ANCHORS[anchors]);
            anchor_scalar = bigint_mod(forced_anchor, a_max_plus_one);
        }
        else {
            uint32_t state = (uint32_t)gen();
            BigInt rand256{};
            for (int i = 0; i < 8; ++i) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                rand256.data[i] = state;
            }
            anchor_scalar = bigint_mod(rand256, a_max_plus_one);
        }

        CHECK_CUDA(cudaMemcpyToSymbol(d_anchor_scalar, &anchor_scalar, sizeof(BigInt)));

        {
            ECPointJac* d_tmp = nullptr;
            CHECK_CUDA(cudaMalloc(&d_tmp, sizeof(ECPointJac)));
            scalar_mul_kernel << <1, 1 >> > (d_tmp, anchor_scalar);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            ECPointJac h_anchor{};
            CHECK_CUDA(cudaMemcpy(&h_anchor, d_tmp, sizeof(ECPointJac), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyToSymbol(d_anchor_point, &h_anchor, sizeof(ECPointJac)));

            CHECK_CUDA(cudaFree(d_tmp));
        }

        {
            BigInt roundBI = bigint_from_u64(rounds);
            BigInt k_round = bigint_mul_u64(roundBI, (uint64_t)STEPS_PER_ROUND);

            ECPointJac* d_tmp = nullptr;
            CHECK_CUDA(cudaMalloc(&d_tmp, sizeof(ECPointJac)));
            scalar_mul_kernel << <1, 1 >> > (d_tmp, k_round);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            ECPointJac h_round{};
            CHECK_CUDA(cudaMemcpy(&h_round, d_tmp, sizeof(ECPointJac), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyToSymbol(d_roundPoint, &h_round, sizeof(ECPointJac)));

            CHECK_CUDA(cudaFree(d_tmp));
        }

        CHECK_CUDA(cudaMemcpy(d_hit_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

        for (uint32_t base = 0; base < TOTAL_SCALARS; base += TDLR_chunksize) {
            uint32_t count = std::min(TDLR_chunksize, TOTAL_SCALARS - base);
            int blocks = (count + threads - 1) / threads;
            if (blocks == 0) blocks = 1;

            hybrid_kernel << <blocks, threads >> > (
                d_db,
                slots,
                TOTAL_SCALARS,
                d_hits,
                d_hit_count,
                MAX_MATCHES_ON_GPU,
                rounds,
                base,
                count);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        anchors++;
        visited += (uint64_t)TOTAL_SCALARS;

        rounds++;
        save_resume_rounds(rounds);

        auto   now = clk::now();
        auto   elapsed_run = std::chrono::duration_cast<std::chrono::seconds>(now - t_run_start).count();
        uint64_t rounds_this_run = (rounds - resume_rounds);
        uint64_t linear_points = rounds_this_run *
            (uint64_t)TOTAL_SCALARS *
            (uint64_t)STEPS_PER_ROUND;
        double   linear_M = (double)linear_points / 1e6;
        double   vm_anchor = (double)visited / 1e6;
        double   rate = (elapsed_run > 0)
            ? (vm_anchor + linear_M) / (double)elapsed_run
            : 0.0;

        std::string anchor_dec = bigint_to_dec(anchor_scalar);
        uint32_t hc = 0;
        CHECK_CUDA(cudaMemcpy(&hc, d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (hc > MAX_MATCHES_ON_GPU) hc = MAX_MATCHES_ON_GPU;

        printf("[Hb] Round=%llu  Anchor(%llu)=%s  AnchorsProcessed=%.3fM  "
            "LinearsProcessed=%.3fM  Speed=%.6fM/s  elapsed=%llus\n",
            (unsigned long long)(rounds - 1),
            (unsigned long long)anchors,
            anchor_dec.c_str(),
            vm_anchor,
            linear_M,
            rate,
            (unsigned long long)elapsed_run);
        fflush(stdout);

        if (hc > 0) {
            std::vector<HitRecord> h_hits(hc);
            CHECK_CUDA(cudaMemcpy(h_hits.data(), d_hits, sizeof(HitRecord) * hc, cudaMemcpyDeviceToHost));

            for (uint32_t i = 0; i < hc; ++i) {
                const HitRecord& h = h_hits[i];

                std::string priv_dec = bigint_to_dec(h.priv);
                std::string hhex = h160_to_hex(h.h160);
                printf("(%u) Match PRIV: %s : %s\n",
                    i + 1,
                    priv_dec.c_str(),
                    hhex.c_str());
                append_match_db(priv_dec + " " + hhex);
            }
        }

        // FULL_PROOF_ROUND is disabled in this clean kernel; proof logic should live in host code.
    }

    return 0;
}

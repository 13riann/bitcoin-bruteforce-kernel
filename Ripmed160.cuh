#ifndef RIPMED160_CUH
#define RIPMED160_CUH
#include <stdint.h>
#include <cuda_runtime.h>
__device__ uint32_t rol(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}
__device__ uint32_t f(uint32_t j, uint32_t x, uint32_t y, uint32_t z) {
    if (j <= 15) return x ^ y ^ z;
    if (j <= 31) return (x & y) | (~x & z);
    if (j <= 47) return (x | ~y) ^ z;
    if (j <= 63) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}
__device__ uint32_t K1(uint32_t j) {
    if (j <= 15) return 0x00000000;
    if (j <= 31) return 0x5A827999;
    if (j <= 47) return 0x6ED9EBA1;
    if (j <= 63) return 0x8F1BBCDC;
    return 0xA953FD4E;
}
__device__ uint32_t K2(uint32_t j) {
    if (j <= 15) return 0x50A28BE6;
    if (j <= 31) return 0x5C4DD124;
    if (j <= 47) return 0x6D703EF3;
    if (j <= 63) return 0x7A6D76E9;
    return 0x00000000;
}
__device__ __constant__ uint32_t R1[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
     3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
     1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
     4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
__device__ __constant__ uint32_t R2[80] = {
     5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
     6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
     8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
__device__ __constant__ uint32_t S1[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
};
__device__ __constant__ uint32_t S2[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
};
__device__ void ripemd160_gpu(const uint8_t* msg, int len, uint8_t* digest) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    uint8_t block[64] = { 0 };
    for (int i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) block[56 + i] = (bit_len >> (8 * i)) & 0xFF;
    uint32_t X[16];
    for (int i = 0; i < 16; i++)
        X[i] = ((uint32_t)block[4 * i]) | ((uint32_t)block[4 * i + 1] << 8) |
        ((uint32_t)block[4 * i + 2] << 16) | ((uint32_t)block[4 * i + 3] << 24);
    uint32_t A1 = h0, B1 = h1, C1 = h2, D1 = h3, E1 = h4;
    uint32_t A2 = h0, B2 = h1, C2 = h2, D2 = h3, E2 = h4;
    for (uint32_t j = 0; j < 80; j++) {
        uint32_t T = rol(A1 + f(j, B1, C1, D1) + X[R1[j]] + K1(j), S1[j]) + E1;
        A1 = E1; E1 = D1; D1 = rol(C1, 10); C1 = B1; B1 = T;
        T = rol(A2 + f(79 - j, B2, C2, D2) + X[R2[j]] + K2(j), S2[j]) + E2;
        A2 = E2; E2 = D2; D2 = rol(C2, 10); C2 = B2; B2 = T;
    }
    uint32_t T = h1 + C1 + D2;
    h1 = h2 + D1 + E2;
    h2 = h3 + E1 + A2;
    h3 = h4 + A1 + B2;
    h4 = h0 + B1 + C2;
    h0 = T;
    uint32_t H[5] = { h0,h1,h2,h3,h4 };
    for (int i = 0; i < 5; i++) {
        digest[4 * i] = H[i] & 0xFF;
        digest[4 * i + 1] = (H[i] >> 8) & 0xFF;
        digest[4 * i + 2] = (H[i] >> 16) & 0xFF;
        digest[4 * i + 3] = (H[i] >> 24) & 0xFF;
    }
}
#endif
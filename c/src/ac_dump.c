/*
 * ac_dump.c -- .npy files and SHA-256 (see ac_dump.h).
 *
 * The .npy format: NumPy Enhancement Proposal 1 (numpy/lib/format.py):
 * the magic string "\x93NUMPY", the version bytes 1 0, the length of the
 * header as a little-endian uint16, the header (a Python dict literal padded
 * with spaces and ended by a newline so that the data begin at a multiple
 * of 64 bytes), then the data.
 * SHA-256: FIPS PUB 180-4, Secs. 4.1.2, 4.2.2, 5.3.3 and 6.2.
 */
#include "ac_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ac_npy_write(const char *path, char type, int nd, const size_t *shape, const void *data)
{
    const uint16_t one = 1;
    if (*(const unsigned char *)&one != 1 || nd < 1 || nd > 2 || (type != 'f' && type != 'i'))
        return -1;
    char dict[160], head[256];
    if (nd == 1)
        snprintf(dict, sizeof dict, "{'descr': '<%c8', 'fortran_order': False, 'shape': (%zu,), }",
                 type, shape[0]);
    else
        snprintf(dict, sizeof dict,
                 "{'descr': '<%c8', 'fortran_order': False, 'shape': (%zu, %zu), }", type, shape[0],
                 shape[1]);
    size_t dl = strlen(dict), total = 10 + dl + 1;
    size_t pad = (64 - total % 64) % 64;
    uint16_t hl = (uint16_t)(dl + pad + 1);
    memcpy(head, "\x93NUMPY\x01\x00", 8);
    head[8] = (char)(hl & 255);
    head[9] = (char)(hl >> 8);
    memcpy(head + 10, dict, dl);
    memset(head + 10 + dl, ' ', pad);
    head[10 + dl + pad] = '\n';
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    size_t count = shape[0] * (nd == 2 ? shape[1] : 1);
    int bad = fwrite(head, 1, 10 + (size_t)hl, f) != 10 + (size_t)hl;
    if (type == 'f')
        bad |= count && fwrite(data, sizeof(double), count, f) != count;
    else {
        const int *v = data;
        int64_t buf[1024];
        for (size_t i = 0; i < count && !bad;) {
            size_t n = count - i < 1024 ? count - i : 1024;
            for (size_t k = 0; k < n; k++)
                buf[k] = v[i + k];
            bad |= fwrite(buf, sizeof buf[0], n, f) != n;
            i += n;
        }
    }
    bad |= fclose(f) != 0;
    return bad ? -1 : 0;
}

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static uint32_t rotr(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

static void sha_block(ac_sha256 *s, const unsigned char *p)
{
    uint32_t w[64], a[8];
    for (int t = 0; t < 16; t++)
        w[t] = (uint32_t)p[4 * t] << 24 | (uint32_t)p[4 * t + 1] << 16 |
               (uint32_t)p[4 * t + 2] << 8 | (uint32_t)p[4 * t + 3];
    for (int t = 16; t < 64; t++) {
        uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
        uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    memcpy(a, s->h, sizeof a);
    for (int t = 0; t < 64; t++) {
        uint32_t S1 = rotr(a[4], 6) ^ rotr(a[4], 11) ^ rotr(a[4], 25);
        uint32_t ch = (a[4] & a[5]) ^ (~a[4] & a[6]);
        uint32_t t1 = a[7] + S1 + ch + K256[t] + w[t];
        uint32_t S0 = rotr(a[0], 2) ^ rotr(a[0], 13) ^ rotr(a[0], 22);
        uint32_t mj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
        uint32_t t2 = S0 + mj;
        a[7] = a[6], a[6] = a[5], a[5] = a[4], a[4] = a[3] + t1;
        a[3] = a[2], a[2] = a[1], a[1] = a[0], a[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++)
        s->h[i] += a[i];
}

void ac_sha256_init(ac_sha256 *s)
{
    static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, h0, sizeof h0);
    s->len = 0;
    s->nbuf = 0;
}

void ac_sha256_add(ac_sha256 *s, const void *data, size_t n)
{
    const unsigned char *p = data;
    s->len += n;
    while (n > 0) {
        size_t k = 64 - (size_t)s->nbuf;
        if (k > n)
            k = n;
        memcpy(s->buf + s->nbuf, p, k);
        s->nbuf += (int)k;
        p += k;
        n -= k;
        if (s->nbuf == 64) {
            sha_block(s, s->buf);
            s->nbuf = 0;
        }
    }
}

void ac_sha256_hex(ac_sha256 *s, char hex[65])
{
    uint64_t bits = s->len * 8;
    unsigned char pad[72] = {0x80}, lenb[8];
    size_t np = (size_t)(s->nbuf < 56 ? 56 - s->nbuf : 120 - s->nbuf);
    for (int i = 0; i < 8; i++)
        lenb[i] = (unsigned char)(bits >> (56 - 8 * i));
    ac_sha256_add(s, pad, np);
    ac_sha256_add(s, lenb, 8);
    for (int i = 0; i < 8; i++)
        snprintf(hex + 8 * i, 9, "%08x", (unsigned)s->h[i]);
}

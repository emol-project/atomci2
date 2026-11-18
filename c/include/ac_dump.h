/*
 * ac_dump.h -- files for programs outside: arrays in NumPy's .npy format
 * and the SHA-256 of a text (acfs "dump=").
 */
#ifndef AC_DUMP_H
#define AC_DUMP_H

#include <stddef.h>
#include <stdint.h>

/* An array as a .npy file of format version 1.0 (little endian, C order):
   type 'f' writes float64 ('<f8') from double data[], 'i' int64 ('<i8')
   from int data[] (widened). nd = 1 or 2 dimensions, shape[nd]. Returns 0,
   or -1 when the file cannot be written (or the machine is big endian). */
int ac_npy_write(const char *path, char type, int nd, const size_t *shape, const void *data);

typedef struct {
    uint32_t h[8];
    uint64_t len;
    unsigned char buf[64];
    int nbuf;
} ac_sha256;

void ac_sha256_init(ac_sha256 *s);
void ac_sha256_add(ac_sha256 *s, const void *data, size_t n);
/* the digest as 64 hexadecimal digits and a terminating 0 */
void ac_sha256_hex(ac_sha256 *s, char hex[65]);

#endif

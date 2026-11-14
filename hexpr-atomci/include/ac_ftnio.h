/*
 * ac_ftnio.h -- gfortran unformatted sequential files.
 *
 * Each record is <int32 length> <payload> <int32 length>, in native byte
 * order. gfortran splits records longer than 2^31-1 bytes into subrecords
 * marked by negative lengths; those are rejected here (ATOMCI's files never
 * come close).
 *
 * A record is read whole into a buffer; ac_ftn_cursor then unpacks its
 * fields in order with bounds checks.
 */
#ifndef AC_FTNIO_H
#define AC_FTNIO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    FILE *fp;
    unsigned char *buf; /* the last record read */
    size_t len;         /* its length in bytes */
    size_t cap;
} ac_ftn_file;

/* Status codes. */
enum {
    AC_FTN_OK = 0,
    AC_FTN_EOF = 1,     /* clean end of file before a record */
    AC_FTN_ERR_IO = -1, /* fopen/fread/fwrite failed */
    AC_FTN_ERR_FMT = -2 /* truncated record, mismatched or negative markers */
};

int ac_ftn_open(ac_ftn_file *f, const char *path, const char *mode);
void ac_ftn_close(ac_ftn_file *f);
int ac_ftn_rewind(ac_ftn_file *f);

/* Read the next record into f->buf / f->len. */
int ac_ftn_read(ac_ftn_file *f);

/* Append one record. */
int ac_ftn_write(ac_ftn_file *f, const void *data, size_t len);

/* Sequential, bounds-checked unpacking of one record. Once a read runs past
   the end, `bad` is set and every later read returns 0. */
typedef struct {
    const unsigned char *p;
    size_t len, pos;
    int bad;
} ac_ftn_cursor;

void ac_ftn_cursor_init(ac_ftn_cursor *c, const ac_ftn_file *f);
int32_t ac_ftn_i32(ac_ftn_cursor *c);
int16_t ac_ftn_i16(ac_ftn_cursor *c);
double ac_ftn_f64(ac_ftn_cursor *c);
/* bytes left in the record */
size_t ac_ftn_left(const ac_ftn_cursor *c);

#endif

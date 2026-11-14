/*
 * ac_ftnio.c -- gfortran unformatted sequential files (see ac_ftnio.h).
 */
#include "ac_ftnio.h"

#include <stdlib.h>
#include <string.h>

int ac_ftn_open(ac_ftn_file *f, const char *path, const char *mode)
{
    memset(f, 0, sizeof *f);
    f->fp = fopen(path, mode);
    return f->fp ? AC_FTN_OK : AC_FTN_ERR_IO;
}

void ac_ftn_close(ac_ftn_file *f)
{
    if (f->fp)
        fclose(f->fp);
    free(f->buf);
    memset(f, 0, sizeof *f);
}

int ac_ftn_rewind(ac_ftn_file *f)
{
    return fseek(f->fp, 0L, SEEK_SET) == 0 ? AC_FTN_OK : AC_FTN_ERR_IO;
}

int ac_ftn_read(ac_ftn_file *f)
{
    int32_t head, tail;
    size_t got = fread(&head, 1, sizeof head, f->fp);
    if (got == 0 && feof(f->fp))
        return AC_FTN_EOF;
    if (got != sizeof head || head < 0)
        return AC_FTN_ERR_FMT;

    size_t n = (size_t)head;
    if (n > f->cap) {
        unsigned char *nb = realloc(f->buf, n);
        if (!nb)
            return AC_FTN_ERR_IO;
        f->buf = nb;
        f->cap = n;
    }
    if (fread(f->buf, 1, n, f->fp) != n)
        return AC_FTN_ERR_FMT;
    if (fread(&tail, 1, sizeof tail, f->fp) != sizeof tail || tail != head)
        return AC_FTN_ERR_FMT;
    f->len = n;
    return AC_FTN_OK;
}

int ac_ftn_write(ac_ftn_file *f, const void *data, size_t len)
{
    if (len > INT32_MAX)
        return AC_FTN_ERR_FMT;
    int32_t m = (int32_t)len;
    if (fwrite(&m, sizeof m, 1, f->fp) != 1 ||
        (len && fwrite(data, 1, len, f->fp) != len) ||
        fwrite(&m, sizeof m, 1, f->fp) != 1)
        return AC_FTN_ERR_IO;
    return AC_FTN_OK;
}

void ac_ftn_cursor_init(ac_ftn_cursor *c, const ac_ftn_file *f)
{
    c->p = f->buf;
    c->len = f->len;
    c->pos = 0;
    c->bad = 0;
}

static int take(ac_ftn_cursor *c, void *dst, size_t n)
{
    if (c->bad || c->len - c->pos < n) {
        c->bad = 1;
        memset(dst, 0, n);
        return 0;
    }
    memcpy(dst, c->p + c->pos, n);
    c->pos += n;
    return 1;
}

int32_t ac_ftn_i32(ac_ftn_cursor *c)
{
    int32_t v;
    take(c, &v, sizeof v);
    return v;
}

int16_t ac_ftn_i16(ac_ftn_cursor *c)
{
    int16_t v;
    take(c, &v, sizeof v);
    return v;
}

double ac_ftn_f64(ac_ftn_cursor *c)
{
    double v;
    take(c, &v, sizeof v);
    return v;
}

size_t ac_ftn_left(const ac_ftn_cursor *c)
{
    return c->bad ? 0 : c->len - c->pos;
}

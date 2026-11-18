/*
 * ac_lsj.c -- the CI in the LSJ basis (see ac_lsj.h).
 *
 * The formulas are derived in the technical note (Secs. 3.6 and 5). Sources
 * cited below by author and year:
 *   Edmonds (1957)   A. R. Edmonds, Angular Momentum in Quantum Mechanics,
 *                    Princeton University Press (1957), Eqs. (7.1.1),
 *                    (7.1.5), (7.1.6)
 *   Neese (2001)     F. Neese, J. Chem. Phys. 115, 11080 (2001)
 *   Pachucki (2004)  K. Pachucki, Phys. Rev. A 69, 052502 (2004): Eq. (32)
 *                    (alpha^2 corrections to g_J), Eqs. (59)-(62)
 *                    (relativistic M1)
 *   Jentschura (2010) U. D. Jentschura, Phys. Rev. A 81, 012512 (2010),
 *                    Eq. (10) (the H6 term, as a check)
 *   Lindgren-Rosen (1974) I. Lindgren and A. Rosen, Case Stud. At. Phys. 4,
 *                    93 (1974) (hyperfine operators)
 *   Sakurai (1967), Messiah (1962)  J. J. Sakurai, Advanced Quantum
 *                    Mechanics (1967), ch. 2; A. Messiah, Quantum Mechanics,
 *                    Vol. II (1962), ch. XXI (radiative rates)
 * Physical constants are those of CODATA 2018 (ac_lsj.h).
 */
/* mkstemp, ftruncate, posix_fallocate, mmap (the rows of w and w2 on disk) */
#define _POSIX_C_SOURCE 200809L
#include "ac_lsj.h"
#include "ac_dump.h"
#include "ac_radial.h"
#include "ac_wigner.h"

#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define AC_LSJ_HAVE_MMAP 1
#endif

static int fail(char *err, int errlen, const char *fmt, ...)
{
    if (err && errlen > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, (size_t)errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* the name of a term, "3F": L as its letter (S P D F G H I K L M N O Q R T U
   V W X Y Z, no J and no P twice), above L = 20 as "3[21]" */
static void term_name(char *buf, size_t n, int ts, int tl)
{
    static const char sym[] = "SPDFGHIKLMNOQRTUVWXYZ";
    if (tl / 2 < (int)sizeof sym - 1)
        snprintf(buf, n, "%d%c", ts + 1, sym[tl / 2]);
    else
        snprintf(buf, n, "%d[%d]", ts + 1, tl / 2);
}

static ac_lab total(const ac_ci *ci, int i)
{
    return ci->csf[i].c.cpl[ci->csf[i].c.nsh];
}

/* The total (S, L) of the CSFs idx[0..dim-1], numbered: lab[p] is the number
   of the label of idx[p], (*labs)[x] the label of number x, *nlab how many
   there are. What a pair of CSFs takes from the recoupling to J depends on
   its two labels only, so it is kept in a table of *nlab^2 numbers instead
   of being computed for every pair. Returns lab (free it and *labs), or
   NULL when out of memory. */
static int *label_numbers(const ac_ci *ci, int dim, const int *idx, ac_lab **labs, int *nlab)
{
    int cap = 16, n = 0;
    int *lab = malloc((size_t)(dim > 0 ? dim : 1) * sizeof *lab);
    ac_lab *l = malloc((size_t)cap * sizeof *l);
    for (int p = 0; p < dim && lab && l; p++) {
        ac_lab a = total(ci, idx[p]);
        int x;
        for (x = 0; x < n; x++)
            if (l[x].ts == a.ts && l[x].tl == a.tl)
                break;
        if (x == n) {
            if (n == cap) {
                ac_lab *nl = realloc(l, (size_t)(cap *= 2) * sizeof *l);
                if (!nl) {
                    free(l);
                    l = NULL;
                    break;
                }
                l = nl;
            }
            l[n++] = a;
        }
        lab[p] = x;
    }
    if (!lab || !l) {
        free(lab);
        free(l);
        return NULL;
    }
    *labs = l;
    *nlab = n;
    return lab;
}

/* the number of electrons CSF j must move to become CSF i (slot lists are
   in increasing orbital order) */
static int moved(const ac_ci *ci, int i, int j)
{
    const ac_ci_csf *F = &ci->csf[i], *G = &ci->csf[j];
    int p = 1, q = 1, nm = 0;
    while (p <= F->c.nsh || q <= G->c.nsh) {
        int sf = p <= F->c.nsh ? F->seq[p] : 1 << 30;
        int sg = q <= G->c.nsh ? G->seq[q] : 1 << 30;
        int d = 0;
        if (sf == sg)
            d = F->c.nel[p++] - G->c.nel[q++];
        else if (sf < sg)
            d = F->c.nel[p++];
        else
            q++;
        if (d > 0)
            nm += d;
    }
    return nm;
}

/* CSFs i and j have the same orbitals with the same occupations */
static int same_config(const ac_ci *ci, int i, int j)
{
    const ac_ci_csf *F = &ci->csf[i], *G = &ci->csf[j];
    if (F->c.nsh != G->c.nsh)
        return 0;
    for (int s = 1; s <= F->c.nsh; s++)
        if (F->seq[s] != G->seq[s] || F->c.nel[s] != G->c.nel[s])
            return 0;
    return 1;
}

/* The CSFs of one configuration (same orbitals and occupations) are
   neighbours in the list: gs[A] .. gs[A+1]-1 is such a run, A < *ng. The
   elements of a pair of runs are computed together (ac_ci_hij_block,
   ac_ci_bp2_block, ac_ci_prop2_block). A run is cut after AC_BP_BLOCK CSFs
   (default 512): a block of two runs needs a few arrays of (run length)^2
   numbers per thread. (The CSFs of a configuration are not in the order of
   their total (S, L), so a run holds several.) NULL when out of memory;
   free with free. */
static int *config_runs(const ac_ci *ci, int *ng)
{
    int n = ci->ncsf, m = 0;
    const char *sb = getenv("AC_BP_BLOCK");
    int maxrun = sb && atoi(sb) > 0 ? atoi(sb) : 512;
    int *gs = malloc(((size_t)n + 1) * sizeof *gs);
    if (!gs)
        return NULL;
    for (int i = 0; i < n; i++)
        if (i == 0 || !same_config(ci, i - 1, i) || i - gs[m - 1] >= maxrun)
            gs[m++] = i;
    gs[m] = n;
    *ng = m;
    return gs;
}

/* ---- the sparse form of h, w and w2 -------------------------------------- */

int ac_lsj_sparse_from = 10000;

/* AC_TIME: the memory of the process now and its highest so far (Linux) */
static void mem_note(const char *what)
{
    if (!getenv("AC_TIME"))
        return;
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long rss = -1, hwm = -1;
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "VmRSS:", 6))
            rss = atol(line + 6);
        else if (!strncmp(line, "VmHWM:", 6))
            hwm = atol(line + 6);
    }
    fclose(f);
    fprintf(stderr, "memory %s: now %.1f MB, highest %.1f MB\n", what, rss / 1024.0, hwm / 1024.0);
}

/* AC_TIME, rows in a file: where the memory of the machine and of the program
   stands and what the program has read from the disk so far (Linux: /proc;
   nothing is printed for what cannot be read) */
static void io_note(const char *what)
{
    static const char *const key[] = {"MemFree:", "MemAvailable:", "Cached:", "Dirty:", "Writeback:",
                                      "RssAnon:", "RssFile:", "RssShmem:", "read_bytes:", NULL};
    static const char *const file[] = {"/proc/meminfo", "/proc/self/status", "/proc/self/io"};
    char line[256];
    fprintf(stderr, "%s:", what);
    for (int k = 0; k < 3; k++) {
        FILE *f = fopen(file[k], "r");
        if (!f)
            continue;
        while (fgets(line, sizeof line, f))
            for (int u = 0; key[u]; u++) {
                size_t len = strlen(key[u]);
                if (!strncmp(line, key[u], len))
                    fprintf(stderr, " %.*s %.2f GB", (int)len - 1, key[u],
                            atof(line + len) / (k == 2 ? 1073741824.0 : 1048576.0));
            }
        fclose(f);
    }
    long maj = -1;
    FILE *f = fopen("/proc/self/stat", "r");
    if (f) {
        /* the fields after the name in parentheses: state ppid pgrp session tty tpgid flags
           minflt cminflt majflt */
        if (fgets(line, sizeof line, f)) {
            const char *p = strrchr(line, ')');
            long v[10];
            char st;
            if (p && sscanf(p + 1, " %c %ld %ld %ld %ld %ld %ld %ld %ld %ld", &st, &v[0], &v[1], &v[2],
                            &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]) == 10)
                maj = v[8];
        }
        fclose(f);
    }
    if (maj >= 0)
        fprintf(stderr, " major faults %ld", maj);
    fprintf(stderr, "\n");
}

static int want_sparse(int n)
{
    if (getenv("AC_LSJ_DENSE"))
        return 0;
    if (getenv("AC_LSJ_SPARSE"))
        return 1;
    return ac_lsj_sparse_from > 0 && n >= ac_lsj_sparse_from;
}

static void csr_free(ac_lsj_csr *A)
{
    free(A->start);
    free(A->col);
    free(A->val);
    A->start = NULL;
    A->col = NULL;
    A->val = NULL;
}

static void file_free(ac_lsj_file *F)
{
#ifdef AC_LSJ_HAVE_MMAP
    if (F->map) {
        munmap(F->map, F->maplen);
        close(F->fd);
    }
#endif
    free(F->at);
    free(F->left);
    free(F->next);
    memset(F, 0, sizeof *F);
}

static void rows_free(ac_lsj_rows *R)
{
    for (int k = 0; R->slab && k < R->nslab; k++)
        free(R->slab[k]);
    free(R->slab);
    free(R->len);
    free(R->col);
    free(R->val);
    memset(R, 0, sizeof *R);
}

/* The file of rows kept on disk: an unnamed file in dir, mapped with room
   for far more than will be used (the file is sparse: only what is
   allocated in it takes space), for rows of nlab labels. 0, or -1. */
#define ROWS_CHUNK ((size_t)256 << 20) /* the file is allocated this much at a time */
#define ROWS_PIECE0 ((size_t)1 << 20)  /* the first piece of a label, */
#define ROWS_PIECE ((size_t)64 << 20)  /* a quarter more for each further one, up to this */
static int file_map(ac_lsj_file *F, const char *dir, int nlab)
{
#ifdef AC_LSJ_HAVE_MMAP
    size_t plen = strlen(dir) + 32, nl = nlab > 0 ? (size_t)nlab : 1;
    char *path = malloc(plen);
    if (!path)
        return -1;
    snprintf(path, plen, "%s/ac_lsj_rows.XXXXXX", dir);
    int fd = mkstemp(path);
    if (fd >= 0)
        unlink(path);
    free(path);
    if (fd < 0)
        return -1;
    F->at = calloc(nl, sizeof *F->at);
    F->left = calloc(nl, sizeof *F->left);
    F->next = calloc(nl, sizeof *F->next);
    F->nlab = nlab;
    /* 1 TiB of addresses if the file system and the address space allow it, else less */
    for (size_t len = (size_t)1 << 40; F->at && F->left && F->next && len >= ROWS_CHUNK; len >>= 1) {
        if (ftruncate(fd, (off_t)len))
            continue;
        void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED)
            continue;
        F->map = p;
        F->maplen = len;
        F->fd = fd;
        for (size_t y = 0; y < nl; y++)
            F->next[y] = ROWS_PIECE0;
        return 0;
    }
    close(fd);
    free(F->at);
    free(F->left);
    free(F->next);
    memset(F, 0, sizeof *F);
#else
    (void)F, (void)dir, (void)nlab;
#endif
    return -1;
}

/* room for `need` bytes in the file F among the rows of label lab: its place,
   or NULL (no room on the disk, or the mapping is used up) */
static void *file_room(ac_lsj_file *F, int lab, size_t need)
{
    void *p = NULL;
#ifdef AC_LSJ_HAVE_MMAP
    need = (need + 7) & ~(size_t)7;
#pragma omp critical(ac_lsj_rows_room)
    {
        int ok = 1;
        if (need > F->left[lab]) {
            /* a new piece of the file for this label (what is left of the old one stays
               unused: less than one row) */
            size_t piece = need > F->next[lab] ? need : F->next[lab];
            size_t end = F->mapused + piece;
            ok = end <= F->maplen;
            /* allocate the blocks of the file before they are written to: a write into a
               mapped hole of a full disk would end the program */
            while (ok && end > F->mapalloc) {
                size_t more = F->mapalloc + ROWS_CHUNK <= F->maplen ? ROWS_CHUNK : F->maplen - F->mapalloc;
                if (more == 0 || posix_fallocate(F->fd, (off_t)F->mapalloc, (off_t)more))
                    ok = 0;
                else
                    F->mapalloc += more;
            }
            if (ok) {
                F->at[lab] = F->map + F->mapused;
                F->left[lab] = piece;
                F->mapused = end;
                /* a quarter more each time: the unused end of the last piece
                   is then a small part of what the label has */
                F->next[lab] += F->next[lab] / 4;
                if (F->next[lab] > ROWS_PIECE)
                    F->next[lab] = ROWS_PIECE;
            }
        }
        if (ok) {
            p = F->at[lab];
            F->at[lab] += need;
            F->left[lab] -= need;
        }
    }
#else
    (void)F, (void)lab, (void)need;
#endif
    return p;
}

/* empty rows of an n x n matrix with room for nslab allocations; 0, or -1 */
static int rows_alloc(ac_lsj_rows *R, int n, int nslab)
{
    size_t N = n > 0 ? (size_t)n : 1, S = nslab > 0 ? (size_t)nslab : 1;
    memset(R, 0, sizeof *R);
    R->len = calloc(N, sizeof *R->len);
    R->col = calloc(N, sizeof *R->col);
    R->val = calloc(N, sizeof *R->val);
    R->slab = calloc(S, sizeof *R->slab);
    R->nslab = nslab;
    if (!R->len || !R->col || !R->val || !R->slab) {
        rows_free(R);
        return -1;
    }
    return 0;
}

/* the nonzero elements of the rows of one configuration as they are
   computed: (row within the configuration, column, value) */
typedef struct {
    int *r, *c;
    double *v;
    size_t n, cap;
} stage;

static int spush(stage *s, int r, int c, double v)
{
    if (s->n == s->cap) {
        size_t cap = s->cap ? 2 * s->cap : 4096;
        int *nr = realloc(s->r, cap * sizeof *nr);
        if (nr)
            s->r = nr;
        int *nc = realloc(s->c, cap * sizeof *nc);
        if (nc)
            s->c = nc;
        double *nv = realloc(s->v, cap * sizeof *nv);
        if (nv)
            s->v = nv;
        if (!nr || !nc || !nv)
            return 1;
        s->cap = cap;
    }
    s->r[s->n] = r, s->c[s->n] = c, s->v[s->n] = v;
    s->n++;
    return 0;
}

static void stage_free(stage *s)
{
    free(s->r);
    free(s->c);
    free(s->v);
    memset(s, 0, sizeof *s);
}

/* The staged elements (emptied) become the rows i0 .. i0+ni-1 of R, in one
   allocation, slab number run; or, with R in a file, each row in the part of
   the file of its label glab[i] (the values, then the columns). The elements
   of a row were staged in the order of their columns and keep it. One thread
   has the whole configuration, so nothing is shared. Returns 0, or 1 when
   out of memory (or out of room in the file). */
static int rows_put(ac_lsj_rows *R, int run, int i0, int ni, stage *s, const int *glab)
{
    size_t ne = s->n;
    s->n = 0;
    if (!ne)
        return 0;
    size_t *at = calloc((size_t)ni + 1, sizeof *at);
    void *slab = R->file || !at ? NULL : malloc(ne * (sizeof(double) + sizeof(int)));
    if (!at || (!slab && !R->file)) {
        free(at);
        return 1;
    }
    for (size_t k = 0; k < ne; k++)
        at[s->r[k] + 1]++;
    if (R->file) {
        /* at[r] becomes the number of the elements of row r that are placed */
        for (int r = 0; r < ni; r++) {
            size_t len = at[r + 1];
            at[r] = 0;
            if (!len)
                continue;
            double *v = file_room(R->file, glab[i0 + r], len * (sizeof(double) + sizeof(int)));
            if (!v) {
                for (int q = 0; q < ni; q++)
                    R->len[i0 + q] = 0;
                free(at);
                return 1;
            }
            R->len[i0 + r] = (int)len;
            R->val[i0 + r] = v;
            R->col[i0 + r] = (int *)(v + len);
        }
        for (size_t k = 0; k < ne; k++) {
            int i = i0 + s->r[k];
            size_t e = at[s->r[k]]++;
            R->col[i][e] = s->c[k];
            R->val[i][e] = s->v[k];
        }
        free(at);
        return 0;
    }
    double *v = slab;
    int *c = (int *)(v + ne);
    for (int r = 0; r < ni; r++) {
        R->len[i0 + r] = (int)at[r + 1];
        at[r + 1] += at[r];
        R->col[i0 + r] = c + at[r];
        R->val[i0 + r] = v + at[r];
    }
    for (size_t k = 0; k < ne; k++) {
        size_t e = at[s->r[k]]++;
        c[e] = s->c[k];
        v[e] = s->v[k];
    }
    R->slab[run] = slab;
    free(at);
    return 0;
}

/* Row i of w and w2 is complete: count its places by the label of the column
   and keep its diagonal elements (m->wcnt, m->wdiag; see ac_lsj.h). A place
   that both matrices have counts once, as in the J block. */
static void rows_count(ac_lsj *m, int i)
{
    const ac_lsj_rows *W[2] = {&m->sw, &m->sw2};
    int n1 = W[0]->len ? W[0]->len[i] : 0, n2 = W[1]->len ? W[1]->len[i] : 0;
    const int *k1 = n1 ? W[0]->col[i] : NULL, *k2 = n2 ? W[1]->col[i] : NULL;
    unsigned *cnt = m->wcnt + (size_t)i * (size_t)m->nglab;
    if (n1 && k1[0] == i)
        m->wdiag[i] = W[0]->val[i][0];
    if (n2 && k2[0] == i)
        m->wdiag[(size_t)m->n + i] = W[1]->val[i][0];
    for (int e1 = 0, e2 = 0; e1 < n1 || e2 < n2;) {
        int col = e2 == n2 || (e1 < n1 && k1[e1] <= k2[e2]) ? k1[e1] : k2[e2];
        if (e1 < n1 && k1[e1] == col)
            e1++;
        if (e2 < n2 && k2[e2] == col)
            e2++;
        if (col != i)
            cnt[m->glab[col]]++;
    }
}

/* a part of the file: [a, b) */
typedef struct {
    size_t a, b;
} span;

static int by_span(const void *x, const void *y)
{
    const span *p = x, *q = y;
    return p->a < q->a ? -1 : p->a > q->a;
}

/* Rows kept in a file: tell the system that the rows row[0 .. nr-1] of w and
   w2 are about to be read, so that it reads them ahead, in the order of the
   file and what lies together in one piece (a disk reads fast in order;
   what is less than ROWS_GAP apart is read through, which costs less than
   moving the head). sp: room for 2 nr spans. */
#define ROWS_GAP ((size_t)1 << 20)
static void rows_ahead(const ac_lsj *m, const int *row, int nr, span *sp)
{
#ifdef AC_LSJ_HAVE_MMAP
    static size_t page;
    const ac_lsj_file *F = &m->wfile;
    const ac_lsj_rows *W[2] = {&m->sw, &m->sw2};
    if (!F->map)
        return;
    if (!page) {
        long ps = sysconf(_SC_PAGESIZE);
        page = ps > 0 ? (size_t)ps : 4096;
    }
    int ns = 0;
    for (int k = 0; k < 2; k++)
        for (int u = 0; W[k]->len && u < nr; u++) {
            int i = row[u];
            if (!W[k]->len[i])
                continue;
            sp[ns].a = (size_t)((const char *)W[k]->val[i] - F->map);
            sp[ns].b = sp[ns].a + (size_t)W[k]->len[i] * (sizeof(double) + sizeof(int));
            ns++;
        }
    qsort(sp, (size_t)ns, sizeof *sp, by_span);
    for (int u = 0; u < ns;) {
        size_t a = sp[u].a & ~(page - 1), b = sp[u].b;
        for (u++; u < ns && sp[u].a <= b + ROWS_GAP; u++)
            if (sp[u].b > b)
                b = sp[u].b;
        (void)posix_madvise(F->map + a, b - a, POSIX_MADV_WILLNEED);
    }
#else
    (void)m, (void)row, (void)nr, (void)sp;
#endif
}

/* (place in the file, row of the block) */
typedef struct {
    size_t key;
    int r;
} rowat;

static int by_rowat(const void *x, const void *y)
{
    const rowat *p = x, *q = y;
    return p->key < q->key ? -1 : p->key > q->key ? 1 : p->r - q->r;
}

static int tri2(int a, int b, int c);

const ac_lsj_hterm *ac_lsj_extra = NULL;
int ac_lsj_nextra = 0;
void (*ac_lsj_sink)(void *ctx, int term, int i, int j, double v) = NULL;
void *ac_lsj_sink_ctx = NULL;
int ac_lsj_nostore = 0, ac_lsj_skip_hc = 0;

int ac_lsj_prepare(ac_lsj *m, ac_ci *ci, char *err, int errlen)
{
    int n = ci->ncsf;
    memset(m, 0, sizeof *m);
    m->n = n;
    const int nostore = ac_lsj_nostore, skip_hc = ac_lsj_skip_hc;
    const int nx = ac_lsj_nextra < AC_LSJ_MAXEXTRA ? ac_lsj_nextra : AC_LSJ_MAXEXTRA;
    void (*const sink)(void *, int, int, int, double) = ac_lsj_sink;
    void *const sctx = ac_lsj_sink_ctx;
    mem_note("before ac_lsj_prepare");
    /* W only with a spin-orbit term (without, ac_ci_so1 and ac_ci_so2 are 0:
       an n x n matrix of zeros, 670 MB at n = 9161, is not kept) */
    int so = ci->in.zeta || ci->in.rkg;
    /* a pass that stores nothing runs as the sparse one (a configuration at a time) */
    int sparse = m->sparse = nostore || want_sparse(n);
    int have_w = so, have_w2 = ci->in.rss != NULL;
    ac_lsj_rows *rows[3] = {&m->sh, &m->sw, &m->sw2};
    int foom = 0;
    if (!sparse && !nostore) {
        m->h = calloc((size_t)n * n, sizeof(double));
        m->w = so ? calloc((size_t)n * n, sizeof(double)) : NULL;
        m->w2 = ci->in.rss ? calloc((size_t)n * n, sizeof(double)) : NULL;
        if (!m->h || (so && !m->w) || (ci->in.rss && !m->w2)) {
            ac_lsj_free(m);
            return fail(err, errlen, "out of memory");
        }
    }
    /* Upper triangle only: H is symmetric, and the reduced elements of the
       Hermitian double tensors W (SO) and W_SS obey
         <j||W||i> = (-1)^((S_i - S_j) + (L_i - L_j)) <i||W||j>
       (checked numerically to 1e-17 on the Si and Ti lists). Pairs that
       differ by more than two electrons have no element at all.
       Rows run in parallel (OpenMP): each thread works on a shallow copy of
       ci with its own Theta cache; the engine's other caches and those of
       the integral callbacks are thread-local. Each element is computed by
       one thread exactly as in a serial run. */
    int fe = 0, fi = 0, fj = 0, fkind = 0;
    long ops = 0, npair = 0;
    /* AC_TIME: the time in each operator, summed over the threads */
    int timed = getenv("AC_TIME") != NULL;
    double th = 0.0, too = 0.0, tso1 = 0.0, tso2 = 0.0, tss = 0.0;
#define TICK(acc, expr)                                                                           \
    do {                                                                                          \
        double t_ = timed ? omp_get_wtime() : 0.0;                                                \
        expr;                                                                                     \
        if (timed)                                                                                \
            acc += omp_get_wtime() - t_;                                                          \
    } while (0)
    /* The Coulomb, two-body spin-orbit and spin-spin elements of a pair of
       runs of one configuration are computed together (ac_ci_hij_block,
       ac_ci_bp2_block, and ac_ci_oo_block for H_OO: the same values, the
       operators set up once); AC_BP_PAIRWISE in the environment computes
       them pair by pair with ac_ci_hij, ac_ci_so2, ac_ci_ss and ac_ci_oo,
       as before. */
    int pairwise = getenv("AC_BP_PAIRWISE") != NULL, ng = 0;
    int *gs = config_runs(ci, &ng);
    if (!gs) {
        ac_lsj_free(m);
        return fail(err, errlen, "out of memory");
    }
    /* the pairs of runs that can have elements, handed out one by one */
    size_t ngp = 0;
    for (int A = 0; A < ng; A++)
        for (int B = A; B < ng; B++)
            ngp += moved(ci, gs[A], gs[B]) <= 2;
    int *gp = malloc((2 * ngp + 1) * sizeof *gp);
    if (!gp) {
        free(gs);
        ac_lsj_free(m);
        return fail(err, errlen, "out of memory");
    }
    /* ga[A]: where the pairs of run A begin in gp */
    size_t *ga = malloc(((size_t)ng + 1) * sizeof *ga);
    if (!ga) {
        free(gp);
        free(gs);
        ac_lsj_free(m);
        return fail(err, errlen, "out of memory");
    }
    ngp = 0;
    for (int A = 0; A < ng; A++) {
        ga[A] = ngp;
        for (int B = A; B < ng; B++)
            if (moved(ci, gs[A], gs[B]) <= 2)
                gp[2 * ngp] = A, gp[2 * ngp + 1] = B, ngp++;
    }
    ga[ng] = ngp;
    /* sparse: the rows of a configuration are made by one thread, which has
       all the pairs of that configuration (B ascending: the columns of a row
       ascend), and stored when the configuration is done */
    if (sparse && !nostore &&
        (rows_alloc(&m->sh, n, ng) || (have_w && rows_alloc(&m->sw, n, ng)) ||
         (have_w2 && rows_alloc(&m->sw2, n, ng)))) {
        free(ga);
        free(gp);
        free(gs);
        ac_lsj_free(m);
        return fail(err, errlen, "out of memory");
    }
    /* what the J blocks need of the rows of w and w2 without reading them */
    if (sparse && !nostore && (have_w || have_w2)) {
        int *all = malloc(((size_t)n + 1) * sizeof *all);
        ac_lab *labs = NULL;
        for (int i = 0; all && i < n; i++)
            all[i] = i;
        m->glab = all ? label_numbers(ci, n, all, &labs, &m->nglab) : NULL;
        free(all);
        free(labs);
        m->wcnt = m->glab ? calloc((size_t)n * (size_t)m->nglab + 1, sizeof *m->wcnt) : NULL;
        m->wdiag = calloc(2 * (size_t)n + 1, sizeof *m->wdiag);
        if (!m->glab || !m->wcnt || !m->wdiag) {
            free(ga);
            free(gp);
            free(gs);
            ac_lsj_free(m);
            return fail(err, errlen, "out of memory");
        }
    }
    /* AC_LSJ_DISK: w and w2 in a file */
    const char *disk = getenv("AC_LSJ_DISK");
    if (sparse && !nostore && disk && *disk && (have_w || have_w2) &&
        file_map(&m->wfile, disk, m->nglab)) {
        free(ga);
        free(gp);
        free(gs);
        ac_lsj_free(m);
        return fail(err, errlen, "AC_LSJ_DISK: cannot make and map a file in %s", disk);
    }
    if (m->wfile.map) {
        if (have_w)
            m->sw.file = &m->wfile;
        if (have_w2)
            m->sw2.file = &m->wfile;
    }
    long nunit = sparse ? (long)ng : (long)ngp, gdone = 0;
    double tstart = omp_get_wtime();
#pragma omp parallel reduction(+ : ops, npair, th, too, tso1, tso2, tss)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
        /* the extra terms: the same CSFs on other integrals */
        ac_ci lx[AC_LSJ_MAXEXTRA];
        for (int k = 0; k < nx; k++) {
            lx[k] = *ci;
            lx[k].thc = NULL;
            lx[k].ops_evaluated = 0;
            lx[k].in.h = ac_lsj_extra[k].h;
            lx[k].in.rk = ac_lsj_extra[k].rk;
            lx[k].in.ctx = ac_lsj_extra[k].ctx;
        }
        double *buf = NULL;
        size_t bufn = 0;
        stage st[3];
        memset(st, 0, sizeof st);
        int oom = 0;
#pragma omp for schedule(dynamic, 1)
        for (long unit = 0; unit < nunit; unit++) {
            long g0 = sparse ? (long)ga[unit] : unit, g1 = sparse ? (long)ga[unit + 1] : unit + 1;
            for (long g = g0; g < g1; g++) {
                int A = gp[2 * g], B = gp[2 * g + 1];
                int i0 = gs[A], ni = gs[A + 1] - i0;
                int j0 = gs[B], nj = gs[B + 1] - j0;
                int e = 0, kind = 0, ei = i0, ej = j0;
                size_t nn = (size_t)ni * (size_t)nj;
                double *bso = NULL, *bss = NULL, *bh = NULL, *boo = NULL;
                int want_so = have_w && loc.in.rkg, want_ss = have_w2 && loc.in.rss;
                /* H_SO1 moves one electron at most: two configurations that differ by
                   two have no element (ac_ci_so1 returns 0 for each of their pairs) */
                int so1_cfg = moved(ci, i0, j0) <= 1;
                double *bx[AC_LSJ_MAXEXTRA] = {NULL};
                if (!pairwise) {
                    if ((4 + (size_t)nx) * nn > bufn) {
                        free(buf);
                        bufn = (4 + (size_t)nx) * nn;
                        buf = malloc(bufn * sizeof *buf);
                    }
                    if (!buf)
                        e = 1;
                    else if (!skip_hc) {
                        bh = buf + 2 * nn;
                        TICK(th, ac_ci_hij_block(&loc, i0, ni, j0, nj, A == B, bh, &e));
                    }
                    if (!e && loc.in.roo) {
                        boo = buf + 3 * nn;
                        TICK(too, ac_ci_oo_block(&loc, i0, ni, j0, nj, A == B, boo, &e));
                    }
                    for (int k = 0; k < nx && !e; k++) {
                        bx[k] = buf + (4 + (size_t)k) * nn;
                        TICK(th, ac_ci_hij_block(&lx[k], i0, ni, j0, nj, A == B, bx[k], &e));
                    }
                    if (!e && (want_so || want_ss)) {
                        double ts[2] = {0.0, 0.0};
                        kind = 1;
                        bso = want_so ? buf : NULL;
                        bss = want_ss ? buf + nn : NULL;
                        ac_ci_bp2_block(&loc, i0, ni, j0, nj, A == B, bso, bss, timed ? ts : NULL,
                                        &e);
                        tso2 += ts[0];
                        tss += ts[1];
                    }
                }
                for (int i = i0; i < i0 + ni && !e; i++)
                    for (int j = A == B ? i : j0; j < j0 + nj; j++) {
                        size_t pq = (size_t)(i - i0) * (size_t)nj + (size_t)(j - j0);
                        npair++;
                        ac_lab a = total(&loc, i), b = total(&loc, j);
                        kind = 0, ei = i, ej = j;
                        if (a.ts == b.ts && a.tl == b.tl) {
                            double v = 0.0, voo = 0.0;
                            if (bh)
                                v = bh[pq];
                            else if (!skip_hc)
                                TICK(th, v = ac_ci_hij(&loc, i, j, &e));
                            if (boo)
                                voo = boo[pq];
                            else if (!e)
                                TICK(too, voo = ac_ci_oo(&loc, i, j, &e));
                            if (sink && !e && v != 0.0)
                                sink(sctx, AC_LSJ_HC, i, j, v);
                            if (sink && !e && voo != 0.0)
                                sink(sctx, AC_LSJ_OO, i, j, voo);
                            v += voo;
                            for (int k = 0; k < nx && !e; k++) {
                                double vx = 0.0;
                                if (bx[k])
                                    vx = bx[k][pq];
                                else
                                    TICK(th, vx = ac_ci_hij(&lx[k], i, j, &e));
                                if (sink && !e && vx != 0.0)
                                    sink(sctx, AC_LSJ_EXTRA + k, i, j, vx);
                                v += vx;
                            }
                            if (nostore)
                                ;
                            else if (!sparse)
                                m->h[(size_t)i * n + j] = m->h[(size_t)j * n + i] = v;
                            else if (!e && v != 0.0)
                                oom |= spush(&st[0], i - i0, j, v);
                        }
                        if (!e) {
                            kind = 1;
                            double s = ((a.ts - b.ts) + (a.tl - b.tl)) / 2 % 2 ? -1.0 : 1.0;
                            if (have_w) {
                                double w = 0.0, wb = 0.0;
                                /* nor without the triangles of the total labels
                                   (the first thing ac_ci_so1 would find) */
                                if (so1_cfg && tri2(a.ts, 2, b.ts) && tri2(a.tl, 2, b.tl))
                                    TICK(tso1, w = ac_ci_so1(&loc, i, j, &e));
                                if (bso)
                                    wb = bso[pq];
                                else if (!e)
                                    TICK(tso2, wb = ac_ci_so2(&loc, i, j, &e));
                                if (sink && !e && w != 0.0)
                                    sink(sctx, AC_LSJ_SO1, i, j, w);
                                if (sink && !e && wb != 0.0)
                                    sink(sctx, AC_LSJ_SO2, i, j, wb);
                                w += wb;
                                if (nostore)
                                    ;
                                else if (!sparse) {
                                    m->w[(size_t)i * n + j] = w;
                                    m->w[(size_t)j * n + i] = s * w;
                                } else if (!e && w != 0.0)
                                    oom |= spush(&st[1], i - i0, j, w);
                            }
                            if (!e && have_w2) {
                                double w2 = 0.0;
                                if (bss)
                                    w2 = bss[pq];
                                else
                                    TICK(tss, w2 = ac_ci_ss(&loc, i, j, &e));
                                if (sink && !e && w2 != 0.0)
                                    sink(sctx, AC_LSJ_SS, i, j, w2);
                                if (nostore)
                                    ;
                                else if (!sparse) {
                                    m->w2[(size_t)i * n + j] = w2;
                                    m->w2[(size_t)j * n + i] = s * w2;
                                } else if (!e && w2 != 0.0)
                                    oom |= spush(&st[2], i - i0, j, w2);
                            }
                        }
                        if (e)
                            break;
                    }
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    if (!fe)
                        fe = e, fi = ei, fj = ej, fkind = kind;
                }
            }
            /* the rows of run `unit` are complete (the diagonal elements of
               w and w2 are stored once; the dense form writes them twice,
               with s = 1) */
            for (int k = 0; sparse && k < 3; k++)
                if (rows[k]->len)
                    oom |= rows_put(rows[k], (int)unit, gs[unit], gs[unit + 1] - gs[unit], &st[k],
                                    m->glab);
            for (int i = gs[unit]; sparse && m->wcnt && i < gs[unit + 1]; i++)
                rows_count(m, i);
            if (timed && sparse) {
                /* AC_TIME: a line for every tenth of the pairs of configurations */
                long d, tenth;
#pragma omp atomic capture
                d = gdone += g1 - g0;
                tenth = (long)(10 * (double)d / (double)(ngp ? ngp : 1));
                if (tenth > (long)(10 * (double)(d - (g1 - g0)) / (double)(ngp ? ngp : 1)))
                    fprintf(stderr, "ac_lsj_prepare: %ld of %zu pairs of configurations, %.0f s\n", d,
                            ngp, omp_get_wtime() - tstart);
            }
        }
        free(buf);
        for (int k = 0; k < 3; k++)
            stage_free(&st[k]);
        ops += loc.ops_evaluated;
        if (timed) {
            size_t ne = 0, nb = ac_ci_cache_bytes(&loc, &ne);
            fprintf(stderr, "ac_lsj_prepare: thread %d: Theta cache %zu entries, %.1f MB\n",
                    omp_get_thread_num(), ne, nb / 1048576.0);
        }
        ac_ci_free_cache(&loc);
        for (int k = 0; k < nx; k++) {
            ops += lx[k].ops_evaluated;
            ac_ci_free_cache(&lx[k]);
        }
#pragma omp critical(ac_lsj_fail)
        foom |= oom;
    }
    free(ga);
    free(gp);
    free(gs);
#undef TICK
    if (sparse) {
        if (foom && !fe) {
            ac_lsj_free(m);
            return fail(err, errlen, disk && *disk ? "out of memory, or no room in AC_LSJ_DISK"
                                                   : "out of memory");
        }
        for (int k = 0; k < 3; k++)
            for (int i = 0; rows[k]->len && i < n; i++)
                rows[k]->nnz += (size_t)rows[k]->len[i];
        if (timed && m->wfile.map)
            fprintf(stderr, "AC_LSJ_DISK: w %.2f GB and w2 %.2f GB in a file of %.2f GB\n",
                    m->sw.nnz * 12 / 1073741824.0, m->sw2.nnz * 12 / 1073741824.0,
                    m->wfile.mapused / 1073741824.0);
    }
    ci->ops_evaluated += ops;
    mem_note("after ac_lsj_prepare");
    if (timed)
        fprintf(stderr,
                "ac_lsj_prepare: %ld CSF pairs of %ld; thread-seconds: H %.2f, OO %.2f, SO1 %.2f, "
                "SO2 %.2f, SS %.2f\n",
                npair, (long)n * (n + 1) / 2, th, too, tso1, tso2, tss);
    if (!fe && getenv("AC_NNZ")) {
        /* AC_NNZ: how sparse the matrices are (upper triangle with the diagonal) */
        const double *mat[3] = {m->h, m->w, m->w2};
        const char *nm[3] = {"h", "w", "w2"};
        double tot = 0.5 * n * ((double)n + 1.0);
        for (int k = 0; k < 3; k++) {
            if (sparse && rows[k]->len)
                fprintf(stderr, "nnz %s: n %d, nonzero %zu (%.5f), sparse; pairs computed %.5f\n",
                        nm[k], n, rows[k]->nnz, rows[k]->nnz / tot, npair / tot);
            if (!mat[k])
                continue;
            long c0 = 0, c14 = 0, c10 = 0;
#pragma omp parallel for reduction(+ : c0, c14, c10) schedule(static)
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) {
                    double v = fabs(mat[k][(size_t)i * n + j]);
                    c0 += v > 0.0, c14 += v > 1e-14, c10 += v > 1e-10;
                }
            fprintf(stderr,
                    "nnz %s: n %d, nonzero %ld (%.5f), > 1e-14 %ld (%.5f), > 1e-10 %ld (%.5f); "
                    "pairs computed %.5f\n",
                    nm[k], n, c0, c0 / tot, c14, c14 / tot, c10, c10 / tot, npair / tot);
        }
    }
    if (fe) {
        ac_lsj_free(m);
        return fail(err, errlen, "%s(%d,%d): engine error %d", fkind ? "W" : "H", fi, fj, fe);
    }
    return 0;
}

void ac_lsj_free(ac_lsj *m)
{
    free(m->h);
    free(m->w);
    free(m->w2);
    rows_free(&m->sh);
    rows_free(&m->sw);
    rows_free(&m->sw2);
    file_free(&m->wfile);
    free(m->glab);
    free(m->wcnt);
    free(m->wdiag);
    memset(m, 0, sizeof *m);
}

static int tri2(int a, int b, int c)
{
    return (a + b + c) % 2 == 0 && c <= a + b && c >= abs(a - b);
}

/* The basis of a block: the CSFs whose (S, L) couple to 2J = tJ; or, while
   ls_block is set (ac_lsj_print_ls: no w and w2, so the J block falls apart
   into the (S, L)), those of the one label ls_lab. */
static int ls_block = 0;
static ac_lab ls_lab;

static int in_block(const ac_ci *ci, int i, int tJ)
{
    ac_lab a = total(ci, i);
    if (ls_block)
        return a.ts == ls_lab.ts && a.tl == ls_lab.tl;
    return tri2(a.ts, a.tl, tJ);
}

/* The elements of row i of a sparse m whose column is in the J block
   (pos[j] >= 0), in the order of the columns: jc[] the columns, hv, wv, w2v
   the three values (0 where a matrix has no element). Returns how many. */
static size_t merged_row(const ac_lsj *m, int i, const int *pos, int *jc, double *hv, double *wv,
                         double *w2v)
{
    const ac_lsj_rows *A[3] = {&m->sh, &m->sw, &m->sw2};
    const int *col[3] = {NULL, NULL, NULL};
    const double *val[3] = {NULL, NULL, NULL};
    int e[3] = {0, 0, 0}, end[3] = {0, 0, 0};
    size_t k = 0;
    for (int t = 0; t < 3; t++)
        if (A[t]->len) {
            end[t] = A[t]->len[i];
            col[t] = A[t]->col[i];
            val[t] = A[t]->val[i];
        }
    for (;;) {
        int j = -1;
        for (int t = 0; t < 3; t++)
            if (e[t] < end[t] && (j < 0 || col[t][e[t]] < j))
                j = col[t][e[t]];
        if (j < 0)
            break;
        double v[3] = {0.0, 0.0, 0.0};
        for (int t = 0; t < 3; t++)
            if (e[t] < end[t] && col[t][e[t]] == j)
                v[t] = val[t][e[t]++];
        if (pos[j] < 0)
            continue;
        jc[k] = j, hv[k] = v[0], wv[k] = v[1], w2v[k] = v[2];
        k++;
    }
    return k;
}

/* The J block of a sparse m: its upper triangle by rows. The element (p, q),
   q > p, is the mean of the two numbers the dense form symmetrizes: the one
   made from <i|.|j> and the one made from <j|.|i> = s <i|.|j>, each with its
   own recoupling factor, added in the same order; so the elements are the
   same numbers as those of the dense matrix (which has zeros elsewhere).
   The CSFs of the block keep their order, so the columns of a row ascend
   and are not below the row. */
int ac_lsj_matrix_csr(const ac_lsj *m, const ac_ci *ci, int tJ, int *idx, ac_lsj_csr *A)
{
    int n = m->n, dim = 0, rc = -1, bad = 0;
    A->start = NULL;
    A->col = NULL;
    A->val = NULL;
    if (!m->sparse)
        return -1;
    int *pos = malloc(((size_t)n + 1) * sizeof *pos);
    if (!pos)
        return -1;
    for (int i = 0; i < n; i++) {
        pos[i] = -1;
        if (in_block(ci, i, tJ))
            pos[i] = dim, idx[dim++] = i;
    }
    ac_lab *labs = NULL;
    int nlab = 0;
    int *lab = label_numbers(ci, dim, idx, &labs, &nlab);
    double *f1 = lab ? malloc((size_t)(nlab ? nlab : 1) * nlab * 2 * sizeof *f1) : NULL;
    double *f2 = f1 ? f1 + (size_t)nlab * nlab : NULL;
    /* sgn[x * nlab + y]: s of the labels x and y */
    double *sgn = f1 ? malloc((size_t)(nlab ? nlab : 1) * nlab * sizeof *sgn) : NULL;
    size_t maxrow = 0;
    const ac_lsj_rows *S[3] = {&m->sh, &m->sw, &m->sw2};
    for (int p = 0; p < dim; p++) {
        size_t len = 0;
        for (int t = 0; t < 3; t++)
            if (S[t]->len)
                len += (size_t)S[t]->len[idx[p]];
        if (len > maxrow)
            maxrow = len;
    }
    size_t *start = calloc((size_t)dim + 2, sizeof *start);
    int *col = NULL;
    double *val = NULL;
    if (!lab || !f1 || !sgn || !start)
        goto out;
    for (int x = 0; x < nlab; x++)
        for (int y = 0; y < nlab; y++) {
            ac_lab a = labs[x], b = labs[y];
            int ph = (b.ts + a.tl + tJ) / 2; /* S' + L + J */
            f1[(size_t)x * nlab + y] =
                -(ph % 2 ? -1.0 : 1.0) / sqrt(3.0) * ac_sixj(tJ, a.tl, a.ts, 2, b.ts, b.tl);
            f2[(size_t)x * nlab + y] =
                (ph % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, a.tl, a.ts, 4, b.ts, b.tl);
            sgn[(size_t)x * nlab + y] = ((a.ts - b.ts) + (a.tl - b.tl)) / 2 % 2 ? -1.0 : 1.0;
        }
    /* how many elements each row has, then the elements; the rows are
       independent of each other */
    for (int pass = 0; pass < 2 && !bad; pass++) {
#pragma omp parallel
        {
            int *jc = malloc((maxrow + 1) * sizeof *jc);
            double *hv = malloc(3 * (maxrow + 1) * sizeof *hv);
            double *wv = hv ? hv + maxrow + 1 : NULL, *w2v = hv ? wv + maxrow + 1 : NULL;
#pragma omp for schedule(dynamic, 64)
            for (int p = 0; p < dim; p++) {
                if (!jc || !hv)
                    continue;
                size_t k = merged_row(m, idx[p], pos, jc, hv, wv, w2v);
                if (!pass) {
                    start[p + 1] = k;
                    continue;
                }
                size_t e = start[p];
                for (size_t u = 0; u < k; u++) {
                    int q = pos[jc[u]];
                    size_t xy = (size_t)lab[p] * nlab + lab[q], yx = (size_t)lab[q] * nlab + lab[p];
                    double a = hv[u], b = hv[u], w = wv[u], w2 = w2v[u], s = sgn[xy];
                    if (w != 0.0)
                        a += f1[xy] * w;
                    if (w2 != 0.0)
                        a += f2[xy] * w2;
                    if (q != p) {
                        if (w != 0.0)
                            b += f1[yx] * (s * w);
                        if (w2 != 0.0)
                            b += f2[yx] * (s * w2);
                        a = 0.5 * (a + b);
                    }
                    col[e] = q;
                    val[e++] = a;
                }
            }
            if (!jc || !hv) {
#pragma omp critical(ac_lsj_fail)
                bad = 1;
            }
            free(jc);
            free(hv);
        }
        if (!pass && !bad) {
            for (int p = 0; p < dim; p++)
                start[p + 1] += start[p];
            col = malloc((start[dim] + 1) * sizeof *col);
            val = malloc((start[dim] + 1) * sizeof *val);
            if (!col || !val)
                bad = 1;
        }
    }
    if (bad)
        goto out;
    A->start = start;
    A->col = col;
    A->val = val;
    start = NULL;
    col = NULL;
    val = NULL;
    rc = dim;
out:
    free(pos);
    free(lab);
    free(labs);
    free(f1);
    free(sgn);
    free(start);
    free(col);
    free(val);
    return rc;
}

/* ---- the J block of a sparse m as an operator ---------------------------
   For Davidson the block is not stored a second time: H x is formed from
   the rows of m. The element (p, q), q >= p, of the block is
     h + c1[x y] w + c2[x y] w2,   c[x y] = (f[x y] + s f[y x]) / 2,
   (x, y the total labels of the two CSFs, f the recoupling factors of
   ac_lsj_matrix_csr, s the sign of the transposed element), which is the
   symmetrized element of the dense form up to rounding.
   Kept for the block: the diagonal; for the elements of h above the
   diagonal only their places in the block (qh, in step with the row of
   m->sh, whose values are used where they are); the elements of w and w2
   inside the block with their factors (wq, wa: few).
   The rows are cut into JOP_PIECES pieces of about equal numbers of
   elements; an element above the diagonal is added to the row of its column
   through the vector of its piece, and the pieces are added in order, so
   H x does not depend on the number of threads. */
#define JOP_PIECES 32
#define JOP_VEC 4
#define JOP_ROWS 64  /* rows of w and w2 a thread takes at a time when the block is made */
#define JOP_AHEAD 16 /* rows in a file: asked for this many steps before they are read */

typedef struct {
    const ac_lsj *m;
    int dim, nlab, np;
    const int *idx;
    int *pos, *lab, *p0;
    double *c1, *c2;
    double *dd;    /* [dim] the diagonal */
    size_t *hst;   /* [dim+1] row r: qh[hst[r] .. hst[r+1]-1] */
    int *qh;       /* place in the block of the elements of the row of h after its diagonal */
    char *hskip;   /* [dim] 1 when the row of h begins with its diagonal element */
    size_t *wst;   /* [dim+1] row r: wq, wa[wst[r] .. wst[r+1]-1] */
    int *wq;
    double *wa;
    size_t *off;   /* in columns: where the vectors of piece c begin in buf */
    double *buf;
    double shift;  /* subtracted from the diagonal */
    const int *part; /* not NULL: jop_mul takes the rows of part[r] == cur only */
    int cur;
} jop;

static void jop_end(jop *J)
{
    free(J->pos);
    free(J->lab);
    free(J->p0);
    free(J->c1);
    free(J->dd);
    free(J->hst);
    free(J->qh);
    free(J->hskip);
    free(J->wst);
    free(J->wq);
    free(J->wa);
    free(J->off);
    free(J->buf);
    memset(J, 0, sizeof *J);
}

/* the element of column j in row i of R, or 0 */
static double rows_find(const ac_lsj_rows *R, int i, int j)
{
    if (!R->len)
        return 0.0;
    const int *c = R->col[i];
    int lo = 0, hi = R->len[i];
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (c[mid] < j)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < R->len[i] && c[lo] == j ? R->val[i][lo] : 0.0;
}

static double jop_elem(void *ctx, int a, int b)
{
    const jop *J = ctx;
    if (a == b)
        return J->dd[a] - J->shift;
    int p = a < b ? a : b, q = a < b ? b : a, i = J->idx[p], j = J->idx[q];
    double v = rows_find(&J->m->sh, i, j);
    /* the part of w and w2 as the block has it (the number jop_mul uses; the
       rows of w and w2 themselves may be in a file) */
    size_t lo = J->wst[p], hi = J->wst[p + 1];
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (J->wq[mid] < q)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < J->wst[p + 1] && J->wq[lo] == q)
        v += J->wa[lo];
    return v;
}

/* The operator of the block 2J = tJ of a sparse m: the basis in idx[] (room
   for ci->ncsf; it must stay until jop_end). Returns dim, or -1 when out of
   memory. The shift is set to the mean of the diagonal. */
static int jop_begin(jop *J, const ac_lsj *m, const ac_ci *ci, int tJ, int *idx)
{
    int n = m->n, dim = 0;
    const ac_lsj_rows *H = &m->sh, *W[2] = {&m->sw, &m->sw2};
    memset(J, 0, sizeof *J);
    J->m = m;
    J->idx = idx;
    J->pos = malloc(((size_t)n + 1) * sizeof *J->pos);
    if (!J->pos)
        return -1;
    for (int i = 0; i < n; i++) {
        J->pos[i] = -1;
        if (in_block(ci, i, tJ))
            J->pos[i] = dim, idx[dim++] = i;
    }
    J->dim = dim;
    ac_lab *labs = NULL;
    int nlab = 0;
    J->lab = label_numbers(ci, dim, idx, &labs, &nlab);
    J->nlab = nlab;
    size_t nl2 = (size_t)(nlab ? nlab : 1) * (size_t)(nlab ? nlab : 1);
    J->c1 = J->lab ? malloc(2 * nl2 * sizeof *J->c1) : NULL;
    J->c2 = J->c1 ? J->c1 + nl2 : NULL;
    int np = dim < JOP_PIECES ? dim : JOP_PIECES;
    J->np = np;
    J->p0 = malloc(((size_t)np + 1) * sizeof *J->p0);
    J->off = malloc(((size_t)np + 1) * sizeof *J->off);
    J->dd = calloc((size_t)dim + 1, sizeof *J->dd);
    J->hst = calloc((size_t)dim + 2, sizeof *J->hst);
    J->wst = calloc((size_t)dim + 2, sizeof *J->wst);
    J->hskip = calloc((size_t)dim + 1, 1);
    if (!J->lab || !J->c1 || !J->p0 || !J->off || !J->dd || !J->hst || !J->wst || !J->hskip) {
        free(labs);
        jop_end(J);
        return -1;
    }
    for (int x = 0; x < nlab; x++)
        for (int y = 0; y < nlab; y++) {
            ac_lab a = labs[x], b = labs[y];
            /* f[x y] and f[y x] as in ac_lsj_matrix_csr */
            int ph = (b.ts + a.tl + tJ) / 2, pt = (a.ts + b.tl + tJ) / 2;
            double s = ((a.ts - b.ts) + (a.tl - b.tl)) / 2 % 2 ? -1.0 : 1.0;
            double f1 = -(ph % 2 ? -1.0 : 1.0) / sqrt(3.0) * ac_sixj(tJ, a.tl, a.ts, 2, b.ts, b.tl);
            double g1 = -(pt % 2 ? -1.0 : 1.0) / sqrt(3.0) * ac_sixj(tJ, b.tl, b.ts, 2, a.ts, a.tl);
            double f2 = (ph % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, a.tl, a.ts, 4, b.ts, b.tl);
            double g2 = (pt % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, b.tl, b.ts, 4, a.ts, a.tl);
            J->c1[(size_t)x * nlab + y] = x == y ? f1 : 0.5 * (f1 + s * g1);
            J->c2[(size_t)x * nlab + y] = x == y ? f2 : 0.5 * (f2 + s * g2);
        }
    free(labs);
    /* count, then fill: the rows are independent of each other.
       The count does not read the rows of w and w2 when the block is made of
       whole labels (it is: in_block looks at the label only; checked here):
       the numbers of places by the label of the column and the diagonal
       elements were kept when the rows were made (m->wcnt, m->wdiag), and
       the sums are the ones the rows give. With the rows in a file this
       halves what is read; the fill asks for its rows ahead (rows_ahead). */
    const int *pos = J->pos, *lab = J->lab;
    char *inb = m->wcnt ? calloc((size_t)m->nglab + 1, 1) : NULL;
    if (inb) {
        int whole = 1;
        for (int r = 0; r < dim; r++)
            inb[m->glab[idx[r]]] = 1;
        for (int i = 0; i < n && whole; i++)
            whole = inb[m->glab[i]] == (pos[i] >= 0);
        if (!whole) {
            free(inb);
            inb = NULL;
        }
    }
    /* With the rows in a file the block takes its rows in the order of the
       file, JOP_ROWS at a time, and asks for those JOP_AHEAD steps on while it
       works (the rows of a label lie together: the disk then reads in order).
       ord[]: the rows of the block in that order; a row without elements goes
       with the one before it. */
    const int ahead = m->wfile.map != NULL;
    int *ord = NULL;
    span *asp = NULL;
    int nth = omp_get_max_threads();
    if (ahead) {
        rowat *ra = malloc(((size_t)dim + 1) * sizeof *ra);
        ord = malloc(((size_t)dim + 1) * sizeof *ord);
        asp = malloc((size_t)nth * 2 * JOP_ROWS * sizeof *asp);
        if (!ra || !ord || !asp) {
            free(ra);
            free(ord);
            free(asp);
            free(inb);
            jop_end(J);
            return -1;
        }
        size_t key = 0;
        for (int r = 0; r < dim; r++) {
            int i = idx[r];
            for (int k = 0; k < 2; k++)
                if (W[k]->len && W[k]->len[i]) {
                    key = (size_t)((const char *)W[k]->val[i] - m->wfile.map);
                    break;
                }
            ra[r].key = key, ra[r].r = r;
        }
        qsort(ra, (size_t)dim, sizeof *ra, by_rowat);
        for (int r = 0; r < dim; r++)
            ord[r] = idx[ra[r].r];
        /* ord[] holds CSFs for rows_ahead; the loop below finds the row of the block by pos[] */
        free(ra);
    }
    double tpass[2] = {0.0, 0.0}, tnote = 0.0;
    for (int pass = 0; pass < 2; pass++) {
        double tp = omp_get_wtime();
        if (pass && ahead && getenv("AC_TIME")) {
            size_t ne = 0;
            for (int r = 0; r < dim; r++)
                for (int k = 0; k < 2; k++)
                    if (W[k]->len)
                        ne += (size_t)W[k]->len[idx[r]];
            fprintf(stderr, "J = %d/2: the rows of the block in the file: %.2f GB\n", tJ,
                    ne * 12 / 1073741824.0);
            io_note("  before the fill");
            tnote = omp_get_wtime();
        }
#pragma omp parallel for schedule(dynamic, 1)
        for (int r0 = 0; r0 < dim; r0 += JOP_ROWS)
            for (int o = r0; o < dim && o < r0 + JOP_ROWS; o++) {
                int r = ord ? pos[ord[o]] : o;
                int i = idx[r];
                int nh = H->len[i], skip = nh > 0 && H->col[i][0] == i;
                size_t nw = 0, eh = J->hst[r], ew = J->wst[r];
                double d = skip ? H->val[i][0] : 0.0;
                const double *c1r = J->c1 + (size_t)lab[r] * nlab;
                const double *c2r = J->c2 + (size_t)lab[r] * nlab;
                if (pass && ahead && o == r0 && tnote > 0.0) {
                    /* AC_TIME: where the fill stands, every 20 s */
#pragma omp critical(ac_lsj_ionote)
                    if (omp_get_wtime() - tnote >= 20.0) {
                        char what[64];
                        tnote = omp_get_wtime();
                        snprintf(what, sizeof what, "  %.0f s, row %d of %d", tnote - tp, r0, dim);
                        io_note(what);
                    }
                }
                if (ahead && o == r0 && (pass || !inb)) {
                    /* the first step asks for the steps before JOP_AHEAD too */
                    int t = omp_get_thread_num();
                    for (int a = r0 ? JOP_AHEAD : 0; a <= JOP_AHEAD; a++) {
                        int o0 = r0 + a * JOP_ROWS;
                        if (o0 < dim)
                            rows_ahead(m, ord + o0, dim - o0 < JOP_ROWS ? dim - o0 : JOP_ROWS,
                                       asp + (size_t)t * 2 * JOP_ROWS);
                    }
                }
                if (pass)
                    for (int e = skip; e < nh; e++)
                        J->qh[eh++] = pos[H->col[i][e]];
                else if (inb) {
                    const unsigned *cnt = m->wcnt + (size_t)i * (size_t)m->nglab;
                    for (int y = 0; y < m->nglab; y++)
                        if (inb[y])
                            nw += cnt[y];
                    double a = c1r[lab[r]] * m->wdiag[i];
                    a += c2r[lab[r]] * m->wdiag[(size_t)n + i];
                    J->hskip[r] = (char)skip;
                    J->dd[r] = d + a;
                    J->hst[r + 1] = (size_t)(nh - skip);
                    J->wst[r + 1] = nw;
                    continue;
                }
                /* the rows of w and w2 together, in the order of the columns: a
                   place that both have becomes one element (with SO2 and SS
                   most places do, and the block needs half the memory) */
                int n1 = W[0]->len ? W[0]->len[i] : 0, n2 = W[1]->len ? W[1]->len[i] : 0;
                const int *k1 = n1 ? W[0]->col[i] : NULL, *k2 = n2 ? W[1]->col[i] : NULL;
                for (int e1 = 0, e2 = 0; e1 < n1 || e2 < n2;) {
                    int col = e2 == n2 || (e1 < n1 && k1[e1] <= k2[e2]) ? k1[e1] : k2[e2];
                    int q = pos[col];
                    double a = 0.0;
                    if (e1 < n1 && k1[e1] == col) {
                        if (q >= 0)
                            a = c1r[lab[q]] * W[0]->val[i][e1];
                        e1++;
                    }
                    if (e2 < n2 && k2[e2] == col) {
                        if (q >= 0)
                            a += c2r[lab[q]] * W[1]->val[i][e2];
                        e2++;
                    }
                    if (q < 0)
                        continue;
                    if (q == r)
                        d += a;
                    else if (pass)
                        J->wq[ew] = q, J->wa[ew++] = a;
                    else
                        nw++;
                }
                if (!pass) {
                    J->hskip[r] = (char)skip;
                    J->dd[r] = d;
                    J->hst[r + 1] = (size_t)(nh - skip);
                    J->wst[r + 1] = nw;
                }
            }
        tpass[pass] = omp_get_wtime() - tp;
        if (pass && ahead && getenv("AC_TIME"))
            io_note("  after the fill");
        if (pass)
            break;
        for (int r = 0; r < dim; r++) {
            J->hst[r + 1] += J->hst[r];
            J->wst[r + 1] += J->wst[r];
        }
        J->qh = malloc((J->hst[dim] + 1) * sizeof *J->qh);
        J->wq = malloc((J->wst[dim] + 1) * sizeof *J->wq);
        J->wa = malloc((J->wst[dim] + 1) * sizeof *J->wa);
        if (!J->qh || !J->wq || !J->wa) {
            free(inb);
            free(ord);
            free(asp);
            jop_end(J);
            return -1;
        }
    }
    free(ord);
    free(asp);
    if (getenv("AC_TIME"))
        fprintf(stderr, "J = %d/2, dim %d: the block from the rows: count %.2f s%s, fill %.2f s\n", tJ,
                dim, tpass[0], inb ? " (rows of w and w2 not read)" : "", tpass[1]);
    if (getenv("AC_TIME") && inb) {
        size_t in = 0, all = 0;
        for (int r = 0; r < dim; r++)
            for (int y = 0; y < m->nglab; y++) {
                size_t c = m->wcnt[(size_t)idx[r] * (size_t)m->nglab + y];
                all += c;
                if (inb[y])
                    in += c;
            }
        fprintf(stderr, "J = %d/2: places of w and w2 in the rows of the block %zu, in the block %zu\n",
                tJ, all, in);
    }
    free(inb);
    /* the pieces: about equal numbers of elements of the rows */
    size_t tot = J->hst[dim] + J->wst[dim];
    int c = 0;
    for (int p = 0; p < dim; p++) {
        size_t acc = J->hst[p] + J->wst[p];
        while (c < np && acc >= tot / (size_t)np * (size_t)c)
            J->p0[c++] = p;
    }
    while (c <= np)
        J->p0[c++] = dim;
    J->off[0] = 0;
    for (c = 0; c < np; c++)
        J->off[c + 1] = J->off[c] + (size_t)(dim - J->p0[c]);
    J->buf = malloc((J->off[np] + 1) * JOP_VEC * sizeof *J->buf);
    if (!J->buf) {
        jop_end(J);
        return -1;
    }
    double shift = 0.0;
    for (int p = 0; p < dim; p++)
        shift += J->dd[p];
    J->shift = dim ? shift / dim : 0.0;
    return dim;
}

/* AV = (H - shift) V for mv columns of length dim, JOP_VEC at a time */
static void jop_mul(void *ctx, int mv, const double *V, double *AV)
{
    const jop *J = ctx;
    const ac_lsj_rows *H = &J->m->sh;
    const int *p0 = J->p0;
    int n = J->dim;
    for (int j = 0; j < mv; j += JOP_VEC) {
        int g = mv - j < JOP_VEC ? mv - j : JOP_VEC;
        /* always JOP_VEC vectors in the loops: the missing ones repeat the
           first with x = 0 and their sums are dropped */
        const double *v[JOP_VEC];
        for (int u = 0; u < JOP_VEC; u++)
            v[u] = V + (size_t)(j + (u < g ? u : 0)) * n;
#pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < J->np; c++) {
            double *b = J->buf + J->off[c] * JOP_VEC - (size_t)p0[c] * JOP_VEC;
            memset(J->buf + J->off[c] * JOP_VEC, 0, (size_t)(n - p0[c]) * JOP_VEC * sizeof *b);
            for (int r = p0[c]; r < p0[c + 1]; r++) {
                if (J->part && J->part[r] != J->cur) {
                    /* a row of another part: the vectors are 0 there and H
                       does not couple it to this part */
                    for (int u = 0; u < g; u++)
                        AV[(size_t)(j + u) * n + r] = 0.0;
                    continue;
                }
                double s[JOP_VEC], x[JOP_VEC], dg = J->dd[r] - J->shift;
                for (int u = 0; u < JOP_VEC; u++) {
                    x[u] = u < g ? v[u][r] : 0.0;
                    s[u] = dg * x[u];
                }
                const double *hv = H->val[J->idx[r]] + J->hskip[r];
                const int *qh = J->qh + J->hst[r];
                size_t nh = J->hst[r + 1] - J->hst[r];
                for (size_t e = 0; e < nh; e++) {
                    int q = qh[e];
                    if (q < 0)
                        continue;
                    double a = hv[e], *bq = b + (size_t)q * JOP_VEC;
                    for (int u = 0; u < JOP_VEC; u++) {
                        s[u] += a * v[u][q];
                        bq[u] += a * x[u];
                    }
                }
                for (size_t e = J->wst[r]; e < J->wst[r + 1]; e++) {
                    double a = J->wa[e], *bq = b + (size_t)J->wq[e] * JOP_VEC;
                    for (int u = 0; u < JOP_VEC; u++) {
                        s[u] += a * v[u][J->wq[e]];
                        bq[u] += a * x[u];
                    }
                }
                for (int u = 0; u < g; u++)
                    AV[(size_t)(j + u) * n + r] = s[u];
            }
        }
#pragma omp parallel for schedule(static)
        for (int q = 0; q < n; q++)
            for (int c = 0; c < J->np && p0[c] <= q; c++) {
                const double *bq = J->buf + (J->off[c] + (size_t)(q - p0[c])) * JOP_VEC;
                for (int u = 0; u < g; u++)
                    AV[(size_t)(j + u) * n + q] += bq[u];
            }
    }
}

static void jop_sel(void *ctx, const int *part, int cur)
{
    jop *J = ctx;
    J->part = part, J->cur = cur;
}

/* The parts of a block that H does not couple at all (the two parities of a
   list that has both; the total (S, L) of a block without spin-dependent
   terms), found from the elements themselves: rows joined by a nonzero
   element are in one part. up[] is the forest of the rows. */
static int part_root(int *up, int r)
{
    while (up[r] != r)
        r = up[r] = up[up[r]];
    return r;
}

static void part_join(int *up, int a, int b)
{
    a = part_root(up, a), b = part_root(up, b);
    if (a != b)
        up[a > b ? a : b] = a > b ? b : a;
}

/* the parts numbered 0, 1, ... in the order of their first rows, into up[];
   returns their number, or -1 when out of memory */
static int part_number(int dim, int *up)
{
    int np = 0, *root = malloc((size_t)(dim ? dim : 1) * sizeof *root);
    if (!root)
        return -1;
    for (int r = 0; r < dim; r++)
        root[r] = part_root(up, r);
    /* the root is the first row of its part (part_join keeps the lower) */
    for (int r = 0; r < dim; r++)
        up[r] = root[r] == r ? np++ : up[root[r]];
    free(root);
    return np;
}

/* of the block of jop J; NULL when out of memory */
static int *jop_parts(const jop *J, int *np)
{
    const ac_lsj_rows *H = &J->m->sh;
    int dim = J->dim, *up = malloc((size_t)(dim ? dim : 1) * sizeof *up);
    if (!up)
        return NULL;
    for (int r = 0; r < dim; r++)
        up[r] = r;
    for (int r = 0; r < dim; r++) {
        const double *hv = H->val[J->idx[r]] + J->hskip[r];
        const int *qh = J->qh + J->hst[r];
        size_t nh = J->hst[r + 1] - J->hst[r];
        for (size_t e = 0; e < nh; e++)
            if (qh[e] >= 0 && hv[e] != 0.0)
                part_join(up, r, qh[e]);
        for (size_t e = J->wst[r]; e < J->wst[r + 1]; e++)
            if (J->wa[e] != 0.0)
                part_join(up, r, J->wq[e]);
    }
    if ((*np = part_number(dim, up)) < 0) {
        free(up);
        return NULL;
    }
    return up;
}

/* of the dense block H */
static int *dense_parts(int dim, const double *H, int *np)
{
    int *up = malloc((size_t)(dim ? dim : 1) * sizeof *up);
    if (!up)
        return NULL;
    for (int r = 0; r < dim; r++)
        up[r] = r;
    for (int p = 0; p < dim; p++)
        for (int q = p + 1; q < dim; q++)
            if (H[(size_t)p * dim + q] != 0.0)
                part_join(up, p, q);
    if ((*np = part_number(dim, up)) < 0) {
        free(up);
        return NULL;
    }
    return up;
}

/* the J block: scalar product of the double tensors W^(kappa kappa)
   (Edmonds (1957), Eq. (7.1.6); technical note, Sec. 3.6) */
int ac_lsj_matrix(const ac_lsj *m, const ac_ci *ci, int tJ, int *idx, double *H,
                  double *asym)
{
    int dim = 0;
    if (m->sparse) {
        /* the sparse block written out */
        ac_lsj_csr A;
        dim = ac_lsj_matrix_csr(m, ci, tJ, idx, &A);
        if (dim < 0)
            return -1;
        memset(H, 0, (size_t)dim * dim * sizeof *H);
        for (int p = 0; p < dim; p++)
            for (size_t e = A.start[p]; e < A.start[p + 1]; e++)
                H[(size_t)A.col[e] * dim + p] = H[(size_t)p * dim + A.col[e]] = A.val[e];
        csr_free(&A);
        if (asym)
            *asym = 0.0;
        return dim;
    }
    for (int i = 0; i < m->n; i++)
        if (in_block(ci, i, tJ))
            idx[dim++] = i;
    /* the factors of w and w2 depend on the total labels of the two CSFs
       only: one table per kappa (label_numbers); without the tables (out of
       memory) they are computed pair by pair. The values are the same. */
    ac_lab *labs = NULL;
    int nlab = 0;
    int *lab = label_numbers(ci, dim, idx, &labs, &nlab);
    double *f1 = lab ? malloc((size_t)(nlab ? nlab : 1) * nlab * 2 * sizeof *f1) : NULL;
    double *f2 = f1 ? f1 + (size_t)nlab * nlab : NULL;
    for (int x = 0; f1 && x < nlab; x++)
        for (int y = 0; y < nlab; y++) {
            ac_lab a = labs[x], b = labs[y];
            int ph = (b.ts + a.tl + tJ) / 2; /* S' + L + J */
            f1[(size_t)x * nlab + y] =
                -(ph % 2 ? -1.0 : 1.0) / sqrt(3.0) * ac_sixj(tJ, a.tl, a.ts, 2, b.ts, b.tl);
            /* kappa = 2: ((-1)^2/sqrt5) (-1)^(S'+L+J) {J L S; 2 S' L'} */
            f2[(size_t)x * nlab + y] =
                (ph % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, a.tl, a.ts, 4, b.ts, b.tl);
        }
#pragma omp parallel for schedule(static)
    for (int p = 0; p < dim; p++)
        for (int q = 0; q < dim; q++) {
            int i = idx[p], j = idx[q];
            double v = m->h[(size_t)i * m->n + j];
            double w = m->w ? m->w[(size_t)i * m->n + j] : 0.0;
            double w2 = m->w2 ? m->w2[(size_t)i * m->n + j] : 0.0;
            if (f1) {
                size_t xy = (size_t)lab[p] * nlab + lab[q];
                if (w != 0.0)
                    v += f1[xy] * w;
                if (w2 != 0.0)
                    v += f2[xy] * w2;
            } else {
                ac_lab a = total(ci, i), b = total(ci, j);
                int ph = (b.ts + a.tl + tJ) / 2;
                if (w != 0.0)
                    v += -(ph % 2 ? -1.0 : 1.0) / sqrt(3.0) *
                         ac_sixj(tJ, a.tl, a.ts, 2, b.ts, b.tl) * w;
                if (w2 != 0.0)
                    v += (ph % 2 ? -1.0 : 1.0) / sqrt(5.0) * ac_sixj(tJ, a.tl, a.ts, 4, b.ts, b.tl) *
                         w2;
            }
            H[(size_t)q * dim + p] = v;
        }
    free(lab);
    free(labs);
    free(f1);
    double d = 0.0;
    for (int p = 0; p < dim; p++)
        for (int q = 0; q < p; q++) {
            double x = H[(size_t)q * dim + p], y = H[(size_t)p * dim + q];
            d = fmax(d, fabs(x - y));
            H[(size_t)q * dim + p] = H[(size_t)p * dim + q] = 0.5 * (x + y);
        }
    if (asym)
        *asym = d;
    return dim;
}

/* Edmonds (1957), Eq. (7.1.5) */
double ac_lsj_reduce(ac_lab a, int tJ, ac_lab b, int tJp, int ks, int kl, int tK, double w)
{
    if (w == 0.0)
        return 0.0;
    return sqrt((double)(tJ + 1) * (tJp + 1) * (tK + 1)) *
           ac_ninej(a.ts, b.ts, ks, a.tl, b.tl, kl, tJ, tJp, tK) * w;
}

/* ---- first-order Zeeman ------------------------------------------------- */

#define ALPHA2 (ac_alpha * ac_alpha)

typedef struct {
    int kind; /* 0 total S, 1 total L, 2 RMC, 3 GC rank (1,0), 4 GC rank (1,2),
                 5 orbital RMC; hyperfine: 6 orbital, 7 spin-dipole, 8 contact,
                 9 electric quadrupole; transitions: 10 E1 length r C^1,
                 11 E1 velocity grad, 12 E2 r^2 C^2; relativistic M1: 13-15
                 s r^2, s p^2, s Z/r, 16-17 l r^2, l p^2, 18-19 s C^2 r^2,
                 s C^2 Z/r, 20 s [p x p]^(2); isotope shift: 21 p^2/2,
                 22 delta(r); transitions: 23 E3 r^3 C^3, 24-25 M2 orbital
                 and spin parts, 26-27 M3 orbital and spin parts */
    double (*r)(void *, int, int);
    void *ctx;
    double (*rk)(void *, int, int, int); /* kinds 10-12: (ctx, kind, a, b) */
} gop;

/* <l||C^k||l'> = (-1)^(l+g) sqrt([l][l']) (l k l'; 0 0 0), 2g = l + k + l'
   (= -[l] F(l,2,l) for k = 2, l' = l) */
static double ckred(int l, int k, int lb)
{
    if ((l + k + lb) % 2 || k < abs(l - lb) || k > l + lb)
        return 0.0;
    return ((l + (l + k + lb) / 2) % 2 ? -1.0 : 1.0) * sqrt((2.0 * l + 1) * (2.0 * lb + 1)) *
           ac_fabc(l, k, lb);
}

static double c2red(int l, int lb)
{
    return ckred(l, 2, lb);
}

/* g_s of the magnetic transition operators (M1, M2, M3): AC_GS, or 2 with the
   environment variable AC_TRANS_GS2 set (ATSP's trans uses g_s = 2; for
   comparisons only) */
static double trans_gs(void)
{
    return getenv("AC_TRANS_GS2") ? 2.0 : AC_GS;
}

/* one-electron factors of the property operators. Sources: kinds 2-5
   (RMC, GC1e, ORMC) Neese (2001) and Pachucki (2004), Eq. (32) (H1, H2);
   6-9 (hyperfine) Lindgren-Rosen (1974); 10-12, 23-27 (multipoles)
   technical note, Sec. 5.3, Edmonds (1957), Eq. (7.1.1); 13-20 (relativistic M1)
   Pachucki (2004), Eqs. (59)-(62), kappa = 0; 21, 22 (isotope shift)
   technical note, Sec. 5.5 */
static double gop_f(void *p, int a, int b, int l, int lb)
{
    const gop *o = p;
    double sl = sqrt(1.5); /* <1/2||s||1/2> */
    if (lb != l && o->kind != 4 && o->kind != 7 && o->kind != 9 && o->kind < 10)
        return 0.0;
    switch (o->kind) {
    case 10: { /* r C^1: <a|r|b> <l||C^1||l'> */
        double c1 = ckred(l, 1, lb);
        return c1 == 0.0 ? 0.0 : o->rk(o->ctx, 0, a, b) * ac_ci_one_coef(sqrt(2.0), c1, 0, 2);
    }
    case 11: { /* grad: <l||C^1||l'> int R_a (R_b' - l' R_b/r) r^2 dr for l = l' + 1,
                  (R_b' + (l'+1) R_b/r) for l = l' - 1 (as ac_ci_rec2) */
        double c1 = ckred(l, 1, lb);
        return c1 == 0.0 ? 0.0
                         : o->rk(o->ctx, l == lb + 1 ? 1 : 2, a, b) *
                               ac_ci_one_coef(sqrt(2.0), c1, 0, 2);
    }
    case 12: { /* r^2 C^2 */
        double c2 = ckred(l, 2, lb);
        return c2 == 0.0 ? 0.0 : o->rk(o->ctx, 3, a, b) * ac_ci_one_coef(sqrt(2.0), c2, 0, 4);
    }
    case 23: { /* r^3 C^3 (rk kind 11) */
        double c3 = ckred(l, 3, lb);
        return c3 == 0.0 ? 0.0 : o->rk(o->ctx, 11, a, b) * ac_ci_one_coef(sqrt(2.0), c3, 0, 6);
    }
    case 24:   /* M2 = grad(r^2 C^2).((2/3) l + g_s s) (g_s of trans_gs) = sqrt6 r [C^1 x ((2/3) l + g_s s)]^(2)
                  (grad(r^k C^k) = sqrt(k(2k-1)) r^(k-1) [C^(k-1) x e]^(k)); orbital part
                  (0, 2): <l||[C^1 x l]^(2)||l'> = (-1)^(l+l') sqrt5 {1 1 2; l' l l'}
                  <l||C^1||l'> <l'||l||l'> (Edmonds (7.1.1)) */
    case 25:   /* spin part (1, 1), coupled to J rank 2 */
    case 26:   /* M3 = grad(r^3 C^3).((1/2) l + g_s s) = sqrt15 r^2 [C^2 x ((1/2) l + g_s s)]^(3);
                  orbital part (0, 3): (-1)^(l+l'+1) sqrt7 {2 1 3; l' l l'} <l||C^2||l'> <l'||l||l'> */
    case 27: { /* spin part (1, 2), coupled to J rank 3 */
        int k = o->kind < 26 ? 2 : 3; /* M k: C^(k-1), r^(k-1) (rk kinds 0, 3) */
        double ck = ckred(l, k - 1, lb);
        if (ck == 0.0)
            return 0.0;
        double r1 = o->rk(o->ctx, k == 2 ? 0 : 3, a, b), pre = sqrt(k * (2.0 * k - 1));
        if (o->kind == 25 || o->kind == 27)
            return pre * trans_gs() * r1 * ac_ci_one_coef(sl, ck, 2, 2 * k - 2);
        if (lb == 0)
            return 0.0;
        double v = ((l + lb + k) % 2 ? -1.0 : 1.0) * sqrt(2.0 * k + 1) *
                   ac_sixj(2 * k - 2, 2, 2 * k, 2 * lb, 2 * l, 2 * lb) * ck *
                   sqrt(lb * (lb + 1) * (2.0 * lb + 1));
        return pre * (2.0 / (k + 1)) * r1 * ac_ci_one_coef(sqrt(2.0), v, 0, 2 * k);
    }
    case 13: /* s f(r, p), f = r^2, p^2, Z/r (rk kinds 3, 4, 5): relativistic M1 */
    case 14:
    case 15:
        return l != lb ? 0.0
                       : o->rk(o->ctx, o->kind - 10, a, b) *
                             ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 16: /* l f(r, p), f = r^2, p^2 (rk kinds 3, 4): relativistic M1, orbital */
    case 17:
        return l != lb ? 0.0
                       : o->rk(o->ctx, o->kind - 13, a, b) *
                             ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    case 18: /* s C^2 f(r), f = r^2, Z/r (rk kinds 3, 5): relativistic M1, tensor */
    case 19: {
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0
                         : o->rk(o->ctx, o->kind == 18 ? 3 : 5, a, b) * ac_ci_one_coef(sl, c2, 2, 4);
    }
    case 20: { /* s [p x p]^(2) = -s [grad x grad]^(2): Edmonds (7.1.1) through l'' = l +- 1,
                  <l''||grad||l> = <l''||C^1||l> D, the radial part by parts
                  int (D R_a)(D R_b) r^2 dr (rk kinds 6-9 = 6 + 2 (l'' > l) + (l'' > l')) */
        double v = 0.0;
        for (int lpp = l - 1; lpp <= l + 1; lpp += 2) {
            if (lpp < 0 || abs(lpp - lb) != 1)
                continue;
            double c = ckred(lpp, 1, l) * ckred(lpp, 1, lb);
            if (c != 0.0)
                v += ac_sixj(2, 2, 4, 2 * lb, 2 * l, 2 * lpp) * c *
                     o->rk(o->ctx, 6 + 2 * (lpp > l) + (lpp > lb), a, b);
        }
        v *= -sqrt(5.0) * ((l + lb) % 2 ? -1.0 : 1.0);
        return v == 0.0 ? 0.0 : ac_ci_one_coef(sl, v, 2, 4);
    }
    case 21: /* p^2/2 (rk kind 4 = <p^2>) */
        return l != lb ? 0.0
                       : 0.5 * o->rk(o->ctx, 4, a, b) *
                             ac_ci_one_coef(sqrt(2.0), sqrt(2.0 * l + 1), 0, 0);
    case 22: /* delta(r): R_a(0) R_b(0)/(4 pi) (rk kind 10) */
        return l != 0 || lb != 0 ? 0.0
                                 : o->rk(o->ctx, 10, a, b) / (16.0 * atan(1.0)) *
                                       ac_ci_one_coef(sqrt(2.0), 1.0, 0, 0);
    case 6: /* l r^-3 (0 for s, whose <r^-3> diverges) */
        return l == 0 ? 0.0
                      : o->r(o->ctx, a, b) *
               ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    case 7: { /* -sqrt(10) (g_s/2) [s x C^2]^(1) r^-3 = (g_s/2)(3 (s.r^) r^ - s) r^-3 */
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0
                         : -sqrt(10.0) * 0.5 * AC_GS * o->r(o->ctx, a, b) *
                               ac_ci_one_coef(sl, c2, 2, 4);
    }
    case 8: /* (g_s/2)(8 pi/3) delta(r) s: <a|delta|b> = R_a(0) R_b(0)/(4 pi), so the
               factor on r00 = R_a(0) R_b(0) is (g_s/2)(2/3) */
        return l == 0 ? AC_GS / 3.0 * o->r(o->ctx, a, b) * ac_ci_one_coef(sl, 1.0, 2, 0) : 0.0;
    case 9: { /* C^2 r^-3 */
        double c2 = c2red(l, lb);
        return c2 == 0.0 ? 0.0 : o->r(o->ctx, a, b) * ac_ci_one_coef(sqrt(2.0), c2, 0, 4);
    }
    case 0:
        return a == b ? ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0) : 0.0;
    case 1:
        return a == b ? ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2) : 0.0;
    case 2:
        return -2.0 * ALPHA2 * o->r(o->ctx, a, b) * ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 3:
        return 0.5 * ALPHA2 * o->r(o->ctx, a, b) * (2.0 / 3.0) *
               ac_ci_one_coef(sl, sqrt(2.0 * l + 1), 2, 0);
    case 5:
        return -ALPHA2 * o->r(o->ctx, a, b) *
               ac_ci_one_coef(sqrt(2.0), sqrt(l * (l + 1) * (2.0 * l + 1)), 0, 2);
    default: {
        /* s_z - (s.r^) z^ = (2/3) s_z - (s.Q)_z, Q_ij = r^_i r^_j - delta_ij/3;
           (s.Q)^(1) = -sqrt(5/3) [s x Q^(2)]^(1), Q^(2) = sqrt(2/3) C^(2);
           <l||C^2||l'> couples l' = l and l +- 2 (s-d, p-f in a correlated CI) */
        double c2 = c2red(l, lb);
        if (c2 == 0.0)
            return 0.0;
        return 0.5 * ALPHA2 * o->r(o->ctx, a, b) * (sqrt(10.0) / 3.0) *
               ac_ci_one_coef(sl, c2, 2, 4);
    }
    }
}

/* the pairs (p, q) of the bases of two levels that a one-body operator can
   connect: c_p, c_q nonzero and at most one electron moved; row p holds
   q[start[p] .. start[p+1]-1], ascending. Made once for all the operators
   between the two levels, before the engine aligns any shells.
   Inside a window of kept elements (pc_begin) the list of the CSF pairs with
   at most one electron moved is there already (ls, lc: row i of the CSF list
   holds the CSFs lc[ls[i] .. ls[i+1]-1], ascending); then only pos is made,
   the place q of a CSF in the basis of B, or -1 (not in the basis, c_q = 0),
   and the same pairs come out of row A->idx[p] in the same order. */
typedef struct {
    size_t *start;
    int *q;
    const size_t *ls;
    const int *lc;
    int *pos;
} pairs1;

static void pairs1_free(pairs1 *P)
{
    free(P->start);
    free(P->q);
    free(P->pos);
    P->start = NULL;
    P->q = NULL;
    P->pos = NULL;
}

static int pc1_list(const ac_ci *ci, const size_t **start, const int **col);

static int pairs1_build(const ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B, pairs1 *P)
{
    P->start = NULL;
    P->q = NULL;
    P->pos = NULL;
    int asc = 1; /* the order of the sum over q is that of the list only then */
    for (int q = 1; q < B->dim && asc; q++)
        asc = B->idx[q - 1] < B->idx[q];
    if (asc && pc1_list(ci, &P->ls, &P->lc)) {
        P->pos = malloc((size_t)(ci->ncsf > 0 ? ci->ncsf : 1) * sizeof *P->pos);
        if (!P->pos)
            return -1;
        for (int j = 0; j < ci->ncsf; j++)
            P->pos[j] = -1;
        for (int q = 0; q < B->dim; q++)
            if (B->c[q] != 0.0)
                P->pos[B->idx[q]] = q;
        return 0;
    }
    P->start = calloc((size_t)A->dim + 1, sizeof *P->start);
    if (!P->start)
        return -1;
#pragma omp parallel for schedule(dynamic, 16)
    for (int p = 0; p < A->dim; p++) {
        size_t m = 0;
        if (A->c[p] != 0.0)
            for (int q = 0; q < B->dim; q++)
                m += B->c[q] != 0.0 && moved(ci, A->idx[p], B->idx[q]) <= 1;
        P->start[p + 1] = m;
    }
    for (int p = 0; p < A->dim; p++)
        P->start[p + 1] += P->start[p];
    P->q = malloc((P->start[A->dim] + 1) * sizeof *P->q);
    if (!P->q) {
        pairs1_free(P);
        return -1;
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (int p = 0; p < A->dim; p++) {
        size_t m = P->start[p];
        if (A->c[p] != 0.0)
            for (int q = 0; q < B->dim; q++)
                if (B->c[q] != 0.0 && moved(ci, A->idx[p], B->idx[q]) <= 1)
                    P->q[m++] = q;
    }
    return 0;
}

/* ac_lsj_reduce without its factor w, for every pair of total labels of the
   bases of two levels (label_numbers): f[la[p] * nlb + lb[q]]. f is NULL
   when it could not be made; then ac_lsj_reduce is called pair by pair. */
typedef struct {
    int *la, *lb, nlb;
    double *f;
} redtab;

static void redtab_free(redtab *r)
{
    free(r->la);
    free(r->lb);
    free(r->f);
    r->la = r->lb = NULL;
    r->f = NULL;
}

static void redtab_make(redtab *r, const ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                        int ks, int kl, int tK)
{
    ac_lab *xa = NULL, *xb = NULL;
    int na = 0, nb = 0;
    r->f = NULL;
    r->la = label_numbers(ci, A->dim, A->idx, &xa, &na);
    r->lb = label_numbers(ci, B->dim, B->idx, &xb, &nb);
    r->nlb = nb;
    if (r->la && r->lb && na && nb)
        r->f = malloc((size_t)na * (size_t)nb * sizeof *r->f);
    for (int x = 0; r->f && x < na; x++)
        for (int y = 0; y < nb; y++)
            r->f[(size_t)x * nb + y] =
                sqrt((double)(A->tJ + 1) * (B->tJ + 1) * (tK + 1)) *
                ac_ninej(xa[x].ts, xb[y].ts, ks, xa[x].tl, xb[y].tl, kl, A->tJ, B->tJ, tK);
    free(xa);
    free(xb);
    if (!r->f)
        redtab_free(r);
}

/* ac_lsj_reduce(a, A->tJ, b, B->tJ, ks, kl, tK, w) of the pair (p, q) */
static inline double redtab_term(const redtab *r, int p, int q, double w)
{
    return w == 0.0 ? 0.0 : r->f[(size_t)r->la[p] * (size_t)r->nlb + (size_t)r->lb[q]] * w;
}

/* the kept one-body elements of CSF pairs (below, with the two-electron tables) */
static double *pc1_table(const ac_ci *ci, const gop *o, int ks, int kl, const size_t **start,
                         const int **col);

/* <psi_A||[W]^(K)||psi_B> (2K = tK) of the operator kind, psi_A over the
   basis idxA of 2J = tJA and psi_B over idxB of tJB (the same level for g_J),
   on the pairs P of pairs1_build */
static double pair_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                           const pairs1 *P, const gop *o, int ks, int kl, int tK, int *err)
{
    /* rows in parallel on copies of ci, as in ac_lsj_prepare; the row sums
       are added in order, so the result does not depend on the threads */
    double *row = calloc((size_t)A->dim, sizeof *row);
    int fe = 0;
    if (!row) {
        *err = 1;
        return 0.0;
    }
    long ops = 0, ncomp = 0, nread = 0;
    double t0 = omp_get_wtime();
    /* the element of a CSF pair is the same for every level: kept in tab at
       the position of the pair in the list (gs, gc), NaN until computed */
    const size_t *gs = NULL;
    const int *gc = NULL;
    double *tab = pc1_table(ci, o, ks, kl, &gs, &gc);
    redtab rt;
    redtab_make(&rt, ci, A, B, ks, kl, tK);
    double t1 = omp_get_wtime();
#pragma omp parallel reduction(+ : ops, ncomp, nread)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
#pragma omp for schedule(dynamic, 4)
        for (int p = 0; p < A->dim; p++) {
            if (A->c[p] == 0.0)
                continue;
            double s = 0.0;
            int i = A->idx[p];
            size_t g = tab ? gs[i] : 0;
            size_t m0 = P->pos ? P->ls[i] : P->start[p];
            size_t m1 = P->pos ? P->ls[i + 1] : P->start[p + 1];
            for (size_t m = m0; m < m1; m++) {
                int q, j, e = 0;
                double w, *slot = NULL;
                if (P->pos) { /* straight from the list of the window */
                    j = P->lc[m];
                    q = P->pos[j];
                    if (q < 0)
                        continue;
                    if (tab)
                        slot = tab + m;
                } else {
                    q = P->q[m];
                    j = B->idx[q];
                    if (tab) { /* row i of the list and the q of P both ascend */
                        while (g < gs[i + 1] && gc[g] < j)
                            g++;
                        if (g < gs[i + 1] && gc[g] == j)
                            slot = tab + g;
                    }
                }
                if (slot && !isnan(*slot)) {
                    w = *slot;
                    nread++;
                } else {
                    w = ac_ci_one_ll(&loc, i, j, ks, kl, gop_f, (void *)o, &e);
                    if (slot && !e)
                        *slot = w;
                    ncomp++;
                }
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    fe = e;
                    break;
                }
                s += B->c[q] * (rt.f ? redtab_term(&rt, p, q, w)
                                     : ac_lsj_reduce(total(&loc, A->idx[p]), A->tJ,
                                                     total(&loc, B->idx[q]), B->tJ, ks, kl, tK, w));
            }
            row[p] = A->c[p] * s;
        }
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    redtab_free(&rt);
    ci->ops_evaluated += ops;
    if (getenv("AC_TIME_PAIR"))
        fprintf(stderr, "pair_reduced kind %d (%d %d): setup %.2f s, sum %.2f s, %ld computed, "
                        "%ld read, table %s\n",
                o->kind, ks, kl, t1 - t0, omp_get_wtime() - t1, ncomp, nread, tab ? "yes" : "no");
    double s = 0.0;
    for (int p = 0; p < A->dim; p++)
        s += row[p];
    free(row);
    *err = fe;
    return fe ? 0.0 : s;
}

/* ---- the CSF-pair elements of the two-electron property operators -------
   pair_reduced_2e needs <i||W||j> of the same CSF pairs again and again: for
   every level of a J block, for every J block that holds both CSFs, and for
   every level pair of the zoff table. Within one call of
   ac_lsj_print_levels_keep or ac_lsj_print_cross (ci and its integrals do
   not change there) each element is kept in a dense n x n table per
   operator, NaN until it is computed. pair_reduced_2e gives row i =
   A->idx[p] to one thread, which alone reads and writes row i, so the tables
   need no locking, and every value is the one an uncached run computes, bit
   for bit. A table is allocated on first use while the total stays below
   AC_PROP_CACHE_MB (default 2048); an operator without a table is computed
   as before.
   The one-electron operators (pair_reduced: g_J, zoff, hfs, trans, is) are
   kept the same way, but sparsely: a one-body operator connects only CSFs
   that differ by at most one electron, so the pairs (i, j) with moved <= 1
   are listed once per window (row i: pc1.col[pc1.start[i] ..
   pc1.start[i+1]-1], ascending) and each operator has one value per listed
   pair. An operator is identified by its kind, ranks and radial callbacks. */
enum { PC_GC0, PC_GC1, PC_GC2, PC_OZ, PC_REC, PC_M10, PC_M12, PC_SMS, PC_N };

static struct {
    const ac_ci *ci;
    int n, depth;
    size_t bytes, limit;
    double *v[PC_N];
    /* the runs of one configuration (config_runs), the run of each CSF, and
       per operator the pairs of runs whose block of the table is filled */
    int ng, *gs, *run;
    unsigned char *done[PC_N];
    unsigned char *near; /* pairs of runs at most two electrons apart (pc_near2) */
    unsigned warned;     /* operators whose table was refused, told once */
} pc;

static int pc_slot(int kp)
{
    switch (kp) {
    case 0:
    case 1:
    case 2:
        return PC_GC0 + kp;
    case -1:
        return PC_OZ;
    case -2:
        return PC_REC;
    case -3:
        return PC_SMS;
    case 10:
        return PC_M10;
    case 12:
        return PC_M12;
    }
    return -1;
}

static void pc_begin(const ac_ci *ci)
{
    if (pc.depth++)
        return;
    const char *s = getenv("AC_PROP_CACHE_MB");
    pc.ci = ci;
    pc.n = ci->ncsf;
    pc.bytes = 0;
    pc.warned = 0;
    pc.limit = (size_t)((s ? atof(s) : 2048.0) * 1048576.0);
}

#define PC1_MAX 48

typedef struct {
    gop o;
    int ks, kl;
    double *v;
} pc1_op;

static struct {
    int state; /* 0 not built, 1 built, -1 cannot be built */
    size_t *start;
    int *col;
    int nop;
    pc1_op op[PC1_MAX];
} pc1;

static void pc_end(void)
{
    if (--pc.depth)
        return;
    for (int k = 0; k < PC_N; k++) {
        free(pc.v[k]);
        free(pc.done[k]);
        pc.v[k] = NULL;
        pc.done[k] = NULL;
    }
    free(pc.gs);
    free(pc.run);
    free(pc.near);
    pc.gs = pc.run = NULL;
    pc.near = NULL;
    pc.ng = 0;
    for (int k = 0; k < pc1.nop; k++)
        free(pc1.op[k].v);
    free(pc1.start);
    free(pc1.col);
    memset(&pc1, 0, sizeof pc1);
    pc.ci = NULL;
}

static int pc_runs(const ac_ci *ci);

/* pc1_build by configurations: how many electrons a pair moves is the same
   for all the CSF pairs of two runs, so it is asked once per pair of runs
   and the rows of the list are written run by run (the same list). Returns
   0 when it cannot be done this way (no runs, or too many of them). */
static int pc1_build_runs(const ac_ci *ci)
{
    int n = ci->ncsf;
    if (!pc_runs(ci) || (size_t)pc.ng * (size_t)pc.ng > ((size_t)1 << 26))
        return 0;
    int ng = pc.ng;
    const int *gs = pc.gs, *run = pc.run;
    unsigned char *ok = malloc((size_t)ng * (size_t)ng + 1);
    size_t *cnt = calloc((size_t)ng + 1, sizeof *cnt);
    size_t *start = calloc((size_t)n + 1, sizeof *start);
    int *col = NULL;
    if (!ok || !cnt || !start)
        goto bad;
#pragma omp parallel for schedule(dynamic, 8)
    for (int ra = 0; ra < ng; ra++) {
        size_t m = 0;
        for (int rb = 0; rb < ng; rb++) {
            int near = moved(ci, gs[ra], gs[rb]) <= 1;
            ok[(size_t)ra * ng + rb] = (unsigned char)near;
            if (near)
                m += (size_t)(gs[rb + 1] - gs[rb]);
        }
        cnt[ra] = m;
    }
    for (int i = 0; i < n; i++)
        start[i + 1] = start[i] + cnt[run[i]];
    size_t sz = ((size_t)n + 1) * sizeof *start + start[n] * sizeof(int);
    col = pc.bytes + sz > pc.limit ? NULL : malloc((start[n] + 1) * sizeof *col);
    if (!col)
        goto bad;
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        size_t m = start[i];
        const unsigned char *row = ok + (size_t)run[i] * ng;
        for (int rb = 0; rb < ng; rb++)
            if (row[rb])
                for (int j = gs[rb]; j < gs[rb + 1]; j++)
                    col[m++] = j;
    }
    free(ok);
    free(cnt);
    pc1.start = start;
    pc1.col = col;
    pc1.state = 1;
    pc.bytes += sz;
    return 1;
bad:
    free(ok);
    free(cnt);
    free(start);
    return 0;
}

/* the list of the CSF pairs with at most one electron moved */
static void pc1_build(const ac_ci *ci)
{
    int n = ci->ncsf;
    double t0 = omp_get_wtime();
    pc1.state = -1;
    if (pc1_build_runs(ci)) {
        if (getenv("AC_TIME"))
            fprintf(stderr, "pc1 list by runs: %zu pairs, %.1f MB, %.2f s (kept %.1f of %.1f MB)\n",
                    pc1.start[n], pc1.start[n] * 4.0 / 1048576.0, omp_get_wtime() - t0,
                    pc.bytes / 1048576.0, pc.limit / 1048576.0);
        return;
    }
    if (getenv("AC_TIME"))
        fprintf(stderr, "pc1 list by runs: not made (%d runs, %.2f s)\n", pc.ng,
                omp_get_wtime() - t0);
    size_t *start = calloc((size_t)n + 1, sizeof *start);
    if (!start)
        return;
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        size_t m = 0;
        for (int j = 0; j < n; j++)
            m += moved(ci, i, j) <= 1;
        start[i + 1] = m;
    }
    for (int i = 0; i < n; i++)
        start[i + 1] += start[i];
    size_t sz = ((size_t)n + 1) * sizeof *start + start[n] * sizeof(int);
    int *col = pc.bytes + sz > pc.limit ? NULL : malloc((start[n] + 1) * sizeof *col);
    if (getenv("AC_TIME"))
        fprintf(stderr, "pc1 list pair by pair: %zu pairs, %.1f MB, %s, %.2f s so far\n", start[n],
                sz / 1048576.0, col ? "kept" : "not kept (over the limit)", omp_get_wtime() - t0);
    if (!col) {
        free(start);
        return;
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        size_t m = start[i];
        for (int j = 0; j < n; j++)
            if (moved(ci, i, j) <= 1)
                col[m++] = j;
    }
    pc1.start = start;
    pc1.col = col;
    pc1.state = 1;
    pc.bytes += sz;
}

/* the list of the window (made on first use); 0 outside a window, for
   another ci or over the limit; called outside the parallel region */
static int pc1_list(const ac_ci *ci, const size_t **start, const int **col)
{
    if (!pc.depth || ci != pc.ci)
        return 0;
    if (!pc1.state)
        pc1_build(ci);
    if (pc1.state < 0)
        return 0;
    *start = pc1.start;
    *col = pc1.col;
    return 1;
}

/* the values of the one-body operator (o, ks, kl) on the listed pairs, with
   the list, or NULL (outside a window, another ci, over the limit); called
   outside the parallel region */
static double *pc1_table(const ac_ci *ci, const gop *o, int ks, int kl, const size_t **start,
                         const int **col)
{
    if (!pc.depth || ci != pc.ci)
        return NULL;
    if (!pc1.state)
        pc1_build(ci);
    if (pc1.state < 0)
        return NULL;
    int k;
    for (k = 0; k < pc1.nop; k++) {
        const pc1_op *t = &pc1.op[k];
        if (t->o.kind == o->kind && t->o.r == o->r && t->o.ctx == o->ctx && t->o.rk == o->rk &&
            t->ks == ks && t->kl == kl)
            break;
    }
    if (k == pc1.nop) {
        size_t nnz = pc1.start[pc.n], sz = nnz * sizeof(double);
        if (getenv("AC_TIME"))
            fprintf(stderr, "pc1 table %d (kind %d, %d %d): %.1f MB, %s (kept %.1f of %.1f MB)\n", k,
                    o->kind, ks, kl, sz / 1048576.0,
                    k == PC1_MAX || pc.bytes + sz > pc.limit ? "not kept" : "kept",
                    pc.bytes / 1048576.0, pc.limit / 1048576.0);
        if (k == PC1_MAX || pc.bytes + sz > pc.limit)
            return NULL;
        double *v = malloc(sz ? sz : 1);
        if (!v)
            return NULL;
        for (size_t i = 0; i < nnz; i++)
            v[i] = NAN;
        pc1.op[k] = (pc1_op){*o, ks, kl, v};
        pc1.nop++;
        pc.bytes += sz;
    }
    *start = pc1.start;
    *col = pc1.col;
    return pc1.op[k].v;
}

/* the table of operator kp, or NULL (outside a window, another ci, over the
   limit); called outside the parallel region */
static double *pc_table(const ac_ci *ci, int kp)
{
    int k = pc_slot(kp);
    if (!pc.depth || ci != pc.ci || k < 0)
        return NULL;
    if (!pc.v[k]) {
        size_t nn = (size_t)pc.n * pc.n, sz = nn * sizeof(double);
        if (pc.bytes + sz > pc.limit) {
            /* said once per operator: the run goes on, but slowly */
            if (!(pc.warned >> k & 1)) {
                pc.warned |= 1u << k;
                fprintf(stderr,
                        "ac_lsj: warning: the table of a two-electron property (%d CSFs, %.2f GB) "
                        "does not fit AC_PROP_CACHE_MB (%.0f MB, %.0f MB in use): its elements "
                        "are computed again for every level or pair of levels that needs them\n",
                        pc.n, sz / 1073741824.0, pc.limit / 1048576.0, pc.bytes / 1048576.0);
            }
            return NULL;
        }
        double *t = malloc(sz);
        if (!t)
            return NULL;
        for (size_t i = 0; i < nn; i++)
            t[i] = NAN;
        pc.v[k] = t;
        pc.bytes += sz;
    }
    return pc.v[k];
}

/* the runs of the window (pc.gs, pc.run), made on first use; 0 outside a
   window, for another ci or when out of memory */
static int pc_runs(const ac_ci *ci)
{
    size_t n = (size_t)ci->ncsf;
    if (!pc.depth || ci != pc.ci)
        return 0;
    if (!pc.gs) {
        pc.gs = config_runs(ci, &pc.ng);
        pc.run = malloc((n ? n : 1) * sizeof *pc.run);
        if (!pc.gs || !pc.run) {
            free(pc.gs);
            free(pc.run);
            pc.gs = pc.run = NULL;
            return 0;
        }
        for (int r = 0; r < pc.ng; r++)
            for (int i = pc.gs[r]; i < pc.gs[r + 1]; i++)
                pc.run[i] = r;
    }
    return 1;
}

/* How many electrons a CSF pair moves is a matter of its two configurations:
   near[ra * ng + rb] is 1 when the runs ra and rb differ by at most two
   electrons, 2 when by more (0: not looked at yet). Filled here for the runs
   the levels A and B touch; NULL when there is no table (then moved is
   asked pair by pair). Called outside the parallel region. */
static const unsigned char *pc_near2(const ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B)
{
    if (!pc_runs(ci))
        return NULL;
    int ng = pc.ng;
    if (!pc.near) {
        size_t sz = (size_t)ng * (size_t)ng;
        if (pc.bytes + sz > pc.limit)
            return NULL;
        pc.near = calloc(sz ? sz : 1, 1);
        if (!pc.near)
            return NULL;
        pc.bytes += sz;
    }
    unsigned char *ha = calloc((size_t)ng + 1, 1), *hb = calloc((size_t)ng + 1, 1);
    if (!ha || !hb) {
        free(ha);
        free(hb);
        return NULL;
    }
    for (int p = 0; p < A->dim; p++)
        if (A->c[p] != 0.0)
            ha[pc.run[A->idx[p]]] = 1;
    for (int q = 0; q < B->dim; q++)
        if (B->c[q] != 0.0)
            hb[pc.run[B->idx[q]]] = 1;
#pragma omp parallel for schedule(dynamic, 8)
    for (int ra = 0; ra < ng; ra++) {
        if (!ha[ra])
            continue;
        unsigned char *row = pc.near + (size_t)ra * (size_t)ng;
        for (int rb = 0; rb < ng; rb++)
            if (hb[rb] && !row[rb])
                row[rb] = moved(ci, pc.gs[ra], pc.gs[rb]) <= 2 ? 1 : 2;
    }
    free(ha);
    free(hb);
    return pc.near;
}

/* Fill the table tab of operator kp for the pairs of runs (configurations)
   that the levels A and B touch and that are not filled yet: all the CSF
   pairs of two runs together (ac_ci_prop2_block; the values are those of
   ac_ci_gc2e and ac_ci_oz2e). Operators without a block form, or
   AC_BP_PAIRWISE in the environment, leave the table to pair_reduced_2e,
   which then computes pair by pair. Returns 0, or the engine's error. */
static int pc_fill_blocks(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B, int kp,
                          double *tab)
{
    int k = pc_slot(kp), ng;
    size_t n = (size_t)ci->ncsf;
    if (!tab || kp < -1 || kp > 2 || getenv("AC_BP_PAIRWISE") || !pc_runs(ci))
        return 0;
    ng = pc.ng;
    if (!pc.done[k]) {
        size_t sz = (size_t)ng * (size_t)ng;
        if (pc.bytes + sz > pc.limit)
            return 0;
        pc.done[k] = calloc(sz ? sz : 1, 1);
        if (!pc.done[k])
            return 0;
        pc.bytes += sz;
    }
    unsigned char *done = pc.done[k];
    unsigned char *ha = calloc((size_t)ng + 1, 1), *hb = calloc((size_t)ng + 1, 1);
    int *ra = malloc(((size_t)ng + 1) * sizeof *ra), *rb = malloc(((size_t)ng + 1) * sizeof *rb);
    int na = 0, nb = 0, fe = 0;
    int *todo = NULL;
    size_t ntodo = 0;
    long ops = 0;
    if (!ha || !hb || !ra || !rb)
        goto out;
    for (int p = 0; p < A->dim; p++)
        if (A->c[p] != 0.0)
            ha[pc.run[A->idx[p]]] = 1;
    for (int q = 0; q < B->dim; q++)
        if (B->c[q] != 0.0)
            hb[pc.run[B->idx[q]]] = 1;
    for (int r = 0; r < ng; r++) {
        if (ha[r])
            ra[na++] = r;
        if (hb[r])
            rb[nb++] = r;
    }
    for (int pass = 0; pass < 2; pass++) {
        ntodo = 0;
        for (int x = 0; x < na; x++)
            for (int y = 0; y < nb; y++) {
                size_t pq = (size_t)ra[x] * (size_t)ng + (size_t)rb[y];
                if (done[pq])
                    continue;
                if (moved(ci, pc.gs[ra[x]], pc.gs[rb[y]]) > 2) {
                    done[pq] = 1; /* no elements: nothing to fill */
                    continue;
                }
                if (pass)
                    todo[2 * ntodo] = ra[x], todo[2 * ntodo + 1] = rb[y];
                ntodo++;
            }
        if (!ntodo)
            goto out;
        if (!pass) {
            todo = malloc(2 * ntodo * sizeof *todo);
            if (!todo)
                goto out;
        }
    }
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
        double *buf = NULL;
        size_t bufn = 0;
#pragma omp for schedule(dynamic, 1)
        for (long g = 0; g < (long)ntodo; g++) {
            int RA = todo[2 * g], RB = todo[2 * g + 1], e = 0;
            int i0 = pc.gs[RA], ni = pc.gs[RA + 1] - i0, j0 = pc.gs[RB], nj = pc.gs[RB + 1] - j0;
            size_t nn = (size_t)ni * (size_t)nj;
            if (nn > bufn) {
                free(buf);
                bufn = nn;
                buf = malloc(bufn * sizeof *buf);
            }
            if (!buf) /* left unfilled: pair_reduced_2e computes these pair by pair */
                continue;
            ac_ci_prop2_block(&loc, kp, i0, ni, j0, nj, buf, &e);
            if (e) {
#pragma omp critical(ac_lsj_fail)
                fe = e;
                continue;
            }
            for (int i = 0; i < ni; i++)
                memcpy(tab + (size_t)(i0 + i) * n + (size_t)j0, buf + (size_t)i * (size_t)nj,
                       (size_t)nj * sizeof *buf);
            done[(size_t)RA * (size_t)ng + (size_t)RB] = 1;
        }
        free(buf);
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
out:
    free(ha);
    free(hb);
    free(ra);
    free(rb);
    free(todo);
    return fe;
}

/* <psi_A||[W^(1 kp)]^(1)||psi_B> of the two-electron gauge correction, or
   <psi_A||W^(0 1)||psi_B> of the orbit-orbit Zeeman term (kp = -1) or of the
   two-body recoil term (kp = -2), or <psi_A||[W^(1 K)]^(1)||psi_B> of the
   spin two-electron M1 term (kp = 10 + K, ac_ci_m1s2), or the scalar
   <psi_A||W^(0 0)||psi_B> of the specific mass shift (kp = -3, ac_ci_sms
   with the gradient factors of rk, ctx) */
static double pair_reduced_2e(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B, int kp,
                              double (*rk)(void *, int, int, int), void *ctx, int *err)
{
    /* in parallel as pair_reduced; each thread has its own Theta cache */
    double *row = calloc((size_t)A->dim, sizeof *row);
    int fe = 0;
    if (!row) {
        *err = 1;
        return 0.0;
    }
    long ops = 0;
    double *tab = pc_table(ci, kp);
    size_t n = (size_t)ci->ncsf;
    /* the blocks of the table that these levels need, filled together */
    if ((fe = pc_fill_blocks(ci, A, B, kp, tab)) != 0) {
        free(row);
        *err = fe;
        return 0.0;
    }
    /* by configuration: which pairs a two-body operator can connect; by
       total label: the recoupling to J */
    const unsigned char *near = pc_near2(ci, A, B);
    const int *run = near ? pc.run : NULL;
    size_t ngr = near ? (size_t)pc.ng : 0;
    int K = kp >= 10 ? kp - 10 : kp;
    redtab rt;
    redtab_make(&rt, ci, A, B, kp == -3 || kp < 0 ? 0 : 2, kp == -3 ? 0 : kp < 0 ? 2 : 2 * K,
                kp == -3 ? 0 : 2);
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
#pragma omp for schedule(dynamic, 4)
        for (int p = 0; p < A->dim; p++) {
            if (A->c[p] == 0.0)
                continue;
            double s = 0.0;
            const unsigned char *nrow = near ? near + (size_t)run[A->idx[p]] * ngr : NULL;
            for (int q = 0; q < B->dim; q++) {
                /* a two-body operator moves at most two electrons */
                if (B->c[q] == 0.0 ||
                    (nrow ? nrow[run[B->idx[q]]] != 1 : moved(&loc, A->idx[p], B->idx[q]) > 2))
                    continue;
                int e = 0, i = A->idx[p], j = B->idx[q];
                double *slot = tab ? tab + (size_t)i * n + j : NULL, w;
                if (slot && !isnan(*slot))
                    w = *slot;
                else {
                    w = kp == -3   ? ac_ci_sms(&loc, i, j, rk, ctx, &e)
                        : kp == -2 ? ac_ci_rec2(&loc, i, j, &e)
                        : kp < 0   ? ac_ci_oz2e(&loc, i, j, &e)
                        : kp >= 10 ? ac_ci_m1s2(&loc, i, j, kp - 10, &e)
                                   : ac_ci_gc2e(&loc, i, j, kp, &e);
                    if (slot && !e)
                        *slot = w;
                }
                if (e) {
#pragma omp critical(ac_lsj_fail)
                    fe = e;
                    break;
                }
                if (rt.f) {
                    s += B->c[q] * redtab_term(&rt, p, q, w);
                    continue;
                }
                ac_lab a = total(&loc, i), b = total(&loc, j);
                s += B->c[q] * (kp == -3 ? ac_lsj_reduce(a, A->tJ, b, B->tJ, 0, 0, 0, w)
                                : kp < 0 ? ac_lsj_reduce(a, A->tJ, b, B->tJ, 0, 2, 2, w)
                                         : ac_lsj_reduce(a, A->tJ, b, B->tJ, 2, 2 * K, 2, w));
            }
            row[p] = A->c[p] * s;
        }
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    redtab_free(&rt);
    ci->ops_evaluated += ops;
    double s = 0.0;
    for (int p = 0; p < A->dim; p++)
        s += row[p];
    free(row);
    *err = fe;
    return fe ? 0.0 : s;
}

static double pair_reduced_gc2e(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                                int kp, int *err)
{
    return pair_reduced_2e(ci, A, B, kp, NULL, NULL, err);
}

/* A radial integral <a|g|b> of the caller (tkin, zr) is asked again for
   every CSF pair and is a quadrature each time: here it is kept per pair of
   orbitals (one table per thread, NaN until asked), so the value is the one
   the caller's function gives. */
typedef struct {
    double (*f)(void *, int, int);
    void *ctx;
    double *tab; /* (AC_ADDR_MAXORB + 1)^2, or NULL: no table */
} rkeep;

static double rkeep_f(void *p, int a, int b)
{
    const rkeep *c = p;
    if (!c->tab || a < 0 || b < 0 || a > AC_ADDR_MAXORB || b > AC_ADDR_MAXORB)
        return c->f(c->ctx, a, b);
    double *s = c->tab + (size_t)a * (AC_ADDR_MAXORB + 1) + b;
    if (isnan(*s))
        *s = c->f(c->ctx, a, b);
    return *s;
}

static double *rkeep_tab(void)
{
    size_t nn = (size_t)(AC_ADDR_MAXORB + 1) * (AC_ADDR_MAXORB + 1);
    double *t = malloc(nn * sizeof *t);
    for (size_t z = 0; t && z < nn; z++)
        t[z] = NAN;
    return t;
}

/* The one-body parts of ac_lsj_zeeman_reduced(A, A) of the levels
   k[0 .. nk-1] of all the J blocks in one pass over the CSF pairs: each
   element <i||W||j> is computed once and added to every level of every J
   block that holds both CSFs, so no table of elements is needed (for a long
   list the tables of pc1 do not fit) and no element is computed twice. The
   levels of one J block follow each other in k and have the same basis
   (ascending). The pairs are taken configuration by configuration
   (pc_runs), ascending in q for every row, and the sums are formed as in
   pair_reduced, so the numbers are the same. g[x] is set (the one-body
   parts; the rest 0).
   Returns 0, 1 when it cannot be done this way (nothing computed; the caller
   goes level by level), -1 on an engine error (*err). */
static int zeeman_1e_all(ac_ci *ci, const ac_lsj_kept *k, int nk,
                         double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                         void *ctx, ac_lsj_g *g, int *err)
{
    enum { NOP = 7 };
    typedef struct {
        int tJ, dim, l0, nl;
        const int *idx;
        size_t roff;
    } grp;
    struct {
        gop o;
        int ks, kl, on;
    } op[NOP] = {{{0, NULL, NULL, NULL}, 2, 0, 1},
                 {{1, NULL, NULL, NULL}, 0, 2, 1},
                 {{2, tkin, ctx, NULL}, 2, 0, tkin != NULL},
                 {{5, tkin, ctx, NULL}, 0, 2, tkin != NULL},
                 {{3, zr, ctx, NULL}, 2, 0, zr != NULL},
                 {{4, zr, ctx, NULL}, 2, 4, zr != NULL},
                 {{20, NULL, ctx, ac_lsj_h6_rk}, 2, 4, ac_lsj_h6_rk != NULL}};
    /* the operators of the same ranks (ks, kl): (2, 0), (0, 2), (2, 4) */
    static const int same[3][3] = {{0, 2, 4}, {1, 3, -1}, {5, 6, -1}};
    *err = 0;
    if (nk <= 0)
        return 0;
    if (!pc_runs(ci))
        return 1;
    const int n = ci->ncsf, ng = pc.ng, *gs = pc.gs, *run = pc.run;
    int ngrp = 0, rc = 1, fe = 0;
    for (int x = 0; x < nk; x++)
        ngrp += x == 0 || k[x].v.tJ != k[x - 1].v.tJ;
    grp *G = calloc((size_t)ngrp, sizeof *G);
    /* pos[j * n + i]: the place of CSF i in the basis of group j, or -1;
       the places of configuration r there: fst[j * ng + r] .. + cnt - 1 */
    int *pos = malloc((size_t)ngrp * (size_t)(n ? n : 1) * sizeof *pos);
    int *fst = malloc((size_t)ngrp * (size_t)ng * sizeof *fst);
    int *cnt = calloc((size_t)ngrp * (size_t)ng, sizeof *cnt);
    redtab *rt = calloc((size_t)NOP * (size_t)ngrp, sizeof *rt);
    double *val = calloc((size_t)NOP * (size_t)nk, sizeof *val), *row = NULL;
    size_t nrow = 0, smax = 0;
    if (!G || !pos || !fst || !cnt || !rt || !val)
        goto out;
    ngrp = 0;
    for (int x = 0; x < nk; x++) {
        if (x == 0 || k[x].v.tJ != k[x - 1].v.tJ) {
            G[ngrp++] = (grp){k[x].v.tJ, k[x].v.dim, x, 0, k[x].v.idx, 0};
        } else if (k[x].v.dim != k[x - 1].v.dim ||
                   memcmp(k[x].v.idx, k[x - 1].v.idx, (size_t)k[x].v.dim * sizeof(int)))
            goto out;
        G[ngrp - 1].nl++;
    }
    for (size_t z = 0; z < (size_t)ngrp * (size_t)n; z++)
        pos[z] = -1;
    for (int j = 0; j < ngrp; j++) {
        const int *idx = G[j].idx;
        for (int p = 0; p < G[j].dim; p++) {
            if (p && idx[p - 1] >= idx[p])
                goto out;
            size_t z = (size_t)j * (size_t)ng + (size_t)run[idx[p]];
            if (!cnt[z]++)
                fst[z] = p;
            pos[(size_t)j * n + idx[p]] = p;
        }
        G[j].roff = nrow;
        nrow += (size_t)G[j].nl * (size_t)G[j].dim;
    }
    for (int r = 0; r < ng; r++) {
        size_t t = 0;
        for (int j = 0; j < ngrp; j++)
            t += (size_t)cnt[(size_t)j * ng + r] * (size_t)G[j].nl;
        if (t > smax)
            smax = t;
    }
    row = malloc((size_t)NOP * (nrow ? nrow : 1) * sizeof *row);
    if (!row)
        goto out;
    for (int x = 0; x < NOP; x++)
        for (int j = 0; j < ngrp && op[x].on; j++) {
            ac_lsj_level A = {G[j].tJ, G[j].dim, G[j].idx, k[G[j].l0].v.c};
            redtab_make(&rt[(size_t)x * ngrp + j], ci, &A, &A, op[x].ks, op[x].kl, 2);
            if (!rt[(size_t)x * ngrp + j].f)
                goto out;
        }
    long ops = 0;
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
        /* s[x * smax + soff[j] + (p - first place) * nl + l]: the sum of row p
           of group j for level l and operator x */
        double *s = malloc((size_t)NOP * (smax ? smax : 1) * sizeof *s);
        size_t *soff = malloc((size_t)ngrp * sizeof *soff);
        int *both = malloc((size_t)ngrp * sizeof *both);
        /* the operators of this thread: their radial integrals kept (rkeep) */
        rkeep kt = {tkin, ctx, rkeep_tab()}, kz = {zr, ctx, rkeep_tab()};
        gop lop[NOP];
        for (int x = 0; x < NOP; x++) {
            lop[x] = op[x].o;
            if (lop[x].r == tkin && tkin)
                lop[x].r = rkeep_f, lop[x].ctx = &kt;
            else if (lop[x].r == zr && zr)
                lop[x].r = rkeep_f, lop[x].ctx = &kz;
        }
#pragma omp for schedule(dynamic, 1)
        for (int RA = 0; RA < ng; RA++) {
            if (!s || !soff || !both || fe)
                continue;
            size_t t = 0;
            int e = 0;
            for (int j = 0; j < ngrp; j++) {
                soff[j] = t;
                t += (size_t)cnt[(size_t)j * ng + RA] * (size_t)G[j].nl;
            }
            if (!t)
                continue;
            for (int x = 0; x < NOP; x++)
                if (op[x].on)
                    memset(s + (size_t)x * smax, 0, t * sizeof *s);
            for (int RB = 0; RB < ng && !e; RB++) {
                int nb = 0;
                for (int j = 0; j < ngrp; j++)
                    if (cnt[(size_t)j * ng + RA] && cnt[(size_t)j * ng + RB])
                        both[nb++] = j;
                /* a one-body operator moves at most one electron */
                if (!nb || moved(&loc, gs[RA], gs[RB]) > 1)
                    continue;
                for (int i = gs[RA]; i < gs[RA + 1] && !e; i++)
                    for (int jc = gs[RB]; jc < gs[RB + 1] && !e; jc++) {
                        int any = 0;
                        for (int y = 0; y < nb && !any; y++)
                            any = pos[(size_t)both[y] * n + i] >= 0 &&
                                  pos[(size_t)both[y] * n + jc] >= 0;
                        if (!any)
                            continue;
                        /* the operators of the same ranks together: one coupling */
                        double wx[NOP] = {0};
                        for (int c = 0; c < 3 && !e; c++) {
                            void *cx[3];
                            double wo[3];
                            int xs[3], nx = 0;
                            for (int y = 0; y < 3; y++)
                                if (same[c][y] >= 0 && op[same[c][y]].on) {
                                    xs[nx] = same[c][y];
                                    cx[nx++] = &lop[same[c][y]];
                                }
                            if (!nx)
                                continue;
                            ac_ci_one_ll_n(&loc, i, jc, op[xs[0]].ks, op[xs[0]].kl, gop_f, nx, cx, wo,
                                           &e);
                            for (int y = 0; y < nx; y++)
                                wx[xs[y]] = wo[y];
                        }
                        for (int x = 0; x < NOP && !e; x++) {
                            const double w = wx[x];
                            if (!op[x].on || w == 0.0)
                                continue;
                            for (int y = 0; y < nb; y++) {
                                const int j = both[y];
                                const int p = pos[(size_t)j * n + i], q = pos[(size_t)j * n + jc];
                                if (p < 0 || q < 0)
                                    continue;
                                double tm = redtab_term(&rt[(size_t)x * ngrp + j], p, q, w);
                                if (tm == 0.0)
                                    continue;
                                const int nl = G[j].nl;
                                const ac_lsj_kept *kl = k + G[j].l0;
                                double *sp = s + (size_t)x * smax + soff[j] +
                                             (size_t)(p - fst[(size_t)j * ng + RA]) * (size_t)nl;
                                for (int l = 0; l < nl; l++)
                                    sp[l] += kl[l].v.c[q] * tm;
                            }
                        }
                    }
            }
            if (e) {
#pragma omp critical(ac_lsj_fail)
                fe = e;
                continue;
            }
            for (int x = 0; x < NOP; x++)
                for (int j = 0; j < ngrp && op[x].on; j++) {
                    const size_t za = (size_t)j * ng + RA;
                    const int dim = G[j].dim, nl = G[j].nl;
                    const ac_lsj_kept *kl = k + G[j].l0;
                    for (int p = fst[za]; p < fst[za] + cnt[za]; p++)
                        for (int l = 0; l < nl; l++)
                            row[(size_t)x * nrow + G[j].roff + (size_t)l * dim + p] =
                                kl[l].v.c[p] * s[(size_t)x * smax + soff[j] +
                                                 (size_t)(p - fst[za]) * (size_t)nl + l];
                }
        }
        if (!s || !soff || !both) {
#pragma omp critical(ac_lsj_fail)
            fe = 1;
        }
        free(s);
        free(soff);
        free(both);
        free(kt.tab);
        free(kz.tab);
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
    if (fe) {
        *err = fe;
        rc = -1;
        goto out;
    }
    for (int x = 0; x < NOP; x++)
        for (int j = 0; j < ngrp && op[x].on; j++)
            for (int l = 0; l < G[j].nl; l++) {
                const double *r = row + (size_t)x * nrow + G[j].roff + (size_t)l * G[j].dim;
                double t = 0.0;
                for (int p = 0; p < G[j].dim; p++)
                    t += r[p];
                val[(size_t)x * nk + G[j].l0 + l] = t;
            }
    for (int x = 0; x < nk; x++) {
        memset(&g[x], 0, sizeof g[x]);
        g[x].gS = val[x];
        g[x].gL = val[(size_t)nk + x];
        g[x].rmc = val[(size_t)2 * nk + x];
        g[x].ormc = val[(size_t)3 * nk + x];
        g[x].gc = val[(size_t)4 * nk + x];
        g[x].gc += val[(size_t)5 * nk + x];
        if (ac_lsj_h6_rk)
            g[x].h6 = ALPHA2 * sqrt(15.0) / 3.0 * val[(size_t)6 * nk + x];
    }
    rc = 0;
out:
    for (int z = 0; rt && z < NOP * ngrp; z++)
        if (rt[z].f)
            redtab_free(&rt[z]);
    free(rt);
    free(G);
    free(pos);
    free(fst);
    free(cnt);
    free(row);
    free(val);
    return rc;
}

/* how many dense tables the two-electron parts of g_J ask of pc_table, and
   whether they fit under AC_PROP_CACHE_MB with what the window holds now */
static int pc_g2_tables(const ac_ci *ci)
{
    return (ci->in.rgc ? 3 : 0) + (ci->in.roz ? 1 : 0) + (ci->in.rrc ? 1 : 0);
}

static int pc_g2_fit(const ac_ci *ci)
{
    double need = (double)pc_g2_tables(ci) * (double)ci->ncsf * (double)ci->ncsf * sizeof(double);
    for (int kp = -2; kp <= 2; kp++)
        if (pc.v[pc_slot(kp)])
            need -= (double)ci->ncsf * (double)ci->ncsf * sizeof(double);
    return (double)pc.bytes + need <= (double)pc.limit;
}

/* The two-electron parts of g_J (gc2e, oz2e, rec2 of
   ac_lsj_zeeman_reduced(A, A); added to g[x]) of the levels k[0 .. nk-1] of
   all the J blocks in one pass over the pairs of configurations: the
   elements of two configurations are computed once (ac_ci_prop2_block, as
   pc_fill_blocks does; rec2 pair by pair), used for every level of every J
   block that holds CSFs of both, and dropped. So no n x n table is needed:
   the memory is that of one pair of configurations and of the sums, one
   number per level and CSF of its block. The levels of one J block follow
   each other in k and have the same basis. The sums are formed as in
   pair_reduced_2e (q ascending in every row, then the rows in order), so
   the numbers are the same.
   Returns 0, 1 when it cannot be done this way (g untouched; the caller goes
   level by level), -1 on an engine error (*err). */
static int zeeman_2e_all(ac_ci *ci, const ac_lsj_kept *k, int nk, ac_lsj_g *g, int *err)
{
    typedef struct {
        int tJ, dim, l0, nl;
        const int *idx;
        size_t roff;
    } grp;
    static const int kps[5] = {0, 1, 2, -1, -2};
    const int on[5] = {ci->in.rgc != NULL, ci->in.rgc != NULL, ci->in.rgc != NULL,
                       ci->in.roz != NULL, ci->in.rrc != NULL};
    const int pairwise = getenv("AC_BP_PAIRWISE") != NULL;
    *err = 0;
    if (!pc_g2_tables(ci) || nk <= 0)
        return 0;
    if (!pc_runs(ci))
        return 1;
    const int ng = pc.ng, *gs = pc.gs, *run = pc.run;
    int ngrp = 0, rc = 1, fe = 0;
    for (int x = 0; x < nk; x++)
        ngrp += x == 0 || k[x].v.tJ != k[x - 1].v.tJ;
    grp *G = calloc((size_t)ngrp, sizeof *G);
    /* the places of configuration r in the basis of group j: fst[j * ng + r]
       .. + cnt[j * ng + r] - 1 */
    int *fst = malloc((size_t)ngrp * (size_t)ng * sizeof *fst);
    int *cnt = calloc((size_t)ngrp * (size_t)ng, sizeof *cnt);
    redtab *rt = calloc((size_t)5 * (size_t)ngrp, sizeof *rt);
    double *val = calloc((size_t)5 * (size_t)nk, sizeof *val), *row = NULL;
    size_t nrow = 0, smax = 0;
    if (!G || !fst || !cnt || !rt || !val)
        goto out;
    ngrp = 0;
    for (int x = 0; x < nk; x++) {
        if (x == 0 || k[x].v.tJ != k[x - 1].v.tJ) {
            G[ngrp++] = (grp){k[x].v.tJ, k[x].v.dim, x, 0, k[x].v.idx, 0};
        } else if (k[x].v.dim != k[x - 1].v.dim ||
                   memcmp(k[x].v.idx, k[x - 1].v.idx, (size_t)k[x].v.dim * sizeof(int)))
            goto out;
        G[ngrp - 1].nl++;
    }
    for (int j = 0; j < ngrp; j++) {
        const int *idx = G[j].idx;
        for (int p = 0; p < G[j].dim; p++) {
            if (p && idx[p - 1] >= idx[p])
                goto out;
            size_t z = (size_t)j * (size_t)ng + (size_t)run[idx[p]];
            if (!cnt[z]++)
                fst[z] = p;
        }
        G[j].roff = nrow;
        nrow += (size_t)G[j].nl * (size_t)G[j].dim;
    }
    for (int r = 0; r < ng; r++) {
        size_t t = 0;
        for (int j = 0; j < ngrp; j++)
            t += (size_t)cnt[(size_t)j * ng + r] * (size_t)G[j].nl;
        if (t > smax)
            smax = t;
    }
    row = malloc((size_t)5 * (nrow ? nrow : 1) * sizeof *row);
    if (!row)
        goto out;
    for (int x = 0; x < 5; x++)
        for (int j = 0; j < ngrp && on[x]; j++) {
            ac_lsj_level A = {G[j].tJ, G[j].dim, G[j].idx, k[G[j].l0].v.c};
            redtab_make(&rt[(size_t)x * ngrp + j], ci, &A, &A, kps[x] < 0 ? 0 : 2,
                        kps[x] < 0 ? 2 : 2 * kps[x], 2);
            if (!rt[(size_t)x * ngrp + j].f)
                goto out;
        }
    long ops = 0;
#pragma omp parallel reduction(+ : ops)
    {
        ac_ci loc = *ci;
        loc.thc = NULL;
        loc.ops_evaluated = 0;
        /* s[x * smax + soff[j] + (p - first place) * nl + l]: the sum of row p
           of group j for level l and operator x */
        double *s = malloc((size_t)5 * (smax ? smax : 1) * sizeof *s), *buf = NULL;
        size_t *soff = malloc((size_t)ngrp * sizeof *soff), bufn = 0;
#pragma omp for schedule(dynamic, 1)
        for (int RA = 0; RA < ng; RA++) {
            if (!s || !soff || fe)
                continue;
            size_t t = 0;
            int e = 0;
            for (int j = 0; j < ngrp; j++) {
                soff[j] = t;
                t += (size_t)cnt[(size_t)j * ng + RA] * (size_t)G[j].nl;
            }
            if (!t)
                continue;
            for (int x = 0; x < 5; x++)
                if (on[x])
                    memset(s + (size_t)x * smax, 0, t * sizeof *s);
            const int i0 = gs[RA], ni = gs[RA + 1] - i0;
            for (int RB = 0; RB < ng && !e; RB++) {
                int both = 0;
                for (int j = 0; j < ngrp && !both; j++)
                    both = cnt[(size_t)j * ng + RA] && cnt[(size_t)j * ng + RB];
                /* a two-body operator moves at most two electrons */
                if (!both || moved(&loc, i0, gs[RB]) > 2)
                    continue;
                const int j0 = gs[RB], nj = gs[RB + 1] - j0;
                for (int x = 0; x < 5 && !e; x++) {
                    if (!on[x])
                        continue;
                    const int kp = kps[x], blk = kp >= -1 && !pairwise;
                    if (blk) {
                        size_t nn = (size_t)ni * (size_t)nj;
                        if (nn > bufn) {
                            free(buf);
                            bufn = nn;
                            buf = malloc(bufn * sizeof *buf);
                        }
                        if (!buf) {
                            bufn = 0;
                            e = 1;
                            break;
                        }
                        ac_ci_prop2_block(&loc, kp, i0, ni, j0, nj, buf, &e);
                        if (e)
                            break;
                    }
                    for (int j = 0; j < ngrp && !e; j++) {
                        const size_t za = (size_t)j * ng + RA, zb = (size_t)j * ng + RB;
                        if (!cnt[za] || !cnt[zb])
                            continue;
                        const int *idx = G[j].idx, dim = G[j].dim, nl = G[j].nl;
                        const redtab *r = &rt[(size_t)x * ngrp + j];
                        const ac_lsj_kept *kl = k + G[j].l0;
                        for (int p = fst[za]; p < fst[za] + cnt[za] && !e; p++) {
                            double *sp = s + (size_t)x * smax + soff[j] +
                                         (size_t)(p - fst[za]) * (size_t)nl;
                            for (int q = fst[zb]; q < fst[zb] + cnt[zb]; q++) {
                                double w = blk ? buf[(size_t)(idx[p] - i0) * (size_t)nj +
                                                     (size_t)(idx[q] - j0)]
                                           : kp == -2 ? ac_ci_rec2(&loc, idx[p], idx[q], &e)
                                           : kp < 0   ? ac_ci_oz2e(&loc, idx[p], idx[q], &e)
                                                      : ac_ci_gc2e(&loc, idx[p], idx[q], kp, &e);
                                if (e)
                                    break;
                                double tm = redtab_term(r, p, q, w);
                                if (tm != 0.0)
                                    for (int l = 0; l < nl; l++)
                                        sp[l] += kl[l].v.c[q] * tm;
                            }
                        }
                        (void)dim;
                    }
                }
            }
            if (e) {
#pragma omp critical(ac_lsj_fail)
                fe = e;
                continue;
            }
            for (int x = 0; x < 5; x++)
                for (int j = 0; j < ngrp && on[x]; j++) {
                    const size_t za = (size_t)j * ng + RA;
                    const int dim = G[j].dim, nl = G[j].nl;
                    const ac_lsj_kept *kl = k + G[j].l0;
                    for (int p = fst[za]; p < fst[za] + cnt[za]; p++)
                        for (int l = 0; l < nl; l++)
                            row[(size_t)x * nrow + G[j].roff + (size_t)l * dim + p] =
                                kl[l].v.c[p] * s[(size_t)x * smax + soff[j] +
                                                 (size_t)(p - fst[za]) * (size_t)nl + l];
                }
        }
        if (!s || !soff) {
#pragma omp critical(ac_lsj_fail)
            fe = 1;
        }
        free(s);
        free(soff);
        free(buf);
        ops += loc.ops_evaluated;
        ac_ci_free_cache(&loc);
    }
    ci->ops_evaluated += ops;
    if (fe) {
        *err = fe;
        rc = -1;
        goto out;
    }
    for (int x = 0; x < 5; x++)
        for (int j = 0; j < ngrp && on[x]; j++)
            for (int l = 0; l < G[j].nl; l++) {
                const double *r = row + (size_t)x * nrow + G[j].roff + (size_t)l * G[j].dim;
                double t = 0.0;
                for (int p = 0; p < G[j].dim; p++)
                    t += r[p];
                val[(size_t)x * nk + G[j].l0 + l] = t;
            }
    for (int x = 0; x < nk; x++) {
        for (int y = 0; y < 3; y++)
            if (on[y])
                g[x].gc2e += val[(size_t)y * nk + x];
        if (on[3])
            g[x].oz2e = val[(size_t)3 * nk + x];
        if (on[4])
            g[x].rec2 = val[(size_t)4 * nk + x];
    }
    rc = 0;
out:
    for (int z = 0; rt && z < 5 * ngrp; z++)
        if (rt[z].f)
            redtab_free(&rt[z]);
    free(rt);
    free(G);
    free(fst);
    free(cnt);
    free(row);
    free(val);
    return rc;
}

double (*ac_lsj_hfs_rm3)(void *, int, int) = NULL;
double (*ac_lsj_hfs_r00)(void *, int, int) = NULL;

/* ---- level listing ---------------------------------------------------- */

typedef struct {
    int tJ;
    double e;
} level;

int ac_lsj_zoff = 0;
double (*ac_lsj_h6_rk)(void *, int, int, int) = NULL;
double (*ac_lsj_is_rk)(void *, int, int, int) = NULL;

/* the printed levels, kept for the table of <A||mu||B> */
typedef ac_lsj_kept kept;

void ac_lsj_kept_free(ac_lsj_kept *k, int n)
{
    for (int i = 0; i < n; i++) {
        free((void *)k[i].v.idx);
        free((void *)k[i].v.c);
    }
    free(k);
}

/* The pairs of the tables: within one list (cross = 0) each pair once, B
   after A (from A itself when diag); between two lists every pair. */
static int pair_first(int cross, int diag, int a)
{
    return cross ? 0 : diag ? a : a + 1;
}

static int keep_level(kept **k, int *nk, int tJ, int num, double e, int dim, const int *idx,
                      const double *c)
{
    kept *t = realloc(*k, (size_t)(*nk + 1) * sizeof *t);
    if (!t)
        return -1;
    *k = t;
    int *ix = malloc((size_t)dim * sizeof *ix);
    double *cc = malloc((size_t)dim * sizeof *cc);
    if (!ix || !cc) {
        free(ix);
        free(cc);
        return -1;
    }
    memcpy(ix, idx, (size_t)dim * sizeof *ix);
    memcpy(cc, c, (size_t)dim * sizeof *cc);
    t[*nk].v = (ac_lsj_level){tJ, dim, ix, cc};
    t[*nk].num = num;
    t[*nk].e = e;
    (*nk)++;
    return 0;
}

static int print_zoff(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                      int cross, double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                      void *ctx, char *err, int errlen)
{
    fprintf(out, "Zeeman reduced matrix elements <A||mu||B> between different levels "
                 "(Edmonds; mu = L + g_s S + alpha^2 terms, in mu_B):\n");
    fprintf(out, "%4s %3s %4s %3s %16s %16s %16s %14s %14s %14s%s%s%s%s\n", "J_A", "#A", "J_B",
            "#B", "<L>", "<S>", "<L + g_s S>", "RMC", "GC", "ORMC",
            ci->in.rgc ? "           GC2e" : "", ci->in.roz ? "           OZ2e" : "",
            ci->in.rrc ? "           rec2" : "", ac_lsj_h6_rk ? "             H6" : "");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 0, a); b < nb; b++) {
            const ac_lsj_level *A = &k[a].v, *B = &kb[b].v;
            if (abs(A->tJ - B->tJ) > 2 || A->tJ + B->tJ < 2)
                continue;
            ac_lsj_g r;
            int e;
            if (ac_lsj_zeeman_reduced(ci, A, B, tkin, zr, ctx, &r, &e))
                return fail(err, errlen, "Zeeman <A||mu||B>: engine error %d", e);
            fprintf(out, "%2d/2 %3d %2d/2 %3d %16.12f %16.12f %16.12f %14.6e %14.6e %14.6e",
                    A->tJ, k[a].num, B->tJ, kb[b].num, r.gL, r.gS, r.gL + AC_GS * r.gS, r.rmc,
                    r.gc, r.ormc);
            if (ci->in.rgc)
                fprintf(out, " %15.8e", r.gc2e);
            if (ci->in.roz)
                fprintf(out, " %15.8e", r.oz2e);
            if (ci->in.rrc)
                fprintf(out, " %15.8e", r.rec2);
            if (ac_lsj_h6_rk)
                fprintf(out, " %15.8e", r.h6);
            fprintf(out, "\n");
        }
    return 0;
}

/* A_J and B_J from the reduced elements (Lindgren-Rosen (1974); technical
   note, Sec. 5.2) */
static int print_hfs(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                     int cross, void *ctx, char *err, int errlen)
{
    fprintf(out, "Hyperfine reduced matrix elements <A||T^(K)||B> (Edmonds, a.u.; T^(1) = "
                 "orb + sd + con, T^(2) = quad); A_J per mu_I/I and B_J per Q:\n");
    fprintf(out, "%4s %3s %4s %3s %16s %16s %16s %16s %16s %14s %14s\n", "J_A", "#A", "J_B", "#B",
            "orb", "sd", "con", "T^(1)", "quad", "A_J(MHz)", "B_J(MHz/b)");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 1, a); b < nb; b++) {
            const ac_lsj_level *A = &k[a].v, *B = &kb[b].v;
            if (abs(A->tJ - B->tJ) > 4 || A->tJ + B->tJ < 2)
                continue;
            ac_lsj_hfs h;
            int e;
            if (ac_lsj_hfs_reduced(ci, A, B, ac_lsj_hfs_rm3, ac_lsj_hfs_r00, ctx, &h, &e))
                return fail(err, errlen, "hyperfine <A||T||B>: engine error %d", e);
            double t1 = h.orb + h.sd + h.con;
            fprintf(out, "%2d/2 %3d %2d/2 %3d %16.9e %16.9e %16.9e %16.9e %16.9e", A->tJ,
                    k[a].num, B->tJ, kb[b].num, h.orb, h.sd, h.con, t1, h.quad);
            if (!cross && a == b) {
                double J = A->tJ / 2.0;
                double aj = AC_HFS_A0 * t1 / sqrt(J * (J + 1) * (2 * J + 1));
                double bj = -2.0 * AC_HFS_B0 * h.quad *
                            sqrt(J * (2 * J - 1) / ((J + 1) * (2 * J + 1) * (2 * J + 3)));
                fprintf(out, " %14.6f %14.6f", aj, bj);
            }
            fprintf(out, "\n");
        }
    return 0;
}

/* mass and field shift (technical note, Sec. 5.5; the field shift is the
   non-relativistic limit of Shabaev, Phys. Rep. 356, 119 (2002), Eqs. (322),
   (324), for a hydrogen-like ns state) */
static int print_is(FILE *out, ac_ci *ci, const kept *k, int nk, void *ctx, char *err,
                    int errlen)
{
    fprintf(out, "Isotope shift (a.u.): K_NMS = <sum p_i^2/2>, K_SMS = <sum_{i<j} p_i.p_j>,"
                 " mass shift (m/M)(K_NMS + K_SMS); rho(0) = <sum delta(r_i)>,"
                 " field shift (2 pi/3) Z rho(0) delta<r^2>:\n");
    fprintf(out, "%4s %3s %22s %22s %22s %22s\n", "J", "#", "K_NMS", "K_SMS", "K_NMS + K_SMS",
            "rho(0)");
    for (int a = 0; a < nk; a++) {
        ac_lsj_is r;
        int e;
        if (ac_lsj_is_level(ci, &k[a].v, ac_lsj_is_rk, ctx, &r, &e))
            return fail(err, errlen, "isotope shift: engine error %d", e);
        fprintf(out, "%2d/2 %3d %22.15f %22.15f %22.15f %22.12f\n", k[a].v.tJ, k[a].num, r.nms,
                r.sms, r.nms + r.sms, r.rho0);
    }
    return 0;
}

double (*ac_lsj_trans_rk)(void *, int, int, int) = NULL;

#define TRLAB "%2d/2 %3d %2d/2 %3d %4s %12.3f"

int ac_lsj_digits = 0;
int ac_lsj_davidson = 500;
/* width and precision of the line strengths */
#define SW (ac_lsj_digits ? 24 : 16)
#define SP (ac_lsj_digits ? 17 : 9)

/* f and A from S (Sakurai (1967), ch. 2; Messiah (1962), ch. XXI;
   technical note, Sec. 5.3) */
static int print_trans(FILE *out, ac_ci *ci, const kept *k, int nk, const kept *kb, int nb,
                       int cross, void *ctx, char *err, int errlen)
{
    const double tau0 = 2.4188843265857e-17, a3 = ac_alpha * ac_alpha * ac_alpha;
    fprintf(out, "Transitions between printed levels (lower -> upper; S = |<||T||>|^2 in a.u.,"
                 " A in s^-1, E1 f of absorption%s):\n",
            cross ? "; last column: the state of the lower level" : "");
    fprintf(out, "%4s %3s %4s %3s %4s %12s %16s %16s %13s %13s %13s%s\n", "J_lo", "#", "J_up", "#",
            "type", "dE(cm-1)", "S_length", "S_velocity", "f_length", "f_velocity", "A",
            cross ? " lo" : "");
    for (int a = 0; a < nk; a++)
        for (int b = pair_first(cross, 0, a); b < nb; b++) {
            const kept *lo = &k[a], *up = &kb[b];
            int swap = up->e < lo->e;
            if (swap) {
                const kept *x = lo;
                lo = up;
                up = x;
            }
            if (abs(lo->v.tJ - up->v.tJ) > 6 || lo->v.tJ + up->v.tJ < 2)
                continue;
            /* within one list the lower level is the bra; between two lists
               the level of the first (S = |<||T||>|^2 either way) */
            const kept *bra = cross ? &k[a] : lo, *ket = cross ? &kb[b] : up;
            ac_lsj_trans t;
            int e;
            if (ac_lsj_trans_reduced(ci, &bra->v, &ket->v, ac_lsj_trans_rk, ctx, &t, &e))
                return fail(err, errlen, "transition <A||T||B>: engine error %d", e);
            double w = up->e - lo->e, gl = lo->v.tJ + 1.0, gu = up->v.tJ + 1.0;
            double cm = w * AC_HARTREE_CM;
            const char *tag = !cross ? "" : swap ? "  2" : "  1";
            if (fabs(t.e1l) > 1e-12 || fabs(t.e1v) > 1e-12) {
                double sl = t.e1l * t.e1l, sv = t.e1v * t.e1v;
                fprintf(out, TRLAB " %*.*e %*.*e %13.6e %13.6e %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E1", cm, SW, SP, sl, SW, SP, sv,
                        2.0 / 3.0 * w * sl / gl,
                        2.0 / 3.0 * sv / (w * gl), 4.0 / 3.0 * a3 * w * w * w * sl / gu / tau0,
                        tag);
            }
            if (fabs(t.m1) > 1e-12) {
                double s = t.m1 * t.m1;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M1", cm, SW, SP, s, "", "", "",
                        a3 * ac_alpha * ac_alpha * w * w * w * s / 3.0 / gu / tau0, tag);
            }
            /* M1 with the relativistic part (Pachucki's Eqs. (60)-(62); the
               r_ab terms when ci->in.rm1 is set); the column after S is
               <||delta mu||> itself */
            double dm = ac_lsj_m1_rel(&t, w);
            if (fabs(dm) > 1e-14) {
                double s = (t.m1 + dm) * (t.m1 + dm);
                fprintf(out, TRLAB " %*.*e %*.*e %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M1r", cm, SW, SP, s, SW, SP, dm, "", "",
                        a3 * ac_alpha * ac_alpha * w * w * w * s / 3.0 / gu / tau0, tag);
            }
            if (fabs(t.e2) > 1e-12) {
                double s = t.e2 * t.e2;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E2", cm, SW, SP, s, "", "", "",
                        a3 * ac_alpha * ac_alpha * pow(w, 5) * s / 15.0 / gu / tau0, tag);
            }
            if (fabs(t.m2) > 1e-12) {
                double s = t.m2 * t.m2;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M2", cm, SW, SP, s, "", "", "",
                        pow(ac_alpha, 7) * pow(w, 5) * s / 60.0 / gu / tau0, tag);
            }
            if (fabs(t.e3) > 1e-12) {
                double s = t.e3 * t.e3;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "E3", cm, SW, SP, s, "", "", "",
                        8.0 / 4725.0 * pow(ac_alpha, 7) * pow(w, 7) * s / gu / tau0, tag);
            }
            if (fabs(t.m3) > 1e-12) {
                double s = t.m3 * t.m3;
                fprintf(out, TRLAB " %*.*e %16s %13s %13s %13.6e%s\n", lo->v.tJ, lo->num,
                        up->v.tJ, up->num, "M3", cm, SW, SP, s, "", "", "",
                        2.0 / 4725.0 * pow(ac_alpha, 9) * pow(w, 7) * s / gu / tau0, tag);
            }
        }
    return 0;
}

int ac_lsj_print_cross(FILE *out, ac_ci *ci, const ac_lsj_kept *ka, int na,
                       const ac_lsj_kept *kb, int nb, double (*tkin)(void *, int, int),
                       double (*zr)(void *, int, int), void *ctx, char *err, int errlen)
{
    int rc = -1;
    pc_begin(ci);
    if (ac_lsj_zoff && print_zoff(out, ci, ka, na, kb, nb, 1, tkin, zr, ctx, err, errlen))
        goto out;
    if (ac_lsj_hfs_rm3 && print_hfs(out, ci, ka, na, kb, nb, 1, ctx, err, errlen))
        goto out;
    if (ac_lsj_trans_rk && print_trans(out, ci, ka, na, kb, nb, 1, ctx, err, errlen))
        goto out;
    rc = 0;
out:
    pc_end();
    return rc;
}

static int by_energy(const void *a, const void *b)
{
    double d = ((const level *)a)->e - ((const level *)b)->e;
    return (d > 0) - (d < 0);
}

/* g_J of one level whose one-body parts are in g already (zeeman_1e_levels):
   the two-electron parts and the factor of ac_lsj_gfactor */
static int gfactor_rest(ac_ci *ci, int dim, const int *idx, int tJ, const double *c, int have2e,
                        ac_lsj_g *g, int *err)
{
    double J = tJ / 2.0, norm = 1.0 / sqrt(J * (J + 1) * (2 * J + 1));
    ac_lsj_level a = {tJ, dim, idx, c};
    *err = 0;
    /* have2e: the two-electron parts are in g too (zeeman_2e_levels) */
    for (int kp = 0; kp <= 2 && !*err && ci->in.rgc && !have2e; kp++)
        g->gc2e += pair_reduced_gc2e(ci, &a, &a, kp, err);
    if (!*err && ci->in.roz && !have2e)
        g->oz2e = pair_reduced_gc2e(ci, &a, &a, -1, err);
    if (!*err && ci->in.rrc && !have2e)
        g->rec2 = pair_reduced_gc2e(ci, &a, &a, -2, err);
    if (*err)
        return -1;
    g->gL *= norm;
    g->gS *= norm;
    g->rmc *= norm;
    g->ormc *= norm;
    g->gc *= norm;
    g->gc2e *= norm;
    g->oz2e *= norm;
    g->rec2 *= norm;
    g->h6 *= norm;
    return 0;
}

/* the line of one level: with its g factors when gg is given */
static void level_line(FILE *out, const ac_ci *ci, int tJ, int num, double e, const char *term,
                       double best, const ac_lsj_g *gg)
{
    fprintf(out, "%2d/2 %3d %22.15f %-6s %6.4f", tJ, num, e, term, best);
    if (gg) {
        fprintf(out, " %16.13f %14.6e %14.6e %14.6e", gg->gL + AC_GS * gg->gS, gg->rmc, gg->gc,
                gg->ormc);
        if (ci->in.rgc)
            fprintf(out, " %17.10e", gg->gc2e);
        if (ci->in.roz)
            fprintf(out, " %17.10e", gg->oz2e);
        if (ci->in.rrc)
            fprintf(out, " %17.10e", gg->rec2);
        if (ac_lsj_h6_rk)
            fprintf(out, " %17.10e", gg->h6);
    }
    fprintf(out, "\n");
    fflush(out); /* a long run shows its levels as they come */
}

/* a level whose line waits for the two-electron parts of g (zeeman_2e_all) */
typedef struct {
    int tJ, num, kx; /* kx: its place in the kept levels, -1 when g is final or absent */
    int hasg;
    double e, best;
    char term[32];
    ac_lsj_g g;
} pending;

double ac_lsj_tol = 1e-11;
const int *ac_lsj_nlev_j = NULL;
int ac_lsj_nlev_j_len = 0;
const int *ac_lsj_nlev_sl = NULL;
int ac_lsj_nlev_sl_n = 0;
const char *ac_lsj_dump = NULL;

static char *dump_json = NULL;
static size_t dump_json_len = 0;

const char *ac_lsj_dump_blocks(void)
{
    return dump_json ? dump_json : "";
}

/* the files of one block (ac_lsj.h); tag "J<2J>" or "S<2S+1>L<L>", head the
   first members of its JSON object */
static int dump_block(const char *tag, const char *head, int dim, const int *idx, int nd,
                      const double *vec, const double *e, const double *res, int it, char *err,
                      int errlen)
{
    static const char *const part[4] = {"idx", "vec", "e", "res"};
    size_t plen = strlen(ac_lsj_dump) + strlen(tag) + 16;
    char *path = malloc(plen);
    if (!path)
        return fail(err, errlen, "out of memory");
    for (int k = 0; k < 4; k++) {
        size_t sh[2] = {k == 0 ? (size_t)dim : (size_t)nd, (size_t)dim};
        const void *data = k == 0 ? (const void *)idx : k == 1 ? (const void *)vec
                           : k == 2 ? (const void *)e : (const void *)res;
        snprintf(path, plen, "%s.%s.%s.npy", ac_lsj_dump, tag, part[k]);
        if (ac_npy_write(path, k == 0 ? 'i' : 'f', k == 1 ? 2 : 1, sh, data)) {
            fail(err, errlen, "cannot write %s", path);
            free(path);
            return -1;
        }
    }
    free(path);
    double rmax = 0.0;
    for (int p = 0; p < nd; p++)
        rmax = fmax(rmax, res[p]);
    char item[512];
    int len = snprintf(item, sizeof item,
                       "%s    {\"tag\": \"%s\", %s, \"dim\": %d, \"nroot\": %d, \"solver\": \"%s\", "
                       "\"iterations\": %d, \"max_residual\": %.3e}",
                       dump_json_len ? ",\n" : "", tag, head, dim, nd, it ? "davidson" : "dsyev", it,
                       rmax);
    char *nj = realloc(dump_json, dump_json_len + (size_t)len + 1);
    if (!nj)
        return fail(err, errlen, "out of memory");
    dump_json = nj;
    memcpy(dump_json + dump_json_len, item, (size_t)len + 1);
    dump_json_len += (size_t)len;
    return 0;
}

/* res[p] = |A x_p - w[p] x_p| for the nd columns x_p of X; A symmetric,
   dim x dim */
static void block_res(int dim, const double *A, int nd, const double *X, const double *w,
                      double *res)
{
    for (int p = 0; p < nd; p++) {
        const double *x = X + (size_t)p * dim;
        double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (int r = 0; r < dim; r++) {
            const double *a = A + (size_t)r * dim;
            double t = -w[p] * x[r];
            for (int q = 0; q < dim; q++)
                t += a[q] * x[q];
            s += t * t;
        }
        res[p] = sqrt(s);
    }
}

/* A check of the levels Davidson returns. Take some CSFs of the block and
   the matrix B of H among them (a principal submatrix), with eigenvalues
   mu_1 <= mu_2 <= ... The n-th level of the block is the least, over all
   spaces of n dimensions, of the largest expectation value of H in the
   space (the min-max principle of Courant and Fischer); the space of the n
   lowest eigenvectors of B, zero on the other CSFs, has the largest value
   mu_n. So lambda_n <= mu_n (Cauchy's interlacing), whether H couples the
   chosen CSFs to the others or not. If B has nd eigenvalues below the nd-th
   level returned, that level is not the nd-th lowest: a level was missed.
   This proves a miss; it cannot prove that there is none.
   The CSFs chosen: of each part of the block, that is each total (S, L) and
   parity, those of lowest diagonal (about 1500 CSFs in all, or
   AC_DAV_CHECK_CSFS of the environment; at most 200 and at least 25 or nd
   of a part), so that a part the start vectors left out is seen. On the
   boron list of tests/s2m4/dump.sh 800 CSFs find all the misses, 400 half
   of them. With one part B would be the start block of Davidson itself, and
   nothing is checked. The eigenvalues are counted (ac_ci_count_below), not
   computed.
   w[nd] are the levels and elem(ctx, a, b) the elements, both of H - shift.
   Returns 0, or -1 with the message in err (a missed level, or no memory).
   AC_DAV_NOCHECK in the environment skips the check. */
typedef struct {
    double d;
    int p;
} diagpos;

static int by_diagpos(const void *a, const void *b)
{
    const diagpos *x = a, *y = b;
    return x->d < y->d ? -1 : x->d > y->d ? 1 : x->p - y->p;
}

typedef struct {
    const double *H;
    int dim;
} dense_el;

static double dense_elem(void *ctx, int a, int b)
{
    const dense_el *d = ctx;
    return d->H[(size_t)a * d->dim + b];
}

static int miss_check(const ac_ci *ci, int tJ, int dim, const int *idx, int nd, const double *w,
                      double shift, double (*elem)(void *, int, int), void *ctx, char *err,
                      int errlen)
{
    if (getenv("AC_DAV_NOCHECK") || nd < 1)
        return 0;
    double t0 = omp_get_wtime();
    ac_lab *labs = NULL;
    int nlab = 0, rc = 0, used = 0, ns = 0;
    int *part = label_numbers(ci, dim, idx, &labs, &nlab);
    if (!part)
        return fail(err, errlen, "out of memory");
    free(labs);
    diagpos *dp = malloc((size_t)dim * sizeof *dp);
    int *cnt = calloc((size_t)2 * nlab, sizeof *cnt), *sel = malloc((size_t)dim * sizeof *sel);
    double *B = NULL;
    if (!dp || !cnt || !sel) {
        rc = fail(err, errlen, "out of memory");
        goto out;
    }
    for (int p = 0; p < dim; p++) {
        const ac_ci_csf *x = &ci->csf[idx[p]];
        int par = 0;
        for (int k = 1; k <= x->c.nsh; k++)
            par += x->c.nel[k] * x->c.l[k];
        part[p] = 2 * part[p] + par % 2;
        used += cnt[part[p]]++ == 0;
        dp[p].d = elem(ctx, p, p);
        dp[p].p = p;
    }
    if (used < 2)
        goto out;
    const char *sc = getenv("AC_DAV_CHECK_CSFS");
    int per = (sc && atoi(sc) > 0 ? atoi(sc) : 1500) / used;
    per = per > 200 ? 200 : per < 25 ? 25 : per;
    if (per < nd)
        per = nd;
    qsort(dp, (size_t)dim, sizeof *dp, by_diagpos);
    memset(cnt, 0, (size_t)2 * nlab * sizeof *cnt);
    for (int q = 0; q < dim; q++)
        if (cnt[part[dp[q].p]] < per) {
            cnt[part[dp[q].p]]++;
            sel[ns++] = dp[q].p;
        }
    if (ns < nd)
        goto out;
    B = malloc((size_t)ns * ns * sizeof *B);
    if (!B) {
        rc = fail(err, errlen, "out of memory");
        goto out;
    }
#pragma omp parallel for schedule(dynamic, 8)
    for (int a = 0; a < ns; a++)
        for (int b = 0; b <= a; b++)
            B[(size_t)b * ns + a] = B[(size_t)a * ns + b] = elem(ctx, sel[a], sel[b]);
    /* how many eigenvalues of B lie below the last level returned (by more
       than 1e-8 hartree: the level itself is converged far better) */
    int below = 0;
    if (ac_ci_count_below(ns, B, w[nd - 1] - 1e-8, &below)) {
        rc = fail(err, errlen, "dsytrf failed");
        goto out;
    }
    if (getenv("AC_TIME"))
        fprintf(stderr,
                "J = %d/2, dim %d: check for missed levels on %d CSFs of %d parts, %.2f s; %d of "
                "their levels below level %d returned (%.10f)\n",
                tJ, dim, ns, used, omp_get_wtime() - t0, below, nd, w[nd - 1] + shift);
    if (below >= nd) {
        rc = fail(err, errlen,
                  "Davidson missed a level in the J = %d/2 block (dimension %d): its level %d is "
                  "returned at %.10f hartree, but %d levels lie below that already among %d CSFs "
                  "(those of lowest diagonal of each (S, L) and parity). Ask for more levels, give "
                  "the two parities in separate lists, or solve by dsyev (dav=0)",
                  tJ, dim, nd, w[nd - 1] + shift, below, ns);
    }
out:
    free(part);
    free(dp);
    free(cnt);
    free(sel);
    free(B);
    return rc;
}

/* The lowest nlev levels of the block 2J = tJ (see in_block): the basis in
   idx[*dimp], the energies in w, the eigenvectors in the first columns of
   *Hp. A dense m takes *Hp with room for n^2 numbers; for a sparse m *Hp is
   freed and allocated anew. *dimp = 0: there is no such block. *itp: the
   Davidson iterations (0 with dsyev). res, when not NULL, receives
   |H c - E c| of each level. Returns 0, or -1 with the message in err. */
static int solve_block(const ac_lsj *m, const ac_ci *ci, int tJ, int nlev, int *idx, double **Hp,
                       double *w, double *res, int *dimp, int *itp, char *err, int errlen)
{
    double asym = 0.0, t1 = omp_get_wtime();
    int n = ci->ncsf, dim = 0, it = 0;
    double *H = *Hp;
    *dimp = *itp = 0;
    if (m->sparse) {
        for (int i = 0; i < n; i++)
            dim += in_block(ci, i, tJ);
        if (dim == 0)
            return 0;
        int nd = nlev < dim ? nlev : dim;
        free(H);
        *Hp = H = NULL;
        if (ac_lsj_davidson > 0 && dim >= ac_lsj_davidson && 4 * nd < dim) {
            /* as below, on the block as an operator (not stored) */
            jop J;
            if (jop_begin(&J, m, ci, tJ, idx) < 0)
                return fail(err, errlen, "out of memory");
            J.part = NULL, J.cur = 0;
            if (getenv("AC_NNZ"))
                fprintf(stderr, "nnz J = %d/2: dim %d, above the diagonal %zu from h, %zu "
                                "from w and w2, sparse\n",
                        tJ, dim, J.hst[dim], J.wst[dim]);
            double t2 = omp_get_wtime();
            double shift = J.shift;
            *Hp = H = malloc((size_t)dim * nd * sizeof *H);
            /* each part that H does not couple to the rest alone (a block of
               one part: as before) */
            int npart = 0, *part = H ? jop_parts(&J, &npart) : NULL;
            if (part && getenv("AC_TIME") && npart > 1)
                fprintf(stderr, "J = %d/2, dim %d: %d parts that H does not couple, each solved "
                                "alone (%.2f s to find them)\n",
                        tJ, dim, npart, omp_get_wtime() - t2);
            int e = part ? ac_ci_davidson_op_parts(dim, jop_mul, jop_elem, jop_sel, &J, npart, part,
                                                   nd, ac_lsj_tol, w, H, &it)
                         : -1;
            int oom = !H || !part;
            free(part);
            if (H && !e && res) {
                /* on H - shift, as the iteration */
                double *av = malloc((size_t)dim * nd * sizeof *av);
                oom = !av;
                if (av) {
                    jop_mul(&J, nd, H, av);
                    for (int p = 0; p < nd; p++) {
                        double s = 0.0;
                        for (int q = 0; q < dim; q++) {
                            double t = av[(size_t)p * dim + q] - w[p] * H[(size_t)p * dim + q];
                            s += t * t;
                        }
                        res[p] = sqrt(s);
                    }
                }
                free(av);
            }
            int missed = H && !e && !oom &&
                         miss_check(ci, tJ, dim, idx, nd, w, shift, jop_elem, &J, err, errlen);
            jop_end(&J);
            if (missed)
                return -1;
            if (oom)
                return fail(err, errlen, "out of memory");
            for (int q = 0; q < nd; q++)
                w[q] += shift;
            if (e)
                return fail(err, errlen, "Davidson failed (%d) in the J = %d/2 block", e, tJ);
            if (getenv("AC_TIME"))
                fprintf(stderr,
                        "J = %d/2, dim %d: matrix %.2f s, eigenvectors %.2f s, Davidson %d "
                        "iterations\n",
                        tJ, dim, t2 - t1, omp_get_wtime() - t2, it);
            *dimp = dim;
            *itp = it;
            return 0;
        }
        *Hp = H = malloc((size_t)dim * dim * sizeof *H);
        if (!H || ac_lsj_matrix(m, ci, tJ, idx, H, &asym) < 0)
            return fail(err, errlen, "out of memory");
    } else
        dim = ac_lsj_matrix(m, ci, tJ, idx, H, &asym);
    if (dim == 0)
        return 0;
    if (asym > 1e-10)
        fprintf(stderr, "warning: J block %d/2 asymmetric by %.1e\n", tJ, asym);
    if (getenv("AC_NNZ")) {
        long c0 = 0, c14 = 0, c10 = 0;
        double tot = 0.5 * dim * ((double)dim + 1.0);
#pragma omp parallel for reduction(+ : c0, c14, c10) schedule(static)
        for (int p = 0; p < dim; p++)
            for (int q = p; q < dim; q++) {
                double v = fabs(H[(size_t)p * dim + q]);
                c0 += v > 0.0, c14 += v > 1e-14, c10 += v > 1e-10;
            }
        fprintf(stderr,
                "nnz J = %d/2: dim %d, nonzero %ld (%.5f), > 1e-14 %ld (%.5f), > 1e-10 %ld "
                "(%.5f)\n",
                tJ, dim, c0, c0 / tot, c14, c14 / tot, c10, c10 / tot);
    }
    double t2 = omp_get_wtime();
    int nd = nlev < dim ? nlev : dim;
    if (ac_lsj_davidson > 0 && dim >= ac_lsj_davidson && 4 * nd < dim) {
        /* the nlev lowest only; their vectors into the first columns of H */
        /* on H - shift: with the total energy (hundreds of hartree) on
           the diagonal, the rounding of H x keeps the residual near the
           tolerance and the iteration can stall just above it */
        double shift = 0.0;
        for (int q = 0; q < dim; q++)
            shift += H[(size_t)q * dim + q];
        shift /= dim;
        for (int q = 0; q < dim; q++)
            H[(size_t)q * dim + q] -= shift;
        double *x = malloc((size_t)dim * nd * sizeof *x);
        /* each part that H does not couple to the rest alone, as above */
        int npart = 0, *part = x ? dense_parts(dim, H, &npart) : NULL;
        if (part && getenv("AC_TIME") && npart > 1)
            fprintf(stderr, "J = %d/2, dim %d: %d parts that H does not couple, each solved alone\n",
                    tJ, dim, npart);
        int e = part ? ac_ci_davidson_parts(dim, H, npart, part, nd, ac_lsj_tol, w, x, &it) : -1;
        free(part);
        if (x && !e && res)
            block_res(dim, H, nd, x, w, res);
        dense_el de = {H, dim};
        if (x && !e && miss_check(ci, tJ, dim, idx, nd, w, shift, dense_elem, &de, err, errlen)) {
            free(x);
            return -1;
        }
        for (int q = 0; q < nd; q++)
            w[q] += shift;
        if (e) {
            free(x);
            return fail(err, errlen, "Davidson failed (%d) in the J = %d/2 block", e, tJ);
        }
        memcpy(H, x, (size_t)dim * nd * sizeof *x);
        free(x);
    } else {
        double *a = res ? malloc((size_t)dim * dim * sizeof *a) : NULL;
        if (res && !a)
            return fail(err, errlen, "out of memory");
        if (a)
            memcpy(a, H, (size_t)dim * dim * sizeof *a);
        if (ac_ci_eigvecs(dim, H, w)) {
            free(a);
            return fail(err, errlen, "dsyev failed");
        }
        if (a)
            block_res(dim, a, nd, H, w, res);
        free(a);
    }
    if (getenv("AC_TIME")) {
        fprintf(stderr, "J = %d/2, dim %d: matrix %.2f s, eigenvectors %.2f s", tJ, dim, t2 - t1,
                omp_get_wtime() - t2);
        fprintf(stderr, it ? ", Davidson %d iterations\n" : "\n", it);
    }
    *dimp = dim;
    *itp = it;
    return 0;
}

/* how many levels of the block 2J = tJ */
static int nlev_of_j(int tJ, int nlev)
{
    if (ac_lsj_nlev_j && tJ < ac_lsj_nlev_j_len && ac_lsj_nlev_j[tJ] >= 0)
        return ac_lsj_nlev_j[tJ];
    return nlev;
}

static int by_label(const void *a, const void *b)
{
    const ac_lab *x = a, *y = b;
    return x->ts != y->ts ? x->ts - y->ts : x->tl - y->tl;
}

int ac_lsj_dumpmat = 0, ac_lsj_proj = 0, ac_lsj_nog = 0;

/* What dump= takes from the sink of ac_lsj_prepare (ac_lsj.h).
   dumpmat: every value, kept as (i, j, v) and written sorted.
   proj: the levels |a> = sum_i c_a(i) |i> of the LS blocks are at hand; a
   value v of a term T at (i, j), i <= j, with i in block X and j in Y adds
     c_a(i) v c_b(j)      to <a|T|b>, a in X, b in Y, and for i != j
     c_b(j) s v c_a(i)    to <b|T|a>, the element (j, i) being s v
   (s = 1 for the terms of h, the sign of ac_lsj.h for those of w and w2).
   Nothing of the matrix of T is kept. Each thread adds into its own copy. */
#define OUT_NTERM (AC_LSJ_EXTRA + AC_LSJ_MAXEXTRA)
typedef struct {
    int i, j;
    double v;
} coo1;

typedef struct {
    int coo_on, proj_on, oom;
    coo1 *coo[OUT_NTERM];
    size_t ncoo[OUT_NTERM], cap[OUT_NTERM];
    /* proj */
    int nb;         /* LS blocks with levels */
    ac_lab *blab;   /* [nb] */
    int *nroot, *bdim; /* [nb] */
    double **vec;   /* [nb] nroot x dim, level by level */
    int *blk, *pos; /* [n] the block of a CSF (-1: none) and its place there */
    long *off[3];   /* [nb*nb] where <X|.|Y> begins, by kind; -1: not allowed */
    size_t tot[3], base[OUT_NTERM], per;
    int nthread;
    double *acc;    /* [nthread][per] */
} outsink;

/* 0: a term of h (within a block); 1, 2: of w, w2 (ranks 1, 2 in S and L) */
static int term_kind(int term)
{
    return term == AC_LSJ_SO1 || term == AC_LSJ_SO2 ? 1 : term == AC_LSJ_SS ? 2 : 0;
}

static const char *term_file(int term)
{
    static const char *const nm[AC_LSJ_EXTRA] = {"h_C", "h_OO", "w_SO1", "w_SO2", "w_SS"};
    return term < AC_LSJ_EXTRA ? nm[term] : ac_lsj_extra[term - AC_LSJ_EXTRA].name;
}

static void out_sink(void *ctx, int term, int i, int j, double v)
{
    outsink *o = ctx;
    if (o->coo_on) {
#pragma omp critical(ac_lsj_coo)
        {
            if (o->ncoo[term] == o->cap[term]) {
                size_t nc = o->cap[term] ? 2 * o->cap[term] : 4096;
                coo1 *t = realloc(o->coo[term], nc * sizeof *t);
                if (t)
                    o->coo[term] = t, o->cap[term] = nc;
                else
                    o->oom = 1;
            }
            if (o->ncoo[term] < o->cap[term]) {
                coo1 *t = &o->coo[term][o->ncoo[term]++];
                t->i = i, t->j = j, t->v = v;
            }
        }
    }
    if (!o->proj_on || term == AC_LSJ_HC)
        return;
    int X = o->blk[i], Y = o->blk[j], kind = term_kind(term);
    if (X < 0 || Y < 0)
        return;
    long oxy = o->off[kind][(size_t)X * o->nb + Y], oyx = o->off[kind][(size_t)Y * o->nb + X];
    if (oxy < 0)
        return;
    double *acc = o->acc + (size_t)omp_get_thread_num() * o->per + o->base[term];
    const double *vx = o->vec[X], *vy = o->vec[Y];
    int nx = o->nroot[X], ny = o->nroot[Y], dx = o->bdim[X], dy = o->bdim[Y];
    int pi = o->pos[i], pj = o->pos[j];
    for (int a = 0; a < nx; a++) {
        double ca = vx[(size_t)a * dx + pi] * v;
        for (int b = 0; b < ny; b++)
            acc[oxy + (long)a * ny + b] += ca * vy[(size_t)b * dy + pj];
    }
    if (i != j) {
        ac_lab la = o->blab[X], lb = o->blab[Y];
        double sv = kind && ((la.ts - lb.ts) + (la.tl - lb.tl)) / 2 % 2 ? -v : v;
        for (int b = 0; b < ny; b++) {
            double cb = vy[(size_t)b * dy + pj] * sv;
            for (int a = 0; a < nx; a++)
                acc[oyx + (long)b * nx + a] += cb * vx[(size_t)a * dx + pi];
        }
    }
}

static void out_free(outsink *o)
{
    for (int t = 0; t < OUT_NTERM; t++)
        free(o->coo[t]);
    for (int x = 0; x < o->nb; x++)
        free(o->vec[x]);
    free(o->blab);
    free(o->nroot);
    free(o->bdim);
    free(o->vec);
    free(o->blk);
    free(o->pos);
    for (int k = 0; k < 3; k++)
        free(o->off[k]);
    free(o->acc);
    memset(o, 0, sizeof *o);
}

static int by_coo(const void *a, const void *b)
{
    const coo1 *x = a, *y = b;
    return x->i != y->i ? x->i - y->i : x->j - y->j;
}

/* dumpmat: PREFIX.<term>.{i,j,v}.npy of every term that has an element */
static int out_write_coo(outsink *o, char *err, int errlen)
{
    if (o->oom)
        return fail(err, errlen, "out of memory");
    for (int t = 0; t < AC_LSJ_EXTRA + ac_lsj_nextra && t < OUT_NTERM; t++) {
        size_t nc = o->ncoo[t], plen = strlen(ac_lsj_dump) + strlen(term_file(t)) + 16;
        if (nc == 0)
            continue;
        qsort(o->coo[t], nc, sizeof(coo1), by_coo);
        int *ij = malloc(nc * sizeof *ij);
        double *v = malloc(nc * sizeof *v);
        char *path = malloc(plen);
        int bad = !ij || !v || !path;
        for (int k = 0; k < 3 && !bad; k++) {
            for (size_t e = 0; e < nc; e++) {
                if (k < 2)
                    ij[e] = k ? o->coo[t][e].j : o->coo[t][e].i;
                else
                    v[e] = o->coo[t][e].v;
            }
            snprintf(path, plen, "%s.%s.%c.npy", ac_lsj_dump, term_file(t), "ijv"[k]);
            bad = ac_npy_write(path, k < 2 ? 'i' : 'f', 1, &nc, k < 2 ? (const void *)ij : (const void *)v);
        }
        if (bad)
            fail(err, errlen, "cannot write the files of %s", term_file(t));
        free(ij);
        free(v);
        free(path);
        if (bad)
            return -1;
    }
    return 0;
}

static char *proj_json = NULL;
static size_t proj_json_len = 0;

const char *ac_lsj_dump_proj(void)
{
    return proj_json ? proj_json : "";
}

/* the terms that ci and ac_lsj_extra hold besides AC_LSJ_HC */
static int term_on(const ac_ci *ci, int t)
{
    return t == AC_LSJ_OO ? ci->in.roo != NULL : t == AC_LSJ_SO1 ? ci->in.zeta != NULL
         : t == AC_LSJ_SO2 ? ci->in.rkg != NULL : t == AC_LSJ_SS ? ci->in.rss != NULL
         : t >= AC_LSJ_EXTRA && t < AC_LSJ_EXTRA + ac_lsj_nextra;
}

/* proj: the tables of the blocks; the levels are put in by the caller
   (o->nb, blab, nroot, bdim, vec, blk, pos) */
static int out_proj_begin(outsink *o, const ac_ci *ci)
{
    int nb = o->nb;
    for (int k = 0; k < 3; k++) {
        o->off[k] = malloc((size_t)(nb ? nb : 1) * nb * sizeof(long));
        if (!o->off[k])
            return -1;
        o->tot[k] = 0;
        for (int x = 0; x < nb; x++)
            for (int y = 0; y < nb; y++) {
                ac_lab a = o->blab[x], b = o->blab[y];
                int ok = k == 0 ? x == y
                                : tri2(a.ts, 2 * k, b.ts) && tri2(a.tl, 2 * k, b.tl);
                o->off[k][(size_t)x * nb + y] = ok ? (long)o->tot[k] : -1;
                if (ok)
                    o->tot[k] += (size_t)o->nroot[x] * o->nroot[y];
            }
    }
    o->per = 0;
    for (int t = 0; t < OUT_NTERM; t++) {
        o->base[t] = o->per;
        if (t != AC_LSJ_HC && term_on(ci, t))
            o->per += o->tot[term_kind(t)];
    }
    o->nthread = omp_get_max_threads();
    o->acc = calloc((size_t)o->nthread * (o->per ? o->per : 1), sizeof *o->acc);
    return o->acc ? 0 : -1;
}

static int out_write_proj(outsink *o, const ac_ci *ci, char *err, int errlen)
{
    int nb = o->nb;
    for (int t = 0; t < OUT_NTERM; t++) {
        if (t == AC_LSJ_HC || !term_on(ci, t))
            continue;
        int kind = term_kind(t);
        for (int th = 1; th < o->nthread; th++)
            for (size_t e = 0; e < o->tot[kind]; e++)
                o->acc[o->base[t] + e] += o->acc[(size_t)th * o->per + o->base[t] + e];
        for (int x = 0; x < nb; x++)
            for (int y = 0; y < nb; y++) {
                long off = o->off[kind][(size_t)x * nb + y];
                if (off < 0 || o->nroot[x] == 0 || o->nroot[y] == 0)
                    continue;
                char name[160], item[400];
                snprintf(name, sizeof name, "proj.%s.S%dL%d.S%dL%d.npy", term_file(t),
                         o->blab[x].ts + 1, o->blab[x].tl / 2, o->blab[y].ts + 1, o->blab[y].tl / 2);
                size_t plen = strlen(ac_lsj_dump) + strlen(name) + 2;
                char *path = malloc(plen);
                size_t sh[2] = {(size_t)o->nroot[x], (size_t)o->nroot[y]};
                if (!path)
                    return fail(err, errlen, "out of memory");
                snprintf(path, plen, "%s.%s", ac_lsj_dump, name);
                int bad = ac_npy_write(path, 'f', 2, sh, o->acc + o->base[t] + off);
                free(path);
                if (bad)
                    return fail(err, errlen, "cannot write %s", name);
                int len = snprintf(item, sizeof item,
                                   "%s    {\"term\": \"%s\", \"bra\": \"S%dL%d\", \"ket\": \"S%dL%d\", "
                                   "\"file\": \"%s\", \"shape\": [%d, %d]}",
                                   proj_json_len ? ",\n" : "", term_file(t), o->blab[x].ts + 1,
                                   o->blab[x].tl / 2, o->blab[y].ts + 1, o->blab[y].tl / 2, name,
                                   o->nroot[x], o->nroot[y]);
                char *nj = realloc(proj_json, proj_json_len + (size_t)len + 1);
                if (!nj)
                    return fail(err, errlen, "out of memory");
                proj_json = nj;
                memcpy(proj_json + proj_json_len, item, (size_t)len + 1);
                proj_json_len += (size_t)len;
            }
    }
    return 0;
}

static int dumpmat_allowed(int n, char *err, int errlen)
{
    if (n > 20000 && ac_lsj_dumpmat < 2)
        return fail(err, errlen,
                    "dumpmat writes whole matrices and is meant for small lists; %d CSFs are "
                    "refused (dumpmat=force writes them all the same)", n);
    return 0;
}

int ac_lsj_print_ls(FILE *out, ac_ci *ci, int nlev, char *err, int errlen)
{
    int want_mat = ac_lsj_dump && ac_lsj_dumpmat, want_proj = ac_lsj_dump && ac_lsj_proj;
    if (want_mat && dumpmat_allowed(ci->ncsf, err, errlen))
        return -1;
    /* the first pass: AC_LSJ_HC alone */
    ac_ci_integrals in0 = ci->in;
    int nextra0 = ac_lsj_nextra, others = nextra0 > 0;
    for (int t = AC_LSJ_OO; t < AC_LSJ_EXTRA; t++)
        others |= term_on(ci, t);
    ci->in.zeta = NULL, ci->in.rkg = NULL, ci->in.rss = NULL, ci->in.roo = NULL;
    ac_lsj_nextra = 0;
    ac_lsj m;
    outsink os;
    memset(&os, 0, sizeof os);
    double t0 = omp_get_wtime();
    int pe = ac_lsj_prepare(&m, ci, err, errlen);
    ci->in = in0;
    ac_lsj_nextra = nextra0;
    if (pe)
        return -1;
    if (getenv("AC_TIME"))
        fprintf(stderr, "ac_lsj_prepare: %.2f s\n", omp_get_wtime() - t0);
    int n = ci->ncsf, rc = -1, nlab = 0;
    int *idx = malloc((size_t)n * sizeof *idx), *lab = NULL;
    double *H = m.sparse ? NULL : malloc((size_t)n * n * sizeof *H);
    double *w = malloc((size_t)n * sizeof *w), *res = malloc((size_t)n * sizeof *res);
    ac_lab *labs = NULL;
    if (!idx || (!H && !m.sparse) || !w || !res) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    for (int i = 0; i < n; i++)
        idx[i] = i;
    lab = label_numbers(ci, n, idx, &labs, &nlab);
    if (!lab) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    qsort(labs, (size_t)nlab, sizeof *labs, by_label);
    if (want_proj) {
        os.blab = malloc((size_t)nlab * sizeof *os.blab);
        os.nroot = malloc((size_t)nlab * sizeof *os.nroot);
        os.bdim = malloc((size_t)nlab * sizeof *os.bdim);
        os.vec = calloc((size_t)nlab, sizeof *os.vec);
        os.blk = malloc((size_t)n * sizeof *os.blk);
        os.pos = malloc((size_t)n * sizeof *os.pos);
        if (!os.blab || !os.nroot || !os.bdim || !os.vec || !os.blk || !os.pos) {
            fail(err, errlen, "out of memory");
            goto out;
        }
        for (int i = 0; i < n; i++)
            os.blk[i] = os.pos[i] = -1;
    }
    fprintf(out, "%-6s %3s %22s\n", "term", "#", "E (hartree)");
    for (int x = 0; x < nlab; x++) {
        int nl = nlev, dim = 0, it = 0;
        for (int k = 0; k < ac_lsj_nlev_sl_n; k++)
            if (ac_lsj_nlev_sl[3 * k] == labs[x].ts && ac_lsj_nlev_sl[3 * k + 1] == labs[x].tl)
                nl = ac_lsj_nlev_sl[3 * k + 2];
        if (nl <= 0)
            continue;
        ls_lab = labs[x];
        ls_block = 1;
        int e = solve_block(&m, ci, labs[x].ts + labs[x].tl, nl, idx, &H, w, res, &dim, &it, err,
                            errlen);
        ls_block = 0;
        if (e)
            goto out;
        int nd = nl < dim ? nl : dim;
        char term[32], tag[32], head[64];
        term_name(term, sizeof term, labs[x].ts, labs[x].tl);
        for (int p = 0; p < nd; p++)
            fprintf(out, "%-6s %3d %22.15f\n", term, p + 1, w[p]);
        fflush(out);
        snprintf(tag, sizeof tag, "S%dL%d", labs[x].ts + 1, labs[x].tl / 2);
        snprintf(head, sizeof head, "\"S2p1\": %d, \"L\": %d", labs[x].ts + 1, labs[x].tl / 2);
        if (ac_lsj_dump && dump_block(tag, head, dim, idx, nd, H, w, res, it, err, errlen))
            goto out;
        if (want_proj) {
            int b = os.nb;
            os.vec[b] = malloc((size_t)(nd ? nd : 1) * dim * sizeof **os.vec);
            if (!os.vec[b]) {
                fail(err, errlen, "out of memory");
                goto out;
            }
            os.nb++;
            memcpy(os.vec[b], H, (size_t)nd * dim * sizeof **os.vec);
            os.blab[b] = labs[x], os.nroot[b] = nd, os.bdim[b] = dim;
            for (int p = 0; p < dim; p++)
                os.blk[idx[p]] = b, os.pos[idx[p]] = p;
        }
    }
    /* the second pass: the other terms (and AC_LSJ_HC again for dumpmat) go
       to the files; nothing is stored */
    if ((want_proj && others) || want_mat) {
        ac_lsj m2;
        os.coo_on = want_mat;
        os.proj_on = want_proj && others;
        if (os.proj_on && out_proj_begin(&os, ci)) {
            fail(err, errlen, "out of memory");
            goto out;
        }
        double t1 = omp_get_wtime();
        ac_lsj_sink = out_sink, ac_lsj_sink_ctx = &os;
        ac_lsj_nostore = 1, ac_lsj_skip_hc = !want_mat;
        int e2 = ac_lsj_prepare(&m2, ci, err, errlen);
        ac_lsj_sink = NULL, ac_lsj_sink_ctx = NULL;
        ac_lsj_nostore = 0, ac_lsj_skip_hc = 0;
        if (e2)
            goto out;
        ac_lsj_free(&m2);
        if (getenv("AC_TIME"))
            fprintf(stderr, "the terms for the files (second pass): %.2f s\n", omp_get_wtime() - t1);
        if (want_mat && out_write_coo(&os, err, errlen))
            goto out;
        if (os.proj_on && out_write_proj(&os, ci, err, errlen))
            goto out;
    }
    rc = 0;
out:
    out_free(&os);
    free(lab);
    free(labs);
    free(idx);
    free(H);
    free(w);
    free(res);
    ac_lsj_free(&m);
    return rc;
}

int ac_lsj_print_levels(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                        double (*zr)(void *, int, int), void *ctx, char *err, int errlen)
{
    return ac_lsj_print_levels_keep(out, ci, nlev, tkin, zr, ctx, NULL, NULL, err, errlen);
}

int ac_lsj_print_levels_keep(FILE *out, ac_ci *ci, int nlev, double (*tkin)(void *, int, int),
                             double (*zr)(void *, int, int), void *ctx, ac_lsj_kept **keep,
                             int *nkeep, char *err, int errlen)
{
    ac_lsj m;
    double t0 = omp_get_wtime();
    int want_mat = ac_lsj_dump && ac_lsj_dumpmat;
    outsink os;
    memset(&os, 0, sizeof os);
    if (want_mat) {
        if (dumpmat_allowed(ci->ncsf, err, errlen))
            return -1;
        os.coo_on = 1;
        ac_lsj_sink = out_sink, ac_lsj_sink_ctx = &os;
    }
    int pe = ac_lsj_prepare(&m, ci, err, errlen);
    ac_lsj_sink = NULL, ac_lsj_sink_ctx = NULL;
    if (!pe && want_mat && out_write_coo(&os, err, errlen)) {
        ac_lsj_free(&m);
        pe = -1;
    }
    out_free(&os);
    if (pe)
        return -1;
    if (getenv("AC_TIME"))
        fprintf(stderr, "ac_lsj_prepare: %.2f s\n", omp_get_wtime() - t0);
    pc_begin(ci);
    int n = ci->ncsf, nel = 0, tjmax = 0, rc = -1;
    for (int s = 1; s <= ci->csf[0].c.nsh; s++)
        nel += ci->csf[0].c.nel[s];
    for (int i = 0; i < n; i++) {
        ac_lab a = total(ci, i);
        if (a.ts + a.tl > tjmax)
            tjmax = a.ts + a.tl;
    }
    int *idx = malloc((size_t)n * sizeof *idx);
    /* sparse: H is made for each J block (the eigenvectors, or a small block for dsyev) */
    double *H = m.sparse ? NULL : malloc((size_t)n * n * sizeof *H);
    double *w = malloc((size_t)n * sizeof *w);
    double *res = ac_lsj_dump ? malloc((size_t)n * sizeof *res) : NULL;
    int maxlev = nlev;
    for (int tJ = 0; tJ <= tjmax && tJ < ac_lsj_nlev_j_len; tJ++)
        if (nlev_of_j(tJ, nlev) > maxlev)
            maxlev = nlev_of_j(tJ, nlev);
    level *all = malloc((size_t)(tjmax / 2 + 1) * (maxlev < n ? maxlev : n) * sizeof *all);
    int nall = 0, nkept = 0;
    const int nlev_all = nlev;
    kept *kp = NULL;
    /* The g factors: for all the J blocks in one pass after the last block
       (gdefer; zeeman_1e_all for the one-body parts, and zeeman_2e_all for
       the two-electron parts when their tables do not fit AC_PROP_CACHE_MB:
       g2pass; from the tables when they fit), and the lines of the levels
       are printed then. AC_G2_PASS=1: the two-electron parts always in one
       pass, =0: never. AC_G_BY_LEVEL: level by level as each J block is
       solved, with the tables of the elements (the way before). */
    const char *g2env = getenv("AC_G2_PASS");
    const int g2pass = !ac_lsj_nog && pc_g2_tables(ci) && !getenv("AC_G_BY_LEVEL") &&
                       (g2env ? atoi(g2env) != 0 : !pc_g2_fit(ci));
    const int gdefer = !ac_lsj_nog && !getenv("AC_G_BY_LEVEL");
    kept *k2 = NULL;
    int nk2 = 0, npend = 0;
    pending *pend = gdefer ? malloc((size_t)(tjmax / 2 + 1) * (maxlev < n ? maxlev : n) *
                                    sizeof *pend + 1)
                           : NULL;
    if (!idx || (!H && !m.sparse) || !w || !all || (ac_lsj_dump && !res) || (gdefer && !pend)) {
        fail(err, errlen, "out of memory");
        goto out;
    }
    if (ac_lsj_nog)
        fprintf(out, "%4s %3s %22s %-6s %6s\n", "J", "#", "E (hartree)", "term", "weight");
    else
    fprintf(out, "%4s %3s %22s %-6s %6s %16s %14s %14s %14s%s%s%s%s\n", "J", "#", "E (hartree)",
            "term", "weight", "g_J", "dg_RMC", "dg_GC", "dg_ORMC",
            ci->in.rgc ? "          dg_GC2e" : "", ci->in.roz ? "          dg_OZ2e" : "",
            ci->in.rrc ? "              rec2" : "", ac_lsj_h6_rk ? "       dg_H6/kappa" : "");
    for (int tJ = nel % 2; tJ <= tjmax; tJ += 2) {
        int dim = 0, it = 0;
        /* the levels of this block (ac_lsj_nlev_j) */
        int nlev = nlev_of_j(tJ, nlev_all);
        if (nlev <= 0)
            continue;
        if (solve_block(&m, ci, tJ, nlev, idx, &H, w, res, &dim, &it, err, errlen))
            goto out;
        if (dim == 0)
            continue;
        if (ac_lsj_dump) {
            char tag[32], head[32];
            snprintf(tag, sizeof tag, "J%d", tJ);
            snprintf(head, sizeof head, "\"twoJ\": %d", tJ);
            if (dump_block(tag, head, dim, idx, nlev < dim ? nlev : dim, H, w, res, it, err, errlen))
                goto out;
        }
        double tg = 0.0;
        for (int p = 0; p < dim && p < nlev; p++) {
            const double *c = H + (size_t)p * dim;
            /* leading LS term */
            double best = 0.0, wsum[64] = {0};
            int key[64], nk = 0, lead = 0;
            for (int q = 0; q < dim; q++) {
                ac_lab a = total(ci, idx[q]);
                int k = a.ts * 1000 + a.tl, r;
                for (r = 0; r < nk && key[r] != k; r++)
                    ;
                if (r == nk && nk < 64)
                    key[nk++] = k;
                if (r < 64)
                    wsum[r] += c[q] * c[q];
            }
            for (int r = 0; r < nk; r++)
                if (wsum[r] > best)
                    best = wsum[r], lead = key[r];
            char term[32];
            term_name(term, sizeof term, lead / 1000, lead % 1000);
            ac_lsj_g gg;
            int hasg = tJ > 0 && !ac_lsj_nog, wait = hasg && gdefer;
            if (hasg && !wait) {
                int e;
                double tg0 = omp_get_wtime();
                if (ac_lsj_gfactor(ci, dim, idx, tJ, c, tkin, zr, ctx, &gg, &e)) {
                    fail(err, errlen, "g factor: engine error %d", e);
                    goto out;
                }
                tg += omp_get_wtime() - tg0;
            }
            if (gdefer) {
                pending *pd = &pend[npend++];
                pd->tJ = tJ, pd->num = p + 1, pd->kx = wait ? nk2 : -1, pd->hasg = hasg;
                pd->e = w[p], pd->best = best;
                memcpy(pd->term, term, sizeof pd->term);
                if (wait && keep_level(&k2, &nk2, tJ, p + 1, w[p], dim, idx, c)) {
                    fail(err, errlen, "out of memory");
                    goto out;
                }
            } else
                level_line(out, ci, tJ, p + 1, w[p], term, best, hasg ? &gg : NULL);
            all[nall].tJ = tJ;
            all[nall++].e = w[p];
            if ((ac_lsj_zoff || ac_lsj_hfs_rm3 || ac_lsj_trans_rk || ac_lsj_is_rk || keep) &&
                keep_level(&kp, &nkept, tJ, p + 1, w[p], dim, idx, c)) {
                fail(err, errlen, "out of memory");
                goto out;
            }
        }
        if (getenv("AC_TIME") && tJ > 0 && !ac_lsj_nog && !gdefer)
            fprintf(stderr, "J = %d/2, dim %d: g factors of %d levels %.2f s\n", tJ, dim,
                    nlev < dim ? nlev : dim, tg);
        if (dim >= 500)
            mem_note("after the J block");
    }
    if (gdefer) {
        /* the g factors of the waiting levels, then all the lines */
        double t1 = omp_get_wtime(), t2;
        int e = 0, r1, r2 = 1;
        ac_lsj_g *g2 = calloc((size_t)nk2 + 1, sizeof *g2);
        if (!g2) {
            fail(err, errlen, "out of memory");
            goto out;
        }
        r1 = zeeman_1e_all(ci, k2, nk2, tkin, zr, ctx, g2, &e);
        t2 = omp_get_wtime();
        if (getenv("AC_TIME"))
            fprintf(stderr, "one-body parts of g of %d levels in one pass: %.2f s%s\n", nk2, t2 - t1,
                    r1 ? " (not done: level by level)" : "");
        if (r1 == 0 && g2pass) {
            r2 = zeeman_2e_all(ci, k2, nk2, g2, &e);
            if (getenv("AC_TIME"))
                fprintf(stderr, "two-electron parts of g of %d levels in one pass: %.2f s%s\n", nk2,
                        omp_get_wtime() - t2, r2 ? " (not done: level by level)" : "");
        }
        int bad = r1 < 0 || r2 < 0;
        for (int x = 0; x < npend && !bad; x++) {
            pending *pd = &pend[x];
            if (pd->kx >= 0) {
                const ac_lsj_level *v = &k2[pd->kx].v;
                pd->g = g2[pd->kx];
                bad = r1 == 0 ? gfactor_rest(ci, v->dim, v->idx, v->tJ, v->c, r2 == 0, &pd->g, &e)
                              : ac_lsj_gfactor(ci, v->dim, v->idx, v->tJ, v->c, tkin, zr, ctx,
                                               &pd->g, &e);
            }
            if (!bad)
                level_line(out, ci, pd->tJ, pd->num, pd->e, pd->term, pd->best,
                           pd->hasg ? &pd->g : NULL);
        }
        free(g2);
        if (bad) {
            fail(err, errlen, "g factor: engine error %d", e);
            goto out;
        }
        if (getenv("AC_TIME"))
            fprintf(stderr, "g factors of %d levels: %.2f s\n", nk2, omp_get_wtime() - t1);
    }
    double tt = omp_get_wtime();
    if (ac_lsj_zoff && print_zoff(out, ci, kp, nkept, kp, nkept, 0, tkin, zr, ctx, err, errlen))
        goto out;
    if (ac_lsj_hfs_rm3 && print_hfs(out, ci, kp, nkept, kp, nkept, 0, ctx, err, errlen))
        goto out;
    if (ac_lsj_trans_rk && print_trans(out, ci, kp, nkept, kp, nkept, 0, ctx, err, errlen))
        goto out;
    if (ac_lsj_is_rk && print_is(out, ci, kp, nkept, ctx, err, errlen))
        goto out;
    if (getenv("AC_TIME") && (ac_lsj_zoff || ac_lsj_hfs_rm3 || ac_lsj_trans_rk || ac_lsj_is_rk))
        fprintf(stderr, "tables (zoff, hfs, trans, is): %.2f s\n", omp_get_wtime() - tt);
    qsort(all, (size_t)nall, sizeof *all, by_energy);
    fprintf(out, "levels relative to the lowest (cm-1):\n");
    for (int i = 0; i < nall; i++)
        fprintf(out, "  J = %2d/2  %16.6f\n", all[i].tJ, (all[i].e - all[0].e) * AC_HARTREE_CM);
    rc = 0;
    if (keep) {
        *keep = kp;
        *nkeep = nkept;
        kp = NULL;
        nkept = 0;
    }
out:
    pc_end();
    ac_lsj_kept_free(kp, nkept);
    ac_lsj_kept_free(k2, nk2);
    free(pend);
    free(all);
    free(idx);
    free(H);
    free(w);
    free(res);
    ac_lsj_free(&m);
    return rc;
}

/* the alpha^2 terms of g_J: Pachucki (2004), Eq. (32) (H1-H6); RMC and GC1e
   as in Neese (2001); the H6 term checked against Jentschura (2010), Eq. (10)
   (technical note, Sec. 5.1) */
int ac_lsj_zeeman_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                          double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                          void *ctx, ac_lsj_g *r, int *err)
{
    memset(r, 0, sizeof *r);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ))
        return 0;
    pairs1 P;
    double tp = omp_get_wtime();
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    if (getenv("AC_TIME_PAIR"))
        fprintf(stderr, "pairs1_build: %.2f s, %s\n", omp_get_wtime() - tp,
                P.pos ? "list of the window" : "own list");
    gop o = {0, NULL, NULL, NULL};
    r->gS = pair_reduced(ci, A, B, &P, &o, 2, 0, 2, err);
    o.kind = 1;
    if (!*err)
        r->gL = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
    if (!*err && tkin) {
        gop t = {2, tkin, ctx, NULL};
        r->rmc = pair_reduced(ci, A, B, &P, &t, 2, 0, 2, err);
        t.kind = 5;
        if (!*err)
            r->ormc = pair_reduced(ci, A, B, &P, &t, 0, 2, 2, err);
    }
    if (!*err && zr) {
        gop u = {3, zr, ctx, NULL};
        r->gc = pair_reduced(ci, A, B, &P, &u, 2, 0, 2, err);
        u.kind = 4;
        if (!*err)
            r->gc += pair_reduced(ci, A, B, &P, &u, 2, 4, 2, err);
    }
    if (!*err && ac_lsj_h6_rk) {
        /* -s^j T_jz = sqrt(5/3) [s x T^(2)]^(1)_z, T^(2) = [p x p]^(2) (kind 20) */
        gop h = {20, NULL, ctx, ac_lsj_h6_rk};
        r->h6 = ALPHA2 * sqrt(15.0) / 3.0 * pair_reduced(ci, A, B, &P, &h, 2, 4, 2, err);
    }
    pairs1_free(&P);
    for (int kp = 0; kp <= 2 && !*err && ci->in.rgc; kp++)
        r->gc2e += pair_reduced_gc2e(ci, A, B, kp, err);
    if (!*err && ci->in.roz)
        r->oz2e = pair_reduced_gc2e(ci, A, B, -1, err);
    if (!*err && ci->in.rrc)
        r->rec2 = pair_reduced_gc2e(ci, A, B, -2, err);
    return *err ? -1 : 0;
}

/* the hyperfine operators of Lindgren-Rosen (1974) */
int ac_lsj_hfs_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                       double (*rm3)(void *, int, int), double (*r00)(void *, int, int),
                       void *ctx, ac_lsj_hfs *h, int *err)
{
    memset(h, 0, sizeof *h);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ) && !tri2(A->tJ, 4, B->tJ))
        return 0;
    pairs1 P;
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    if (tri2(A->tJ, 2, B->tJ)) {
        gop o = {6, rm3, ctx, NULL};
        h->orb = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        o.kind = 7;
        if (!*err)
            h->sd = pair_reduced(ci, A, B, &P, &o, 2, 4, 2, err);
        gop c = {8, r00, ctx, NULL};
        if (!*err)
            h->con = pair_reduced(ci, A, B, &P, &c, 2, 0, 2, err);
    }
    if (!*err && tri2(A->tJ, 4, B->tJ)) {
        gop q = {9, rm3, ctx, NULL};
        h->quad = pair_reduced(ci, A, B, &P, &q, 0, 4, 4, err);
    }
    pairs1_free(&P);
    return *err ? -1 : 0;
}

int ac_lsj_is_level(ac_ci *ci, const ac_lsj_level *A, double (*rk)(void *, int, int, int),
                    void *ctx, ac_lsj_is *r, int *err)
{
    memset(r, 0, sizeof *r);
    *err = 0;
    pairs1 P;
    if (pairs1_build(ci, A, A, &P)) {
        *err = 1;
        return -1;
    }
    /* J-scalars: <J||O||J> = sqrt(2J+1) <O> */
    double norm = 1.0 / sqrt(A->tJ + 1.0);
    gop o = {21, NULL, ctx, rk};
    r->nms = norm * pair_reduced(ci, A, A, &P, &o, 0, 0, 0, err);
    o.kind = 22;
    if (!*err)
        r->rho0 = norm * pair_reduced(ci, A, A, &P, &o, 0, 0, 0, err);
    pairs1_free(&P);
    if (!*err)
        r->sms = norm * pair_reduced_2e(ci, A, A, -3, rk, ctx, err);
    return *err ? -1 : 0;
}

int ac_lsj_trans_reduced(ac_ci *ci, const ac_lsj_level *A, const ac_lsj_level *B,
                         double (*rk)(void *, int, int, int), void *ctx, ac_lsj_trans *t,
                         int *err)
{
    memset(t, 0, sizeof *t);
    *err = 0;
    if (!tri2(A->tJ, 2, B->tJ) && !tri2(A->tJ, 4, B->tJ) && !tri2(A->tJ, 6, B->tJ))
        return 0;
    pairs1 P;
    if (pairs1_build(ci, A, B, &P)) {
        *err = 1;
        return -1;
    }
    gop o = {10, NULL, ctx, rk};
    if (tri2(A->tJ, 2, B->tJ)) {
        t->e1l = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        o.kind = 11;
        if (!*err)
            t->e1v = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        gop s = {0, NULL, NULL, NULL};
        double rs = 0.0, rl = 0.0;
        if (!*err)
            rs = pair_reduced(ci, A, B, &P, &s, 2, 0, 2, err);
        s.kind = 1;
        if (!*err)
            rl = pair_reduced(ci, A, B, &P, &s, 0, 2, 2, err);
        t->m1 = rl + trans_gs() * rs;
        double *m[3] = {&t->m1r2, &t->m1p2, &t->m1zr};
        for (int k = 0; k < 3 && !*err; k++) {
            o.kind = 13 + k;
            *m[k] = pair_reduced(ci, A, B, &P, &o, 2, 0, 2, err);
        }
        double *ml[2] = {&t->m1lr2, &t->m1lp2};
        for (int k = 0; k < 2 && !*err; k++) {
            o.kind = 16 + k;
            *ml[k] = pair_reduced(ci, A, B, &P, &o, 0, 2, 2, err);
        }
        double *mt[3] = {&t->m1tr2, &t->m1tzr, &t->m1tpp};
        for (int k = 0; k < 3 && !*err; k++) {
            o.kind = 18 + k;
            *mt[k] = pair_reduced(ci, A, B, &P, &o, 2, 4, 2, err);
        }
        /* the two-electron terms, when their integrals are there */
        if (!*err && ci->in.rm1 && (ac_lsj_m1_2e_parts & 1)) {
            t->m1s0 = pair_reduced_gc2e(ci, A, B, 10, err);
            if (!*err)
                t->m1s2 = pair_reduced_gc2e(ci, A, B, 12, err);
        }
        if (!*err && ci->in.rm1 && ci->in.roz && (ac_lsj_m1_2e_parts & 2))
            t->m1oz = pair_reduced_gc2e(ci, A, B, -1, err);
    }
    if (!*err && tri2(A->tJ, 4, B->tJ)) {
        o.kind = 12;
        t->e2 = pair_reduced(ci, A, B, &P, &o, 0, 4, 4, err);
        double ml = 0.0, ms = 0.0;
        o.kind = 24;
        if (!*err)
            ml = pair_reduced(ci, A, B, &P, &o, 0, 4, 4, err);
        o.kind = 25;
        if (!*err)
            ms = pair_reduced(ci, A, B, &P, &o, 2, 2, 4, err);
        t->m2 = ml + ms;
    }
    if (!*err && tri2(A->tJ, 6, B->tJ)) {
        o.kind = 23;
        t->e3 = pair_reduced(ci, A, B, &P, &o, 0, 6, 6, err);
        double ml = 0.0, ms = 0.0;
        o.kind = 26;
        if (!*err)
            ml = pair_reduced(ci, A, B, &P, &o, 0, 6, 6, err);
        o.kind = 27;
        if (!*err)
            ms = pair_reduced(ci, A, B, &P, &o, 2, 4, 6, err);
        t->m3 = ml + ms;
    }
    pairs1_free(&P);
    return *err ? -1 : 0;
}

/* Pachucki (2004), Eqs. (59), (60), kappa = 0 */
double ac_lsj_m1_spin_scalar(const ac_lsj_trans *t, double w)
{
    return -4.0 * ALPHA2 * (w * w * t->m1r2 / 12.0 + t->m1p2 / 3.0 - t->m1zr / 6.0);
}

double ac_lsj_m1_rel1e(const ac_lsj_trans *t, double w)
{
    /* Pachucki (2004), Eqs. (61), (62).
       orbital: -2 T_vec -> -L (w^2 r^2/10 + p^2/2); tensor: -2 T^ij sigma_j =
       -4 (T.s) = 4 sqrt(5/3) [s x T^(2)]^(1), T^(2) = -(w^2/20) [r x r]^(2)
       - (1/8) [p x p]^(2) + (Z/4) [r x r]^(2)/r^3, [r x r]^(2) = sqrt(2/3) r^2 C^2
       (the delta_ij parts of Eq. (62) cancel) */
    return ac_lsj_m1_spin_scalar(t, w) +
           ALPHA2 * (-w * w * t->m1lr2 / 10.0 - t->m1lp2 / 2.0 -
                     sqrt(10.0) / 15.0 * w * w * t->m1tr2 + sqrt(10.0) / 3.0 * t->m1tzr -
                     sqrt(15.0) / 6.0 * t->m1tpp);
}

int ac_lsj_m1_2e_parts = 3;

double ac_lsj_m1_rel(const ac_lsj_trans *t, double w)
{
    /* Pachucki (2004), Eqs. (60)-(62). The two-electron terms carry alpha^2
       already (ac_ci_m1s2, ac_ci_oz2e:
       the orbital term of Eq. (61) is the orbit-orbit Zeeman operator) */
    return ac_lsj_m1_rel1e(t, w) + t->m1s0 + t->m1s2 + t->m1oz;
}

int ac_lsj_gfactor(ac_ci *ci, int dim, const int *idx, int tJ, const double *c,
                   double (*tkin)(void *, int, int), double (*zr)(void *, int, int),
                   void *ctx, ac_lsj_g *g, int *err)
{
    memset(g, 0, sizeof *g);
    *err = 0;
    if (tJ <= 0)
        return -1;
    /* <J J|T^1_0|J J> = J / sqrt(J(J+1)(2J+1)) <J||T||J>, then divided by J */
    double J = tJ / 2.0, norm = 1.0 / sqrt(J * (J + 1) * (2 * J + 1));
    ac_lsj_level a = {tJ, dim, idx, c};
    if (ac_lsj_zeeman_reduced(ci, &a, &a, tkin, zr, ctx, g, err))
        return -1;
    g->gL *= norm;
    g->gS *= norm;
    g->rmc *= norm;
    g->ormc *= norm;
    g->gc *= norm;
    g->gc2e *= norm;
    g->oz2e *= norm;
    g->rec2 *= norm;
    g->h6 *= norm;
    return 0;
}

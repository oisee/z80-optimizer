// CPU focused brute-force for i3 (no CUDA) — small pools, deep search
// Build: gcc -O3 -march=native -o cpu_focused cpu_focused_i3.c -lpthread -lm
// Usage: ./cpu_focused [max-len]  (default 22)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define MAX_OPS 13
#define NUM_TEST 256
#define NUM_THREADS 4

static const char *allOpNames[] = {
    "MUL3", "MUL5", "MUL7", "SHR", "NEG",
    "SAVE", "SUB_B", "SBC_MASK", "AND_0F",
    "XOR_B", "AND_F0", "RLCA", "RRCA"
};

static inline uint8_t run_op(uint8_t a, uint8_t *b, int *carry, int op) {
    uint16_t r;
    switch (op) {
    case 0:  r = (uint16_t)a * 3; *carry = r > 0xFF; return (uint8_t)r;
    case 1:  r = (uint16_t)a * 5; *carry = r > 0xFF; return (uint8_t)r;
    case 2:  r = (uint16_t)a * 7; *carry = r > 0xFF; return (uint8_t)r;
    case 3:  *carry = a & 1; return a >> 1;
    case 4:  *carry = (a != 0); return (uint8_t)(0 - a);
    case 5:  *b = a; return a;
    case 6:  *carry = a < *b; return a - *b;
    case 7:  { int cc = *carry; return cc ? 0xFF : 0x00; }
    case 8:  *carry = 0; return a & 0x0F;
    case 9:  *carry = 0; return a ^ *b;
    case 10: *carry = 0; return a & 0xF0;
    case 11: *carry = (a >> 7) & 1; return ((a << 1) | (a >> 7)) & 0xFF;
    case 12: *carry = a & 1; return ((a >> 1) | (a << 7)) & 0xFF;
    }
    return a;
}

static inline uint8_t run_seq(const uint8_t *ops, const uint8_t *opMap, int len, uint8_t input) {
    uint8_t a = input, b = 0; int carry = 0;
    for (int i = 0; i < len; i++)
        a = run_op(a, &b, &carry, opMap[ops[i]]);
    return a;
}

typedef struct {
    const char *name;
    uint8_t target[256];
    uint8_t pool[MAX_OPS];
    int poolSize;
    int maxDepth;
} Target;

static void gen_log2_f3_5(uint8_t *out) {
    for (int i = 0; i < 256; i++)
        out[i] = (i == 0) ? 0 : (uint8_t)(log2((double)i) * 32.0);
}

static void gen_bin2bcd(uint8_t *out) {
    for (int i = 0; i < 256; i++)
        out[i] = (i > 99) ? 0 : (uint8_t)(((i / 10) << 4) | (i % 10));
}

static volatile int global_best_err;
static volatile int found_exact;

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

typedef struct {
    int thread_id;
    int len;
    int poolSize;
    const uint8_t *opMap;
    const uint8_t *target;
} ThreadArg;

static void *search_thread(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;
    int len = ta->len, poolSize = ta->poolSize;
    uint64_t total = ipow(poolSize, len);
    uint64_t chunk = (total + NUM_THREADS - 1) / NUM_THREADS;
    uint64_t start = (uint64_t)ta->thread_id * chunk;
    uint64_t end = start + chunk;
    if (end > total) end = total;

    uint8_t ops[24];
    int best_local = global_best_err;

    for (uint64_t idx = start; idx < end && !found_exact; idx++) {
        uint64_t tmp = idx;
        for (int i = len - 1; i >= 0; i--) { ops[i] = tmp % poolSize; tmp /= poolSize; }

        // Quick check
        uint8_t qc[] = {0, 1, 64, 128, 255};
        int max_err = 0, reject = 0;
        for (int q = 0; q < 5; q++) {
            int e = (int)run_seq(ops, ta->opMap, len, qc[q]) - (int)ta->target[qc[q]];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            if (max_err >= best_local) { reject = 1; break; }
        }
        if (reject) continue;

        // Full verify
        max_err = 0;
        for (int i = 0; i < 256; i++) {
            int e = (int)run_seq(ops, ta->opMap, len, (uint8_t)i) - (int)ta->target[i];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            if (max_err >= best_local) break;
        }

        if (max_err < best_local) {
            best_local = max_err;
            if (max_err < global_best_err) global_best_err = max_err;
            best_local = global_best_err;

            printf("%-14s len=%d err=%d:", "target", len, max_err);
            for (int i = 0; i < len; i++) printf(" %s", allOpNames[ta->opMap[ops[i]]]);
            if (max_err == 0) { printf(" [EXACT]\n"); found_exact = 1; }
            else printf(" [approx, max_err=%d]\n", max_err);
            fflush(stdout);
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    // Target 0: log2_f3.5 with 5 ops
    // Target 1: bin2bcd with 5 ops
    Target tgts[2];

    tgts[0].name = "log2_f3.5";
    tgts[0].pool[0] = 0; tgts[0].pool[1] = 4; tgts[0].pool[2] = 3;
    tgts[0].pool[3] = 5; tgts[0].pool[4] = 9;  // MUL3 NEG SHR SAVE XOR_B
    tgts[0].poolSize = 5; tgts[0].maxDepth = 22;
    gen_log2_f3_5(tgts[0].target);

    tgts[1].name = "bin2bcd";
    tgts[1].pool[0] = 8; tgts[1].pool[1] = 0; tgts[1].pool[2] = 1;
    tgts[1].pool[3] = 7; tgts[1].pool[4] = 3;  // AND_0F MUL3 MUL5 SBC_MASK SHR
    tgts[1].poolSize = 5; tgts[1].maxDepth = 22;
    gen_bin2bcd(tgts[1].target);

    for (int t = 0; t < 2; t++) {
        Target *tgt = &tgts[t];
        global_best_err = 255;
        found_exact = 0;

        printf("\n=== %s: %d ops, max depth %d ===\n", tgt->name, tgt->poolSize, tgt->maxDepth);
        printf("Pool:");
        for (int i = 0; i < tgt->poolSize; i++) printf(" %s", allOpNames[tgt->pool[i]]);
        printf("\n");
        fprintf(stderr, "Starting %s...\n", tgt->name);

        for (int len = 1; len <= tgt->maxDepth && !found_exact; len++) {
            uint64_t total = ipow(tgt->poolSize, len);
            time_t t0 = time(NULL);
            fprintf(stderr, "  len=%d: %.2e candidates\n", len, (double)total);

            pthread_t threads[NUM_THREADS];
            ThreadArg args[NUM_THREADS];
            for (int i = 0; i < NUM_THREADS; i++) {
                args[i] = (ThreadArg){i, len, tgt->poolSize, tgt->pool, tgt->target};
                pthread_create(&threads[i], NULL, search_thread, &args[i]);
            }
            for (int i = 0; i < NUM_THREADS; i++)
                pthread_join(threads[i], NULL);

            time_t t1 = time(NULL);
            fprintf(stderr, "    done (%lds)\n", (long)(t1 - t0));
        }

        if (found_exact)
            fprintf(stderr, ">>> EXACT for %s!\n", tgt->name);
        else
            fprintf(stderr, ">>> Max depth for %s, best err=%d\n", tgt->name, global_best_err);
    }
    return 0;
}

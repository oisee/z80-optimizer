// Minimal gray_decode brute-force — 5-op pool, single target, CPU
// Build: gcc -O3 -march=native -o z80_graydec_mini z80_graydec_mini.c -lpthread
// Usage: ./z80_graydec_mini [max-len]  (default 18)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define NUM_OPS 5
#define NUM_TEST 256
#define NUM_THREADS 4

// Gray decode: x ^= x>>1; x ^= x>>2; x ^= x>>4
static uint8_t gray_decode_ref[NUM_TEST];

static void gen_target(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t x = (uint8_t)i;
        x ^= (x >> 1);
        x ^= (x >> 2);
        x ^= (x >> 4);
        gray_decode_ref[i] = x;
    }
}

static inline uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    for (int i = 0; i < len; i++) {
        switch (ops[i]) {
        case 0: b = a; break;                                    // SAVE
        case 1: a = a >> 1; break;                               // SHR
        case 2: a ^= b; break;                                   // XOR_B
        case 3: { uint8_t t = (a << 1) | (a >> 7); a = t; } break; // RLCA
        case 4: { uint8_t t = (a >> 1) | (a << 7); a = t; } break; // RRCA
        }
    }
    return a;
}

static const char *opNames[] = {"SAVE", "SHR", "XOR_B", "RLCA", "RRCA"};

static volatile int global_best_err = 255;
static volatile int found_exact = 0;
static int max_len_global = 18;

typedef struct { int thread_id; } thread_arg_t;

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

static void search_len(int len, int thread_id, int num_threads) {
    uint64_t total = ipow(NUM_OPS, len);
    uint64_t chunk = (total + num_threads - 1) / num_threads;
    uint64_t start = (uint64_t)thread_id * chunk;
    uint64_t end = start + chunk;
    if (end > total) end = total;

    uint8_t ops[24];
    int best_err_local = global_best_err;

    for (uint64_t idx = start; idx < end; idx++) {
        if (found_exact) return;

        // Decode index to ops
        uint64_t tmp = idx;
        for (int i = len - 1; i >= 0; i--) {
            ops[i] = tmp % NUM_OPS;
            tmp /= NUM_OPS;
        }

        // Quick check: inputs 0, 1, 128, 255
        uint8_t qc[] = {0, 1, 128, 255};
        int max_err = 0;
        int reject = 0;
        for (int q = 0; q < 4; q++) {
            int e = (int)run_seq(ops, len, qc[q]) - (int)gray_decode_ref[qc[q]];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            if (max_err >= best_err_local) { reject = 1; break; }
        }
        if (reject) continue;

        // Full verify
        max_err = 0;
        for (int i = 0; i < 256; i++) {
            int e = (int)run_seq(ops, len, (uint8_t)i) - (int)gray_decode_ref[i];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
            if (max_err >= best_err_local) break;
        }

        if (max_err < best_err_local) {
            best_err_local = max_err;
            // Atomically update global
            __sync_val_compare_and_swap(&global_best_err, global_best_err, max_err);
            // Better: just store if we improved
            if (max_err < global_best_err) global_best_err = max_err;
            best_err_local = global_best_err;

            printf("gray_dec  len=%d err=%d:", len, max_err);
            for (int i = 0; i < len; i++) printf(" %s", opNames[ops[i]]);
            if (max_err == 0) { printf(" [EXACT]\n"); found_exact = 1; }
            else printf(" [approx, max_err=%d]\n", max_err);
            fflush(stdout);
        }
    }
}

static void *thread_func(void *arg) {
    thread_arg_t *ta = (thread_arg_t *)arg;
    for (int len = 1; len <= max_len_global; len++) {
        if (found_exact) break;
        search_len(len, ta->thread_id, NUM_THREADS);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc > 1) max_len_global = atoi(argv[1]);

    gen_target();

    // Verify target
    printf("Gray decode target: [0]=%d [1]=%d [2]=%d [128]=%d [255]=%d\n",
           gray_decode_ref[0], gray_decode_ref[1], gray_decode_ref[2],
           gray_decode_ref[128], gray_decode_ref[255]);
    printf("5-op pool: SAVE SHR XOR_B RLCA RRCA\n");
    printf("Max depth: %d (5^%d = %.2e)\n", max_len_global, max_len_global,
           (double)ipow(NUM_OPS, max_len_global));
    fprintf(stderr, "Searching with %d threads...\n", NUM_THREADS);

    pthread_t threads[NUM_THREADS];
    thread_arg_t args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        pthread_create(&threads[i], NULL, thread_func, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    if (!found_exact)
        printf("\nBest error: %d (not exact)\n", global_best_err);

    return found_exact ? 0 : 1;
}

/*
 * bb_search.cu — GPU port of Introspec's BB brute-force search
 *
 * Exact reimplementation of the algorithm from bb_brute_search_2.zip:
 *   - 24-bit Galois LFSR (8-bit high + 16-bit low), polynomial 0xDB
 *   - 2×2 pixel XOR plots on ZX Spectrum screen (256×192)
 *   - 66 layers, layer N draws N×2 random points
 *   - 3 weighted masks for fitness (face region gets 4× weight)
 *   - High byte of LFSR carries between layers
 *   - Each layer: brute-force all 65536 16-bit seeds
 *
 * Build: nvcc -O3 -o cuda/bb_search cuda/bb_search.cu
 * Usage: ./cuda/bb_search --target image.scr [--mask0 m0.scr] [--mask1 m1.scr] [--mask2 m2.scr]
 *        ./cuda/bb_search --target-pgm image.pgm  (auto-converts 128x96 to 256x192)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define SCR_SIZE 6144     /* ZX Spectrum screen: 256x192, 1 bit/pixel */
#define NLAYERS 66
#define POINTS_MULT 2     /* points per layer = layer_num * POINTS_MULT */

/* ====== 24-bit Galois LFSR (exact copy from Introspec) ====== */
/* State: FG24a (8-bit high) + FG24b (16-bit low) */
/* Output: mixed bits from FG24b */

struct LFSRState {
    uint8_t a;    /* high 8 bits */
    uint16_t b;   /* low 16 bits */
};

__device__ __host__ uint8_t lfsr24_next(LFSRState* st) {
    uint32_t t = (((uint32_t)st->a << 16) + st->b) << 1;
    st->a = (t >> 16) & 0xFF;
    uint32_t s = (t >> 24) & 1;
    /* Polynomial feedback: if top bit was set, XOR with 0xDB */
    uint16_t mask = (uint16_t)(-(int16_t)s) & 0x00DB;
    st->b = (t & 0xFFFF) ^ mask;
    return ((st->b & 0xAA) + ((st->b >> 8) & 0xFF)) & 0xFF;
}

__device__ __host__ void lfsr24_seed(LFSRState* st, uint16_t seed16, uint8_t prev_a) {
    st->a = prev_a;
    st->b = seed16;
}

/* ====== Spectrum screen address (interleaved) ====== */
__device__ __host__ int line2addr(int line) {
    int zz  = line & 0xC0;
    int xxx = line & 0x38;
    int nnn = line & 0x07;
    return (zz + (nnn << 3) + (xxx >> 3)) << 5;
}

/* ====== Plot 2×2 XOR point (exact copy from Introspec) ====== */
__device__ void plot2x2(uint8_t* scr, int x, int y) {
    /* 2×2 pixel block: pairs of bits */
    static const uint8_t pnts[4] = {0xC0, 0x30, 0x0C, 0x03};
    int byte_col = x >> 3;
    int bit_pair = (x & 0x06) >> 1;
    uint8_t mask = pnts[bit_pair];
    int addr0 = line2addr(y & 0xFE) + byte_col;
    int addr1 = line2addr((y & 0xFE) + 1) + byte_col;
    if (addr0 >= 0 && addr0 < SCR_SIZE) scr[addr0] ^= mask;
    if (addr1 >= 0 && addr1 < SCR_SIZE) scr[addr1] ^= mask;
}

/* Host version */
void host_plot2x2(uint8_t* scr, int x, int y) {
    static const uint8_t pnts[4] = {0xC0, 0x30, 0x0C, 0x03};
    int byte_col = x >> 3;
    int bit_pair = (x & 0x06) >> 1;
    uint8_t mask = pnts[bit_pair];
    int addr0 = line2addr(y & 0xFE) + byte_col;
    int addr1 = line2addr((y & 0xFE) + 1) + byte_col;
    if (addr0 >= 0 && addr0 < SCR_SIZE) scr[addr0] ^= mask;
    if (addr1 >= 0 && addr1 < SCR_SIZE) scr[addr1] ^= mask;
}

/* ====== Draw N random 2×2 XOR points ====== */
__device__ void draw_rndpoints(uint8_t* scr, LFSRState* st, int npoints) {
    for (int i = 0; i < npoints; i++) {
        int x = lfsr24_next(st);
        int y = lfsr24_next(st);
        if (y < 192) plot2x2(scr, x, y);
    }
}

void host_draw_rndpoints(uint8_t* scr, LFSRState* st, int npoints) {
    for (int i = 0; i < npoints; i++) {
        int x = lfsr24_next(st);
        int y = lfsr24_next(st);
        if (y < 192) host_plot2x2(scr, x, y);
    }
}

/* ====== Kernel: test all 65536 seeds for one layer ====== */
__global__ void search_layer_kernel(
    const uint8_t* __restrict__ prev_screen,  /* accumulated XOR of all previous layers */
    const uint8_t* __restrict__ target,
    const uint8_t* __restrict__ mask0,
    const uint8_t* __restrict__ mask1,
    const uint8_t* __restrict__ mask2,
    uint32_t* __restrict__ errors,
    uint8_t* __restrict__ out_last_a,    /* carry: last LFSR high byte per seed */
    uint8_t prev_a,                      /* LFSR high byte from previous layer */
    int npoints                          /* points for this layer */
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= 65536) return;

    /* Copy previous screen to local buffer */
    uint8_t scr[SCR_SIZE];
    for (int i = 0; i < SCR_SIZE; i++)
        scr[i] = prev_screen[i];

    /* Init LFSR and draw points */
    LFSRState st;
    lfsr24_seed(&st, (uint16_t)seed, prev_a);
    draw_rndpoints(scr, &st, npoints);

    /* Compute weighted error (exact Introspec formula) */
    /* diff = error0 + error1 + error2 + error2 */
    uint32_t err0 = 0, err1 = 0, err2 = 0;
    for (int i = 0; i < SCR_SIZE / 4; i++) {
        uint32_t diff = ((uint32_t*)scr)[i] ^ ((uint32_t*)target)[i];
        diff &= ((uint32_t*)mask0)[i];
        err0 += __popc(diff);
        err1 += __popc(diff & ((uint32_t*)mask1)[i]);
        err2 += __popc(diff & ((uint32_t*)mask2)[i]);
    }

    errors[seed] = err0 + err1 + err2 + err2;
    out_last_a[seed] = st.a;
}

/* ====== I/O ====== */
int load_scr(const char* path, uint8_t* buf) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < SCR_SIZE) { fclose(f); return -2; }
    fread(buf, 1, SCR_SIZE, f);
    fclose(f);
    return 0;
}

void save_scr(const char* path, const uint8_t* buf) {
    FILE* f = fopen(path, "wb");
    fwrite(buf, 1, SCR_SIZE, f);
    /* Pad to 6912 with attributes (white on black) */
    for (int i = 0; i < 768; i++) {
        uint8_t attr = 0x38;
        fwrite(&attr, 1, 1, f);
    }
    fclose(f);
}

/* Convert PGM 128x96 to Spectrum .scr format (2x scale) */
int pgm_to_scr(const char* pgm_path, uint8_t* scr) {
    FILE* f = fopen(pgm_path, "rb");
    if (!f) return -1;
    char magic[4]; int w, h, maxval;
    fscanf(f, "%2s", magic);
    int c; while ((c=fgetc(f))!=EOF) { if (c=='#') while((c=fgetc(f))!=EOF&&c!='\n'); else if(c>' ') {ungetc(c,f);break;} }
    fscanf(f, "%d %d %d", &w, &h, &maxval); fgetc(f);
    if (w != 128 || h != 96) { fclose(f); return -2; }
    uint8_t* raw = (uint8_t*)malloc(w * h);
    fread(raw, 1, w * h, f); fclose(f);

    memset(scr, 0, SCR_SIZE);
    for (int y = 0; y < 96; y++) {
        for (int x = 0; x < 128; x++) {
            if (raw[y * 128 + x] > maxval / 2) {
                /* Plot 2×2 at (x*2, y*2) — but actually map to Spectrum addr */
                for (int dy = 0; dy < 2; dy++) {
                    int sy = y * 2 + dy;
                    int sx = x; /* NOT *2 — Spectrum is 256 wide, our image maps to left 128 */
                    /* Actually: for 128x96→256x192, double both */
                    sx = x * 2;
                    int addr = line2addr(sy) + (sx >> 3);
                    /* Set 2 bits for 2x horizontal */
                    int bit = 6 - (sx & 6);
                    if (addr < SCR_SIZE) {
                        scr[addr] |= (0xC0 >> (sx & 6));
                    }
                }
            }
        }
    }
    free(raw);
    return 0;
}

void scr_to_pgm(const char* path, const uint8_t* scr) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n256 192\n255\n");
    for (int y = 0; y < 192; y++) {
        int addr = line2addr(y);
        for (int byte_col = 0; byte_col < 32; byte_col++) {
            uint8_t bv = scr[addr + byte_col];
            for (int bit = 0; bit < 8; bit++) {
                uint8_t v = (bv & (0x80 >> bit)) ? 255 : 0;
                fwrite(&v, 1, 1, f);
            }
        }
    }
    fclose(f);
}

/* ====== Main ====== */
int main(int argc, char** argv) {
    int device_id = 0;
    const char* target_path = NULL;
    const char* target_pgm = NULL;
    const char* mask0_path = NULL;
    const char* mask1_path = NULL;
    const char* mask2_path = NULL;
    const char* output_dir = "media/prng_images/bb_search";
    int s0_start = 0, s0_end = 256;  /* search range for initial high byte */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1<argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--target-pgm") && i+1<argc) target_pgm = argv[++i];
        else if (!strcmp(argv[i], "--mask0") && i+1<argc) mask0_path = argv[++i];
        else if (!strcmp(argv[i], "--mask1") && i+1<argc) mask1_path = argv[++i];
        else if (!strcmp(argv[i], "--mask2") && i+1<argc) mask2_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i+1<argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1<argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--s0") && i+1<argc) s0_start = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--s0-end") && i+1<argc) s0_end = atoi(argv[++i]);
    }

    if (!target_path && !target_pgm) {
        fprintf(stderr, "Usage: %s --target image.scr [--mask0/1/2 mask.scr]\n"
                        "       %s --target-pgm image.pgm\n", argv[0], argv[0]);
        return 1;
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU: %s\n", prop.name);

    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir); system(cmd);

    /* Load target */
    uint8_t h_target[SCR_SIZE];
    if (target_pgm) {
        if (pgm_to_scr(target_pgm, h_target) != 0) {
            fprintf(stderr, "Failed to load PGM %s\n", target_pgm); return 1;
        }
        printf("Target: %s (PGM→SCR)\n", target_pgm);
    } else {
        if (load_scr(target_path, h_target) != 0) {
            fprintf(stderr, "Failed to load %s\n", target_path); return 1;
        }
        printf("Target: %s\n", target_path);
    }

    /* Load or generate masks */
    uint8_t h_mask0[SCR_SIZE], h_mask1[SCR_SIZE], h_mask2[SCR_SIZE];
    if (mask0_path && load_scr(mask0_path, h_mask0) == 0) {
        printf("Mask0: %s\n", mask0_path);
    } else {
        memset(h_mask0, 0xFF, SCR_SIZE);  /* all pixels count */
    }
    if (mask1_path && load_scr(mask1_path, h_mask1) == 0) {
        printf("Mask1: %s\n", mask1_path);
    } else {
        memset(h_mask1, 0xFF, SCR_SIZE);
    }
    if (mask2_path && load_scr(mask2_path, h_mask2) == 0) {
        printf("Mask2: %s\n", mask2_path);
    } else {
        memcpy(h_mask2, h_mask1, SCR_SIZE);
    }

    /* GPU allocations */
    uint8_t *d_screen, *d_target, *d_mask0, *d_mask1, *d_mask2, *d_last_a;
    uint32_t *d_errors;
    cudaMalloc(&d_screen, SCR_SIZE);
    cudaMalloc(&d_target, SCR_SIZE);
    cudaMalloc(&d_mask0, SCR_SIZE);
    cudaMalloc(&d_mask1, SCR_SIZE);
    cudaMalloc(&d_mask2, SCR_SIZE);
    cudaMalloc(&d_errors, 65536 * sizeof(uint32_t));
    cudaMalloc(&d_last_a, 65536);

    cudaMemcpy(d_target, h_target, SCR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask0, h_mask0, SCR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask1, h_mask1, SCR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask2, h_mask2, SCR_SIZE, cudaMemcpyHostToDevice);

    uint32_t h_errors[65536];
    uint8_t h_last_a[65536];

    printf("Layers: %d, Points multiplier: %d\n", NLAYERS, POINTS_MULT);
    printf("Search s0: %d..%d\n", s0_start, s0_end);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint32_t global_best_score = 0xFFFFFFFF;
    uint16_t global_best_seeds[NLAYERS];
    uint8_t global_best_s0 = 0;

    for (int s0 = s0_start; s0 < s0_end; s0++) {
        uint8_t h_screen[SCR_SIZE];
        memset(h_screen, 0, SCR_SIZE);

        uint16_t best_seeds[NLAYERS];
        uint8_t prev_a = (uint8_t)s0;

        for (int n = NLAYERS; n > 0; n--) {
            int layer_idx = NLAYERS - n;
            int npoints = n * POINTS_MULT;

            cudaMemcpy(d_screen, h_screen, SCR_SIZE, cudaMemcpyHostToDevice);

            search_layer_kernel<<<256, 256>>>(
                d_screen, d_target, d_mask0, d_mask1, d_mask2,
                d_errors, d_last_a, prev_a, npoints);
            cudaDeviceSynchronize();

            cudaMemcpy(h_errors, d_errors, 65536 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_last_a, d_last_a, 65536, cudaMemcpyDeviceToHost);

            /* Find best seed */
            uint32_t best_err = 0xFFFFFFFF;
            uint16_t best_seed = 0;
            for (int k = 0; k < 65536; k++) {
                if (h_errors[k] < best_err) {
                    best_err = h_errors[k];
                    best_seed = (uint16_t)k;
                }
            }

            /* Apply best seed to host screen */
            LFSRState st;
            lfsr24_seed(&st, best_seed, prev_a);
            host_draw_rndpoints(h_screen, &st, npoints);
            prev_a = st.a;
            best_seeds[layer_idx] = best_seed;

            if (layer_idx % 8 == 0 || n == 1) {
                printf("  s0=%d L%02d: n=%d pts=%d seed=0x%04X err=%d\n",
                       s0, layer_idx, n, npoints, best_seed, best_err);
            }
        }

        /* Final score */
        uint32_t final_score = 0;
        for (int i = 0; i < SCR_SIZE / 4; i++) {
            uint32_t diff = ((uint32_t*)h_screen)[i] ^ ((uint32_t*)h_target)[i];
            diff &= ((uint32_t*)h_mask0)[i];
            final_score += __builtin_popcount(diff);
        }

        printf("s0=%d: final weighted error = score, raw pixel diff = %d\n", s0, final_score);

        if (final_score < global_best_score) {
            global_best_score = final_score;
            memcpy(global_best_seeds, best_seeds, sizeof(best_seeds));
            global_best_s0 = (uint8_t)s0;

            /* Save checkpoint */
            char path[512];
            snprintf(path, sizeof(path), "%s/s0_%03d_result.scr", output_dir, s0);
            save_scr(path, h_screen);
            snprintf(path, sizeof(path), "%s/s0_%03d_result.pgm", output_dir, s0);
            scr_to_pgm(path, h_screen);
            printf("  → NEW BEST! Saved s0=%d\n", s0);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n=== RESULT ===\n");
    printf("Best s0: %d, error: %d\n", global_best_s0, global_best_score);
    printf("Time: %.1fs\n", elapsed);

    /* Save seed table */
    char path[512];
    snprintf(path, sizeof(path), "%s/best_seeds.txt", output_dir);
    FILE* f = fopen(path, "w");
    fprintf(f, "; BB-style search result, s0=%d\n", global_best_s0);
    fprintf(f, "; %d layers, points_mult=%d\n", NLAYERS, POINTS_MULT);
    for (int i = 0; i < NLAYERS; i++)
        fprintf(f, "layer %2d: 0x%04X  (n=%d pts=%d)\n",
                i, global_best_seeds[i], NLAYERS - i, (NLAYERS - i) * POINTS_MULT);
    fclose(f);
    printf("Seeds: %s\n", path);
    printf("Output: %s/\n", output_dir);

    cudaFree(d_screen); cudaFree(d_target);
    cudaFree(d_mask0); cudaFree(d_mask1); cudaFree(d_mask2);
    cudaFree(d_errors); cudaFree(d_last_a);
    return 0;
}

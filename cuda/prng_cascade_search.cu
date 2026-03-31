/*
 * prng_cascade_search.cu — LFSR-16 AND-cascade image search (GPU)
 *
 * Direct port of /tmp/buf_foveal_cascade.go to CUDA.
 * Algorithm:
 *   - LFSR-16, poly 0xB400 (maximal-length, 65535 states)
 *   - makeBuf: 768 blocks = 32×24 grid; each block = AND of andN consecutive LFSR bits
 *   - applyBuf: XOR flips at (ox, oy, blk) offset
 *   - Greedy: each step finds best (seed, posIdx) over all 65535 seeds × all positions
 *   - Position search: ox/oy snapped to GRID=8px, patch must fit in 128×96 canvas
 *
 * Cascade phase schedule (same as Go version):
 *   L0: AND3, blk=4, 1 seed    (1 position: full screen)
 *   L1: AND3, blk=2, 8 seeds   (63 positions)
 *   L2: AND4, blk=1, 16 seeds  (130 positions)
 *   L3: AND5, blk=1, 128 seeds
 *   L4: AND6, blk=1, 256 seeds
 *   L5: AND7, blk=1, 800 seeds
 *
 * Build:
 *   nvcc -O3 -o cuda/prng_cascade_search cuda/prng_cascade_search.cu -lm
 *
 * Usage:
 *   ./cuda/prng_cascade_search --target <file.pgm> [--gpu 0] [--out seeds.json]
 *
 * Verification:
 *   Run Go version and CUDA version on same target; compare seed lists.
 *   Both should produce identical (seed, ox, oy) at each greedy step.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define W       128
#define H        96
#define PS      (W * H / 8)  /* 1536 packed bytes */
#define GRID      8           /* position snap pixels */
#define BUF_N   768           /* 32×24 blocks */

/* ====================================================================
 * LFSR-16, poly 0xB400 (same as Go: bit = s&1; s>>=1; if bit: s^=0xB400)
 * ==================================================================== */
__device__ __host__ __forceinline__ uint16_t lfsr16(uint16_t s) {
    uint16_t bit = s & 1u;
    s >>= 1;
    if (bit) s ^= 0xB400u;
    return s;
}

/* ====================================================================
 * makeBuf: generate 768-element buffer for given seed/warmup/andN
 * buf[i] = AND of andN consecutive LFSR bits (1 if all 1, else 0)
 * Sparsity: P(buf[i]=1) = 0.5^andN
 * ==================================================================== */
__device__ void makeBuf(uint16_t seed, int warmup, int andN, uint8_t buf[BUF_N]) {
    uint16_t s = seed ? seed : 1u;
    for (int i = 0; i < warmup; i++) s = lfsr16(s);
    for (int i = 0; i < BUF_N; i++) {
        uint16_t acc = 1u;
        for (int k = 0; k < andN; k++) {
            s = lfsr16(s);
            acc &= (s & 1u);
        }
        buf[i] = (uint8_t)acc;
    }
}

/* ====================================================================
 * Position table: all (ox, oy) snapped to GRID for given blk.
 * Returns count, fills ox_out[]/oy_out[] (host).
 * ==================================================================== */
int buildPositions(int blk, int ox_out[512], int oy_out[512]) {
    int pw = 32 * blk;
    int ph = 24 * blk;
    int n = 0;
    for (int oy = 0; oy + ph <= H; oy += GRID)
        for (int ox = 0; ox + pw <= W; ox += GRID) {
            ox_out[n] = ox;
            oy_out[n] = oy;
            n++;
        }
    return n;
}

/* ====================================================================
 * computeBaseErrors (host): precompute error count inside each patch
 * region given current canvas vs target.
 * ==================================================================== */
void computeBaseErrors(const uint8_t *canvas, const uint8_t *target,
                       int npos, const int *ox_arr, const int *oy_arr,
                       int blk, int *baseErr) {
    int pw = 32 * blk;
    int ph = 24 * blk;
    for (int pi = 0; pi < npos; pi++) {
        int ox = ox_arr[pi], oy = oy_arr[pi];
        int err = 0;
        for (int y = oy; y < oy + ph && y < H; y++) {
            for (int x = ox; x < ox + pw && x < W; x++) {
                uint8_t cb = (canvas[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
                uint8_t tb = (target[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
                err += (cb != tb);
            }
        }
        baseErr[pi] = err;
    }
}

/* ====================================================================
 * CUDA kernel: one thread per seed (1..65535)
 * Each thread:
 *   1. Generates buf[768] for its seed
 *   2. For each position: computes delta (flip gain) using baseErr
 *   3. Tracks best (seed, posIdx, newErr)
 *   4. Writes result to out_seed[threadId], out_pos[threadId], out_err[threadId]
 *
 * baseErr[npos], ox_d[npos], oy_d[npos] — device arrays
 * canvas_d[PS], target_d[PS] — packed bit images
 * ==================================================================== */
__global__ void searchKernel(
    const uint8_t * __restrict__ canvas_d,
    const uint8_t * __restrict__ target_d,
    const int     * __restrict__ baseErr_d,
    const int     * __restrict__ ox_d,
    const int     * __restrict__ oy_d,
    int npos,
    int blk,
    int andN,
    int warmup,
    uint16_t * __restrict__ out_seed,
    int      * __restrict__ out_pos,
    int      * __restrict__ out_err
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = tid + 1;  /* seeds 1..65535 */
    if (seed > 65535) return;

    /* Generate buffer in registers/local memory */
    uint8_t buf[BUF_N];
    makeBuf((uint16_t)seed, warmup, andN, buf);

    int bestPos = 0, bestDelta = 0x7fffffff;

    for (int pi = 0; pi < npos; pi++) {
        int ox = ox_d[pi], oy = oy_d[pi];
        int delta = 0;
        for (int by = 0; by < 24; by++) {
            for (int bx = 0; bx < 32; bx++) {
                if (!buf[by*32+bx]) continue;
                for (int dy = 0; dy < blk; dy++) {
                    for (int dx = 0; dx < blk; dx++) {
                        int x = ox + bx*blk + dx;
                        int y = oy + by*blk + dy;
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int bidx = y*(W/8) + x/8;
                            int bbit = 7 - (x%8);
                            uint8_t cb = (canvas_d[bidx] >> bbit) & 1u;
                            uint8_t tb = (target_d[bidx]  >> bbit) & 1u;
                            delta += (cb == tb) ? 1 : -1;  /* flip hurts / helps */
                        }
                    }
                }
            }
        }
        /* Foveal: minimize delta (= maximize improvement), not absolute newErr.
         * This ensures we pick the position with the most error-dense region,
         * not the already-well-optimized region that happens to have low baseErr. */
        if (delta < bestDelta) {
            bestDelta = delta;
            bestPos   = pi;
        }
    }

    out_seed[tid] = (uint16_t)seed;
    out_pos[tid]  = bestPos;
    out_err[tid]  = bestDelta;  /* now stores best delta for this seed */
}

/* ====================================================================
 * applyBuf (host): XOR canvas with buf at (ox, oy, blk)
 * ==================================================================== */
void applyBuf(uint8_t *canvas, const uint8_t buf[BUF_N], int ox, int oy, int blk) {
    for (int by = 0; by < 24; by++) {
        for (int bx = 0; bx < 32; bx++) {
            if (!buf[by*32+bx]) continue;
            for (int dy = 0; dy < blk; dy++) {
                for (int dx = 0; dx < blk; dx++) {
                    int x = ox + bx*blk + dx;
                    int y = oy + by*blk + dy;
                    if (x >= 0 && x < W && y >= 0 && y < H)
                        canvas[y*(W/8)+x/8] ^= (1u << (7-(x%8)));
                }
            }
        }
    }
}

/* ====================================================================
 * lBin: binary error rate
 * ==================================================================== */
double lBin(const uint8_t *canvas, const uint8_t *target) {
    int err = 0;
    for (int i = 0; i < PS; i++) err += __builtin_popcount(canvas[i] ^ target[i]);
    return (double)err / (W * H);
}

/* ====================================================================
 * PGM loader (P5 binary, 8-bit gray)
 * ==================================================================== */
int loadPGM(const char *path, uint8_t *target_bin, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return -1; }
    int w, h, maxval;
    char magic[8];
    fscanf(f, "%7s %d %d %d", magic, &w, &h, &maxval);
    fgetc(f);  /* consume newline after header */
    if (strcmp(magic, "P5") != 0) { fclose(f); fprintf(stderr, "Not P5 PGM\n"); return -1; }
    *out_w = w; *out_h = h;
    int npix = w * h;
    uint8_t *raw = (uint8_t*)malloc(npix);
    if ((int)fread(raw, 1, npix, f) != npix) { free(raw); fclose(f); return -1; }
    fclose(f);
    /* binarize and pack */
    memset(target_bin, 0, PS);
    for (int i = 0; i < w*h; i++) {
        if (raw[i] > maxval/2)
            target_bin[i/8] |= (1u << (7-(i%8)));
    }
    free(raw);
    return 0;
}

/* ====================================================================
 * PGM saver (canvas → grayscale)
 * ==================================================================== */
void savePGM(const char *path, const uint8_t *canvas) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t b = (canvas[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
            fputc(b ? 255 : 0, f);
        }
    }
    fclose(f);
}

/* ====================================================================
 * JSON seed log
 * ==================================================================== */
typedef struct {
    int    step;
    uint16_t seed;
    int    ox, oy, blk, and_n, warmup;
    char   label[32];
} SeedRecord;

void writeJSON(const char *path, SeedRecord *recs, int nrecs) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n  \"lfsr16_poly\": \"0xB400\",\n  \"canvas_w\": %d,\n  \"canvas_h\": %d,\n", W, H);
    fprintf(f, "  \"position_grid\": %d,\n  \"seeds\": [\n", GRID);
    for (int i = 0; i < nrecs; i++) {
        SeedRecord *r = &recs[i];
        fprintf(f, "    {\"step\":%d,\"seed\":%u,\"ox\":%d,\"oy\":%d,\"blk\":%d,\"and_n\":%d,\"warmup\":%d,\"label\":\"%s\"}%s\n",
            r->step, r->seed, r->ox, r->oy, r->blk, r->and_n, r->warmup, r->label,
            (i < nrecs-1) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("Seeds written: %s (%d entries)\n", path, nrecs);
}

/* ====================================================================
 * Phase descriptor
 * ==================================================================== */
typedef struct { int andN, blk, count; const char *label; } Phase;

static Phase phases[] = {
    {3, 4,   1, "L0-AND3"},
    {3, 2,   8, "L1-AND3"},
    {4, 1,  16, "L2-AND4"},
    {5, 1, 128, "L3-AND5"},
    {6, 1, 256, "L4-AND6"},
    {7, 1, 800, "L5-AND7"},
};
static int nphases = 6;

/* ====================================================================
 * main
 * ==================================================================== */
int main(int argc, char **argv) {
    const char *target_path      = NULL;
    const char *out_json         = "/tmp/cuda_cascade_seeds.json";
    const char *init_canvas_path = NULL;   /* --init-canvas: start from existing canvas */
    int first_phase = 0;                   /* --phase-from N: skip phases 0..N-1 */
    int gpu_id = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target")       && i+1 < argc) target_path      = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) out_json         = argv[++i];
        else if (!strcmp(argv[i], "--gpu")     && i+1 < argc) gpu_id           = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--init-canvas") && i+1 < argc) init_canvas_path = argv[++i];
        else if (!strcmp(argv[i], "--phase-from")  && i+1 < argc) first_phase  = atoi(argv[++i]);
    }
    if (!target_path) {
        target_path = "/home/alice/dev/z80-optimizer/media/prng_images/targets/che.pgm";
    }

    cudaSetDevice(gpu_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("GPU %d: %s\n", gpu_id, prop.name);

    /* Load target */
    uint8_t target_bin[PS];
    int tw, th;
    if (loadPGM(target_path, target_bin, &tw, &th) < 0) return 1;
    if (tw != W || th != H) { fprintf(stderr, "Size mismatch: got %dx%d, want %dx%d\n", tw, th, W, H); return 1; }
    printf("Target: %s (%dx%d)\n", target_path, W, H);

    /* Phase summary */
    int total_steps = 0;
    for (int p = 0; p < nphases; p++) total_steps += phases[p].count;
    printf("Cascade: %d total steps\n\n", total_steps);

    /* Allocate device memory for canvas and target */
    uint8_t *canvas_d, *target_d;
    cudaMalloc(&canvas_d, PS);
    cudaMalloc(&target_d, PS);
    cudaMemset(canvas_d, 0, PS);
    cudaMemcpy(target_d, target_bin, PS, cudaMemcpyHostToDevice);

    /* Device arrays for per-seed results */
    const int NSEEDS = 65535;
    uint16_t *out_seed_d; int *out_pos_d, *out_err_d;
    cudaMalloc(&out_seed_d, NSEEDS * sizeof(uint16_t));
    cudaMalloc(&out_pos_d,  NSEEDS * sizeof(int));
    cudaMalloc(&out_err_d,  NSEEDS * sizeof(int));

    /* Host mirrors */
    uint16_t out_seed_h[NSEEDS];
    int      out_pos_h[NSEEDS];
    int      out_err_h[NSEEDS];

    /* Position arrays (max 512) */
    int  pos_ox_h[512], pos_oy_h[512];
    int *pos_ox_d, *pos_oy_d, *baseErr_d;
    cudaMalloc(&pos_ox_d,   512 * sizeof(int));
    cudaMalloc(&pos_oy_d,   512 * sizeof(int));
    cudaMalloc(&baseErr_d,  512 * sizeof(int));

    uint8_t canvas_h[PS];
    memset(canvas_h, 0, PS);
    /* --init-canvas: load existing canvas as starting point for delta mode */
    if (init_canvas_path) {
        int iw, ih;
        if (loadPGM(init_canvas_path, canvas_h, &iw, &ih) == 0 && iw == W && ih == H) {
            printf("Init canvas: %s (delta mode, err=%.2f%%)\n",
                   init_canvas_path, lBin(canvas_h, target_bin)*100);
        } else {
            fprintf(stderr, "Warning: could not load --init-canvas %s, starting blank\n", init_canvas_path);
            memset(canvas_h, 0, PS);
        }
    }

    SeedRecord *seedLog = (SeedRecord*)malloc(total_steps * sizeof(SeedRecord));
    int nlog = 0;

    printf("%-6s  %-8s  %-8s  %-12s  %s\n", "step", "L_bin", "elapsed", "pos", "label");
    printf("------  --------  --------  ------------  ------\n");

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int step = 0;

    const int BLOCK = 128;
    const int GRID_DIM = (NSEEDS + BLOCK - 1) / BLOCK;

    /* Milestones */
    int milestones[] = {1,5,9,25,50,100,153,213,256,405,512,597,total_steps};
    int nmiles = (int)(sizeof(milestones)/sizeof(milestones[0]));

    if (first_phase > 0)
        printf("Delta mode: starting from phase %d (%s)\n\n", first_phase, phases[first_phase].label);

    for (int pi = first_phase; pi < nphases; pi++) {
        Phase *ph = &phases[pi];
        int npos = buildPositions(ph->blk, pos_ox_h, pos_oy_h);
        cudaMemcpy(pos_ox_d, pos_ox_h, npos*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(pos_oy_d, pos_oy_h, npos*sizeof(int), cudaMemcpyHostToDevice);

        for (int si = 0; si < ph->count; si++) {
            int warmup = step;

            /* Precompute baseErr on host (cheap, once per step) */
            int baseErr_h[512];
            computeBaseErrors(canvas_h, target_bin, npos, pos_ox_h, pos_oy_h, ph->blk, baseErr_h);
            cudaMemcpy(baseErr_d, baseErr_h, npos*sizeof(int), cudaMemcpyHostToDevice);

            /* Upload current canvas */
            cudaMemcpy(canvas_d, canvas_h, PS, cudaMemcpyHostToDevice);

            /* Launch kernel: one thread per seed */
            searchKernel<<<GRID_DIM, BLOCK>>>(
                canvas_d, target_d, baseErr_d, pos_ox_d, pos_oy_d,
                npos, ph->blk, ph->andN, warmup,
                out_seed_d, out_pos_d, out_err_d);
            cudaDeviceSynchronize();

            /* Download results, find global best */
            cudaMemcpy(out_seed_h, out_seed_d, NSEEDS*sizeof(uint16_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_pos_h,  out_pos_d,  NSEEDS*sizeof(int),      cudaMemcpyDeviceToHost);
            cudaMemcpy(out_err_h,  out_err_d,  NSEEDS*sizeof(int),      cudaMemcpyDeviceToHost);

            uint16_t bestSeed = 1; int bestPos = 0, bestDelta = 0x7fffffff;
            for (int s = 0; s < NSEEDS; s++) {
                if (out_err_h[s] < bestDelta) {  /* out_err now holds delta */
                    bestDelta = out_err_h[s];
                    bestSeed  = out_seed_h[s];
                    bestPos   = out_pos_h[s];
                }
            }

            /* Apply only if it improves global error */
            uint8_t buf[BUF_N];
            uint16_t ss = bestSeed ? bestSeed : 1;
            uint16_t state = ss;
            for (int w = 0; w < warmup; w++) state = lfsr16(state);
            for (int i = 0; i < BUF_N; i++) {
                uint16_t acc = 1;
                for (int k = 0; k < ph->andN; k++) { state = lfsr16(state); acc &= (state&1u); }
                buf[i] = (uint8_t)acc;
            }

            uint8_t test[PS]; memcpy(test, canvas_h, PS);
            applyBuf(test, buf, pos_ox_h[bestPos], pos_oy_h[bestPos], ph->blk);
            if (lBin(test, target_bin) < lBin(canvas_h, target_bin)) {
                applyBuf(canvas_h, buf, pos_ox_h[bestPos], pos_oy_h[bestPos], ph->blk);
                SeedRecord *r = &seedLog[nlog++];
                r->step   = nlog;
                r->seed   = bestSeed;
                r->ox     = pos_ox_h[bestPos];
                r->oy     = pos_oy_h[bestPos];
                r->blk    = ph->blk;
                r->and_n  = ph->andN;
                r->warmup = warmup;
                strncpy(r->label, ph->label, 31);
            }

            step++;

            /* Check milestone */
            int is_mile = 0;
            for (int m = 0; m < nmiles; m++) if (milestones[m] == step) { is_mile = 1; break; }
            if (is_mile || step == total_steps) {
                struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec)*1e-9;
                double lb = lBin(canvas_h, target_bin);
                char snap[256];
                snprintf(snap, sizeof(snap), "/tmp/cuda_cascade_s%04d.pgm", step);
                savePGM(snap, canvas_h);
                printf("%-6d  %6.2f%%  %7.1fs  (%3d,%2d) blk=%d  %s\n",
                    step, lb*100, elapsed, pos_ox_h[bestPos], pos_oy_h[bestPos], ph->blk, ph->label);
                fflush(stdout);
            }
        }
    }

    printf("\nApplied %d / %d seeds\n", nlog, total_steps);
    writeJSON(out_json, seedLog, nlog);
    savePGM("/tmp/cuda_cascade_result.pgm", canvas_h);
    printf("Result: /tmp/cuda_cascade_result.pgm\n");

    free(seedLog);
    cudaFree(canvas_d); cudaFree(target_d);
    cudaFree(out_seed_d); cudaFree(out_pos_d); cudaFree(out_err_d);
    cudaFree(pos_ox_d); cudaFree(pos_oy_d); cudaFree(baseErr_d);
    return 0;
}

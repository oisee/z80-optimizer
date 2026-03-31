/*
 * prng_joint2.cu — Joint u32 seed search for two overlapping foveal layers.
 *
 * For a pair of layers (i, j) with FIXED positions (ox1,oy1,blk1) and (ox2,oy2,blk2):
 *   1. Undo both seeds by XOR-applying them again (trivial reversibility)
 *   2. Jointly enumerate ALL (s1, s2) in 1..65535 × 1..65535 = 4.3B combos
 *   3. Find (s1*, s2*) that minimizes global error after applying both buffers
 *   4. Apply new seeds if total error improves
 *
 * The u32 search is done in batched 2D kernels:
 *   outer batch: S1_BATCH seeds of s1 at a time
 *   inner: 65535 threads for s2
 *   Each thread evaluates the combined effect of (s1_in_batch, s2) on the canvas.
 *
 * Interaction term: pixels in the OVERLAP region are flipped by both buffers.
 * XOR of both buffers at overlap pixels: if both set → cancel; if one set → flip.
 * This is the key nonlinearity that joint search exploits vs. greedy sequential.
 *
 * Build:
 *   nvcc -O3 -o cuda/prng_joint2 cuda/prng_joint2.cu -lm
 *
 * Usage:
 *   ./cuda/prng_joint2 --seeds data/foveal_cascade_seeds.json \
 *                      --pairs 0,24 0,8 0,2 0,31 \
 *                      --gpu 0
 *
 * Output: improved seeds written to /tmp/joint2_seeds.json
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
#define PS      (W * H / 8)
#define BUF_N   768
#define S1_BATCH 128   /* outer batch size — tune for occupancy */

/* ====================================================================
 * LFSR-16
 * ==================================================================== */
__device__ __host__ __forceinline__ uint16_t lfsr16(uint16_t s) {
    uint16_t bit = s & 1u; s >>= 1; if (bit) s ^= 0xB400u; return s;
}

/* ====================================================================
 * makeBuf: generate 768-element AND-N buffer for (seed, warmup, andN)
 * ==================================================================== */
__device__ __host__ void makeBuf(uint16_t seed, int warmup, int andN, uint8_t buf[BUF_N]) {
    uint16_t s = seed ? seed : 1u;
    for (int i = 0; i < warmup; i++) s = lfsr16(s);
    for (int i = 0; i < BUF_N; i++) {
        uint16_t acc = 1u;
        for (int k = 0; k < andN; k++) { s = lfsr16(s); acc &= (s & 1u); }
        buf[i] = (uint8_t)acc;
    }
}

/* ====================================================================
 * applyBuf (host): XOR canvas with buf at (ox, oy, blk)
 * ==================================================================== */
void applyBufHost(uint8_t *canvas, const uint8_t buf[BUF_N], int ox, int oy, int blk) {
    for (int by = 0; by < 24; by++)
        for (int bx = 0; bx < 32; bx++) {
            if (!buf[by*32+bx]) continue;
            for (int dy = 0; dy < blk; dy++)
                for (int dx = 0; dx < blk; dx++) {
                    int x = ox+bx*blk+dx, y = oy+by*blk+dy;
                    if (x<W && y<H) canvas[y*(W/8)+x/8] ^= (1u<<(7-(x%8)));
                }
        }
}

double lBin(const uint8_t *c, const uint8_t *t) {
    int err=0; for(int i=0;i<PS;i++) err+=__builtin_popcount(c[i]^t[i]);
    return (double)err/(W*H);
}

/* ====================================================================
 * CUDA kernel: joint u32 search
 *
 * Grid:  (S1_BATCH, 1, 1)   — one block per s1 value in the current batch
 * Block: (512, 1, 1)         — 512 threads each handling ceil(65535/512) s2 values
 *
 * For each (s1_in_batch, s2):
 *   1. Generate buf1 for s1 at (pos1, warmup1, andN1) → shared memory
 *   2. Apply buf1 to base_canvas → temp pixel region
 *   3. Generate buf2 for s2 at (pos2, warmup2, andN2)
 *   4. Compute combined delta (both regions, including overlap interaction)
 *   5. Track best (s2, total_delta) per s1
 *
 * Output: out_s2[S1_BATCH], out_delta[S1_BATCH]
 * ==================================================================== */
__global__ void jointSearchKernel(
    const uint8_t * __restrict__ base_d,   /* canvas with BOTH seeds undone */
    const uint8_t * __restrict__ target_d,
    int s1_base,          /* first seed index in this batch (s1 = s1_base+blockIdx.x+1) */
    int warmup1, int andN1, int ox1, int oy1, int blk1,
    int warmup2, int andN2, int ox2, int oy2, int blk2,
    int   * __restrict__ out_s2,      /* best s2 for each s1 in batch */
    int   * __restrict__ out_delta    /* best combined delta for each s1 */
) {
    /* Which s1 does this block own? */
    int s1 = s1_base + (int)blockIdx.x + 1;  /* 1-indexed */
    if (s1 > 65535) return;

    /* Generate buf1 in shared memory (only once per block) */
    __shared__ uint8_t sbuf1[BUF_N];
    if (threadIdx.x == 0) makeBuf((uint16_t)s1, warmup1, andN1, sbuf1);
    __syncthreads();

    /* Each thread searches a range of s2 values */
    const int S2_TOTAL = 65535;
    const int S2_PER_THREAD = (S2_TOTAL + blockDim.x - 1) / blockDim.x;
    const int s2_start = threadIdx.x * S2_PER_THREAD + 1;
    const int s2_end   = min(s2_start + S2_PER_THREAD, S2_TOTAL + 1);

    int bestS2 = s2_start, bestDelta = 0x7fffffff;

    int pw1 = 32*blk1, ph1 = 24*blk1;
    int pw2 = 32*blk2, ph2 = 24*blk2;

    for (int s2 = s2_start; s2 < s2_end; s2++) {
        uint8_t buf2[BUF_N];
        makeBuf((uint16_t)s2, warmup2, andN2, buf2);

        int delta = 0;

        /* ---- Region 1 (buf1 effect on pos1 region) ---- */
        for (int by = 0; by < 24; by++) {
            for (int bx = 0; bx < 32; bx++) {
                if (!sbuf1[by*32+bx]) continue;
                for (int dy = 0; dy < blk1; dy++) {
                    for (int dx = 0; dx < blk1; dx++) {
                        int x = ox1+bx*blk1+dx, y = oy1+by*blk1+dy;
                        if (x<0||x>=W||y<0||y>=H) continue;
                        int bidx = y*(W/8)+x/8, bbit = 7-(x%8);
                        uint8_t cb = (base_d[bidx]>>bbit)&1u;
                        uint8_t tb = (target_d[bidx]>>bbit)&1u;
                        /* Check if buf2 also flips this pixel (overlap interaction) */
                        int bx2 = (x-ox2)/blk2, by2 = (y-oy2)/blk2;
                        int in2 = (x>=ox2 && x<ox2+pw2 && y>=oy2 && y<oy2+ph2
                                   && bx2>=0 && bx2<32 && by2>=0 && by2<24);
                        uint8_t flip2 = in2 ? buf2[by2*32+bx2] : 0u;
                        /* Combined flip: buf1 XOR buf2 */
                        uint8_t net_flip = 1u ^ flip2; /* buf1=1, so net = 1 XOR buf2 */
                        if (net_flip) {
                            delta += (cb==tb) ? 1 : -1;
                        }
                        /* else: buf2 cancels buf1 here → no flip, no delta contribution */
                    }
                }
            }
        }

        /* ---- Region 2 (buf2 effect on pos2 region, excluding overlap already counted) ---- */
        for (int by = 0; by < 24; by++) {
            for (int bx = 0; bx < 32; bx++) {
                if (!buf2[by*32+bx]) continue;
                for (int dy = 0; dy < blk2; dy++) {
                    for (int dx = 0; dx < blk2; dx++) {
                        int x = ox2+bx*blk2+dx, y = oy2+by*blk2+dy;
                        if (x<0||x>=W||y<0||y>=H) continue;
                        /* Skip pixels already counted in region 1 */
                        int bx1 = (x-ox1)/blk1, by1 = (y-oy1)/blk1;
                        int in1 = (x>=ox1 && x<ox1+pw1 && y>=oy1 && y<oy1+ph1
                                   && bx1>=0 && bx1<32 && by1>=0 && by1<24
                                   && sbuf1[by1*32+bx1]);
                        if (in1) continue; /* already counted in region 1 loop */
                        int bidx = y*(W/8)+x/8, bbit = 7-(x%8);
                        uint8_t cb = (base_d[bidx]>>bbit)&1u;
                        uint8_t tb = (target_d[bidx]>>bbit)&1u;
                        delta += (cb==tb) ? 1 : -1;
                    }
                }
            }
        }

        if (delta < bestDelta) { bestDelta = delta; bestS2 = s2; }
    }

    /* Block reduction: find best (s2, delta) across all threads */
    __shared__ int sh_delta[512];
    __shared__ int sh_s2[512];
    sh_delta[threadIdx.x] = bestDelta;
    sh_s2[threadIdx.x]    = bestS2;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            if (sh_delta[threadIdx.x+stride] < sh_delta[threadIdx.x]) {
                sh_delta[threadIdx.x] = sh_delta[threadIdx.x+stride];
                sh_s2[threadIdx.x]    = sh_s2[threadIdx.x+stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_s2[(int)blockIdx.x]    = sh_s2[0];
        out_delta[(int)blockIdx.x] = sh_delta[0];
    }
}

/* ====================================================================
 * Minimal JSON reader for our seeds format
 * ==================================================================== */
#define MAX_SEEDS 2000
typedef struct {
    int step, ox, oy, blk, and_n, warmup;
    uint16_t seed;
    char label[32];
} SeedRec;

static SeedRec gSeeds[MAX_SEEDS];
static int gNSeeds = 0;

int readSeeds(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char*)malloc(sz+1); fread(buf, 1, sz, f); buf[sz]=0; fclose(f);

    char *p = buf;
    gNSeeds = 0;
    while ((p = strstr(p, "\"step\"")) != NULL) {
        SeedRec *r = &gSeeds[gNSeeds];
        sscanf(p, "\"step\":%d", &r->step);
        char *q;
        if ((q=strstr(p,"\"seed\"")))   sscanf(q, "\"seed\":%hu", &r->seed);
        if ((q=strstr(p,"\"ox\"")))     sscanf(q, "\"ox\":%d",    &r->ox);
        if ((q=strstr(p,"\"oy\"")))     sscanf(q, "\"oy\":%d",    &r->oy);
        if ((q=strstr(p,"\"blk\"")))    sscanf(q, "\"blk\":%d",   &r->blk);
        if ((q=strstr(p,"\"and_n\"")))  sscanf(q, "\"and_n\":%d", &r->and_n);
        if ((q=strstr(p,"\"warmup\""))) sscanf(q, "\"warmup\":%d",&r->warmup);
        if ((q=strstr(p,"\"label\""))) {
            q += 9; char *e = strchr(q, '"');
            if (e) { int n=e-q<31?e-q:31; memcpy(r->label,q,n); r->label[n]=0; }
        }
        gNSeeds++; if (gNSeeds >= MAX_SEEDS) break;
        p++;
    }
    free(buf);
    return gNSeeds;
}

void writeSeeds(const char *path) {
    FILE *f = fopen(path, "w");
    fprintf(f, "{\n  \"lfsr16_poly\":\"0xB400\",\"canvas_w\":%d,\"canvas_h\":%d,\n"
               "  \"position_grid\":8,\n  \"seeds\":[\n", W, H);
    for (int i = 0; i < gNSeeds; i++) {
        SeedRec *r = &gSeeds[i];
        fprintf(f, "    {\"step\":%d,\"seed\":%u,\"ox\":%d,\"oy\":%d,"
                   "\"blk\":%d,\"and_n\":%d,\"warmup\":%d,\"label\":\"%s\"}%s\n",
                r->step,r->seed,r->ox,r->oy,r->blk,r->and_n,r->warmup,r->label,
                i<gNSeeds-1?",":"");
    }
    fprintf(f, "  ]\n}\n"); fclose(f);
}

/* ====================================================================
 * Reconstruct canvas by replaying all seeds
 * ==================================================================== */
void buildCanvas(uint8_t *canvas) {
    memset(canvas, 0, PS);
    for (int i = 0; i < gNSeeds; i++) {
        SeedRec *r = &gSeeds[i];
        uint8_t buf[BUF_N];
        makeBuf(r->seed, r->warmup, r->and_n, buf);
        applyBufHost(canvas, buf, r->ox, r->oy, r->blk);
    }
}

/* ====================================================================
 * Joint search for one pair (idx_i, idx_j)
 * ==================================================================== */
void jointOptPair(int idx_i, int idx_j,
                  const uint8_t *canvas_full, const uint8_t *target,
                  int gpu_id) {
    SeedRec *ri = &gSeeds[idx_i];
    SeedRec *rj = &gSeeds[idx_j];

    printf("Pair [%d×%d]  pos1=(%d,%d)blk%d  pos2=(%d,%d)blk%d  andN=%d+%d  warmup=%d+%d\n",
        idx_i, idx_j, ri->ox, ri->oy, ri->blk, rj->ox, rj->oy, rj->blk,
        ri->and_n, rj->and_n, ri->warmup, rj->warmup);

    /* Build base canvas: full canvas with BOTH seeds undone */
    uint8_t base_canvas[PS];
    memcpy(base_canvas, canvas_full, PS);
    uint8_t bufi[BUF_N], bufj[BUF_N];
    makeBuf(ri->seed, ri->warmup, ri->and_n, bufi);
    makeBuf(rj->seed, rj->warmup, rj->and_n, bufj);
    applyBufHost(base_canvas, bufi, ri->ox, ri->oy, ri->blk); /* undo i */
    applyBufHost(base_canvas, bufj, rj->ox, rj->oy, rj->blk); /* undo j */

    double base_err = lBin(base_canvas, target);
    double orig_err = lBin(canvas_full, target);
    printf("  base (both undone): %.4f%%  original: %.4f%%\n",
           base_err*100, orig_err*100);

    /* Upload to GPU */
    uint8_t *base_d, *target_d;
    cudaMalloc(&base_d,   PS); cudaMemcpy(base_d,   base_canvas, PS, cudaMemcpyHostToDevice);
    cudaMalloc(&target_d, PS); cudaMemcpy(target_d, target,      PS, cudaMemcpyHostToDevice);

    const int NBATCHES = (65535 + S1_BATCH - 1) / S1_BATCH;
    int *out_s2_d, *out_delta_d;
    cudaMalloc(&out_s2_d,    S1_BATCH * sizeof(int));
    cudaMalloc(&out_delta_d, S1_BATCH * sizeof(int));
    int out_s2_h[S1_BATCH], out_delta_h[S1_BATCH];

    int global_best_s1 = ri->seed, global_best_s2 = rj->seed;
    int global_best_delta = 0x7fffffff;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int b = 0; b < NBATCHES; b++) {
        int s1_base = b * S1_BATCH;
        int batch_n = min(S1_BATCH, 65535 - s1_base);
        if (batch_n <= 0) break;

        jointSearchKernel<<<batch_n, 512>>>(
            base_d, target_d,
            s1_base,
            ri->warmup, ri->and_n, ri->ox, ri->oy, ri->blk,
            rj->warmup, rj->and_n, rj->ox, rj->oy, rj->blk,
            out_s2_d, out_delta_d);
        cudaDeviceSynchronize();

        cudaMemcpy(out_s2_h,    out_s2_d,    batch_n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_delta_h, out_delta_d, batch_n*sizeof(int), cudaMemcpyDeviceToHost);

        for (int k = 0; k < batch_n; k++) {
            if (out_delta_h[k] < global_best_delta) {
                global_best_delta = out_delta_h[k];
                global_best_s1    = s1_base + k + 1;
                global_best_s2    = out_s2_h[k];
            }
        }

        if (b % 64 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
            printf("  batch %d/%d  elapsed %.1fs  best_delta=%d\r",
                   b+1, NBATCHES, elapsed, global_best_delta);
            fflush(stdout);
        }
    }
    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    /* Apply best (s1*, s2*) to base_canvas and measure improvement */
    uint8_t test[PS]; memcpy(test, base_canvas, PS);
    uint8_t new_bufi[BUF_N], new_bufj[BUF_N];
    makeBuf((uint16_t)global_best_s1, ri->warmup, ri->and_n, new_bufi);
    makeBuf((uint16_t)global_best_s2, rj->warmup, rj->and_n, new_bufj);
    applyBufHost(test, new_bufi, ri->ox, ri->oy, ri->blk);
    applyBufHost(test, new_bufj, rj->ox, rj->oy, rj->blk);
    double new_err = lBin(test, target);

    printf("  Search time: %.1fs\n", elapsed);
    printf("  Old: seed_i=%u seed_j=%u  err=%.4f%%\n",
           ri->seed, rj->seed, orig_err*100);
    printf("  New: seed_i=%u seed_j=%u  err=%.4f%%  Δ=%.4f%%\n",
           global_best_s1, global_best_s2, new_err*100,
           (orig_err-new_err)*100);

    if (new_err < orig_err) {
        printf("  ✓ IMPROVEMENT!  applying new seeds.\n");
        ri->seed = (uint16_t)global_best_s1;
        rj->seed = (uint16_t)global_best_s2;
    } else {
        printf("  ✗ No improvement from joint search (greedy was already optimal for this pair).\n");
    }

    cudaFree(base_d); cudaFree(target_d);
    cudaFree(out_s2_d); cudaFree(out_delta_d);
}

/* ====================================================================
 * loadPGM
 * ==================================================================== */
int loadPGM(const char *path, uint8_t *target_bin, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    int w, h, maxval; char magic[8];
    fscanf(f, "%7s %d %d %d", magic, &w, &h, &maxval); fgetc(f);
    *out_w=w; *out_h=h;
    uint8_t *raw=(uint8_t*)malloc(w*h); fread(raw,1,w*h,f); fclose(f);
    memset(target_bin,0,PS);
    for(int i=0;i<w*h;i++) if(raw[i]>maxval/2) target_bin[i/8]|=(1u<<(7-(i%8)));
    free(raw); return 0;
}

/* ====================================================================
 * main
 * ==================================================================== */
int main(int argc, char **argv) {
    const char *seeds_path  = "data/foveal_cascade_seeds.json";
    const char *target_path = "media/prng_images/targets/che.pgm";
    const char *out_path    = "/tmp/joint2_seeds.json";
    int gpu_id = 0;

    /* Parse args: --seeds, --target, --out, --gpu, --pairs i,j ... */
    int pair_i[8], pair_j[8], npairs = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a],"--seeds")  && a+1<argc) seeds_path  = argv[++a];
        else if (!strcmp(argv[a],"--target") && a+1<argc) target_path = argv[++a];
        else if (!strcmp(argv[a],"--out")    && a+1<argc) out_path    = argv[++a];
        else if (!strcmp(argv[a],"--gpu")    && a+1<argc) gpu_id      = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--pairs")) {
            while (a+1<argc && argv[a+1][0]!='-') {
                a++;
                if (sscanf(argv[a],"%d,%d",&pair_i[npairs],&pair_j[npairs])==2)
                    npairs++;
            }
        }
    }

    cudaSetDevice(gpu_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,gpu_id);
    printf("GPU %d: %s\n", gpu_id, prop.name);

    int n = readSeeds(seeds_path);
    if (n <= 0) return 1;
    printf("Loaded %d seeds from %s\n", n, seeds_path);

    uint8_t target_bin[PS];
    int tw, th;
    if (loadPGM(target_path, target_bin, &tw, &th) < 0) {
        fprintf(stderr, "Cannot load target %s\n", target_path); return 1;
    }

    /* Build full canvas */
    uint8_t canvas_full[PS];
    buildCanvas(canvas_full);
    printf("Full canvas err: %.4f%%\n", lBin(canvas_full, target_bin)*100);

    /* Default pairs if none specified */
    if (npairs == 0) {
        /* 4 pairs: sequential near top, random from top-50, cross-half */
        int defaults[][2] = {{0,24},{0,8},{0,2},{0,31}};
        for (int k = 0; k < 4; k++) { pair_i[k]=defaults[k][0]; pair_j[k]=defaults[k][1]; }
        npairs = 4;
    }

    printf("\nJoint u32 search: %d pairs, %.1fB combos per pair\n",
           npairs, 65535.0*65535.0/1e9);
    printf("Search space per pair: 65535 × 65535 = %llu combinations\n\n",
           (unsigned long long)65535*65535);

    for (int k = 0; k < npairs; k++) {
        if (pair_i[k] >= gNSeeds || pair_j[k] >= gNSeeds) {
            printf("Pair [%d×%d]: index out of range (have %d seeds)\n",
                   pair_i[k], pair_j[k], gNSeeds); continue;
        }
        /* After each pair optimization, rebuild canvas with updated seeds */
        buildCanvas(canvas_full);
        jointOptPair(pair_i[k], pair_j[k], canvas_full, target_bin, gpu_id);
        printf("\n");
    }

    /* Final result */
    buildCanvas(canvas_full);
    printf("=== Final canvas err after joint-2: %.4f%% ===\n",
           lBin(canvas_full, target_bin)*100);

    writeSeeds(out_path);
    printf("Updated seeds: %s\n", out_path);
    return 0;
}

/*
 * prng_hybrid_gpu.cu — CUDA hybrid image generator for evolutionary search
 *
 * 57-byte genome: pRNG seed + tile masks + basis shapes + threshold + symmetry
 * All 6 layers computed on GPU, fitness (MSE + edges + blocks) on GPU
 * Evolution (selection + mutation) on CPU
 *
 * Build:  nvcc -O3 -o cuda/prng_hybrid_gpu cuda/prng_hybrid_gpu.cu -lm
 * Usage:  ./cuda/prng_hybrid_gpu [--pop 4096] [--gens 1000] [--gpu 0]
 *                                [--target target.pgm] [--output dir/]
 *
 * Target format: PGM P5, 128x96, 8-bit grayscale
 *   Convert: convert input.png -resize 128x96 -colorspace Gray -depth 8 target.pgm
 *
 * Performance target: ~500K-1M images/sec on RTX 4060 Ti
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define W 128
#define H 96
#define BW (W / 8)       /* 16 bytes per row */
#define PIXELS (W * H)   /* 12288 */
#define THREADS 256

/* ====== Genome: 128 bytes (dual-layer) ====== */
/*
 * Dual-layer architecture:
 *   Layer A: seed_a + tile_mask_a → H-mirror (vertical axis symmetry)
 *            → eyes, ears, nose, facial symmetry
 *   Layer B: seed_b + tile_mask_b → configurable sym (V-mirror, none, etc.)
 *            → mouth, brow lines, horizontal features
 *   Result:  A | B | circle | stripes → threshold → final
 *
 * Each layer has independent noise, independent symmetry.
 * OR blend = additive (draw on black), natural for demoscene.
 */
struct __align__(16) Genome {
    /* Layer A: H-mirror (left-right symmetry, faces) */
    uint64_t seed_a;          /*  8 bytes */
    uint8_t  tile_mask_a[12]; /* 12 bytes (96 tiles = half, mirrored) */
    uint8_t  density_a;       /*  1 byte: noise density */

    /* Layer B: configurable symmetry */
    uint64_t seed_b;          /*  8 bytes */
    uint8_t  tile_mask_b[12]; /* 12 bytes */
    uint8_t  density_b;       /*  1 byte */
    uint8_t  sym_b;           /*  1 byte: 0=none, 1=V-mirror, 2=4-fold */

    /* Layer C: no symmetry, fine detail */
    uint64_t seed_c;          /*  8 bytes */
    uint8_t  density_c;       /*  1 byte */

    /* Shared basis shapes (always H-mirrored) */
    uint8_t  circle[4];       /*  4 bytes: cx, cy, r, fill */
    uint8_t  gradient[4];     /*  4 bytes: angle, threshold, strength, offset */
    uint8_t  stripes[4];      /*  4 bytes: period, phase, angle, thickness */
    uint8_t  shapes[4];       /*  4 bytes: extra shape params */

    /* Regional threshold */
    uint8_t  threshold[6];    /*  6 bytes: 2x3 density */

    /* Legacy single-layer mode */
    uint8_t  symmetry;        /*  1 byte: used in single-layer mode */
    uint8_t  mode;            /*  1 byte: 0=single-layer, 1=dual-layer */
    uint8_t  _pad[30];        /*  padding to 128 */
};

/* ====== GPU: splitmix64-based hash noise (parallel-friendly) ====== */
__device__ __forceinline__ uint32_t hash_noise(uint64_t seed, uint32_t idx) {
    uint64_t h = seed ^ (uint64_t)idx * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return (uint32_t)h;
}

/* ====== GPU: generate + threshold + symmetry in one kernel ====== */
__global__ void generate_and_score_kernel(
    const Genome* __restrict__ genomes,
    const float*  __restrict__ target,
    float* __restrict__ fitness,
    float* __restrict__ best_images,  /* only top-1 slot, or NULL */
    int num_genomes,
    int best_genome_idx              /* -1 = don't save images */
) {
    int gid = blockIdx.x;
    if (gid >= num_genomes) return;

    const Genome* g = &genomes[gid];

    /* Precompute genome parameters (shared across all pixels) */
    float cx     = g->circle[0] * ((float)W / 256.0f);
    float cy     = g->circle[1] * ((float)H / 256.0f);
    float cr     = g->circle[2] * (fminf(W, H) / 512.0f) + 5.0f;
    float cfill  = g->circle[3] / 255.0f;

    float g_angle    = g->gradient[0] * (3.14159265f / 128.0f);
    float g_thresh   = g->gradient[1] / 255.0f;
    float g_strength = g->gradient[2] / 255.0f * 0.5f;
    float g_cos      = cosf(g_angle);
    float g_sin      = sinf(g_angle);

    float s_period = fmaxf(g->stripes[0] / 16.0f + 2.0f, 2.0f);
    float s_phase  = g->stripes[1] / 255.0f * 6.28318530f;
    float s_angle  = g->stripes[2] * (3.14159265f / 128.0f);
    float s_thick  = g->stripes[3] / 255.0f * 0.5f;
    float s_cos    = cosf(s_angle);
    float s_sin    = sinf(s_angle);

    uint8_t sym = g->symmetry;
    uint8_t dual = g->mode;

    /* Shared memory for fitness reduction (static: 3KB) */
    __shared__ float s_mse[THREADS];
    __shared__ float s_edge[THREADS];
    __shared__ float s_block[THREADS];

    /* Image in dynamic shared memory (49KB, allocated at launch) */
    extern __shared__ float s_img[];

    /* ---- Phase 1: generate all pixels ---- */
    for (int pid = threadIdx.x; pid < PIXELS; pid += THREADS) {
        int x = pid % W;
        int y = pid / W;
        float val = 0.0f;

        if (dual) {
            /* ======== DUAL-LAYER MODE ======== */

            /* Layer A: H-mirror noise (vertical axis symmetry) */
            int ax = (x > W/2) ? (W - 1 - x) : x;  /* fold left */
            int ay = y;
            int a_byte = ay * BW + (ax / 8);
            int a_bit  = 7 - (ax % 8);
            uint32_t na = hash_noise(g->seed_a, a_byte);
            float da = g->density_a / 255.0f;
            if (((na >> a_bit) & 1) && ((na & 0xFF00) >> 8) < (uint32_t)(da * 256))
                val = 1.0f;

            /* Layer A tile mask (12 bytes = 96 tiles, left half mirrored) */
            int atx = ax / 8;  /* 0-7 (left half) */
            int aty = ay / 8;  /* 0-11 */
            int a_tile = aty * 8 + atx;
            if (atx < 8 && a_tile/8 < 12) {
                if ((g->tile_mask_a[a_tile/8] >> (a_tile%8)) & 1) {
                    uint32_t tn = hash_noise(g->seed_a + 0xAA, ay * W + ax);
                    if ((tn & 0xFF) < 100u) val = 1.0f;
                }
            }

            /* Layer B: configurable symmetry noise */
            int bx = x, by = y;
            if (g->sym_b == 1) { if (by > H/2) by = H - 1 - by; }  /* V-mirror */
            if (g->sym_b == 2) { /* 4-fold */
                if (bx >= W/2) bx = W - 1 - bx;
                if (by >= H/2) by = H - 1 - by;
            }
            int b_byte = by * BW + (bx / 8);
            int b_bit  = 7 - (bx % 8);
            uint32_t nb = hash_noise(g->seed_b, b_byte);
            float db = g->density_b / 255.0f;
            if (((nb >> b_bit) & 1) && ((nb & 0xFF00) >> 8) < (uint32_t)(db * 256))
                val = fmaxf(val, 1.0f);  /* OR */

            /* Layer B tile mask */
            int btx = bx / 8, bty = by / 8;
            int b_tile = bty * 8 + (btx < 8 ? btx : 15 - btx);
            if (b_tile/8 < 12) {
                if ((g->tile_mask_b[b_tile/8] >> (b_tile%8)) & 1) {
                    uint32_t tn = hash_noise(g->seed_b + 0xBB, by * W + bx);
                    if ((tn & 0xFF) < 100u) val = fmaxf(val, 1.0f);
                }
            }

            /* Layer C: detail noise (no symmetry) */
            {
                int c_byte = y * BW + (x / 8);
                int c_bit  = 7 - (x % 8);
                uint32_t nc = hash_noise(g->seed_c, c_byte);
                float dc = g->density_c / 255.0f;
                if (((nc >> c_bit) & 1) && ((nc & 0xFF00) >> 8) < (uint32_t)(dc * 256))
                    val = fmaxf(val, 1.0f);
            }

            /* Shared shapes (H-mirrored for faces) */
            int sx = (x > W/2) ? (W - 1 - x) : x;
            float dx = sx - cx, dy = y - cy;
            float dist = sqrtf(dx * dx + dy * dy);
            if (fabsf(dist - cr) < 2.0f) val = fmaxf(val, cfill);
            if (dist < cr * 0.3f)        val = fmaxf(val, cfill * 0.5f);

            float grad_val = (g_cos * x / (float)W + g_sin * y / (float)H) * g_strength;
            if (grad_val > g_thresh) val = fminf(val + 0.2f, 1.0f);

            float sv = sinf((s_cos * x + s_sin * y) / s_period + s_phase);
            if (sv > 1.0f - s_thick * 2.0f) val = fmaxf(val, 0.8f);

            if (g->shapes[0] > 128) {
                float sr = g->shapes[1] / 16.0f + 3.0f;
                float swv = sinf(dist / sr + g->shapes[2] / 32.0f);
                float sth = g->shapes[3] / 255.0f;
                if (swv > sth) val = fmaxf(val, 0.6f);
            }

        } else {
            /* ======== SINGLE-LAYER MODE (legacy) ======== */
            int sx = x, sy = y;
            if (sym == 1 || sym == 3) { if (sx > W/2) sx = W - 1 - sx; }
            if (sym == 2 || sym == 3) { if (sy > H/2) sy = H - 1 - sy; }
            if (sym == 4) {
                if (sx >= W/2) sx = W - 1 - sx;
                if (sy >= H/2) sy = H - 1 - sy;
            }

            int byte_idx = sy * BW + (sx / 8);
            int bit_idx  = 7 - (sx % 8);
            uint32_t noise = hash_noise(g->seed_a, byte_idx);
            if ((noise >> bit_idx) & 1) val = 1.0f;

            int tx = sx / 8, ty2 = sy / 8;
            int tile_idx = ty2 * 16 + tx;
            int mask_byte = tile_idx / 8;
            int mask_bit  = tile_idx % 8;
            if (mask_byte < 12 && ((g->tile_mask_a[mask_byte] >> mask_bit) & 1)) {
                uint32_t tile_noise = hash_noise(g->seed_a + 0x12345ULL, sy * W + sx);
                if ((tile_noise & 0xFF) < 77u) val = 1.0f;
            }

            float dx = sx - cx, dy = sy - cy;
            float dist = sqrtf(dx * dx + dy * dy);
            if (fabsf(dist - cr) < 2.0f) val = fmaxf(val, cfill);
            if (dist < cr * 0.3f)        val = fmaxf(val, cfill * 0.5f);

            float grad_val = (g_cos * sx / (float)W + g_sin * sy / (float)H) * g_strength;
            if (grad_val > g_thresh) val = fminf(val + 0.2f, 1.0f);

            float sv = sinf((s_cos * sx + s_sin * sy) / s_period + s_phase);
            if (sv > 1.0f - s_thick * 2.0f) val = fmaxf(val, 0.8f);

            if (g->shapes[0] > 128) {
                float sr = g->shapes[1] / 16.0f + 3.0f;
                float swv = sinf(dist / sr + g->shapes[2] / 32.0f);
                float sth = g->shapes[3] / 255.0f;
                if (swv > sth) val = fmaxf(val, 0.6f);
            }
        }

        val = fminf(val, 1.0f);

        /* Regional threshold (binarize) */
        int ry = y * 2 / H;
        int rx = x * 3 / W;
        float region_thresh = g->threshold[ry * 3 + rx] / 255.0f;
        val = (val > region_thresh) ? 1.0f : 0.0f;

        s_img[pid] = val;
    }
    __syncthreads();

    /* ---- Phase 2: compute fitness against target ---- */
    float local_mse  = 0.0f;
    float local_edge = 0.0f;
    float local_block = 0.0f;

    for (int pid = threadIdx.x; pid < PIXELS; pid += THREADS) {
        float diff = s_img[pid] - target[pid];
        local_mse += diff * diff;

        /* Edge similarity (Sobel-like) */
        int x = pid % W, y = pid / W;
        if (x > 0 && x < W - 1 && y > 0 && y < H - 1) {
            float gx_i = s_img[y * W + x + 1] - s_img[y * W + x - 1];
            float gy_i = s_img[(y+1) * W + x] - s_img[(y-1) * W + x];
            float gx_t = target[y * W + x + 1] - target[y * W + x - 1];
            float gy_t = target[(y+1) * W + x] - target[(y-1) * W + x];
            local_edge += (gx_i - gx_t) * (gx_i - gx_t) +
                          (gy_i - gy_t) * (gy_i - gy_t);
        }
    }

    /* Block-level MSE (8x8 → 16x12 = 192 blocks) */
    for (int bid = threadIdx.x; bid < 192; bid += THREADS) {
        int bx = (bid % 16) * 8;
        int by = (bid / 16) * 8;
        float sum_i = 0, sum_t = 0;
        for (int dy = 0; dy < 8; dy++)
            for (int dx = 0; dx < 8; dx++) {
                int idx = (by + dy) * W + bx + dx;
                sum_i += s_img[idx];
                sum_t += target[idx];
            }
        float bdiff = (sum_i - sum_t) * (1.0f / 64.0f);
        local_block += bdiff * bdiff;
    }

    s_mse[threadIdx.x]   = local_mse;
    s_edge[threadIdx.x]  = local_edge;
    s_block[threadIdx.x] = local_block;
    __syncthreads();

    /* Warp-level reduction */
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            s_mse[threadIdx.x]   += s_mse[threadIdx.x + s];
            s_edge[threadIdx.x]  += s_edge[threadIdx.x + s];
            s_block[threadIdx.x] += s_block[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        fitness[gid] = s_mse[0] / PIXELS +
                       s_edge[0] / PIXELS * 0.5f +
                       s_block[0] / 192.0f * 2.0f;
    }

    /* Optionally save image to global memory */
    if (gid == best_genome_idx && best_images != NULL) {
        for (int pid = threadIdx.x; pid < PIXELS; pid += THREADS)
            best_images[pid] = s_img[pid];
    }
}

/* ====== Host utilities ====== */

static void random_genome(Genome* g, unsigned long long base, int dual_mode) {
    memset(g, 0, sizeof(*g));
    srand48((long)base);
    g->seed_a = ((uint64_t)lrand48() << 32) | (uint64_t)lrand48();
    g->seed_b = ((uint64_t)lrand48() << 32) | (uint64_t)lrand48();
    g->seed_c = ((uint64_t)lrand48() << 32) | (uint64_t)lrand48();
    for (int i = 0; i < 12; i++) g->tile_mask_a[i] = lrand48() & 0xFF;
    for (int i = 0; i < 12; i++) g->tile_mask_b[i] = lrand48() & 0xFF;
    g->density_a = 80 + lrand48() % 176;   /* 80-255: H-mirror layer */
    g->density_b = 40 + lrand48() % 176;   /* 40-215: secondary layer */
    g->density_c = lrand48() % 80;          /* 0-79: detail (sparse) */
    g->sym_b = lrand48() % 3;              /* 0=none, 1=V-mirror, 2=4-fold */
    for (int i = 0; i < 4; i++)  g->circle[i]    = lrand48() & 0xFF;
    for (int i = 0; i < 4; i++)  g->gradient[i]   = lrand48() & 0xFF;
    for (int i = 0; i < 4; i++)  g->stripes[i]    = lrand48() & 0xFF;
    for (int i = 0; i < 4; i++)  g->shapes[i]     = lrand48() & 0xFF;
    for (int i = 0; i < 6; i++)  g->threshold[i]  = lrand48() & 0xFF;
    g->symmetry = lrand48() % 5;
    g->mode = dual_mode ? 1 : 0;
}

static void mutate_genome(const Genome* parent, Genome* child, int strength) {
    *child = *parent;
    for (int s = 0; s < strength; s++) {
        int what = rand() % 14;
        int idx;
        switch (what) {
        case 0:  child->seed_a ^= 1ULL << (rand() % 64); break;
        case 1:  child->seed_b ^= 1ULL << (rand() % 64); break;
        case 2:  child->seed_c ^= 1ULL << (rand() % 64); break;
        case 3:  child->tile_mask_a[rand() % 12] ^= 1 << (rand() % 8); break;
        case 4:  child->tile_mask_b[rand() % 12] ^= 1 << (rand() % 8); break;
        case 5:  child->density_a = (child->density_a + rand()%41 - 20) & 0xFF; break;
        case 6:  child->density_b = (child->density_b + rand()%41 - 20) & 0xFF; break;
        case 7:  child->density_c = (child->density_c + rand()%31 - 15) & 0xFF; break;
        case 8:  child->sym_b = rand() % 3; break;
        case 9:  idx = rand()%4; child->circle[idx]   = (child->circle[idx]   + rand()%41 - 20) & 0xFF; break;
        case 10: idx = rand()%4; child->gradient[idx]  = (child->gradient[idx]  + rand()%41 - 20) & 0xFF; break;
        case 11: idx = rand()%4; child->stripes[idx]   = (child->stripes[idx]   + rand()%41 - 20) & 0xFF; break;
        case 12: idx = rand()%6; child->threshold[idx]  = (child->threshold[idx]  + rand()%61 - 30) & 0xFF; break;
        case 13: idx = rand()%4; child->shapes[idx]    = (child->shapes[idx]    + rand()%41 - 20) & 0xFF; break;
        }
    }
}

/* Crossover: uniform mix of two parents */
static void crossover_genome(const Genome* a, const Genome* b, Genome* child) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t* pc = (uint8_t*)child;
    for (int i = 0; i < (int)sizeof(Genome); i++)
        pc[i] = (rand() & 1) ? pa[i] : pb[i];
}

static void synthetic_cat(float* target) {
    memset(target, 0, PIXELS * sizeof(float));
    float cx0 = W / 2.0f, cy0 = H / 2.0f;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float dx = x - cx0, dy = y - cy0;
            float dist = sqrtf(dx * dx + dy * dy);
            /* Head */
            if (dist < 20) target[y * W + x] = 1.0f;
            /* Left ear */
            float ex = x - (cx0 - 12), ey = y - (cy0 - 22);
            if (ex * ex / 36.0f + ey * ey / 100.0f < 1.0f) target[y * W + x] = 1.0f;
            /* Right ear */
            ex = x - (cx0 + 12);
            if (ex * ex / 36.0f + ey * ey / 100.0f < 1.0f) target[y * W + x] = 1.0f;
            /* Eyes (holes) */
            float lx = x - (cx0 - 8), ly = y - (cy0 - 3);
            float rx2 = x - (cx0 + 8), ry = y - (cy0 - 3);
            if (lx * lx + ly * ly < 9) target[y * W + x] = 0.0f;
            if (rx2 * rx2 + ry * ry < 9) target[y * W + x] = 0.0f;
        }
    }
}

static void synthetic_skull(float* target) {
    memset(target, 0, PIXELS * sizeof(float));
    float cx0 = W / 2.0f, cy0 = H / 2.0f - 5;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float dx = x - cx0, dy = y - cy0;
            /* Cranium (wide ellipse) */
            if (dx*dx/900.0f + dy*dy/625.0f < 1.0f) target[y*W+x] = 1.0f;
            /* Left eye socket */
            float ex = x-(cx0-10), ey = y-(cy0+2);
            if (ex*ex/25.0f + ey*ey/36.0f < 1.0f) target[y*W+x] = 0.0f;
            /* Right eye socket */
            ex = x-(cx0+10);
            if (ex*ex/25.0f + ey*ey/36.0f < 1.0f) target[y*W+x] = 0.0f;
            /* Nose */
            float nx = x-cx0, ny = y-(cy0+12);
            if (fabsf(nx) < 3 && ny > 0 && ny < 6) target[y*W+x] = 0.0f;
            /* Jaw */
            if (dy > 18 && dy < 28 && fabsf(dx) < 18)
                target[y*W+x] = ((int)(x/4) % 2 == 0) ? 1.0f : 0.0f;
        }
    }
}

static int load_pgm(const char* path, float* img) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4] = {0};
    int w = 0, h = 0, maxval = 0;
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) { fclose(f); return -2; }
    /* Skip comments */
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); }
        else if (c > ' ') { ungetc(c, f); break; }
    }
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) { fclose(f); return -3; }
    fgetc(f); /* consume single whitespace after maxval */
    if (w != W || h != H) {
        fprintf(stderr, "PGM size %dx%d, need %dx%d\n", w, h, W, H);
        fclose(f); return -4;
    }
    uint8_t* buf = (uint8_t*)malloc(w * h);
    if ((int)fread(buf, 1, w * h, f) != w * h) { free(buf); fclose(f); return -5; }
    fclose(f);
    for (int i = 0; i < w * h; i++)
        img[i] = buf[i] / (float)maxval;
    free(buf);
    return 0;
}

static void save_pgm(const char* path, const float* img) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int i = 0; i < PIXELS; i++) {
        uint8_t v = (uint8_t)(fminf(fmaxf(img[i], 0.0f), 1.0f) * 255.0f);
        fwrite(&v, 1, 1, f);
    }
    fclose(f);
}

/* Side-by-side comparison (target | generated) */
static void save_comparison_pgm(const char* path, const float* target, const float* gen) {
    int cw = W * 2 + 4;
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", cw, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < cw; x++) {
            uint8_t v;
            if (x < W) v = (uint8_t)(target[y*W+x] * 255);
            else if (x < W + 4) v = 128;  /* separator */
            else v = (uint8_t)(gen[y*W + (x - W - 4)] * 255);
            fwrite(&v, 1, 1, f);
        }
    }
    fclose(f);
}

/* ====== Sort helper ====== */
typedef struct { float f; int i; } FI;
static int cmp_fi(const void* a, const void* b) {
    float fa = ((const FI*)a)->f, fb = ((const FI*)b)->f;
    return (fa > fb) - (fa < fb);
}

/* ====== Main ====== */
int main(int argc, char** argv) {
    int pop_size     = 4096;
    int generations  = 1000;
    int device_id    = 0;
    int save_every   = 100;
    int elite_pct    = 20;
    int crossover_pct = 10;
    int num_islands  = 8;
    int migrate_every = 50;
    int restart_stall = 200;   /* restart island if no improvement for N gens */
    int dual_mode    = 0;
    const char* target_path = NULL;
    const char* output_dir  = "media/prng_images/hybrid_gpu";
    const char* synth_mode  = NULL;  /* "cat", "skull" */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--pop")      && i+1<argc) pop_size     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gens")     && i+1<argc) generations  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")      && i+1<argc) device_id    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target")   && i+1<argc) target_path  = argv[++i];
        else if (!strcmp(argv[i], "--output")   && i+1<argc) output_dir   = argv[++i];
        else if (!strcmp(argv[i], "--save-every")&& i+1<argc) save_every  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--elite")    && i+1<argc) elite_pct    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crossover")&& i+1<argc) crossover_pct= atoi(argv[++i]);
        else if (!strcmp(argv[i], "--synthetic") && i+1<argc) synth_mode  = argv[++i];
        else if (!strcmp(argv[i], "--islands")  && i+1<argc) num_islands = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--migrate")  && i+1<argc) migrate_every= atoi(argv[++i]);
        else if (!strcmp(argv[i], "--restart-stall")&& i+1<argc) restart_stall= atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dual")) dual_mode = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: %s [options]\n"
                   "  --pop N          Population size (default 4096)\n"
                   "  --gens N         Generations (default 1000)\n"
                   "  --gpu N          CUDA device (default 0)\n"
                   "  --target F.pgm   Target image (P5, 128x96, 8-bit)\n"
                   "  --synthetic MODE  Built-in target: cat, skull\n"
                   "  --output DIR     Output directory\n"
                   "  --save-every N   Save checkpoint every N gens\n"
                   "  --elite N        Elite percentage (default 20)\n"
                   "  --crossover N    Crossover percentage (default 10)\n",
                   argv[0]);
            return 0;
        }
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU: %s (%d SMs, %dMB, shared/block=%dKB)\n",
           prop.name, prop.multiProcessorCount,
           (int)(prop.totalGlobalMem >> 20),
           (int)(prop.sharedMemPerBlock >> 10));

    printf("Pop: %d, Gens: %d, Image: %dx%d = %d pixels, Mode: %s\n",
           pop_size, generations, W, H, PIXELS,
           dual_mode ? "DUAL-LAYER (A=H-mirror | B=config | C=detail)" : "single-layer");

    /* Memory */
    size_t genomes_bytes = pop_size * sizeof(Genome);
    size_t fitness_bytes = pop_size * sizeof(float);
    printf("Memory: genomes=%.1fMB, fitness=%.1fKB\n",
           genomes_bytes / 1e6, fitness_bytes / 1e3);

    Genome* h_genomes  = (Genome*)malloc(genomes_bytes);
    float*  h_fitness  = (float*)malloc(fitness_bytes);
    float*  h_target   = (float*)malloc(PIXELS * sizeof(float));
    float*  h_best_img = (float*)malloc(PIXELS * sizeof(float));

    Genome *d_genomes;  float *d_target, *d_fitness, *d_best_img;
    cudaMalloc(&d_genomes, genomes_bytes);
    cudaMalloc(&d_target,  PIXELS * sizeof(float));
    cudaMalloc(&d_fitness, fitness_bytes);
    cudaMalloc(&d_best_img, PIXELS * sizeof(float));

    /* Load target */
    if (target_path) {
        if (load_pgm(target_path, h_target) != 0) {
            fprintf(stderr, "Failed to load %s\n", target_path);
            return 1;
        }
        printf("Target: %s\n", target_path);
    } else if (synth_mode && !strcmp(synth_mode, "skull")) {
        synthetic_skull(h_target);
        printf("Target: synthetic skull\n");
    } else {
        synthetic_cat(h_target);
        printf("Target: synthetic cat\n");
    }
    cudaMemcpy(d_target, h_target, PIXELS * sizeof(float), cudaMemcpyHostToDevice);

    /* Output directory */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);
    char path[512];
    snprintf(path, sizeof(path), "%s/target.pgm", output_dir);
    save_pgm(path, h_target);

    /* Initialize population */
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++)
        random_genome(&h_genomes[i], (unsigned long long)rand() * 65536ULL + i, dual_mode);

    /* Request extended dynamic shared memory (49KB for image pixels) */
    size_t dyn_smem = PIXELS * sizeof(float);  /* 49152 bytes */
    cudaError_t attr_err = cudaFuncSetAttribute(
        generate_and_score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem);
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "Warning: cannot set dynamic shared memory to %zu: %s\n",
                dyn_smem, cudaGetErrorString(attr_err));
    }
    printf("Dynamic shared memory: %zu bytes (%.1fKB)\n", dyn_smem, dyn_smem/1024.0);

    /* Force symmetry diversity: first 5 islands get fixed symmetry (single-layer only) */
    int force_sym = dual_mode ? 0 : 1;
    int island_size = pop_size / num_islands;
    pop_size = island_size * num_islands;  /* round down */
    genomes_bytes = pop_size * sizeof(Genome);
    fitness_bytes = pop_size * sizeof(float);
    printf("Islands: %d x %d = %d (sym-diverse=%d)\n",
           num_islands, island_size, pop_size, force_sym);

    float best_fitness = 1e30f;
    Genome best_genome;
    memset(&best_genome, 0, sizeof(best_genome));

    /* Per-island tracking */
    float* island_best  = (float*)malloc(num_islands * sizeof(float));
    int*   island_stall = (int*)calloc(num_islands, sizeof(int));
    for (int i = 0; i < num_islands; i++) island_best[i] = 1e30f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gen = 0; gen <= generations; gen++) {
        /* Upload genomes */
        cudaMemcpy(d_genomes, h_genomes, genomes_bytes, cudaMemcpyHostToDevice);

        /* Single fused kernel: generate + threshold + symmetry + fitness */
        generate_and_score_kernel<<<pop_size, THREADS, dyn_smem>>>(
            d_genomes, d_target, d_fitness, d_best_img, pop_size, -1);
        cudaDeviceSynchronize();

        /* Download fitness */
        cudaMemcpy(h_fitness, d_fitness, fitness_bytes, cudaMemcpyDeviceToHost);

        /* Per-island sort */
        FI* sorted = (FI*)malloc(pop_size * sizeof(FI));
        for (int isl = 0; isl < num_islands; isl++) {
            int base = isl * island_size;
            for (int j = 0; j < island_size; j++) {
                sorted[base + j].f = h_fitness[base + j];
                sorted[base + j].i = base + j;
            }
            qsort(&sorted[base], island_size, sizeof(FI), cmp_fi);

            /* Stall detection */
            if (sorted[base].f < island_best[isl] - 0.0001f) {
                island_best[isl] = sorted[base].f;
                island_stall[isl] = 0;
            } else {
                island_stall[isl]++;
            }
        }

        /* Global best */
        for (int isl = 0; isl < num_islands; isl++) {
            int base = isl * island_size;
            if (sorted[base].f < best_fitness) {
                best_fitness = sorted[base].f;
                best_genome  = h_genomes[sorted[base].i];

                /* Re-run to capture image */
                Genome tmp = best_genome;
                cudaMemcpy(d_genomes, &tmp, sizeof(Genome), cudaMemcpyHostToDevice);
                generate_and_score_kernel<<<1, THREADS, dyn_smem>>>(
                    d_genomes, d_target, d_fitness, d_best_img, 1, 0);
                cudaDeviceSynchronize();
                cudaMemcpy(h_best_img, d_best_img, PIXELS * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }

        /* Status */
        if (gen % 20 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            double ips = (double)(gen + 1) * pop_size / elapsed;
            printf("Gen %4d: best=%.6f [", gen, best_fitness);
            for (int isl = 0; isl < num_islands; isl++)
                printf("%.4f%s", island_best[isl], isl < num_islands-1 ? " " : "");
            printf("] %.0f img/s %.1fs\n", ips, elapsed);
        }

        /* Save checkpoint */
        if (gen > 0 && gen % save_every == 0) {
            snprintf(path, sizeof(path), "%s/gen%04d_f%.4f.pgm",
                     output_dir, gen, best_fitness);
            save_pgm(path, h_best_img);
            snprintf(path, sizeof(path), "%s/gen%04d_compare.pgm",
                     output_dir, gen);
            save_comparison_pgm(path, h_target, h_best_img);
            printf("  Saved checkpoint gen %d\n", gen);
        }

        /* Evolution */
        if (gen < generations) {
            int keep      = island_size * elite_pct / 100;
            int cross_num = island_size * crossover_pct / 100;
            if (keep < 1) keep = 1;

            Genome* next = (Genome*)malloc(genomes_bytes);

            for (int isl = 0; isl < num_islands; isl++) {
                int base = isl * island_size;

                /* Restart stalled islands */
                if (island_stall[isl] >= restart_stall) {
                    printf("  Island %d stalled, restarting\n", isl);
                    next[base] = best_genome;  /* seed with global best */
                    for (int j = 1; j < island_size; j++)
                        random_genome(&next[base + j],
                            (unsigned long long)rand() * 65536ULL + gen * 1000 + j, dual_mode);
                    island_best[isl] = 1e30f;
                    island_stall[isl] = 0;
                    continue;
                }

                /* Elite */
                for (int j = 0; j < keep; j++)
                    next[base + j] = h_genomes[sorted[base + j].i];

                /* Crossover */
                for (int j = keep; j < keep + cross_num && j < island_size; j++) {
                    int a = rand() % keep, b = rand() % keep;
                    crossover_genome(&next[base + a], &next[base + b], &next[base + j]);
                }

                /* Mutants */
                for (int j = keep + cross_num; j < island_size; j++) {
                    int parent = rand() % keep;
                    int strength = 1 + rand() % 4;
                    mutate_genome(&next[base + parent], &next[base + j], strength);
                }
            }

            /* Force symmetry on first 5 islands (0=none,1=H,2=V,3=both,4=kaleido) */
            if (force_sym) {
                for (int isl = 0; isl < num_islands && isl < 5; isl++) {
                    int base = isl * island_size;
                    for (int j = 0; j < island_size; j++)
                        next[base + j].symmetry = isl;
                }
            }

            /* Migration: ring topology, top-3 → next island */
            if (gen > 0 && gen % migrate_every == 0) {
                for (int isl = 0; isl < num_islands; isl++) {
                    int src = isl * island_size;
                    int dst = ((isl + 1) % num_islands) * island_size;
                    for (int k = 0; k < 3 && k < keep; k++) {
                        next[dst + island_size - 1 - k] = next[src + k];
                        /* Fix symmetry for forced islands */
                        int dst_isl = (isl + 1) % num_islands;
                        if (force_sym && dst_isl < 5)
                            next[dst + island_size - 1 - k].symmetry = dst_isl;
                    }
                }
            }

            memcpy(h_genomes, next, genomes_bytes);
            free(next);
        }

        free(sorted);
    }

    free(island_best);
    free(island_stall);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    /* Save final results */
    snprintf(path, sizeof(path), "%s/final_best.pgm", output_dir);
    save_pgm(path, h_best_img);
    snprintf(path, sizeof(path), "%s/final_compare.pgm", output_dir);
    save_comparison_pgm(path, h_target, h_best_img);

    printf("\n=== RESULT ===\n");
    printf("Best fitness: %.6f\n", best_fitness);
    printf("Symmetry: %d (0=none 1=H 2=V 3=both 4=kaleidoscope)\n", best_genome.symmetry);
    printf("Circle: cx=%d cy=%d r=%d fill=%d\n",
           best_genome.circle[0], best_genome.circle[1],
           best_genome.circle[2], best_genome.circle[3]);
    printf("Gradient: angle=%d thresh=%d str=%d\n",
           best_genome.gradient[0], best_genome.gradient[1], best_genome.gradient[2]);
    printf("Stripes: period=%d phase=%d angle=%d thick=%d\n",
           best_genome.stripes[0], best_genome.stripes[1],
           best_genome.stripes[2], best_genome.stripes[3]);
    printf("Total images: %lld\n", (long long)(generations + 1) * pop_size);
    printf("Time: %.1fs (%.0f img/s)\n", total, (generations + 1) * pop_size / total);
    printf("Output: %s/\n", output_dir);

    /* Cleanup */
    free(h_genomes); free(h_fitness); free(h_target); free(h_best_img);
    cudaFree(d_genomes); cudaFree(d_target); cudaFree(d_fitness); cudaFree(d_best_img);

    return 0;
}

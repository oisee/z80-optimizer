/*
 * prng_segmented_search.cu — Hierarchical segmented LFSR image search
 *
 * Instead of each layer covering the whole image, we split into segments:
 *   Level 0: 1 seed  covers whole 128x96 at 8x8 block resolution
 *   Level 1: 4 seeds cover quadrants at 4x4 block resolution (XOR correction)
 *   Level 2: 16 seeds cover 32x24 tiles at 2x2 block resolution
 *   Level 3: 64 seeds cover 16x12 tiles at 1x1 pixel resolution
 *
 * Each seed only needs to match ~192 bits → 65536 candidates is plenty.
 * Total: 85 seeds × 2 bytes = 170 bytes data.
 *
 * Build: nvcc -O3 -o cuda/prng_segmented_search cuda/prng_segmented_search.cu
 * Usage: ./cuda/prng_segmented_search --target file.pgm [--gpu 0]
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
#define PIXELS (W * H)
#define PACKED_SIZE (PIXELS / 8)  /* 1536 bytes */

/* ====== LFSR ====== */
__device__ __host__ uint32_t lfsr_step(uint32_t state) {
    uint32_t bit = state & 1;
    state >>= 1;
    if (bit) state ^= 0xB4BCD35C;
    return state;
}

/* ====== Block-scan draw: 1 LFSR bit = 1 block, sequential scan ====== */
__device__ void draw_segment(
    uint8_t* canvas,
    uint16_t seed,
    int seg_id,        /* unique per segment for LFSR init */
    int rx, int ry,    /* rectangle top-left */
    int rw, int rh,    /* rectangle size */
    int block_size,    /* pixel block size */
    int num_points     /* ignored in block-scan mode */
) {
    /* 16-bit LFSR: x^16 + x^14 + x^13 + x^11 + 1 */
    uint16_t state = seed;
    if (state == 0) state = 1;

    /* Warm up LFSR with seg_id for uniqueness */
    for (int i = 0; i < (seg_id & 15) + 4; i++) {
        uint16_t bit = state & 1;
        state >>= 1;
        if (bit) state ^= 0xB400;
    }

    int nbx = rw / block_size;
    int nby = rh / block_size;

    /* Scan all blocks in region; 1 LFSR bit per block */
    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            /* Step LFSR */
            uint16_t bit = state & 1;
            state >>= 1;
            if (bit) state ^= 0xB400;

            if (state & 1) {  /* use next bit as the XOR decision */
                int px = rx + bx * block_size;
                int py = ry + by * block_size;
                for (int dy = 0; dy < block_size && (py + dy) < H; dy++) {
                    for (int dx = 0; dx < block_size && (px + dx) < W; dx++) {
                        int x = px + dx;
                        int y = py + dy;
                        int byte_idx = y * (W / 8) + (x / 8);
                        int bit_idx = 7 - (x % 8);
                        canvas[byte_idx] ^= (1 << bit_idx);
                    }
                }
            }
        }
    }
}

/* ====== Kernel: test all 65536 seeds for one segment ====== */
__global__ void search_segment_kernel(
    const uint8_t* __restrict__ current,
    const uint8_t* __restrict__ target,
    uint32_t* __restrict__ errors,
    int seg_id,
    int rx, int ry, int rw, int rh,
    int block_size, int num_points
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= 65536) return;

    /* Copy current canvas */
    uint8_t canvas[PACKED_SIZE];
    for (int i = 0; i < PACKED_SIZE; i++)
        canvas[i] = current[i];

    /* Draw this seed's contribution */
    draw_segment(canvas, (uint16_t)seed, seg_id, rx, ry, rw, rh, block_size, num_points);

    /* Count errors ONLY within the rectangle (don't penalize outside) */
    uint32_t err = 0;
    for (int y = ry; y < ry + rh && y < H; y++) {
        for (int x = rx; x < rx + rw && x < W; x++) {
            int byte_idx = y * (W / 8) + (x / 8);
            int bit_idx = 7 - (x % 8);
            int gen_bit = (canvas[byte_idx] >> bit_idx) & 1;
            int tgt_bit = (target[byte_idx] >> bit_idx) & 1;
            if (gen_bit != tgt_bit) err++;
        }
    }

    errors[seed] = err;
}

/* ====== Host draw (block-scan, matches GPU) ====== */
void host_draw_segment(uint8_t* canvas, uint16_t seed, int seg_id,
                       int rx, int ry, int rw, int rh, int block_size, int num_points) {
    uint16_t state = seed;
    if (state == 0) state = 1;
    for (int i = 0; i < (seg_id & 15) + 4; i++) {
        uint16_t bit = state & 1; state >>= 1; if (bit) state ^= 0xB400;
    }
    int nbx = rw / block_size;
    int nby = rh / block_size;
    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            uint16_t bit = state & 1; state >>= 1; if (bit) state ^= 0xB400;
            if (state & 1) {
                int px = rx + bx * block_size;
                int py = ry + by * block_size;
                for (int dy = 0; dy < block_size && (py + dy) < H; dy++) {
                    for (int dx = 0; dx < block_size && (px + dx) < W; dx++) {
                        int x = px + dx;
                        int y = py + dy;
                        int byte_idx = y * (W / 8) + (x / 8);
                        int bit_idx = 7 - (x % 8);
                        canvas[byte_idx] ^= (1 << bit_idx);
                    }
                }
            }
        }
    }
}

/* ====== Segment definition ====== */
struct Segment {
    int rx, ry, rw, rh;
    int block_size;
    int num_points;
    int level;
};

/* ====== I/O (same as layered) ====== */
int load_pgm_binary(const char* path, uint8_t* packed) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4] = {0};
    int w = 0, h = 0, maxval = 0;
    fscanf(f, "%2s", magic);
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); }
        else if (c > ' ') { ungetc(c, f); break; }
    }
    fscanf(f, "%d %d %d", &w, &h, &maxval); fgetc(f);
    if (w != W || h != H) { fclose(f); return -4; }
    memset(packed, 0, PACKED_SIZE);
    uint8_t* raw = (uint8_t*)malloc(w * h);
    fread(raw, 1, w * h, f); fclose(f);
    for (int i = 0; i < w * h; i++) {
        if (raw[i] > maxval / 2) {
            packed[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    free(raw);
    return 0;
}

void save_pgm(const char* path, const uint8_t* packed) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t v = (packed[y*(W/8)+(x/8)] >> (7-(x%8))) & 1 ? 255 : 0;
            fwrite(&v, 1, 1, f);
        }
    fclose(f);
}

void save_comparison(const char* path, const uint8_t* target, const uint8_t* gen) {
    int cw = W * 2 + 4;
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", cw, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < cw; x++) {
            uint8_t v;
            if (x < W) { int px=x; v = (target[y*(W/8)+(px/8)]>>(7-(px%8)))&1 ? 255:0; }
            else if (x < W+4) v = 128;
            else { int px=x-W-4; v = (gen[y*(W/8)+(px/8)]>>(7-(px%8)))&1 ? 255:0; }
            fwrite(&v, 1, 1, f);
        }
    fclose(f);
}

int count_errors(const uint8_t* a, const uint8_t* b) {
    int err = 0;
    for (int i = 0; i < PACKED_SIZE; i++) err += __builtin_popcount(a[i] ^ b[i]);
    return err;
}

/* ====== Main ====== */
int main(int argc, char** argv) {
    int device_id = 0;
    const char* target_path = NULL;
    const char* output_dir = "media/prng_images/segmented";
    int pts_per_pixel = 3;  /* points per pixel in segment */
    const char* mode = "quadtree";  /* quadtree, foveal, mondrian, golden */
    int foveal_seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1<argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i+1<argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1<argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--density") && i+1<argc) pts_per_pixel = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i+1<argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i+1<argc) foveal_seed = atoi(argv[++i]);
    }
    if (!target_path) {
        fprintf(stderr, "Usage: %s --target file.pgm [--gpu 0] [--output dir] [--density 3]\n", argv[0]);
        return 1;
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU: %s\n", prop.name);

    uint8_t h_target[PACKED_SIZE];
    if (load_pgm_binary(target_path, h_target) != 0) {
        fprintf(stderr, "Failed to load %s\n", target_path); return 1;
    }

    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir); system(cmd);
    char path[512];
    snprintf(path, sizeof(path), "%s/target.pgm", output_dir);
    save_pgm(path, h_target);

    /* Build segment hierarchy */
    Segment segments[1024];
    int num_segments = 0;

    printf("Mode: %s\n", mode);

    if (!strcmp(mode, "foveal") || !strcmp(mode, "golden") || !strcmp(mode, "mondrian")) {
        /* ====== Foveal/Golden/Mondrian: asymmetric focus regions ====== */
        /* Levels: 1+2+3+4 = 10 seeds (40 bytes) or 1+3+5+7 = 16 seeds (64 bytes) */
        /* Configurable seeds-per-level via density:
           density 1: 1+2+3+4   = 10 seeds (20 bytes)
           density 2: 1+3+5+7   = 16 seeds (32 bytes)
           density 3: 1+4+12+24 = 41 seeds (82 bytes)
           density 4: 1+5+15+40 = 61 seeds (122 bytes)
           density 5: 1+8+24+64 = 97 seeds (194 bytes) */
        int spl_table[][4] = {
            {1, 2, 3, 4},    /* density 1 */
            {1, 3, 5, 7},    /* density 2 */
            {1, 4, 12, 24},  /* density 3 */
            {1, 5, 15, 40},  /* density 4 */
            {1, 8, 24, 64},  /* density 5 */
        };
        int di = pts_per_pixel - 1;
        if (di < 0) di = 0;
        if (di > 4) di = 4;
        int *spl = spl_table[di];
        int blocks[] = {8, 4, 2, 1};

        /* Simple PRNG for region placement */
        unsigned rng = (unsigned)foveal_seed;
        #define FRNG() (rng = rng * 1103515245 + 12345, (rng >> 16) & 0x7FFF)

        double phi = 1.6180339887;

        for (int lv = 0; lv < 4; lv++) {
            int block = blocks[lv];
            int n = spl[lv];

            for (int i = 0; i < n; i++) {
                int rx, ry, rw, rh;

                if (lv == 0) {
                    /* Full image */
                    rx = 0; ry = 0; rw = W; rh = H;
                } else if (!strcmp(mode, "golden")) {
                    /* Golden ratio spiral */
                    double angle = i * 2.0 * 3.14159 / phi + lv * 0.7;
                    double radius = (0.35 - 0.06*lv) * (W < H ? W : H);
                    double cx = W / phi + radius * cos(angle) * (0.4 + 0.15*i);
                    double cy = H / phi + radius * sin(angle) * (0.4 + 0.15*i);
                    rw = (int)(W * (0.6 - 0.1*lv));
                    rh = (int)(H * (0.6 - 0.1*lv));
                    rx = (int)(cx - rw/2);
                    ry = (int)(cy - rh/2);
                } else if (!strcmp(mode, "mondrian")) {
                    /* Mondrian: random rectangles, center-biased */
                    int cx = W/2 + (int)((FRNG() % 60 - 30) * W / 256.0);
                    int cy = (int)(H*0.42) + (int)((FRNG() % 40 - 20) * H / 256.0);
                    rw = (int)(W * (0.3 + (FRNG() % 30) / 100.0 - 0.07*lv));
                    rh = (int)(H * (0.3 + (FRNG() % 30) / 100.0 - 0.07*lv));
                    rx = cx - rw/2;
                    ry = cy - rh/2;
                } else { /* foveal = center-focused */
                    int cx = W/2 + (i - n/2) * W/12;
                    int cy = (int)(H*0.42) + (lv - 2) * H/15;
                    rw = (int)(W * (0.65 - 0.12*lv));
                    rh = (int)(H * (0.65 - 0.12*lv));
                    rx = cx - rw/2;
                    ry = cy - rh/2;
                }

                /* Clamp and align to block grid */
                if (rx < 0) rx = 0;
                if (ry < 0) ry = 0;
                if (rx + rw > W) rw = W - rx;
                if (ry + rh > H) rh = H - ry;
                rw = (rw / block) * block;
                rh = (rh / block) * block;
                if (rw < block) rw = block;
                if (rh < block) rh = block;
                if (rx + rw > W) rx = W - rw;
                if (ry + rh > H) ry = H - rh;

                int npx = (rw/block) * (rh/block);
                int npts = npx * pts_per_pixel;

                segments[num_segments++] = {rx, ry, rw, rh, block, npts, lv};
            }
        }

    } else if (!strcmp(mode, "face")) {
        /* ====== Face-aware: dense on features, sparse on background ====== */
        /* L0: whole image 8×8 */
        segments[num_segments++] = {0, 0, W, H, 8, 576, 0};
        /* L1: 4 grid + 4 face overlaps (4×4) */
        segments[num_segments++] = {0, 0, 64, 48, 4, 576, 1};
        segments[num_segments++] = {64, 0, 64, 48, 4, 576, 1};
        segments[num_segments++] = {0, 48, 64, 48, 4, 576, 1};
        segments[num_segments++] = {64, 48, 64, 48, 4, 576, 1};
        segments[num_segments++] = {24, 8, 80, 64, 4, 960, 1};   /* face center */
        segments[num_segments++] = {16, 20, 96, 24, 4, 432, 1};  /* eyes band */
        segments[num_segments++] = {32, 44, 64, 24, 4, 288, 1};  /* mouth area */
        segments[num_segments++] = {24, 8, 80, 20, 4, 300, 1};   /* forehead */
        /* L2: 4 background corners + 9 face regions (2×2) */
        segments[num_segments++] = {0, 0, 40, 32, 2, 960, 2};    /* bg corners */
        segments[num_segments++] = {88, 0, 40, 32, 2, 960, 2};
        segments[num_segments++] = {0, 64, 48, 32, 2, 1152, 2};
        segments[num_segments++] = {80, 64, 48, 32, 2, 1152, 2};
        segments[num_segments++] = {32, 24, 32, 16, 2, 384, 2};  /* left eye */
        segments[num_segments++] = {64, 24, 32, 16, 2, 384, 2};  /* right eye */
        segments[num_segments++] = {40, 28, 48, 12, 2, 432, 2};  /* eye bridge */
        segments[num_segments++] = {52, 34, 24, 20, 2, 360, 2};  /* nose */
        segments[num_segments++] = {44, 48, 40, 16, 2, 480, 2};  /* mouth */
        segments[num_segments++] = {48, 48, 32, 12, 2, 288, 2};  /* lips */
        segments[num_segments++] = {26, 22, 20, 40, 2, 600, 2};  /* face left */
        segments[num_segments++] = {82, 22, 20, 40, 2, 600, 2};  /* face right */
        segments[num_segments++] = {28, 8, 72, 16, 2, 864, 2};   /* hairline */
        /* L3: 1×1 pixel on eyes, nose, mouth, brows (overlapping!) */
        segments[num_segments++] = {32, 27, 20, 10, 1, 600, 3};  /* L eye outer */
        segments[num_segments++] = {44, 29, 12, 8, 1, 288, 3};   /* L eye inner */
        segments[num_segments++] = {44, 30, 8, 6, 1, 144, 3};    /* L pupil */
        segments[num_segments++] = {76, 27, 20, 10, 1, 600, 3};  /* R eye outer */
        segments[num_segments++] = {72, 29, 12, 8, 1, 288, 3};   /* R eye inner */
        segments[num_segments++] = {76, 30, 8, 6, 1, 144, 3};    /* R pupil */
        segments[num_segments++] = {56, 44, 16, 8, 1, 384, 3};   /* nose tip */
        segments[num_segments++] = {50, 53, 28, 6, 1, 504, 3};   /* mouth line */
        segments[num_segments++] = {52, 50, 24, 6, 1, 432, 3};   /* upper lip */
        segments[num_segments++] = {52, 56, 24, 6, 1, 432, 3};   /* lower lip */
        segments[num_segments++] = {38, 24, 20, 6, 1, 360, 3};   /* L brow */
        segments[num_segments++] = {70, 24, 20, 6, 1, 360, 3};   /* R brow */
        segments[num_segments++] = {48, 62, 32, 8, 1, 768, 3};   /* chin line */
        segments[num_segments++] = {56, 47, 16, 6, 1, 288, 3};   /* nostrils */

    } else if (!strcmp(mode, "hybrid")) {
        /* ====== Hybrid: foveal center + quadtree background ====== */
        /* L0: full image (1 seed) */
        segments[num_segments++] = {0, 0, W, H, 8, (W/8)*(H/8)*pts_per_pixel, 0};

        /* L1: 4 quadrants (grid) + 2 foveal face regions = 6 seeds */
        for (int qy = 0; qy < 2; qy++)
            for (int qx = 0; qx < 2; qx++)
                segments[num_segments++] = {qx*W/2, qy*H/2, W/2, H/2, 4,
                                            (W/8)*(H/8)*pts_per_pixel/2, 1};
        /* Face center + eyes region */
        segments[num_segments++] = {W/4, H/6, W/2, H*2/3, 4, (W/8)*(H/4)*pts_per_pixel, 1};
        segments[num_segments++] = {W/4, H/5, W/2, H/3, 4, (W/8)*(H/6)*pts_per_pixel, 1};

        /* L2: 16 grid + 4 foveal = 20 seeds */
        for (int ty = 0; ty < 4; ty++)
            for (int tx = 0; tx < 4; tx++)
                segments[num_segments++] = {tx*W/4, ty*H/4, W/4, H/4, 2,
                                            (W/16)*(H/16)*pts_per_pixel, 2};
        /* Extra foveal: eyes, nose, mouth */
        segments[num_segments++] = {W/4, H/4, W/2, H/4, 2, (W/8)*(H/8)*pts_per_pixel, 2};
        segments[num_segments++] = {W/3, H/3, W/3, H/4, 2, (W/6)*(H/8)*pts_per_pixel, 2};
        segments[num_segments++] = {W/3, H/2, W/3, H/5, 2, (W/6)*(H/10)*pts_per_pixel, 2};
        segments[num_segments++] = {W/4, H/6, W/2, H/3, 2, (W/8)*(H/6)*pts_per_pixel, 2};

        /* L3: 64 grid + 8 foveal = 72 seeds */
        for (int ty = 0; ty < 8; ty++)
            for (int tx = 0; tx < 8; tx++)
                segments[num_segments++] = {tx*W/8, ty*H/8, W/8, H/8, 1,
                                            (W/8)*(H/8)*pts_per_pixel/4, 3};
        /* Extra foveal detail at 1x1 */
        unsigned rng2 = (unsigned)foveal_seed;
        for (int i = 0; i < 8; i++) {
            rng2 = rng2 * 1103515245 + 12345;
            int cx = W/3 + (int)(((rng2 >> 16) & 0xFF) * W / 3 / 256);
            rng2 = rng2 * 1103515245 + 12345;
            int cy = H/5 + (int)(((rng2 >> 16) & 0xFF) * H * 3 / 5 / 256);
            int rw = W/6, rh = H/6;
            int rx = cx - rw/2; if (rx < 0) rx = 0; if (rx+rw>W) rx=W-rw;
            int ry = cy - rh/2; if (ry < 0) ry = 0; if (ry+rh>H) ry=H-rh;
            segments[num_segments++] = {rx, ry, rw, rh, 1, rw*rh*pts_per_pixel/3, 3};
        }

    } else {
        /* ====== Original quadtree mode ====== */

        /* Level 0: 1 segment = whole image, 8x8 blocks */
        segments[num_segments++] = {0, 0, W, H, 8, (W/8)*(H/8)*pts_per_pixel, 0};

        /* Level 1: 4 quadrants, 4x4 blocks */
        for (int qy = 0; qy < 2; qy++)
            for (int qx = 0; qx < 2; qx++)
                segments[num_segments++] = {qx*W/2, qy*H/2, W/2, H/2, 4,
                                            (W/8)*(H/8)*pts_per_pixel/2, 1};

        /* Level 2: 16 tiles (4x4 grid), 2x2 blocks */
        for (int ty = 0; ty < 4; ty++)
            for (int tx = 0; tx < 4; tx++)
                segments[num_segments++] = {tx*W/4, ty*H/4, W/4, H/4, 2,
                                            (W/16)*(H/16)*pts_per_pixel, 2};

        /* Level 3: 64 tiles (8x8 grid), 1x1 pixels */
        for (int ty = 0; ty < 8; ty++)
            for (int tx = 0; tx < 8; tx++)
                segments[num_segments++] = {tx*W/8, ty*H/8, W/8, H/8, 1,
                                            (W/8)*(H/8)*pts_per_pixel/4, 3};

        /* Level 4: 256 tiles (16x16 grid), 1x1 pixels, fine correction */
        for (int ty = 0; ty < 16; ty++)
            for (int tx = 0; tx < 16; tx++)
                segments[num_segments++] = {tx*W/16, ty*H/16, W/16, H/16, 1,
                                            (W/16)*(H/16)*pts_per_pixel/3, 4};

        /* Level 5: 256 more tiles (shifted by half), 1x1, overlap correction */
        for (int ty = 0; ty < 16; ty++)
            for (int tx = 0; tx < 16; tx++) {
                int rx = tx*W/16 + W/32;
                int ry = ty*H/16 + H/32;
                int rw = W/16, rh = H/16;
                if (rx + rw > W) rw = W - rx;
                if (ry + rh > H) rh = H - ry;
                if (rw > 0 && rh > 0)
                    segments[num_segments++] = {rx, ry, rw, rh, 1,
                                                rw*rh*pts_per_pixel/4, 5};
            }
    }

    printf("Segments: %d (L0=1, L1=4, L2=16, L3=64)\n", num_segments);
    printf("Data: %d seeds × 2 bytes = %d bytes\n", num_segments, num_segments * 2);

    /* GPU alloc */
    uint8_t *d_current, *d_target;
    uint32_t *d_errors;
    cudaMalloc(&d_current, PACKED_SIZE);
    cudaMalloc(&d_target, PACKED_SIZE);
    cudaMalloc(&d_errors, 65536 * sizeof(uint32_t));
    cudaMemcpy(d_target, h_target, PACKED_SIZE, cudaMemcpyHostToDevice);

    uint8_t h_canvas[PACKED_SIZE];
    memset(h_canvas, 0, sizeof(h_canvas));
    uint16_t best_seeds[1024];
    uint32_t h_errors[65536];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int initial_err = count_errors(h_canvas, h_target);
    printf("Initial error: %d / %d (%.1f%%)\n", initial_err, PIXELS, 100.0*initial_err/PIXELS);

    int prev_level = -1;
    for (int s = 0; s < num_segments; s++) {
        Segment* seg = &segments[s];

        if (seg->level != prev_level) {
            printf("\n--- Level %d: %dx%d blocks, %d segments ---\n",
                   seg->level, seg->block_size, seg->block_size,
                   seg->level == 0 ? 1 : seg->level == 1 ? 4 : seg->level == 2 ? 16 : 64);
            prev_level = seg->level;
        }

        cudaMemcpy(d_current, h_canvas, PACKED_SIZE, cudaMemcpyHostToDevice);

        search_segment_kernel<<<256, 256>>>(
            d_current, d_target, d_errors, s,
            seg->rx, seg->ry, seg->rw, seg->rh,
            seg->block_size, seg->num_points);
        cudaDeviceSynchronize();

        cudaMemcpy(h_errors, d_errors, 65536 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t best_err = 0xFFFFFFFF;
        uint16_t best_seed = 0;
        for (int k = 0; k < 65536; k++)
            if (h_errors[k] < best_err) { best_err = h_errors[k]; best_seed = k; }

        int prev_err = count_errors(h_canvas, h_target);
        host_draw_segment(h_canvas, best_seed, s,
                          seg->rx, seg->ry, seg->rw, seg->rh,
                          seg->block_size, seg->num_points);
        int new_err = count_errors(h_canvas, h_target);
        best_seeds[s] = best_seed;

        int seg_pixels = seg->rw * seg->rh;
        printf("  Seg %3d [%3d,%2d %3dx%2d] blk=%d pts=%3d seed=0x%04X  err %d→%d (-%d) seg_err=%d/%d(%.0f%%)\n",
               s, seg->rx, seg->ry, seg->rw, seg->rh,
               seg->block_size, seg->num_points, best_seed,
               prev_err, new_err, prev_err - new_err,
               best_err, seg_pixels, 100.0*best_err/seg_pixels);

        /* Save after each level */
        if (s == num_segments - 1 || segments[s+1].level != seg->level) {
            snprintf(path, sizeof(path), "%s/level%d_err%d.pgm", output_dir, seg->level, new_err);
            save_pgm(path, h_canvas);
            snprintf(path, sizeof(path), "%s/level%d_compare.pgm", output_dir, seg->level);
            save_comparison(path, h_target, h_canvas);
            printf("  → Saved level %d: %d errors (%.1f%%)\n", seg->level, new_err, 100.0*new_err/PIXELS);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    int final_err = count_errors(h_canvas, h_target);
    printf("\n=== RESULT ===\n");
    printf("Final error: %d / %d (%.1f%%)\n", final_err, PIXELS, 100.0*final_err/PIXELS);
    printf("Segments: %d, Data: %d bytes\n", num_segments, num_segments * 2);
    printf("Time: %.1fs\n", elapsed);

    /* Save seeds */
    snprintf(path, sizeof(path), "%s/seeds.bin", output_dir);
    FILE* f = fopen(path, "wb"); fwrite(best_seeds, 2, num_segments, f); fclose(f);

    snprintf(path, sizeof(path), "%s/seeds.txt", output_dir);
    f = fopen(path, "w");
    fprintf(f, "# Segmented seeds: %d segments, %d bytes\n", num_segments, num_segments*2);
    for (int i = 0; i < num_segments; i++)
        fprintf(f, "seg %3d: 0x%04X  L%d [%d,%d %dx%d] blk=%d pts=%d\n",
                i, best_seeds[i], segments[i].level,
                segments[i].rx, segments[i].ry, segments[i].rw, segments[i].rh,
                segments[i].block_size, segments[i].num_points);
    fclose(f);

    printf("Output: %s/\n", output_dir);

    cudaFree(d_current); cudaFree(d_target); cudaFree(d_errors);
    return 0;
}

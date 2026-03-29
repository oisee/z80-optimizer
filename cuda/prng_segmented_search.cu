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

/* ====== Draw LFSR points within a rectangle, XOR onto canvas ====== */
__device__ void draw_segment(
    uint8_t* canvas,
    uint16_t seed,
    int seg_id,        /* unique per segment for LFSR init */
    int rx, int ry,    /* rectangle top-left */
    int rw, int rh,    /* rectangle size */
    int block_size,    /* pixel block size */
    int num_points
) {
    uint32_t state = ((uint32_t)seed << 16) | ((uint32_t)(seg_id * 13 + 0xBEEF));
    for (int i = 0; i < 8; i++) state = lfsr_step(state);

    for (int p = 0; p < num_points; p++) {
        state = lfsr_step(state);

        /* Map to within rectangle */
        int lx = ((state >> 0) & 0xFFFF) % rw;
        int ly = ((state >> 16) & 0xFFFF) % rh;

        /* Align to block grid */
        lx = (lx / block_size) * block_size;
        ly = (ly / block_size) * block_size;

        /* XOR block */
        for (int dy = 0; dy < block_size && (ry + ly + dy) < H; dy++) {
            for (int dx = 0; dx < block_size && (rx + lx + dx) < W; dx++) {
                int x = rx + lx + dx;
                int y = ry + ly + dy;
                int byte_idx = y * (W / 8) + (x / 8);
                int bit_idx = 7 - (x % 8);
                canvas[byte_idx] ^= (1 << bit_idx);
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

/* ====== Host draw ====== */
void host_draw_segment(uint8_t* canvas, uint16_t seed, int seg_id,
                       int rx, int ry, int rw, int rh, int block_size, int num_points) {
    uint32_t state = ((uint32_t)seed << 16) | ((uint32_t)(seg_id * 13 + 0xBEEF));
    for (int i = 0; i < 8; i++) {
        uint32_t bit = state & 1; state >>= 1; if (bit) state ^= 0xB4BCD35C;
    }
    for (int p = 0; p < num_points; p++) {
        uint32_t bit = state & 1; state >>= 1; if (bit) state ^= 0xB4BCD35C;
        int lx = ((state >> 0) & 0xFFFF) % rw;
        int ly = ((state >> 16) & 0xFFFF) % rh;
        lx = (lx / block_size) * block_size;
        ly = (ly / block_size) * block_size;
        for (int dy = 0; dy < block_size && (ry + ly + dy) < H; dy++) {
            for (int dx = 0; dx < block_size && (rx + lx + dx) < W; dx++) {
                int x = rx + lx + dx;
                int y = ry + ly + dy;
                int byte_idx = y * (W / 8) + (x / 8);
                int bit_idx = 7 - (x % 8);
                canvas[byte_idx] ^= (1 << bit_idx);
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

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1<argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i+1<argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1<argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--density") && i+1<argc) pts_per_pixel = atoi(argv[++i]);
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
            int rx = tx*W/16 + W/32;  /* shifted by half-tile */
            int ry = ty*H/16 + H/32;
            int rw = W/16, rh = H/16;
            if (rx + rw > W) rw = W - rx;
            if (ry + rh > H) rh = H - ry;
            if (rw > 0 && rh > 0)
                segments[num_segments++] = {rx, ry, rw, rh, 1,
                                            rw*rh*pts_per_pixel/4, 5};
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

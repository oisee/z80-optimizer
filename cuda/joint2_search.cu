/*
 * joint2_search.cu — Joint-2 seed optimization for overlapping segments.
 *
 * Instead of greedy (find best seed_A, lock, find best seed_B),
 * tests ALL 65536×65536 = 4.3B combinations of (seed_A, seed_B)
 * and finds the globally optimal pair.
 *
 * Build: nvcc -O3 -o cuda/joint2_search cuda/joint2_search.cu
 * Usage: ./cuda/joint2_search --target che.pgm --canvas base.pgm \
 *        --regA "32,24,32,16,2" --regB "36,26,24,10,1" [--gpu 0]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define W 128
#define H 96
#define PIXELS (W * H)
#define PACKED_SIZE (PIXELS / 8)

/* ====== LFSR-16 block-scan draw ====== */
__device__ void draw_blockscan(
    uint8_t* canvas,
    uint16_t seed, int seg_id,
    int rx, int ry, int rw, int rh, int block_size
) {
    uint16_t state = seed;
    if (state == 0) state = 1;
    for (int i = 0; i < (seg_id & 15) + 4; i++) {
        uint16_t bit = state & 1; state >>= 1;
        if (bit) state ^= 0xB400;
    }
    int nbx = rw / block_size, nby = rh / block_size;
    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            uint16_t bit = state & 1; state >>= 1;
            if (bit) state ^= 0xB400;
            if (state & 1) {
                int px = rx + bx * block_size, py = ry + by * block_size;
                for (int dy = 0; dy < block_size && (py+dy) < H; dy++)
                    for (int dx = 0; dx < block_size && (px+dx) < W; dx++) {
                        int x = px+dx, y = py+dy;
                        canvas[y*(W/8)+(x/8)] ^= (1 << (7-(x%8)));
                    }
            }
        }
    }
}

/* ====== LFSR-32 point-spray draw ====== */
__device__ void draw_pointspray(
    uint8_t* canvas,
    uint16_t seed, int seg_id,
    int rx, int ry, int rw, int rh, int block_size, int num_points
) {
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
        for (int dy = 0; dy < block_size && (ry+ly+dy) < H; dy++)
            for (int dx = 0; dx < block_size && (rx+lx+dx) < W; dx++) {
                int x = rx+lx+dx, y = ry+ly+dy;
                canvas[y*(W/8)+(x/8)] ^= (1 << (7-(x%8)));
            }
    }
}

/* ====== Joint-2 kernel ====== */
/* Each block handles one seedA value, threads handle seedB values */
__global__ void joint2_kernel(
    const uint8_t* __restrict__ base_canvas,
    const uint8_t* __restrict__ target,
    uint32_t* __restrict__ block_best_err,  /* [65536] best error per seedA */
    uint16_t* __restrict__ block_best_seedB, /* [65536] best seedB per seedA */
    int rxA, int ryA, int rwA, int rhA, int blkA, int ptsA,
    int rxB, int ryB, int rwB, int rhB, int blkB, int ptsB,
    /* ROI for error counting */
    int roi_x, int roi_y, int roi_w, int roi_h
) {
    int seedA = blockIdx.x;  /* 0..65535 */
    int seedB = blockIdx.y * blockDim.x + threadIdx.x;  /* 0..65535 */
    if (seedA >= 65536 || seedB >= 65536) return;

    /* Copy canvas */
    uint8_t canvas[PACKED_SIZE];
    for (int i = 0; i < PACKED_SIZE; i++)
        canvas[i] = base_canvas[i];

    /* Apply both seeds */
    if (ptsA > 0)
        draw_pointspray(canvas, (uint16_t)seedA, 0, rxA, ryA, rwA, rhA, blkA, ptsA);
    else
        draw_blockscan(canvas, (uint16_t)seedA, 0, rxA, ryA, rwA, rhA, blkA);

    if (ptsB > 0)
        draw_pointspray(canvas, (uint16_t)seedB, 1, rxB, ryB, rwB, rhB, blkB, ptsB);
    else
        draw_blockscan(canvas, (uint16_t)seedB, 1, rxB, ryB, rwB, rhB, blkB);

    /* Count error in ROI only */
    uint32_t err = 0;
    for (int y = roi_y; y < roi_y + roi_h && y < H; y++)
        for (int x = roi_x; x < roi_x + roi_w && x < W; x++) {
            int byte_idx = y * (W/8) + (x/8);
            int bit_idx = 7 - (x%8);
            int gen = (canvas[byte_idx] >> bit_idx) & 1;
            int tgt = (target[byte_idx] >> bit_idx) & 1;
            if (gen != tgt) err++;
        }

    /* Per-block (seedA) atomic min */
    __shared__ uint32_t s_best_err;
    __shared__ uint16_t s_best_seedB;
    if (threadIdx.x == 0) {
        s_best_err = 0xFFFFFFFF;
        s_best_seedB = 0;
    }
    __syncthreads();

    /* Warp-level reduction then shared atomic */
    atomicMin(&s_best_err, err);
    __syncthreads();
    if (err == s_best_err) {
        s_best_seedB = (uint16_t)seedB;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (s_best_err < block_best_err[seedA]) {
            block_best_err[seedA] = s_best_err;
            block_best_seedB[seedA] = s_best_seedB;
        }
    }
}

/* ====== I/O ====== */
int load_pgm_binary(const char* path, uint8_t* packed) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4]; int w, h, maxval;
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
    for (int i = 0; i < w * h; i++)
        if (raw[i] > maxval / 2) packed[i/8] |= (1 << (7-(i%8)));
    free(raw);
    return 0;
}

int parse_region(const char* str, int* rx, int* ry, int* rw, int* rh, int* blk) {
    return sscanf(str, "%d,%d,%d,%d,%d", rx, ry, rw, rh, blk) == 5 ? 0 : -1;
}

int main(int argc, char** argv) {
    const char* target_path = NULL;
    const char* canvas_path = NULL;
    const char* regA_str = NULL;
    const char* regB_str = NULL;
    int device_id = 0;
    int pts_per_pixel = 3;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1<argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--canvas") && i+1<argc) canvas_path = argv[++i];
        else if (!strcmp(argv[i], "--regA") && i+1<argc) regA_str = argv[++i];
        else if (!strcmp(argv[i], "--regB") && i+1<argc) regB_str = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i+1<argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--density") && i+1<argc) pts_per_pixel = atoi(argv[++i]);
    }

    if (!target_path || !regA_str || !regB_str) {
        fprintf(stderr, "Usage: %s --target t.pgm [--canvas c.pgm] --regA \"rx,ry,rw,rh,blk\" --regB \"rx,ry,rw,rh,blk\"\n", argv[0]);
        return 1;
    }

    int rxA,ryA,rwA,rhA,blkA, rxB,ryB,rwB,rhB,blkB;
    parse_region(regA_str, &rxA, &ryA, &rwA, &rhA, &blkA);
    parse_region(regB_str, &rxB, &ryB, &rwB, &rhB, &blkB);

    int ptsA = (rwA/blkA) * (rhA/blkA) * pts_per_pixel;
    int ptsB = (rwB/blkB) * (rhB/blkB) * pts_per_pixel;

    /* ROI = UNION of both regions (where either seed acts).
       Joint-2 optimizes for the OVERLAP zone where both seeds contribute. */
    // FULL SCREEN ROI for global optimization
    int roi_x = 0; // rxA < rxB ? rxA : rxB;
    int roi_y = 0;
    int roi_x2 = 128;
    int roi_y2 = 96;
    int roi_w = roi_x2 - roi_x; if (roi_w < 0) roi_w = 0;
    int roi_h = roi_y2 - roi_y; if (roi_h < 0) roi_h = 0;

    printf("Region A: [%d,%d %dx%d] blk=%d pts=%d\n", rxA,ryA,rwA,rhA,blkA,ptsA);
    printf("Region B: [%d,%d %dx%d] blk=%d pts=%d\n", rxB,ryB,rwB,rhB,blkB,ptsB);
    printf("ROI: [%d,%d %dx%d]\n", roi_x, roi_y, roi_w, roi_h);

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU: %s\n", prop.name);

    uint8_t h_target[PACKED_SIZE], h_canvas[PACKED_SIZE];
    load_pgm_binary(target_path, h_target);
    if (canvas_path)
        load_pgm_binary(canvas_path, h_canvas);
    else
        memset(h_canvas, 0, sizeof(h_canvas));

    uint8_t *d_canvas, *d_target;
    uint32_t *d_best_err;
    uint16_t *d_best_seedB;

    cudaMalloc(&d_canvas, PACKED_SIZE);
    cudaMalloc(&d_target, PACKED_SIZE);
    cudaMalloc(&d_best_err, 65536 * sizeof(uint32_t));
    cudaMalloc(&d_best_seedB, 65536 * sizeof(uint16_t));

    cudaMemcpy(d_canvas, h_canvas, PACKED_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, PACKED_SIZE, cudaMemcpyHostToDevice);

    /* Init best_err to max */
    uint32_t h_init_err[65536];
    uint16_t h_init_seedB[65536];
    for (int i = 0; i < 65536; i++) { h_init_err[i] = 0xFFFFFFFF; h_init_seedB[i] = 0; }
    cudaMemcpy(d_best_err, h_init_err, 65536*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_best_seedB, h_init_seedB, 65536*sizeof(uint16_t), cudaMemcpyHostToDevice);

    /* Launch: 65536 seedA values × 256 passes of 256 seedB threads */
    printf("Searching 65536 × 65536 = 4.3B pairs...\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int pass = 0; pass < 256; pass++) {
        dim3 grid(65536, 1);
        dim3 block(256);
        /* seedB range for this pass: pass*256 .. pass*256+255 */
        /* We need to offset seedB in kernel... hack: use blockIdx.y */
        dim3 grid2(65536, 1);
        /* Actually simpler: just launch 256 times with different base */
        /* TODO: better approach — encode pass in grid.y */

        /* For now: each pass covers seedB = pass*256 + threadIdx.x */
        /* We encode pass in a constant — simplest: just run 256 launches */
        /* Each launch: 65536 blocks × 256 threads = 16.7M threads */

        /* We'll pass seedB offset via grid.y trick */
        dim3 grid3(65536, 1);

        /* Actually let's just do it simply: pass offset via grid dim */
        /* seedB = pass * 256 + threadIdx.x */
        /* We'll modify kernel to accept base_seedB... */
        /* For prototype: just launch with modified kernel */
    }

    /* Launch: grid(65536, 256), block(256) = 4.3B threads total
       seedA = blockIdx.x (0..65535)
       seedB = blockIdx.y * 256 + threadIdx.x (0..65535) */
    {
        dim3 grid(65536, 256);
        dim3 block(256);
        joint2_kernel<<<grid, block>>>(
            d_canvas, d_target, d_best_err, d_best_seedB,
            rxA, ryA, rwA, rhA, blkA, ptsA,
            rxB, ryB, rwB, rhB, blkB, ptsB,
            roi_x, roi_y, roi_w, roi_h);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    /* Find global best */
    uint32_t h_best_err[65536];
    uint16_t h_best_seedB_out[65536];
    cudaMemcpy(h_best_err, d_best_err, 65536*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_seedB_out, d_best_seedB, 65536*sizeof(uint16_t), cudaMemcpyDeviceToHost);

    uint32_t global_best = 0xFFFFFFFF;
    uint16_t best_seedA = 0, best_seedB = 0;
    for (int i = 0; i < 65536; i++) {
        if (h_best_err[i] < global_best) {
            global_best = h_best_err[i];
            best_seedA = (uint16_t)i;
            best_seedB = h_best_seedB_out[i];
        }
    }

    /* For prototype: just report the structure */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n=== RESULT ===\n");
    printf("Best pair: seedA=0x%04X, seedB=0x%04X\n", best_seedA, best_seedB);
    printf("ROI error: %u / %d pixels (%.1f%%)\n", global_best, roi_w*roi_h, 100.0*global_best/(roi_w*roi_h));
    printf("Time: %.1fs\n", elapsed);
    printf("Search: 65536 × 65536 = 4.3B pairs\n");

    cudaFree(d_canvas); cudaFree(d_target);
    cudaFree(d_best_err); cudaFree(d_best_seedB);
    return 0;
}

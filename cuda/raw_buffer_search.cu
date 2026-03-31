/*
 * raw_buffer_search.cu — Raw LFSR buffer search with grayscale loss.
 * 
 * Architecture: seed(u16) → 768 bits LFSR → 32×24 buffer → XOR at block_size
 * Joint-2: (seedA@blkA, seedB@blkB) searched as u32
 *
 * Build: nvcc -O3 -o cuda/raw_buffer_search cuda/raw_buffer_search.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define W 128
#define H 96
#define BW 32
#define BH 24
#define BUFSIZE 768

/* LFSR-16 fill buffer: seed → 768 bits */
__device__ void lfsr_fill(uint16_t seed, uint8_t* buf, int warmup) {
    uint16_t state = seed;
    if (state == 0) state = 1;
    for (int i = 0; i < warmup; i++) {
        uint16_t bit = state & 1; state >>= 1;
        if (bit) state ^= 0xB400;
    }
    for (int i = 0; i < BUFSIZE; i++) {
        uint16_t bit = state & 1; state >>= 1;
        if (bit) state ^= 0xB400;
        buf[i] = state & 1;
    }
}

/* Apply buffer: XOR solid blocks onto packed canvas */
__device__ void apply_buffer(uint8_t* canvas, const uint8_t* buf, int block_size, int ox=0, int oy=0) {
    for (int by = 0; by < BH; by++) {
        for (int bx = 0; bx < BW; bx++) {
            if (buf[by * BW + bx]) {
                int px = ox + bx * block_size;
                int py = oy + by * block_size;
                for (int dy = 0; dy < block_size && (py+dy) < H; dy++) {
                    for (int dx = 0; dx < block_size && (px+dx) < W; dx++) {
                        int x = px + dx, y = py + dy;
                        canvas[y * (W/8) + (x/8)] ^= (1 << (7 - (x%8)));
                    }
                }
            }
        }
    }
}

/* Grayscale loss: count pixels per 2×2, compare with target gray */
__device__ uint32_t grayscale_loss(const uint8_t* canvas, const uint8_t* target_gray) {
    uint32_t loss = 0;
    for (int by = 0; by < H/2; by++) {
        for (int bx = 0; bx < W/2; bx++) {
            int density = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int x = bx*2+dx, y = by*2+dy;
                    int bit = (canvas[y*(W/8)+(x/8)] >> (7-(x%8))) & 1;
                    density += bit;
                }
            }
            int target_d = target_gray[by * (W/2) + bx];
            int diff = density - target_d;
            loss += (diff < 0) ? -diff : diff;
        }
    }
    return loss;
}

#define PACKED_SIZE (W * H / 8)

/* Joint-2 kernel: seedA@blkA + seedB@blkB */
__global__ void joint2_raw_kernel(
    const uint8_t* __restrict__ base_canvas,
    const uint8_t* __restrict__ target_gray,
    uint32_t* __restrict__ block_best_err,
    uint16_t* __restrict__ block_best_seedB,
    int blkA, int warmupA,
    int blkB, int warmupB
) {
    int seedA = blockIdx.x;
    int seedB = blockIdx.y * blockDim.x + threadIdx.x;
    if (seedA >= 65536 || seedB >= 65536) return;

    uint8_t canvas[PACKED_SIZE];
    for (int i = 0; i < PACKED_SIZE; i++) canvas[i] = base_canvas[i];

    uint8_t bufA[BUFSIZE], bufB[BUFSIZE];
    lfsr_fill((uint16_t)seedA, bufA, warmupA);
    lfsr_fill((uint16_t)seedB, bufB, warmupB);

    apply_buffer(canvas, bufA, blkA);
    apply_buffer(canvas, bufB, blkB);

    uint32_t err = grayscale_loss(canvas, target_gray);

    /* Per-seedA reduction */
    __shared__ uint32_t s_best;
    __shared__ uint16_t s_seedB;
    if (threadIdx.x == 0) { s_best = 0xFFFFFFFF; s_seedB = 0; }
    __syncthreads();
    atomicMin(&s_best, err);
    __syncthreads();
    if (err == s_best) s_seedB = (uint16_t)seedB;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_best < block_best_err[seedA]) {
            block_best_err[seedA] = s_best;
            block_best_seedB[seedA] = s_seedB;
        }
    }
}

/* Single-seed greedy kernel */
__global__ void greedy_raw_kernel(
    const uint8_t* __restrict__ base_canvas,
    const uint8_t* __restrict__ target_gray,
    const uint8_t* __restrict__ target_bin,
    uint32_t* __restrict__ errors,
    int blk, int warmup, int ox, int oy
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= 65536) return;

    uint8_t canvas[PACKED_SIZE];
    for (int i = 0; i < PACKED_SIZE; i++) canvas[i] = base_canvas[i];

    uint8_t buf[BUFSIZE];
    lfsr_fill((uint16_t)seed, buf, warmup);
    apply_buffer(canvas, buf, blk, ox, oy);

    {
        uint32_t berr = 0;
        for (int i = 0; i < PACKED_SIZE; i++) berr += __builtin_popcount(canvas[i] ^ d_target_bin[i]);
        errors[seed] = berr;
    }
}

/* I/O */
int load_pgm_binary(const char* path, uint8_t* packed) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4]; int w, h, maxval;
    fscanf(f, "%2s", magic);
    int c; while ((c = fgetc(f)) != EOF) { if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); } else if (c > ' ') { ungetc(c, f); break; } }
    fscanf(f, "%d %d %d", &w, &h, &maxval); fgetc(f);
    memset(packed, 0, PACKED_SIZE);
    uint8_t* raw = (uint8_t*)malloc(w * h);
    fread(raw, 1, w * h, f); fclose(f);
    for (int i = 0; i < w * h; i++)
        if (raw[i] > maxval / 2) packed[i/8] |= (1 << (7-(i%8)));
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

void make_grayscale_target(const uint8_t* target_packed, uint8_t* target_gray) {
    /* Count white pixels per 2×2 block → 0-4 */
    for (int by = 0; by < H/2; by++) {
        for (int bx = 0; bx < W/2; bx++) {
            int count = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    int x = bx*2+dx, y = by*2+dy;
                    count += (target_packed[y*(W/8)+(x/8)] >> (7-(x%8))) & 1;
                }
            target_gray[by * (W/2) + bx] = count;
        }
    }
}

int main(int argc, char** argv) {
    const char* target_path = NULL;
    const char* output_path = "result.pgm";
    int gpu = 0;
    int mode = 0; /* 0=greedy 4-layer, 1=joint L0+L1, 2=joint L2+L3, 3=full joint */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target")) target_path = argv[++i];
        else if (!strcmp(argv[i], "--output")) output_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu")) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")) mode = atoi(argv[++i]);
    }
    if (!target_path) { fprintf(stderr, "Usage: %s --target t.pgm [--output o.pgm] [--mode 0-3]\n", argv[0]); return 1; }

    cudaSetDevice(gpu);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu);
    printf("GPU: %s\n", prop.name);

    uint8_t h_target[PACKED_SIZE];
    load_pgm_binary(target_path, h_target);

    uint8_t h_target_gray[H/2 * W/2];
    make_grayscale_target(h_target, h_target_gray);

    uint8_t *d_canvas, *d_target_gray, *d_target_bin;
    uint32_t *d_errors, *d_best_err;
    uint16_t *d_best_seedB;

    cudaMalloc(&d_canvas, PACKED_SIZE);
    cudaMalloc(&d_target_gray, H/2 * W/2);
    cudaMalloc(&d_target_bin, PACKED_SIZE);
    cudaMemcpy(d_target_bin, h_target, PACKED_SIZE, cudaMemcpyHostToDevice);
    cudaMalloc(&d_errors, 65536 * sizeof(uint32_t));
    cudaMalloc(&d_best_err, 65536 * sizeof(uint32_t));
    cudaMalloc(&d_best_seedB, 65536 * sizeof(uint16_t));
    cudaMemcpy(d_target_gray, h_target_gray, H/2 * W/2, cudaMemcpyHostToDevice);

    uint8_t h_canvas[PACKED_SIZE];
    memset(h_canvas, 0, sizeof(h_canvas));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (mode == 0) {
        /* Quadtree: 1×8×8 + 4×4×4 + 16×2×2 + 64×1×1 = 85 seeds */
        struct Seg { int ox, oy, blk, warmup; };
        Seg segs[85];
        int ns = 0;
        /* L0 */
        segs[ns++] = {0, 0, 8, 0};
        /* L1: 4 quadrants */
        for (int qy=0;qy<2;qy++) for(int qx=0;qx<2;qx++)
            segs[ns++] = {qx*64, qy*48, 4, 10+qy*2+qx};
        /* L2: 16 tiles */
        for (int ty=0;ty<4;ty++) for(int tx=0;tx<4;tx++)
            segs[ns++] = {tx*32, ty*24, 2, 20+ty*4+tx};
        /* L3: 64 tiles */
        for (int ty=0;ty<8;ty++) for(int tx=0;tx<8;tx++)
            segs[ns++] = {tx*16, ty*12, 1, 40+ty*8+tx};
        
        printf("Quadtree: %d segments\n", ns);
        uint16_t all_seeds[85];
        
        for (int layer = 0; layer < ns; layer++) {
            int blk = segs[layer].blk;
            int blks[] = {blk}; /* dummy for compat */
            cudaMemcpy(d_canvas, h_canvas, PACKED_SIZE, cudaMemcpyHostToDevice);
            
            uint32_t h_errors[65536];
            greedy_raw_kernel<<<256, 256>>>(d_canvas, d_target_gray, d_target_bin, d_errors, segs[layer].blk, segs[layer].warmup, segs[layer].ox, segs[layer].oy);
            cudaDeviceSynchronize();
            cudaMemcpy(h_errors, d_errors, 65536*sizeof(uint32_t), cudaMemcpyDeviceToHost);

            uint32_t best = 0xFFFFFFFF; uint16_t best_s = 0;
            for (int s = 0; s < 65536; s++)
                if (h_errors[s] < best) { best = h_errors[s]; best_s = s; }

            all_seeds[layer] = best_s;

            /* Apply to host canvas */
            uint8_t buf[BUFSIZE];
            uint16_t state = best_s; if (!state) state = 1;
            for (int i = 0; i < segs[layer].warmup; i++) { uint16_t b=state&1;state>>=1;if(b)state^=0xB400; }
            for (int i = 0; i < BUFSIZE; i++) { uint16_t b=state&1;state>>=1;if(b)state^=0xB400;buf[i]=state&1; }
            int sblk=segs[layer].blk, sox=segs[layer].ox, soy=segs[layer].oy;
            for (int by = 0; by < BH; by++)
                for (int bx = 0; bx < BW; bx++)
                    if (buf[by*BW+bx])
                        for (int dy=0;dy<sblk&&soy+by*sblk+dy<H;dy++)
                            for (int dx=0;dx<sblk&&sox+bx*sblk+dx<W;dx++) {
                                int x=sox+bx*sblk+dx, y=soy+by*sblk+dy;
                                h_canvas[y*(W/8)+(x/8)] ^= (1<<(7-(x%8)));
                            }

            /* Count binary error */
            int err = 0;
            for (int i = 0; i < PACKED_SIZE; i++) err += __builtin_popcount(h_canvas[i] ^ h_target[i]);
            printf("L%d (blk=%d): seed=0x%04X, gray_loss=%u, binary_err=%d/12288 (%.1f%%)\n",
                   layer, blks[layer], best_s, best, err, 100.0*err/12288);
        }

        printf("\n%d seeds = %d bytes\n", ns, ns*2);

    } else if (mode == 1 || mode == 2) {
        /* Joint L0+L1 or L2+L3 */
        int blkA = (mode == 1) ? 8 : 2;
        int blkB = (mode == 1) ? 4 : 1;
        int wA = (mode == 1) ? 4 : 6;
        int wB = (mode == 1) ? 5 : 7;

        printf("Joint-2: blk=%d + blk=%d\n", blkA, blkB);

        cudaMemcpy(d_canvas, h_canvas, PACKED_SIZE, cudaMemcpyHostToDevice);

        uint32_t h_init[65536]; uint16_t h_initB[65536];
        for (int i = 0; i < 65536; i++) { h_init[i] = 0xFFFFFFFF; h_initB[i] = 0; }
        cudaMemcpy(d_best_err, h_init, 65536*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_best_seedB, h_initB, 65536*sizeof(uint16_t), cudaMemcpyHostToDevice);

        dim3 grid(65536, 256);
        dim3 block(256);
        joint2_raw_kernel<<<grid, block>>>(d_canvas, d_target_gray, d_best_err, d_best_seedB, blkA, wA, blkB, wB);
        cudaDeviceSynchronize();

        uint32_t h_berr[65536]; uint16_t h_bseedB[65536];
        cudaMemcpy(h_berr, d_best_err, 65536*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bseedB, d_best_seedB, 65536*sizeof(uint16_t), cudaMemcpyDeviceToHost);

        uint32_t gbest = 0xFFFFFFFF; uint16_t gA=0, gB=0;
        for (int i = 0; i < 65536; i++)
            if (h_berr[i] < gbest) { gbest = h_berr[i]; gA = i; gB = h_bseedB[i]; }

        printf("Best: seedA=0x%04X seedB=0x%04X gray_loss=%u\n", gA, gB, gbest);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;

    save_pgm(output_path, h_canvas);
    printf("Time: %.1fs\nOutput: %s\n", elapsed, output_path);

    cudaFree(d_canvas); cudaFree(d_target_gray); cudaFree(d_target_bin);
    cudaFree(d_errors); cudaFree(d_best_err); cudaFree(d_best_seedB);
    return 0;
}

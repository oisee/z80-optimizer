// z80_image_search.cu — GPU brute-force: find pRNG seed that draws target image
// Inspired by Introspec's BB (Big Brother) 256-byte intro
//
// Approach: pRNG with SEED → generates "random" pixels → compare to target
// GPU tests billions of seeds/second with block-based similarity metric
//
// Build: nvcc -O3 -o z80_image_search z80_image_search.cu
// Usage: ./z80_image_search --target face.bin [--mode exhaust|hill] [--gpu N]
//
// Target format: raw 1-bit mono, 128×96 pixels = 1536 bytes (16 bytes/row)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <curand_kernel.h>

#define IMG_W 128
#define IMG_H 96
#define IMG_BW (IMG_W / 8)  // 16 bytes per row
#define IMG_SIZE (IMG_BW * IMG_H)  // 1536 bytes

#define BLOCK_W 8   // comparison block size
#define BLOCK_H 8
#define N_BLOCKS_X (IMG_W / BLOCK_W)   // 16
#define N_BLOCKS_Y (IMG_H / BLOCK_H)   // 12
#define N_BLOCKS (N_BLOCKS_X * N_BLOCKS_Y)  // 192

// ===== Patrik Rak CMWC pRNG =====
struct PRNG {
    uint8_t table[8];
    uint8_t idx;
    uint8_t carry;
};

__device__ void prng_init(PRNG *p, uint64_t seed) {
    p->idx = 0;
    p->carry = (seed >> 56) & 0xFF;
    for (int i = 0; i < 8; i++) {
        p->table[i] = (seed >> (i * 7)) & 0xFF;
        if (p->table[i] == 0) p->table[i] = i + 1;
    }
}

__device__ uint8_t prng_next(PRNG *p) {
    p->idx = (p->idx + 1) & 7;
    uint8_t y = p->table[p->idx];
    uint16_t t = (uint16_t)y * 253 + p->carry;
    p->carry = (uint8_t)(t >> 8);
    uint8_t x = ~(uint8_t)(t & 0xFF);
    p->table[p->idx] = x;
    return x;
}

// ===== Similarity metrics =====

__constant__ uint8_t d_target[IMG_SIZE];

// Pre-computed target block features (set by host)
__constant__ int d_target_popcount[N_BLOCKS];  // bits set per block

// Block Hamming: count mismatched bits per 8×8 block, weighted
__device__ int score_block_hamming(const uint8_t *img) {
    int total = 0;
    for (int by = 0; by < N_BLOCKS_Y; by++) {
        for (int bx = 0; bx < N_BLOCKS_X; bx++) {
            int block_diff = 0;
            for (int y = 0; y < BLOCK_H; y++) {
                int row = by * BLOCK_H + y;
                // Each block is 1 byte wide (8 pixels / 8 = 1 byte per block column)
                int byte_idx = row * IMG_BW + bx;
                block_diff += __popc(img[byte_idx] ^ d_target[byte_idx]);
            }
            // Weight: center blocks matter more (face is in center)
            int cx = bx - N_BLOCKS_X / 2;
            int cy = by - N_BLOCKS_Y / 2;
            int dist2 = cx * cx + cy * cy;
            int weight = (dist2 < 16) ? 4 : (dist2 < 36) ? 2 : 1;
            total += block_diff * weight;
        }
    }
    return total;  // lower = better
}

// Edge-aware: XOR adjacent pixels, compare edge maps
__device__ int score_edges(const uint8_t *img) {
    int total = 0;
    for (int y = 0; y < IMG_H - 1; y++) {
        for (int x = 0; x < IMG_BW; x++) {
            int idx = y * IMG_BW + x;
            // Vertical edges
            uint8_t edge_gen = img[idx] ^ img[idx + IMG_BW];
            uint8_t edge_tgt = d_target[idx] ^ d_target[idx + IMG_BW];
            total += __popc(edge_gen ^ edge_tgt);
        }
    }
    return total;
}

// Combined score
__device__ int combined_score(const uint8_t *img) {
    int hamming = score_block_hamming(img);
    int edges = score_edges(img);
    return hamming * 2 + edges;  // weight block similarity higher
}

// ===== Mode 1: Exhaustive search =====
__device__ uint32_t d_bestScore;
__device__ uint64_t d_bestSeed;

__global__ void search_exhaust(uint64_t offset, uint32_t count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seed = offset + tid;

    PRNG prng;
    prng_init(&prng, seed);

    uint8_t img[IMG_SIZE];
    for (int i = 0; i < IMG_SIZE; i++) {
        img[i] = prng_next(&prng);
    }

    int score = combined_score(img);

    uint32_t old = atomicMin(&d_bestScore, (uint32_t)score);
    if ((uint32_t)score <= old) {
        atomicExch((unsigned long long *)&d_bestSeed, (unsigned long long)seed);
    }
}

// ===== Mode 2: Hill climbing =====
__global__ void search_hill_climb(uint64_t *seeds, int *scores, int n_pop,
                                   int n_mutations, unsigned int rng_seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pop) return;

    curandState rng;
    curand_init(rng_seed, tid, 0, &rng);

    uint64_t seed = seeds[tid];
    int best_score = scores[tid];

    for (int m = 0; m < n_mutations; m++) {
        // Mutate: flip 1-3 random bits in seed
        uint64_t new_seed = seed;
        int n_flips = 1 + (curand(&rng) % 3);
        for (int f = 0; f < n_flips; f++) {
            int bit = curand(&rng) % 64;
            new_seed ^= (1ULL << bit);
        }

        PRNG prng;
        prng_init(&prng, new_seed);

        uint8_t img[IMG_SIZE];
        for (int i = 0; i < IMG_SIZE; i++) {
            img[i] = prng_next(&prng);
        }

        int score = combined_score(img);

        if (score < best_score) {
            best_score = score;
            seed = new_seed;
        }
    }

    seeds[tid] = seed;
    scores[tid] = best_score;
}

// ===== Host =====

void dump_ascii(uint64_t seed) {
    // Regenerate on CPU
    PRNG prng;
    // CPU version of prng_init
    prng.idx = 0;
    prng.carry = (seed >> 56) & 0xFF;
    for (int i = 0; i < 8; i++) {
        prng.table[i] = (seed >> (i * 7)) & 0xFF;
        if (prng.table[i] == 0) prng.table[i] = i + 1;
    }

    uint8_t img[IMG_SIZE];
    for (int i = 0; i < IMG_SIZE; i++) {
        prng.idx = (prng.idx + 1) & 7;
        uint8_t y = prng.table[prng.idx];
        uint16_t t = (uint16_t)y * 253 + prng.carry;
        prng.carry = (uint8_t)(t >> 8);
        uint8_t x = ~(uint8_t)(t & 0xFF);
        prng.table[prng.idx] = x;
        img[i] = x;
    }

    printf("\n  Generated image (SEED=0x%016llX):\n", (unsigned long long)seed);
    for (int y = 0; y < IMG_H; y += 2) {
        printf("  ");
        for (int x = 0; x < IMG_W; x += 2) {
            int bx = x / 8;
            int bit = 7 - (x % 8);
            int px = (img[y * IMG_BW + bx] >> bit) & 1;
            printf("%c", px ? '#' : ' ');
        }
        printf("\n");
    }

    // Save raw
    char fname[64];
    snprintf(fname, sizeof(fname), "result_%016llX.bin", (unsigned long long)seed);
    FILE *f = fopen(fname, "wb");
    if (f) { fwrite(img, 1, IMG_SIZE, f); fclose(f); }
    printf("  Saved: %s\n", fname);
}

int main(int argc, char *argv[]) {
    int gpuId = 0;
    int mode = 0;  // 0=exhaust, 1=hill
    const char *targetPath = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpuId = atoi(argv[++i]);
        if (!strcmp(argv[i], "--target") && i + 1 < argc) targetPath = argv[++i];
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            if (!strcmp(argv[i + 1], "hill")) mode = 1;
            i++;
        }
    }

    if (!targetPath) {
        fprintf(stderr, "Usage: z80_image_search --target image.bin [--mode exhaust|hill] [--gpu N]\n");
        fprintf(stderr, "  image.bin: 1536 bytes, 128x96 mono (1 bit/pixel, 16 bytes/row)\n");
        return 1;
    }

    cudaSetDevice(gpuId);

    // Load target
    uint8_t target[IMG_SIZE];
    FILE *f = fopen(targetPath, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", targetPath); return 1; }
    fread(target, 1, IMG_SIZE, f);
    fclose(f);
    cudaMemcpyToSymbol(d_target, target, IMG_SIZE);

    // Pre-compute target block popcounts
    int target_popcounts[N_BLOCKS];
    for (int by = 0; by < N_BLOCKS_Y; by++)
        for (int bx = 0; bx < N_BLOCKS_X; bx++) {
            int pc = 0;
            for (int y = 0; y < BLOCK_H; y++)
                pc += __builtin_popcount(target[(by * BLOCK_H + y) * IMG_BW + bx]);
            target_popcounts[by * N_BLOCKS_X + bx] = pc;
        }
    cudaMemcpyToSymbol(d_target_popcount, target_popcounts, sizeof(target_popcounts));

    int target_bits = 0;
    for (int i = 0; i < IMG_SIZE; i++) target_bits += __builtin_popcount(target[i]);
    printf("Target: %s (%d bytes, %d bits set = %.1f%% density)\n",
           targetPath, IMG_SIZE, target_bits, target_bits * 100.0 / (IMG_W * IMG_H));

    if (mode == 0) {
        // Exhaustive: 4-byte seed space
        printf("Mode: exhaustive (4-byte seed = 4.3B)\n");
        uint64_t total = (uint64_t)1 << 32;
        uint32_t bs = 256;
        uint64_t batch = (uint64_t)bs * 65535;

        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initSeed = 0;
        cudaMemcpyToSymbol(d_bestScore, &initScore, sizeof(initScore));
        cudaMemcpyToSymbol(d_bestSeed, &initSeed, sizeof(initSeed));

        time_t t0 = time(NULL);
        for (uint64_t off = 0; off < total; off += batch) {
            uint32_t cnt = (uint32_t)((total - off < batch) ? total - off : batch);
            search_exhaust<<<(cnt + bs - 1) / bs, bs>>>(off, cnt);
            cudaDeviceSynchronize();

            if ((off % (256 * 1024 * 1024)) == 0 && off > 0) {
                uint32_t cur; uint64_t curSeed;
                cudaMemcpyFromSymbol(&cur, d_bestScore, sizeof(cur));
                cudaMemcpyFromSymbol(&curSeed, d_bestSeed, sizeof(curSeed));
                double pct = off * 100.0 / total;
                fprintf(stderr, "%.1f%% — best score=%d seed=0x%016llX\n",
                        pct, cur, (unsigned long long)curSeed);
            }
        }

        uint32_t bestScore; uint64_t bestSeed;
        cudaMemcpyFromSymbol(&bestScore, d_bestScore, sizeof(bestScore));
        cudaMemcpyFromSymbol(&bestSeed, d_bestSeed, sizeof(bestSeed));

        printf("\n=== RESULT (exhaustive) ===\n");
        printf("Best score: %d\nBest seed: 0x%016llX\n", bestScore, (unsigned long long)bestSeed);
        printf("Time: %lds\n", (long)(time(NULL) - t0));
        dump_ascii(bestSeed);

    } else {
        // Hill climbing: large seed space, genetic approach
        printf("Mode: hill climbing (64-bit seed, population=10000, 1000 generations)\n");
        int pop_size = 10000;
        int generations = 1000;
        int mutations_per_gen = 100;

        // Init random population
        uint64_t *h_seeds = (uint64_t *)malloc(pop_size * sizeof(uint64_t));
        int *h_scores = (int *)malloc(pop_size * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < pop_size; i++) {
            h_seeds[i] = ((uint64_t)rand() << 32) | rand();
            h_scores[i] = 999999;
        }

        uint64_t *d_seeds; int *d_scores;
        cudaMalloc(&d_seeds, pop_size * sizeof(uint64_t));
        cudaMalloc(&d_scores, pop_size * sizeof(int));

        // Initial scoring
        cudaMemcpy(d_seeds, h_seeds, pop_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scores, h_scores, pop_size * sizeof(int), cudaMemcpyHostToDevice);

        time_t t0 = time(NULL);
        int best_ever = 999999;
        uint64_t best_seed_ever = 0;

        for (int gen = 0; gen < generations; gen++) {
            search_hill_climb<<<(pop_size + 255) / 256, 256>>>(
                d_seeds, d_scores, pop_size, mutations_per_gen, gen * 1337);
            cudaDeviceSynchronize();

            // Find best in population
            cudaMemcpy(h_seeds, d_seeds, pop_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_scores, d_scores, pop_size * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < pop_size; i++) {
                if (h_scores[i] < best_ever) {
                    best_ever = h_scores[i];
                    best_seed_ever = h_seeds[i];
                }
            }

            if (gen % 100 == 0) {
                fprintf(stderr, "Gen %d: best=%d seed=0x%016llX\n",
                        gen, best_ever, (unsigned long long)best_seed_ever);
            }

            // Selection: replace worst 50% with mutated copies of best 50%
            // Simple: sort by score, copy top half to bottom half
            // (Quick hack: just let hill climbing do its thing per individual)
        }

        printf("\n=== RESULT (hill climbing) ===\n");
        printf("Best score: %d\nBest seed: 0x%016llX\n", best_ever, (unsigned long long)best_seed_ever);
        printf("Time: %lds (%d generations × %d population × %d mutations)\n",
               (long)(time(NULL) - t0), generations, pop_size, mutations_per_gen);
        dump_ascii(best_seed_ever);

        free(h_seeds); free(h_scores);
        cudaFree(d_seeds); cudaFree(d_scores);
    }

    return 0;
}

// z80_prng_search.cu — GPU brute-force pRNG SEED search
// Simulates ZX Spectrum pRNG → screen fill → fitness scoring
// Finds SEEDs that produce visually interesting patterns
//
// Build: nvcc -O3 -o z80_prng_search z80_prng_search.cu
// Usage: ./z80_prng_search [--target image.bin] [--method cmwc|xorshift] [--gpu N]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define SCREEN_SIZE 6144    // pixel area: $4000-$57FF
#define ATTR_SIZE   768     // attribute area: $5800-$5AFF
#define TOTAL_SCREEN (SCREEN_SIZE + ATTR_SIZE)  // 6912 bytes

// ===== Patrik Rak CMWC pRNG (best quality) =====
// Exact Z80 implementation from Hole #17 by .ded^RMDA
// State: 8-byte table + 1-byte index + 1-byte carry = 10 bytes
// Algorithm: t = 253 * y + carry; carry = t>>8; result = ~(t&0xFF)
// Period: ~2^66
struct PRNGState {
    uint8_t table[8];   // q[0..7]
    uint8_t idx;        // current index (0-7)
    uint8_t carry;      // carry byte
};

__device__ uint8_t cmwc_next(PRNGState *s) {
    s->idx = (s->idx + 1) & 7;
    uint8_t y = s->table[s->idx];
    // Exact Z80 method: 253*y = 256*y - 3*y
    // Z80 does: ba = 256*y + carry; then subtract y three times
    // We just do the multiply directly on GPU
    uint16_t t = (uint16_t)y * 253 + s->carry;
    s->carry = (uint8_t)(t >> 8);
    uint8_t x = ~(uint8_t)(t & 0xFF);  // CPL in Z80
    s->table[s->idx] = x;
    return x;
}

// ===== XORShift 16-bit pRNG (fast, mediocre quality) =====
__device__ uint8_t xorshift_next(uint16_t *state) {
    uint16_t s = *state;
    s ^= s << 7;
    s ^= s >> 9;
    s ^= s << 8;
    *state = s;
    return (uint8_t)(s >> 8);
}

// ===== Fitness scoring =====

// Score 1: Hamming distance to target (lower = better match)
__device__ int score_hamming(const uint8_t *screen, const uint8_t *target, int size) {
    int dist = 0;
    for (int i = 0; i < size; i++) {
        dist += __popc(screen[i] ^ target[i]);
    }
    return dist;  // 0 = perfect match, max = size * 8
}

// Score 2: Visual entropy (how "interesting" the pattern looks)
// Sweet spot: not too uniform (noise), not too sparse (blank)
__device__ int score_entropy(const uint8_t *screen, int size) {
    // Count byte value distribution
    int counts[16] = {};  // nibble histogram (cheaper than full byte)
    for (int i = 0; i < size; i++) {
        counts[screen[i] >> 4]++;
        counts[screen[i] & 0xF]++;
    }
    // Compute simplified entropy: sum of |count - expected|
    int expected = size * 2 / 16;
    int deviation = 0;
    for (int i = 0; i < 16; i++) {
        int d = counts[i] - expected;
        if (d < 0) d = -d;
        deviation += d;
    }
    return deviation;  // lower = more uniform distribution
}

// Score 3: Horizontal structure (autocorrelation)
// Patterns with horizontal coherence look intentional
__device__ int score_structure(const uint8_t *screen, int width, int height) {
    int correlation = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 1; x < width; x++) {
            // XOR adjacent bytes: 0 = identical, 0xFF = opposite
            uint8_t diff = screen[y * width + x] ^ screen[y * width + x - 1];
            correlation += 8 - __popc(diff);  // higher = more correlated
        }
    }
    return correlation;  // higher = more horizontal structure
}

// Score 4: Vertical symmetry (patterns that look designed)
__device__ int score_symmetry(const uint8_t *screen, int width, int height) {
    int sym = 0;
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t top = screen[y * width + x];
            uint8_t bot = screen[(height - 1 - y) * width + x];
            sym += 8 - __popc(top ^ bot);
        }
    }
    return sym;  // higher = more vertically symmetric
}

// Combined fitness: lower = better
__device__ int combined_score(const uint8_t *screen, const uint8_t *target, int hasTarget) {
    int score = 0;

    if (hasTarget) {
        // Primary: match target image
        score = score_hamming(screen, target, SCREEN_SIZE);
    } else {
        // No target: find "interesting" patterns
        int entropy = score_entropy(screen, SCREEN_SIZE);
        int structure = score_structure(screen, 32, 192);
        int symmetry = score_symmetry(screen, 32, 192);

        // Penalize boring patterns (too uniform or too random)
        // Sweet spot: moderate entropy + high structure + some symmetry
        score = entropy * 2 - structure - symmetry / 2;
    }

    return score;
}

// ===== CALL-chain simulation (Hole 17 style) =====
// Simulates: fill RAM with pRNG → generate CALL chain → execute CALLs → screen output
__device__ void simulate_call_chain(PRNGState *prng, uint8_t *screen) {
    // Simplified Hole 17 simulation:
    // 1. Generate random CALL targets
    // 2. Each CALL pushes 2 bytes (its address) to screen via SP

    // We model the key behavior: CALL pushes return address to stack
    // SP starts at end of screen, moves down with each CALL push

    // Generate a sequence of CALL targets using pRNG
    uint16_t sp = SCREEN_SIZE;  // SP starts at end of screen area

    // Clear screen
    for (int i = 0; i < SCREEN_SIZE; i++) screen[i] = 0;

    // Generate and execute ~3000 CALLs
    for (int i = 0; i < 3072 && sp >= 2; i++) {
        // Generate random target address ($5B00-$FFFF)
        uint8_t hi = cmwc_next(prng);
        uint8_t lo = cmwc_next(prng);

        // Clamp high byte to valid range (≥$5B) using CPL trick from Hole 17
        if (hi < 0x5B) hi = ~hi;  // CPL: maps $00-$5A → $FF-$A5

        // CALL pushes return address (= our "address" as 2-byte pixel data)
        sp -= 2;
        screen[sp] = lo;       // low byte
        screen[sp + 1] = hi;   // high byte
    }
}

// ===== GPU Kernel =====
__constant__ uint8_t d_target[SCREEN_SIZE];
__device__ uint32_t d_bestScore;
__device__ uint32_t d_bestSeed;

// Mode 3: CALL-chain simulation (most faithful to Hole 17)
__global__ void search_seeds_callchain(uint32_t seed_offset, uint32_t count, int hasTarget) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t seed = seed_offset + tid;

    PRNGState prng;
    prng.idx = 0;
    prng.carry = (seed >> 24) & 0xFF;
    for (int i = 0; i < 8; i++) {
        prng.table[i] = ((seed >> (i * 4)) ^ (seed >> (i * 3 + 1))) & 0xFF;
        if (prng.table[i] == 0) prng.table[i] = i + 1;
    }

    uint8_t screen[SCREEN_SIZE];
    simulate_call_chain(&prng, screen);

    int score = combined_score(screen, hasTarget ? d_target : NULL, hasTarget);

    uint32_t packed = ((uint32_t)(score & 0xFFFF) << 16) | (tid & 0xFFFF);
    atomicMin(&d_bestScore, packed);
    if (packed <= d_bestScore) {
        atomicExch(&d_bestSeed, seed);
    }
}

__global__ void search_seeds_cmwc(uint32_t seed_offset, uint32_t count, int hasTarget) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t seed = seed_offset + tid;

    // Init pRNG state from 4-byte seed
    PRNGState prng;
    prng.idx = 0;
    prng.carry = (seed >> 24) & 0xFF;
    // Fill table from seed bytes + simple hash
    for (int i = 0; i < 8; i++) {
        prng.table[i] = ((seed >> (i * 4)) ^ (seed >> (i * 3 + 1))) & 0xFF;
        if (prng.table[i] == 0) prng.table[i] = i + 1;  // avoid zero
    }

    // Generate screen content
    uint8_t screen[SCREEN_SIZE];
    for (int i = 0; i < SCREEN_SIZE; i++) {
        screen[i] = cmwc_next(&prng);
    }

    // Score
    int score = combined_score(screen, hasTarget ? d_target : NULL, hasTarget);

    // Atomic best
    uint32_t packed = ((uint32_t)(score & 0xFFFF) << 16) | (tid & 0xFFFF);
    atomicMin(&d_bestScore, packed);
    if (packed <= d_bestScore) {
        atomicExch(&d_bestSeed, seed);
    }
}

__global__ void search_seeds_xorshift(uint32_t seed_offset, uint32_t count, int hasTarget) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t seed = seed_offset + tid;
    if (seed == 0) seed = 1;  // xorshift can't be 0

    uint16_t state = (uint16_t)seed;

    uint8_t screen[SCREEN_SIZE];
    for (int i = 0; i < SCREEN_SIZE; i++) {
        screen[i] = xorshift_next(&state);
    }

    int score = combined_score(screen, hasTarget ? d_target : NULL, hasTarget);

    uint32_t packed = ((uint32_t)(score & 0xFFFF) << 16) | (tid & 0xFFFF);
    atomicMin(&d_bestScore, packed);
    if (packed <= d_bestScore) {
        atomicExch(&d_bestSeed, seed);
    }
}

// ===== Host code =====

static void dump_screen(uint32_t seed, int method) {
    // Regenerate screen on CPU for the winning seed
    uint8_t screen[SCREEN_SIZE];

    if (method == 0) {
        PRNGState prng;
        prng.idx = 0;
        prng.carry = (seed >> 24) & 0xFF;
        for (int i = 0; i < 8; i++) {
            prng.table[i] = ((seed >> (i * 4)) ^ (seed >> (i * 3 + 1))) & 0xFF;
            if (prng.table[i] == 0) prng.table[i] = i + 1;
        }
        for (int i = 0; i < SCREEN_SIZE; i++) {
            // Inline CMWC
            prng.idx = (prng.idx + 1) & 7;
            uint8_t y = prng.table[prng.idx];
            uint16_t t = (uint16_t)y * 253 + prng.carry;
            prng.carry = t >> 8;
            uint8_t x = ~(uint8_t)(t & 0xFF);
            prng.table[prng.idx] = x;
            screen[i] = x;
        }
    } else {
        uint16_t state = (uint16_t)seed;
        if (state == 0) state = 1;
        for (int i = 0; i < SCREEN_SIZE; i++) {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            screen[i] = (uint8_t)(state >> 8);
        }
    }

    // Print ASCII art preview (32 chars wide × 24 lines, sampling every 8 rows)
    printf("\n  Screen preview (SEED=0x%08X):\n", seed);
    for (int y = 0; y < 192; y += 8) {
        printf("  ");
        for (int x = 0; x < 32; x++) {
            uint8_t b = screen[y * 32 + x];
            int bits = __builtin_popcount(b);
            char c = " .:-=+*#@"[bits];
            printf("%c", c);
        }
        printf("\n");
    }

    // Save raw screen binary
    char fname[64];
    snprintf(fname, sizeof(fname), "screen_%08X.bin", seed);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(screen, 1, SCREEN_SIZE, f);
        fclose(f);
        printf("  Saved: %s (%d bytes)\n", fname, SCREEN_SIZE);
    }
}

int main(int argc, char *argv[]) {
    int gpuId = 0;
    int method = 0;  // 0=cmwc, 1=xorshift
    const char *targetPath = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);
        if (!strcmp(argv[i], "--method") && i+1 < argc) {
            if (!strcmp(argv[i+1], "xorshift")) method = 1;
            if (!strcmp(argv[i+1], "callchain")) method = 2;
            i++;
        }
        if (!strcmp(argv[i], "--target") && i+1 < argc) targetPath = argv[++i];
    }

    cudaSetDevice(gpuId);

    int hasTarget = 0;
    if (targetPath) {
        FILE *f = fopen(targetPath, "rb");
        if (f) {
            uint8_t target[SCREEN_SIZE];
            fread(target, 1, SCREEN_SIZE, f);
            fclose(f);
            cudaMemcpyToSymbol(d_target, target, SCREEN_SIZE);
            hasTarget = 1;
            printf("Loaded target: %s\n", targetPath);
        }
    }

    const char *methodNames[] = {"CMWC", "XORShift", "CALL-chain"};
    printf("pRNG SEED search: method=%s, hasTarget=%d, GPU=%d\n",
           methodNames[method], hasTarget, gpuId);

    // Search 4-byte seed space exhaustively: 2^32 = 4.3 billion
    uint64_t totalSeeds = (uint64_t)1 << 32;
    uint32_t bs = 256;
    uint32_t batch = bs * 65535;

    uint32_t initScore = 0xFFFFFFFF;
    uint32_t initSeed = 0;
    cudaMemcpyToSymbol(d_bestScore, &initScore, sizeof(initScore));
    cudaMemcpyToSymbol(d_bestSeed, &initSeed, sizeof(initSeed));

    time_t t0 = time(NULL);
    uint64_t processed = 0;

    for (uint64_t off = 0; off < totalSeeds; off += batch) {
        uint32_t cnt = batch;
        if (off + cnt > totalSeeds) cnt = (uint32_t)(totalSeeds - off);
        uint32_t nblocks = (cnt + bs - 1) / bs;

        if (method == 0)
            search_seeds_cmwc<<<nblocks, bs>>>((uint32_t)off, cnt, hasTarget);
        else if (method == 1)
            search_seeds_xorshift<<<nblocks, bs>>>((uint32_t)off, cnt, hasTarget);
        else
            search_seeds_callchain<<<nblocks, bs>>>((uint32_t)off, cnt, hasTarget);

        cudaDeviceSynchronize();
        processed += cnt;

        // Progress every ~256M seeds
        if ((off % (256 * 1024 * 1024)) == 0 && off > 0) {
            time_t now = time(NULL);
            double elapsed = difftime(now, t0);
            double rate = processed / elapsed / 1e6;
            double eta = (totalSeeds - processed) / (rate * 1e6);

            uint32_t curScore, curSeed;
            cudaMemcpyFromSymbol(&curScore, d_bestScore, sizeof(curScore));
            cudaMemcpyFromSymbol(&curSeed, d_bestSeed, sizeof(curSeed));

            fprintf(stderr, "%.1f%% done, %.0fM seeds/s, ETA %.0fs, best score=%d seed=0x%08X\n",
                    processed * 100.0 / totalSeeds, rate, eta,
                    curScore >> 16, curSeed);
        }
    }

    // Final results
    uint32_t bestScore, bestSeed;
    cudaMemcpyFromSymbol(&bestScore, d_bestScore, sizeof(bestScore));
    cudaMemcpyFromSymbol(&bestSeed, d_bestSeed, sizeof(bestSeed));

    time_t t1 = time(NULL);
    printf("\n=== RESULTS ===\n");
    printf("Searched: %llu seeds in %lds (%.0fM/s)\n",
           (unsigned long long)processed, (long)(t1-t0),
           processed / difftime(t1, t0) / 1e6);
    printf("Best score: %d\n", bestScore >> 16);
    printf("Best seed:  0x%08X\n", bestSeed);

    dump_screen(bestSeed, method);

    return 0;
}

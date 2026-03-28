// z80_focused.cu — Sequential focused brute-force with minimal per-target op pools
// Each target runs to max depth with its own trimmed op set, then next target starts.
// Build: nvcc -O3 -o z80_focused z80_focused.cu
// Usage: ./z80_focused [--gpu N] [--max-len N]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define MAX_OPS 13
#define NUM_TEST 256

// ===== All 13 ops (superset) =====
// 0=MUL3 1=MUL5 2=MUL7 3=SHR 4=NEG 5=SAVE 6=SUB_B 7=SBC_MASK 8=AND_0F 9=XOR_B 10=AND_F0 11=RLCA 12=RRCA

__device__ uint8_t run_seq_mapped(const uint8_t *ops, int len, uint8_t input, const uint8_t *opMap, int numOps) {
    uint8_t a = input, b = 0;
    int carry = 0;
    for (int i = 0; i < len; i++) {
        uint16_t r;
        switch (opMap[ops[i]]) {
        case 0:  r = (uint16_t)a * 3; a = (uint8_t)r; carry = r > 0xFF; break;
        case 1:  r = (uint16_t)a * 5; a = (uint8_t)r; carry = r > 0xFF; break;
        case 2:  r = (uint16_t)a * 7; a = (uint8_t)r; carry = r > 0xFF; break;
        case 3:  carry = a & 1; a = a >> 1; break;
        case 4:  carry = (a != 0); a = (uint8_t)(0 - a); break;
        case 5:  b = a; break;
        case 6:  carry = a < b; a = a - b; break;
        case 7:  { int cc = carry; a = cc ? 0xFF : 0x00; } break;
        case 8:  a &= 0x0F; carry = 0; break;
        case 9:  a ^= b; carry = 0; break;
        case 10: a &= 0xF0; carry = 0; break;
        case 11: carry = (a >> 7) & 1; a = ((a << 1) | (a >> 7)) & 0xFF; break;
        case 12: carry = a & 1; a = ((a >> 1) | (a << 7)) & 0xFF; break;
        }
    }
    return a;
}

__constant__ uint8_t d_target[NUM_TEST];
__constant__ uint8_t d_opMap[MAX_OPS];
__device__ uint32_t d_bestScore;
__device__ uint64_t d_bestIdx;

__global__ void focused_kernel(int seqLen, int numOps, uint64_t offset, uint64_t count) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[24];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % numOps);
        tmp /= numOps;
    }

    uint32_t cur_best = d_bestScore;
    int best_err = (cur_best == 0xFFFFFFFF) ? 255 : (cur_best >> 16);

    // Quick check
    uint8_t qc[] = {0, 1, 64, 128, 255};
    int max_err = 0;
    for (int q = 0; q < 5; q++) {
        uint8_t out = run_seq_mapped(ops, seqLen, qc[q], d_opMap, numOps);
        int err = (int)out - (int)d_target[qc[q]];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (max_err >= best_err) return;
    }

    // Full verify
    max_err = 0;
    for (int i = 0; i < 256; i++) {
        uint8_t out = run_seq_mapped(ops, seqLen, (uint8_t)i, d_opMap, numOps);
        int err = (int)out - (int)d_target[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (max_err >= best_err) break;
    }

    uint32_t score = ((uint32_t)max_err << 16) | ((uint32_t)seqLen << 8);
    uint32_t old = atomicMin(&d_bestScore, score);
    if (score <= old) {
        atomicExch((unsigned long long*)&d_bestIdx, (unsigned long long)seqIdx);
    }
}

// ===== Op names =====
static const char *allOpNames[] = {
    "MUL3", "MUL5", "MUL7", "SHR", "NEG",
    "SAVE", "SUB_B", "SBC_MASK", "AND_0F",
    "XOR_B", "AND_F0", "RLCA", "RRCA"
};

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

// ===== Target definitions =====
struct Target {
    const char *name;
    uint8_t pool[MAX_OPS];   // indices into allOpNames
    int poolSize;
    int maxDepth;            // max depth for this pool size
};

static void gen_target(const char *name, uint8_t out[256]) {
    for (int i = 0; i < 256; i++) {
        uint8_t x = (uint8_t)i;
        double xd = (double)x;

        if (!strcmp(name, "log2_f3.5")) {
            out[i] = (x == 0) ? 0 : (uint8_t)(log2(xd) * 32.0);
        } else if (!strcmp(name, "bin2bcd")) {
            // Binary to BCD: high nibble = tens, low nibble = ones
            out[i] = (x > 99) ? 0 : (uint8_t)(((x / 10) << 4) | (x % 10));
        } else if (!strcmp(name, "log2_x28")) {
            out[i] = (x == 0) ? 0 : (uint8_t)(log2(xd) * 28.0);
        } else if (!strcmp(name, "recip")) {
            out[i] = (x == 0) ? 255 : (uint8_t)(256.0 / xd);
        } else if (!strcmp(name, "popcnt_x32")) {
            int p = 0;
            for (int b = 0; b < 8; b++) if (x & (1 << b)) p++;
            out[i] = (uint8_t)(p * 32);
        } else if (!strcmp(name, "cbrt_f2.6")) {
            out[i] = (uint8_t)(cbrt(xd / 64.0) * 64.0);
        } else if (!strcmp(name, "log2_f4.4")) {
            out[i] = (x == 0) ? 0 : (uint8_t)(log2(xd) * 16.0);
        } else if (!strcmp(name, "sqrt_f4.4")) {
            out[i] = (uint8_t)(sqrt(xd) * 16.0);
        } else if (!strcmp(name, "bcd2bin")) {
            uint8_t tens = (x >> 4) & 0xF, ones = x & 0xF;
            out[i] = (tens > 9 || ones > 9) ? 0 : (uint8_t)(tens * 10 + ones);
        } else if (!strcmp(name, "sqrt_f3.5")) {
            out[i] = (uint8_t)(sqrt(xd) * 32.0);
        }
        // === 3D graphics primitives ===
        else if (!strcmp(name, "sqr_lo")) {
            // (x*x) & 0xFF — low byte of square table for fast multiply
            out[i] = (uint8_t)((x * x) & 0xFF);
        } else if (!strcmp(name, "sqr_hi")) {
            // (x*x) >> 8 — high byte of square table
            out[i] = (uint8_t)((x * x) >> 8);
        } else if (!strcmp(name, "sin_q1")) {
            // sin(x * pi/128) * 255 for x=0..63, 0 for rest
            // Quarter sine table for rotation matrix
            if (x < 64) out[i] = (uint8_t)(sin(x * M_PI / 128.0) * 255.0);
            else out[i] = 0;
        } else if (!strcmp(name, "sin_full")) {
            // sin(x * 2pi/256) * 127 + 128, unsigned full circle
            double angle = x * 2.0 * M_PI / 256.0;
            out[i] = (uint8_t)(sin(angle) * 127.0 + 128.0);
        } else if (!strcmp(name, "log_fast")) {
            // log2(x) * 32 for perspective division via log subtraction
            out[i] = (x == 0) ? 0 : (uint8_t)(log2(xd) * 32.0);
        } else if (!strcmp(name, "antilog")) {
            // 2^(x/32) for perspective: antilog(log(a)-log(b)) = a/b
            out[i] = (uint8_t)(pow(2.0, xd / 32.0));
            if (pow(2.0, xd / 32.0) > 255.0) out[i] = 255;
        } else if (!strcmp(name, "smoothstep")) {
            // 3x²-2x³ (smooth interpolation for blending)
            double t = xd / 255.0;
            out[i] = (uint8_t)(t * t * (3.0 - 2.0 * t) * 255.0);
        } else if (!strcmp(name, "gamma22")) {
            // (x/255)^2.2 * 255 — sRGB gamma for brightness
            out[i] = (uint8_t)(pow(xd / 255.0, 2.2) * 255.0);
        } else if (!strcmp(name, "inv_gamma")) {
            // (x/255)^(1/2.2) * 255 — linear to sRGB
            out[i] = (uint8_t)(pow(xd / 255.0, 1.0/2.2) * 255.0);
        }
    }
}

// Pipeline: grouped by GPU assignment
// i3 Vulkan (weakest):  0-1  — small pools, deep search
// i5 RTX 2070 (mid):    2-4  — 7 ops each
// i7 GPU0 4060Ti:       5-7  — 8-10 ops (heavy)
// i7 GPU1 4060Ti:       8-9  — 11-12 ops (heaviest)
static Target targets[] = {
    // --- i3 Vulkan: --start 0 --end 2 ---
    {"log2_f3.5",    {0, 4, 3, 5, 9},                   5, 22},  // MUL3 NEG SHR SAVE XOR_B → 5^22=2.4T
    {"bin2bcd",      {8, 0, 1, 7, 3},                   5, 22},  // AND_0F MUL3 MUL5 SBC_MASK SHR → 5^22=2.4T
    // --- i5 RTX 2070: --start 2 --end 5 ---
    {"log2_x28",     {8, 1, 2, 4, 12, 3, 6},            7, 16},  // AND_0F MUL5 MUL7 NEG RRCA SHR SUB_B
    {"recip",        {8, 2, 4, 11, 12, 3, 9},            7, 16},  // AND_0F MUL7 NEG RLCA RRCA SHR XOR_B
    {"popcnt_x32",   {8, 0, 2, 12, 5, 3, 6},             7, 16},  // AND_0F MUL3 MUL7 RRCA SAVE SHR SUB_B
    // --- i7 GPU0: --start 5 --end 8 ---
    {"cbrt_f2.6",    {8, 10, 0, 4, 12, 7, 3, 6},         8, 15},  // AND_0F AND_F0 MUL3 NEG RRCA SBC_MASK SHR SUB_B
    {"bcd2bin",      {8, 10, 0, 4, 11, 12, 5, 7, 3, 6, 9}, 11, 13},
    {"log2_f4.4",    {8, 10, 0, 1, 4, 11, 12, 3, 6, 9},  10, 14},
    // --- i7 GPU1: --start 8 --end 10 ---
    {"sqrt_f4.4",    {8, 0, 1, 2, 4, 11, 12, 5, 3, 6, 9}, 11, 13},
    {"sqrt_f3.5",    {8, 0, 1, 2, 4, 11, 12, 5, 7, 3, 6, 9}, 12, 13},
    // === 3D Graphics batch: --start 10 --end 17 ===
    // Small pools — can go deep!
    {"sqr_lo",       {0, 1, 3, 5, 9, 11},                  6, 18},  // MUL3 MUL5 SHR SAVE XOR_B RLCA
    {"sqr_hi",       {0, 1, 3, 5, 9, 11},                  6, 18},  // same pool
    {"sin_full",     {0, 2, 3, 4, 5, 6, 9},                7, 16},  // MUL3 MUL7 SHR NEG SAVE SUB_B XOR_B
    {"sin_q1",       {0, 2, 3, 4, 5, 6, 9},                7, 16},  // same as sin_full
    {"smoothstep",   {0, 1, 3, 5, 9, 8},                   6, 18},  // MUL3 MUL5 SHR SAVE XOR_B AND_0F
    {"gamma22",      {0, 1, 3, 5, 9, 8},                   6, 18},  // same — gamma ≈ x²
    {"antilog",      {0, 1, 2, 3, 5, 11, 12},              7, 16},  // MUL3 MUL5 MUL7 SHR SAVE RLCA RRCA
};
#define NUM_TARGETS (sizeof(targets)/sizeof(targets[0]))

int main(int argc, char *argv[]) {
    int gpuId = 0;
    int maxLenOverride = 0;
    int startTarget = 0;
    int endTarget = (int)NUM_TARGETS;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLenOverride = atoi(argv[++i]);
        if (!strcmp(argv[i], "--start") && i+1 < argc) startTarget = atoi(argv[++i]);
        if (!strcmp(argv[i], "--end") && i+1 < argc) endTarget = atoi(argv[++i]);
    }

    cudaSetDevice(gpuId);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    fprintf(stderr, "Focused sequential search on GPU %d, targets %d..%d\n", gpuId, startTarget, endTarget-1);

    for (int t = startTarget; t < endTarget && t < (int)NUM_TARGETS; t++) {
        Target *tgt = &targets[t];
        int maxLen = maxLenOverride > 0 ? maxLenOverride : tgt->maxDepth;

        // Check if search space is reasonable (< 100T)
        uint64_t maxSpace = ipow(tgt->poolSize, maxLen);
        while (maxSpace > 100000000000000ULL && maxLen > 12) {
            maxLen--;
            maxSpace = ipow(tgt->poolSize, maxLen);
        }

        uint8_t target[256];
        gen_target(tgt->name, target);

        fprintf(stderr, "\n=== [%d/%d] %s: %d ops, max depth %d ===\n",
                t+1, (int)NUM_TARGETS, tgt->name, tgt->poolSize, maxLen);
        fprintf(stderr, "Pool:");
        for (int i = 0; i < tgt->poolSize; i++)
            fprintf(stderr, " %s", allOpNames[tgt->pool[i]]);
        fprintf(stderr, "\n");

        cudaMemcpyToSymbol(d_target, target, sizeof(target));
        cudaMemcpyToSymbol(d_opMap, tgt->pool, tgt->poolSize);
        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initIdx = 0;
        cudaMemcpyToSymbol(d_bestScore, &initScore, sizeof(initScore));
        cudaMemcpyToSymbol(d_bestIdx, &initIdx, sizeof(initIdx));

        int found_exact = 0;
        for (int len = 1; len <= maxLen && !found_exact; len++) {
            uint64_t total = ipow(tgt->poolSize, len);
            time_t t0 = time(NULL);
            fprintf(stderr, "  len=%d: %.2e candidates...\n", len, (double)total);

            int bs = 256;
            uint64_t batch = (uint64_t)bs * 65535;
            for (uint64_t off = 0; off < total; off += batch) {
                uint64_t cnt = total - off;
                if (cnt > batch) cnt = batch;
                focused_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, tgt->poolSize, off, cnt);
                cudaDeviceSynchronize();
            }

            uint32_t score;
            uint64_t bestIdx;
            cudaMemcpyFromSymbol(&score, d_bestScore, sizeof(score));
            cudaMemcpyFromSymbol(&bestIdx, d_bestIdx, sizeof(bestIdx));

            time_t t1 = time(NULL);

            if (score != 0xFFFFFFFF) {
                int err = score >> 16;
                int rlen = (score >> 8) & 0xFF;
                if (rlen == len) {
                    uint8_t ops[24];
                    uint64_t tmp = bestIdx;
                    for (int i = rlen-1; i >= 0; i--) { ops[i] = tmp % tgt->poolSize; tmp /= tgt->poolSize; }

                    printf("%-14s len=%d err=%d:", tgt->name, rlen, err);
                    for (int i = 0; i < rlen; i++) printf(" %s", allOpNames[tgt->pool[ops[i]]]);
                    if (err == 0) { printf(" [EXACT]"); found_exact = 1; }
                    else printf(" [approx, max_err=%d]", err);
                    printf("  (%lds)\n", (long)(t1 - t0));
                    fflush(stdout);
                }
            }
            fprintf(stderr, "    len=%d done (%lds)\n", len, (long)(t1 - t0));
        }

        if (found_exact)
            fprintf(stderr, "  >>> EXACT found for %s! Moving to next target.\n", tgt->name);
        else
            fprintf(stderr, "  >>> Max depth reached for %s.\n", tgt->name);
    }

    printf("\n=== ALL TARGETS COMPLETE ===\n");
    return 0;
}

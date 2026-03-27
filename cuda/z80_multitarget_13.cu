// z80_multitarget.cu — Multi-target brute-force with approximate matching
// ONE search pass tests against ALL non-linear functions simultaneously.
// Reports exact matches AND best approximations (min max-error).
//
// 12-op prime basis + core ops. Depth 12 in ~1.2h.
// Build: nvcc -O3 -o cuda/z80_multitarget cuda/z80_multitarget.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define NUM_OPS 13
#define NUM_TARGETS 15
#define NUM_TEST 256

// 13-op trimmed pool: removed ADD_B, SHL, RESTORE (never/rarely used in best results)
// MUL3, MUL5, MUL7, SHR, NEG, SAVE, SUB_B, SBC_MASK, AND_0F, XOR_B, AND_F0, RLCA, RRCA
// 13^12 = 23.3B → 3.2h. vs 16^12 = 281T → 39h. 12x faster!

__device__ uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    int carry = 0;

    for (int i = 0; i < len; i++) {
        uint16_t r;
        switch (ops[i]) {
        case 0:  // MUL3
            r = (uint16_t)a * 3; a = (uint8_t)r; carry = r > 0xFF; break;
        case 1:  // MUL5
            r = (uint16_t)a * 5; a = (uint8_t)r; carry = r > 0xFF; break;
        case 2:  // MUL7
            r = (uint16_t)a * 7; a = (uint8_t)r; carry = r > 0xFF; break;
        case 3:  // SHR: A >>= 1
            carry = a & 1; a = a >> 1; break;
        case 4:  // NEG: A = -A
            carry = (a != 0); a = (uint8_t)(0 - a); break;
        case 5:  // SAVE: B = A
            b = a; break;
        case 6:  // SUB_B: A = A - B
            carry = a < b; a = a - b; break;
        case 7:  // SBC_MASK: A = carry ? 0xFF : 0x00
            { int cc = carry; carry = cc; a = cc ? 0xFF : 0x00; } break;
        case 8:  // AND_0F: A &= 0x0F
            a &= 0x0F; carry = 0; break;
        case 9:  // XOR_B: A ^= B
            a ^= b; carry = 0; break;
        case 10: // AND_F0: A &= 0xF0
            a &= 0xF0; carry = 0; break;
        case 11: // RLCA: rotate left circular
            carry = (a >> 7) & 1; a = ((a << 1) | (a >> 7)) & 0xFF; break;
        case 12: // RRCA: rotate right circular
            carry = a & 1; a = ((a >> 1) | (a << 7)) & 0xFF; break;
        }
    }
    return a;
}

// Target functions
__constant__ uint8_t d_targets[NUM_TARGETS][NUM_TEST];

// Target names (host side)
static const char *targetNames[] = {
    "log2_f3.5", // log2(x)*32, f3.5 fixed-point
    "sqrt_f4.4", // sqrt(x)*16, f4.4 fixed-point
    "cbrt_f2.6", // cbrt(x)*40, scaled
    "recip",     // floor(256/x), 0 for x=0
    "popcnt×32", // popcount*32 (scaled)
    "clz×32",    // clz*32 (scaled)
    "ctz×32",    // ctz*32 (scaled)
    "is_pow2",   // 1 if power of 2, 0 otherwise
    "bcd2bin",   // packed BCD → binary (valid for 0x00-0x99)
    "bin2bcd",   // binary → packed BCD (valid for 0-99)
    "gray_enc",  // Gray code encode: x ^ (x>>1)
    "gray_dec",  // Gray code decode
    "log2_f4.4", // log2(x)*16, range 0-127
    "sqrt_f3.5", // sqrt(x)*8, range 0-127
    "log2_×28",  // (log2(x)+1)*28, range 28-252
};

static void gen_targets(uint8_t targets[NUM_TARGETS][NUM_TEST]) {
    for (int i = 0; i < 256; i++) {
        uint8_t x = (uint8_t)i;

        // log2 as f3.5: log2(x)*32, range 0-255
        if (x == 0) targets[0][i] = 0;
        else { double l = log2((double)x); targets[0][i] = (uint8_t)(l * 32.0); }

        // sqrt as f4.4: sqrt(x)*16, range 0-255
        { double s = sqrt((double)x); targets[1][i] = (uint8_t)(s * 16.0); }

        // cbrt as f2.6: cbrt(x)*40, range 0-253
        { double c = cbrt((double)x); int v = (int)(c * 40.0); targets[2][i] = (uint8_t)(v > 255 ? 255 : v); }

        // recip: 256/x (already fills range)
        targets[3][i] = (x == 0) ? 0 : (uint8_t)(256 / x);

        // popcount × 32 (range 0-256, scale to fill byte)
        { uint8_t v=x, c=0; while(v){c+=v&1;v>>=1;} targets[4][i] = c * 32; }

        // clz × 32 (range 0-256)
        if (x == 0) targets[5][i] = 255;
        else { int c=0; uint8_t v=x; while(!(v&0x80)){v<<=1;c++;} targets[5][i] = c * 32; }

        // ctz × 32 (range 0-256)
        if (x == 0) targets[6][i] = 255;
        else { int c=0; uint8_t v=x; while(!(v&1)){v>>=1;c++;} targets[6][i] = c * 32; }

        // is_pow2
        targets[7][i] = (x && !(x & (x-1))) ? 1 : 0;

        // bcd2bin (only valid for BCD inputs, rest don't matter for matching)
        { uint8_t hi = x >> 4, lo = x & 0xF;
          targets[8][i] = (hi <= 9 && lo <= 9) ? hi*10 + lo : 0; }

        // bin2bcd (only valid for 0-99)
        targets[9][i] = (x < 100) ? ((x/10)<<4)|(x%10) : 0;

        // gray encode
        targets[10][i] = x ^ (x >> 1);

        // gray decode
        { uint8_t v=x; v ^= v>>4; v ^= v>>2; v ^= v>>1; targets[11][i] = v; }

        // log2 as f4.4: log2(x)*16, range 0-127
        if (x == 0) targets[12][i] = 0;
        else { double l = log2((double)x); targets[12][i] = (uint8_t)(l * 16.0); }

        // sqrt as f3.5: sqrt(x)*8, range 0-127 (coarser but smoother)
        { double s = sqrt((double)x); targets[13][i] = (uint8_t)(s * 8.0); }

        // log2 as integer × 32 but shifted: (log2(x)+1)*28 — fills more of 0-255
        if (x == 0) targets[14][i] = 0;
        else { double l = log2((double)x); targets[14][i] = (uint8_t)((l + 1.0) * 28.0); }
    }
}

// Per-target: best score (len<<16 | max_error<<8 | cost)
// and best index
__device__ uint32_t d_bestScore[NUM_TARGETS];
__device__ uint64_t d_bestIdx[NUM_TARGETS];

__global__ void multi_kernel(int seqLen, uint64_t offset, uint64_t count) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[20];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % NUM_OPS);
        tmp /= NUM_OPS;
    }

    // Compute output for 8 quick-check inputs
    uint8_t qc_inputs[] = {0, 1, 2, 4, 15, 16, 100, 255};
    uint8_t qc_out[8];
    for (int q = 0; q < 8; q++)
        qc_out[q] = run_seq(ops, seqLen, qc_inputs[q]);

    // For each target, quick-check then full verify
    for (int t = 0; t < NUM_TARGETS; t++) {
        // Read current best error for this target
        uint32_t cur_best = d_bestScore[t];
        int best_err = (cur_best == 0xFFFFFFFF) ? 255 : (cur_best >> 16);

        // Quick reject: if any QC input already exceeds current best, skip
        int max_err = 0;
        bool reject = false;
        for (int q = 0; q < 4; q++) {
            int err = (int)qc_out[q] - (int)d_targets[t][qc_inputs[q]];
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
            if (max_err >= best_err) { reject = true; break; }
        }
        if (reject) continue;

        // Full verify: compute max error over all 256 inputs
        int cur_best_err = best_err;  // from quick-check above

        max_err = 0;
        for (int i = 0; i < 256; i++) {
            uint8_t out = run_seq(ops, seqLen, (uint8_t)i);
            int err = (int)out - (int)d_targets[t][i];
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
            if (max_err >= cur_best_err) break;  // can't beat current best, bail
        }

        // Score: pack (max_error, seqLen, 0) — ERROR first, then length
        // Lower error always wins, ties broken by shorter sequence
        uint32_t score = ((uint32_t)max_err << 16) | ((uint32_t)seqLen << 8);

        uint32_t old = atomicMin(&d_bestScore[t], score);
        if (score <= old) {
            atomicExch((unsigned long long*)&d_bestIdx[t], (unsigned long long)seqIdx);
        }
    }
}

static const char *opNames[] = {
    "MUL3", "MUL5", "MUL7", "SHR", "NEG",
    "SAVE", "SUB_B", "SBC_MASK", "AND_0F",
    "XOR_B", "AND_F0", "RLCA", "RRCA"
};

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

int main(int argc, char *argv[]) {
    int maxLen = 12;
    int gpuId = 0;
    int splitId = 0;   // this partition (0-based)
    int splitTotal = 1; // total partitions

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split") && i+1 < argc) {
            // format: N/M (partition N of M)
            sscanf(argv[++i], "%d/%d", &splitId, &splitTotal);
        }
    }

    cudaSetDevice(gpuId);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Generate all target tables
    uint8_t targets[NUM_TARGETS][NUM_TEST];
    gen_targets(targets);
    cudaMemcpyToSymbol(d_targets, targets, sizeof(targets));

    // Init best scores to max
    uint32_t initScores[NUM_TARGETS];
    uint64_t initIdx[NUM_TARGETS];
    for (int t = 0; t < NUM_TARGETS; t++) {
        initScores[t] = 0xFFFFFFFF;
        initIdx[t] = 0;
    }
    cudaMemcpyToSymbol(d_bestScore, initScores, sizeof(initScores));
    cudaMemcpyToSymbol(d_bestIdx, initIdx, sizeof(initIdx));

    fprintf(stderr, "Multi-target search: %d targets, %d ops, max-len %d, GPU %d, split %d/%d\n",
            NUM_TARGETS, NUM_OPS, maxLen, gpuId, splitId, splitTotal);
    for (int t = 0; t < NUM_TARGETS; t++)
        fprintf(stderr, "  [%d] %s\n", t, targetNames[t]);

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);
        if (total > 500000000000000ULL) {
            fprintf(stderr, "len=%d: %.2e exceeds limit\n", len, (double)total);
            break;
        }

        // Reset scores for this length (keep best from shorter)
        // Actually, don't reset — we want overall best across all lengths

        // Split range: this GPU handles [partStart, partEnd)
        uint64_t partSize = (total + splitTotal - 1) / splitTotal;
        uint64_t partStart = (uint64_t)splitId * partSize;
        uint64_t partEnd = partStart + partSize;
        if (partEnd > total) partEnd = total;
        if (partStart >= total) { fprintf(stderr, "len=%d: skip (partition empty)\n", len); continue; }

        fprintf(stderr, "len=%d: searching %.2e of %.2e (range %llu-%llu)\n",
                len, (double)(partEnd - partStart), (double)total,
                (unsigned long long)partStart, (unsigned long long)partEnd);

        int bs = 256;
        uint64_t batch = (uint64_t)bs * 65535;
        for (uint64_t off = partStart; off < partEnd; off += batch) {
            uint64_t cnt = partEnd - off;
            if (cnt > batch) cnt = batch;
            multi_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt);
            cudaDeviceSynchronize();
        }

        // Read back results after each length
        uint32_t scores[NUM_TARGETS];
        uint64_t idxs[NUM_TARGETS];
        cudaMemcpyFromSymbol(scores, d_bestScore, sizeof(scores));
        cudaMemcpyFromSymbol(idxs, d_bestIdx, sizeof(idxs));

        int any_new = 0;
        for (int t = 0; t < NUM_TARGETS; t++) {
            if (scores[t] != 0xFFFFFFFF) {
                int rerr = scores[t] >> 16;
                int rlen = (scores[t] >> 8) & 0xFF;
                if (rlen == len) {  // found at this length
                    any_new = 1;
                    uint8_t ops[20];
                    uint64_t tmp = idxs[t];
                    for (int i = rlen-1; i >= 0; i--) { ops[i] = tmp % NUM_OPS; tmp /= NUM_OPS; }

                    printf("%-12s len=%d err=%d:", targetNames[t], rlen, rerr);
                    for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
                    if (rerr == 0) printf(" [EXACT]");
                    else printf(" [approx, max_err=%d]", rerr);
                    printf("\n");
                    fflush(stdout);
                }
            }
        }

        // Print summary
        fprintf(stderr, "  After len=%d: ", len);
        for (int t = 0; t < NUM_TARGETS; t++) {
            if (scores[t] != 0xFFFFFFFF) {
                int rerr = scores[t] >> 16;
                fprintf(stderr, "%s=%s%d ", targetNames[t], rerr==0?"✓":"≈", rerr);
            }
        }
        fprintf(stderr, "\n");
    }

    // Final summary
    printf("\n=== FINAL RESULTS ===\n");
    uint32_t scores[NUM_TARGETS];
    uint64_t idxs[NUM_TARGETS];
    cudaMemcpyFromSymbol(scores, d_bestScore, sizeof(scores));
    cudaMemcpyFromSymbol(idxs, d_bestIdx, sizeof(idxs));

    for (int t = 0; t < NUM_TARGETS; t++) {
        if (scores[t] == 0xFFFFFFFF) {
            printf("%-12s: NOT FOUND\n", targetNames[t]);
        } else {
            int rerr = scores[t] >> 16;
            int rlen = (scores[t] >> 8) & 0xFF;
            uint8_t ops[20];
            uint64_t tmp = idxs[t];
            for (int i = rlen-1; i >= 0; i--) { ops[i] = tmp % NUM_OPS; tmp /= NUM_OPS; }

            printf("%-12s len=%d err=%d:", targetNames[t], rlen, rerr);
            for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
            if (rerr == 0) printf(" [EXACT]");
            else printf(" [max_err=%d]", rerr);
            printf("\n");
        }
    }

    return 0;
}

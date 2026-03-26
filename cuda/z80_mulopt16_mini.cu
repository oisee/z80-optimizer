// z80_mulopt16_mini.cu — Minimal 3-7 op u16 multiply search
// Exploits insight: optimal u16 multiplications use only 3 core ops
// at lengths up to 7 (validated). Enables search to len-20.
//
// Build: nvcc -O3 -o z80_mulopt16_mini z80_mulopt16_mini.cu
// Usage: z80_mulopt16_mini [--max-len 20] [--k 42] [--ops 3|5|7] [--json]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Op pools (nested: 3 ⊂ 5 ⊂ 7)
// Pool 3: ADD_HL_HL(0), ADD_HL_BC(1), LD_C_A(2)
// Pool 5: + NEG(3), SBC_A_B(4)
// Pool 7: + LD_L_A(5), LD_H_A(6)

__device__ uint16_t run_seq(const uint8_t *ops, int len, uint8_t input, int numOps) {
    // Initial state: A=input, B=0, C=0, H=0, L=input (preamble: LD L,A / LD H,0)
    uint8_t a = input, b = 0, c = 0, h = 0, l = input;
    int carry = 0;
    
    for (int i = 0; i < len; i++) {
        uint16_t hl, bc, r;
        switch (ops[i]) {
        case 0: // ADD HL,HL
            hl = ((uint16_t)h << 8) | l;
            r = hl + hl;
            carry = r < hl ? 1 : 0; // overflow
            h = (uint8_t)(r >> 8);
            l = (uint8_t)r;
            break;
        case 1: // ADD HL,BC (B=0 or saved, C=input or saved)
            hl = ((uint16_t)h << 8) | l;
            bc = ((uint16_t)b << 8) | c;
            r = hl + bc;
            carry = r < hl ? 1 : 0;
            h = (uint8_t)(r >> 8);
            l = (uint8_t)r;
            break;
        case 2: // LD C,A (save input for ADD HL,BC)
            c = a;
            break;
        // -- Pool 4: add SWAP_HL --
        case 3: // SWAP_HL: HL = L * 256 (byte swap: H=L, L=0)
            // Materializes to: LD H,L / LD L,0 (2 insts, 11T)
            h = l;
            l = 0;
            break;
        // -- Pool 5: add SUB_HL_BC --
        case 4: // SUB_HL_BC: HL -= BC
            // Materializes to: OR A / SBC HL,BC (2 insts, 15T)
            { uint16_t hl2 = ((uint16_t)h << 8) | l;
              uint16_t bc2 = ((uint16_t)b << 8) | c;
              carry = hl2 < bc2 ? 1 : 0;
              hl2 = hl2 - bc2;
              h = (uint8_t)(hl2 >> 8);
              l = (uint8_t)hl2; }
            break;
        // -- Pool 7: add 8-bit helpers --
        case 5: // NEG (8-bit)
            carry = (a != 0) ? 1 : 0;
            a = (uint8_t)(0 - a);
            break;
        case 6: // SBC A,B
            { int cc = carry ? 1 : 0;
              carry = ((int)a - (int)b - cc) < 0 ? 1 : 0;
              a = a - b - (uint8_t)cc; }
            break;
        case 7: // LD L,A
            l = a;
            break;
        case 8: // LD H,A
            h = a;
            break;
        }
    }
    return ((uint16_t)h << 8) | l;
}

__global__ void kernel(uint8_t k, int seqLen, uint64_t offset, uint64_t count,
                       int numOps, uint32_t *bestScore, uint64_t *bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint64_t seqIdx = offset + tid;
    uint8_t ops[25];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % numOps);
        tmp /= numOps;
    }
    
    // QuickCheck: 4 inputs
    uint16_t target;
    target = 1 * (uint16_t)k; if (run_seq(ops, seqLen, 1, numOps) != target) return;
    target = 2 * (uint16_t)k; if (run_seq(ops, seqLen, 2, numOps) != target) return;
    target = 127 * (uint16_t)k; if (run_seq(ops, seqLen, 127, numOps) != target) return;
    target = 255 * (uint16_t)k; if (run_seq(ops, seqLen, 255, numOps) != target) return;
    
    // Full verify
    for (int inp = 0; inp < 256; inp++) {
        if (run_seq(ops, seqLen, (uint8_t)inp, numOps) != (uint16_t)inp * k) return;
    }
    
    // Cost (approximate — all ops are 11T except NEG=8T, LD=4T)
    uint16_t cost = 0;
    // Costs: real T-states for materialized sequence
    const uint8_t costs[] = {11, 11, 4, 11, 15, 8, 4, 4, 4};
    // pool 3: ADD_HL_HL=11, ADD_HL_BC=11, LD_C_A=4
    // pool 5: SWAP_HL=11(LD H,L+LD L,0), SUB_HL_BC=15(OR A+SBC HL,BC)
    // pool 9: NEG=8, SBC_A_B=4, LD_L_A=4, LD_H_A=4
    for (int i = 0; i < seqLen; i++) cost += costs[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;
    
    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) {
        atomicExch((unsigned long long *)bestIdx, (unsigned long long)seqIdx);
    }
}

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; i++) r *= base;
    return r;
}

static const char *opName(uint8_t op) {
    static const char *names[] = {
        "ADD HL,HL", "ADD HL,BC", "LD C,A",      // pool 3: core
        "SWAP_HL", "SUB HL,BC",                    // pool 5: +virtual 16-bit
        "NEG", "SBC A,B", "LD L,A", "LD H,A"      // pool 9: +8-bit helpers
    };
    return op < 9 ? names[op] : "?";
}

int main(int argc, char *argv[]) {
    int maxLen = 20, singleK = 0, numOps = 3, jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) singleK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ops") && i+1 < argc) numOps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) jsonMode = 1;
    }
    
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);
    
    // Warm up
    uint32_t dummy = 0;
    cudaMemcpy(d_best, &dummy, 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    int startK = singleK > 0 ? singleK : 2;
    int endK = singleK > 0 ? singleK : 255;
    int solved = 0;
    
    fprintf(stderr, "mulopt16-mini: %d ops, max-len %d\n", numOps, maxLen);
    if (jsonMode) printf("[\n");
    
    for (int k = startK; k <= endK; k++) {
        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initIdx = 0;
        int found = 0;
        
        for (int len = 1; len <= maxLen; len++) {
            uint64_t total = ipow(numOps, len);
            if (total > 50000000000ULL) break; // 50B limit
            
            cudaMemcpy(d_best, &initScore, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, &initIdx, 8, cudaMemcpyHostToDevice);
            
            int bs = 256;
            uint64_t batch = (uint64_t)bs * 65535;
            for (uint64_t off = 0; off < total; off += batch) {
                uint64_t cnt = total - off;
                if (cnt > batch) cnt = batch;
                uint64_t grid = (cnt + bs - 1) / bs;
                kernel<<<(unsigned int)grid, bs>>>((uint8_t)k, len, off, cnt,
                                                    numOps, d_best, d_idx);
                cudaDeviceSynchronize();
            }
            
            uint32_t bestScore;
            uint64_t bestIdx;
            cudaMemcpy(&bestScore, d_best, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&bestIdx, d_idx, 8, cudaMemcpyDeviceToHost);
            
            if (bestScore != 0xFFFFFFFF) {
                found = 1; solved++;
                int rlen = bestScore >> 16;
                int rcost = bestScore & 0xFFFF;
                uint8_t ops[25];
                uint64_t tmp = bestIdx;
                for (int i = rlen - 1; i >= 0; i--) {
                    ops[i] = (uint8_t)(tmp % numOps);
                    tmp /= numOps;
                }
                
                if (jsonMode) {
                    printf("%s {\"k\": %d, \"ops\": [", solved > 1 ? "," : "", k);
                    for (int i = 0; i < rlen; i++)
                        printf("%s\"%s\"", i ? "," : "", opName(ops[i]));
                    printf("], \"length\": %d, \"tstates\": %d}\n", rlen, rcost);
                } else {
                    printf("x%d:", k);
                    for (int i = 0; i < rlen; i++) printf(" %s", opName(ops[i]));
                    printf(" (%d insts, %dT)\n", rlen, rcost);
                }
                fflush(stdout);
                break;
            }
        }
        fprintf(stderr, "x%d/%d (%d solved)%s\n", k, endK, solved, found ? "" : " NOT FOUND");
    }
    
    if (jsonMode) printf("]\n");
    fprintf(stderr, "Done: %d/%d solved\n", solved, endK - startK + 1);
    
    cudaFree(d_best); cudaFree(d_idx);
    return 0;
}

// z80_bcd21.cu — BCD arithmetic brute-force with proper H/N flag support for DAA
// 19-op pool including DAA with half-carry and subtract flag tracking.
// Build: nvcc -O3 -o cuda/z80_bcd21 cuda/z80_bcd21.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_OPS 19

// State: a, b, carry, halfcarry, subflag
// Packed into: a(8) + b(8) + flags(3) = fits in uint32

__device__ uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    int carry = 0, hc = 0, nf = 0;  // carry, half-carry, subtract flag

    for (int i = 0; i < len; i++) {
        uint16_t r;
        switch (ops[i]) {
        case 0:  // ADD A,A (4T)
            hc = ((a & 0xF) + (a & 0xF)) > 0xF; r = (uint16_t)a + a; carry = r > 0xFF; a = (uint8_t)r; nf = 0; break;
        case 1:  // ADD A,B (4T)
            hc = ((a & 0xF) + (b & 0xF)) > 0xF; r = (uint16_t)a + b; carry = r > 0xFF; a = (uint8_t)r; nf = 0; break;
        case 2:  // SUB B (4T)
            hc = (a & 0xF) < (b & 0xF); carry = a < b; a = a - b; nf = 1; break;
        case 3:  // LD B,A (4T)
            b = a; break;
        case 4:  // SRL A (8T)
            carry = a & 1; a = a >> 1; hc = 0; nf = 0; break;
        case 5:  // RLCA (4T)
            carry = (a >> 7) & 1; a = ((a << 1) | (a >> 7)) & 0xFF; hc = 0; nf = 0; break;
        case 6:  // RRCA (4T)
            carry = a & 1; a = ((a >> 1) | (a << 7)) & 0xFF; hc = 0; nf = 0; break;
        case 7:  // RLA (4T)
            { int bit = carry; carry = (a >> 7) & 1; a = ((a << 1) | bit) & 0xFF; hc = 0; nf = 0; } break;
        case 8:  // RRA (4T)
            { int bit = carry ? 0x80 : 0; carry = a & 1; a = (a >> 1) | bit; hc = 0; nf = 0; } break;
        case 9:  // NEG (8T)
            hc = (a & 0xF) != 0; carry = a != 0; a = (uint8_t)(0 - a); nf = 1; break;
        case 10: // AND 0x0F (7T)
            a &= 0x0F; carry = 0; hc = 1; nf = 0; break;  // AND sets H=1 on Z80
        case 11: // AND 0xF0 (7T)
            a &= 0xF0; carry = 0; hc = 1; nf = 0; break;
        case 12: // DAA (4T) — the whole point of this kernel
            {
                uint8_t corr = 0;
                int new_c = carry;
                if (!nf) {
                    // After addition
                    if (hc || (a & 0x0F) > 9) corr |= 0x06;
                    if (carry || a > 0x99) { corr |= 0x60; new_c = 1; }
                    a = (a + corr) & 0xFF;
                } else {
                    // After subtraction
                    if (hc) corr |= 0x06;
                    if (carry) corr |= 0x60;
                    a = (a - corr) & 0xFF;
                }
                carry = new_c; hc = 0;
                // nf unchanged
            }
            break;
        case 13: // CPL (4T)
            a ^= 0xFF; hc = 1; nf = 1; break;
        case 14: // SCF (4T)
            carry = 1; hc = 0; break;  // nf unchanged? Actually SCF clears H, N on Z80
        case 15: // OR B (4T)
            a |= b; carry = 0; hc = 0; nf = 0; break;
        case 16: // XOR B (4T)
            a ^= b; carry = 0; hc = 0; nf = 0; break;
        case 17: // INC A (4T) — no carry change!
            hc = (a & 0xF) == 0xF; a++; nf = 0; break;
        case 18: // DEC A (4T) — no carry change!
            hc = (a & 0xF) == 0x0; a--; nf = 1; break;
        }
    }
    return a;
}

__constant__ uint8_t d_target[256];

static const char *opNames[] = {
    "ADD A,A", "ADD A,B", "SUB B", "LD B,A", "SRL A",
    "RLCA", "RRCA", "RLA", "RRA", "NEG",
    "AND 0x0F", "AND 0xF0", "DAA", "CPL", "SCF",
    "OR B", "XOR B", "INC A", "DEC A"
};

__constant__ uint8_t opCost[] = {
    4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 7, 7, 4, 4, 4, 4, 4, 4, 4
};

__global__ void bcd_kernel(int seqLen, uint64_t offset, uint64_t count,
                            uint32_t *bestScore, uint64_t *bestIdx,
                            const uint8_t *d_inputs, int numInputs) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[20];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % NUM_OPS);
        tmp /= NUM_OPS;
    }

    // QuickCheck on 4 inputs
    if (run_seq(ops, seqLen, d_inputs[0]) != d_target[d_inputs[0]]) return;
    if (run_seq(ops, seqLen, d_inputs[1]) != d_target[d_inputs[1]]) return;
    if (run_seq(ops, seqLen, d_inputs[numInputs/2]) != d_target[d_inputs[numInputs/2]]) return;
    if (run_seq(ops, seqLen, d_inputs[numInputs-1]) != d_target[d_inputs[numInputs-1]]) return;

    // Full verify
    for (int j = 0; j < numInputs; j++) {
        uint8_t inp = d_inputs[j];
        if (run_seq(ops, seqLen, inp) != d_target[inp]) return;
    }

    uint16_t cost = 0;
    for (int i = 0; i < seqLen; i++) cost += opCost[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;

    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) atomicExch((unsigned long long*)bestIdx, (unsigned long long)seqIdx);
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

int main(int argc, char *argv[]) {
    int maxLen = 10;
    const char *target = "bcd_to_bin";
    int gpuId = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1 < argc) target = argv[++i];
        else if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);
    }

    // Valid BCD inputs: 0x00-0x09, 0x10-0x19, ..., 0x90-0x99
    uint8_t bcd_inputs[100];
    int num_bcd = 0;
    for (int hi = 0; hi < 10; hi++)
        for (int lo = 0; lo < 10; lo++)
            bcd_inputs[num_bcd++] = hi * 16 + lo;

    // Binary inputs: 0-99
    uint8_t bin_inputs[100];
    for (int i = 0; i < 100; i++) bin_inputs[i] = i;

    uint8_t tgt[256] = {0};
    uint8_t *inputs;
    int numInputs;

    if (!strcmp(target, "bcd_to_bin")) {
        // BCD → binary: 0x42 → 42
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            tgt[bcd] = (bcd >> 4) * 10 + (bcd & 0x0F);
        }
    } else if (!strcmp(target, "bin_to_bcd")) {
        // Binary → BCD: 42 → 0x42
        inputs = bin_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            tgt[i] = ((i / 10) << 4) | (i % 10);
        }
    } else if (!strcmp(target, "bcd_x2")) {
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            int val = ((bcd >> 4) * 10 + (bcd & 0xF)) * 2;
            tgt[bcd] = ((val % 100) / 10 << 4) | (val % 10);
        }
    } else if (!strcmp(target, "bcd_x10")) {
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            int val = ((bcd >> 4) * 10 + (bcd & 0xF)) * 10;
            tgt[bcd] = ((val % 100) / 10 << 4) | (val % 10);
        }
    } else if (!strcmp(target, "bcd_add1")) {
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            int val = ((bcd >> 4) * 10 + (bcd & 0xF)) + 1;
            tgt[bcd] = ((val % 100) / 10 << 4) | (val % 10);
        }
    } else if (!strcmp(target, "bcd_sub1")) {
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            int val = ((bcd >> 4) * 10 + (bcd & 0xF)) - 1;
            if (val < 0) val = 99;
            tgt[bcd] = ((val % 100) / 10 << 4) | (val % 10);
        }
    } else if (!strcmp(target, "bcd_complement")) {
        inputs = bcd_inputs; numInputs = 100;
        for (int i = 0; i < 100; i++) {
            uint8_t bcd = bcd_inputs[i];
            int val = 100 - ((bcd >> 4) * 10 + (bcd & 0xF));
            if (val == 100) val = 0;
            tgt[bcd] = ((val % 100) / 10 << 4) | (val % 10);
        }
    } else {
        fprintf(stderr, "Unknown target: %s\n", target);
        fprintf(stderr, "Available: bcd_to_bin, bin_to_bcd, bcd_x2, bcd_x10, bcd_add1, bcd_sub1, bcd_complement\n");
        return 1;
    }

    cudaSetDevice(gpuId);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    fprintf(stderr, "Target: %s (max-len %d, %d ops, %d inputs, GPU %d)\n",
            target, maxLen, NUM_OPS, numInputs, gpuId);

    // Upload
    cudaMemcpyToSymbol(d_target, tgt, 256);
    uint8_t *d_inputs;
    cudaMalloc(&d_inputs, numInputs);
    cudaMemcpy(d_inputs, inputs, numInputs, cudaMemcpyHostToDevice);

    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);

    uint32_t initScore = 0xFFFFFFFF;
    uint64_t initIdx = 0;
    int found = 0;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);
        if (total > 500000000000000ULL) {
            fprintf(stderr, "len=%d: %.2e exceeds limit\n", len, (double)total);
            break;
        }

        cudaMemcpy(d_best, &initScore, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx, &initIdx, 8, cudaMemcpyHostToDevice);

        fprintf(stderr, "len=%d: %.2e sequences...\n", len, (double)total);

        int bs = 256;
        uint64_t batch = (uint64_t)bs * 65535;
        for (uint64_t off = 0; off < total; off += batch) {
            uint64_t cnt = total - off;
            if (cnt > batch) cnt = batch;
            bcd_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt, d_best, d_idx, d_inputs, numInputs);
            cudaDeviceSynchronize();
        }

        uint32_t bestScore;
        uint64_t bestIdx;
        cudaMemcpy(&bestScore, d_best, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestIdx, d_idx, 8, cudaMemcpyDeviceToHost);

        if (bestScore != 0xFFFFFFFF) {
            found = 1;
            int rlen = bestScore >> 16;
            int rcost = bestScore & 0xFFFF;
            uint8_t ops[20];
            uint64_t tmp = bestIdx;
            for (int i = rlen-1; i >= 0; i--) { ops[i] = tmp % NUM_OPS; tmp /= NUM_OPS; }

            printf("%s:", target);
            for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
            printf(" (%d ops, %dT)\n", rlen, rcost);
            fflush(stdout);
            break;
        }
    }
    if (!found) printf("%s: NOT FOUND at len %d\n", target, maxLen);

    cudaFree(d_best); cudaFree(d_idx); cudaFree(d_inputs);
    return 0;
}

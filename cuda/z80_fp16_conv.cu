// z80_fp16_conv.cu — GPU brute-force FP16 format conversions for Z80
// Uses 16-bit HL input (testing representative HL values across the FP16 space).
// Target: HL = f(HL_input) for 512 representative FP16 values.
// Build: nvcc -O3 -o cuda/z80_fp16_conv cuda/z80_fp16_conv.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 33-op pool: same as z80_arith16.cu
#define NUM_OPS 33

// 512 representative FP16 input values
#define NUM_INPUTS 512
#define NUM_QC 8

__device__ uint16_t run_seq(const uint8_t *ops, int len, uint16_t input) {
    uint8_t a = 0, b = 0, c = 0, d = 0, e = 0;
    uint8_t h = (uint8_t)(input >> 8), l = (uint8_t)(input & 0xFF);
    int carry = 0;

    for (int i = 0; i < len; i++) {
        uint16_t hl, bc, de, r;
        switch (ops[i]) {
        case 0: // ADD HL,HL (11T)
            hl = ((uint16_t)h << 8) | l;
            r = hl + hl;
            h = (uint8_t)(r >> 8); l = (uint8_t)r;
            break;
        case 1: // ADD HL,BC (11T)
            hl = ((uint16_t)h << 8) | l;
            bc = ((uint16_t)b << 8) | c;
            r = hl + bc;
            h = (uint8_t)(r >> 8); l = (uint8_t)r;
            break;
        case 2: // LD C,A (4T)
            c = a;
            break;
        case 3: // SWAP_HL: H=L, L=0 (11T)
            h = l; l = 0;
            break;
        case 4: // SUB HL,BC (15T)
            hl = ((uint16_t)h << 8) | l;
            bc = ((uint16_t)b << 8) | c;
            hl = hl - bc;
            h = (uint8_t)(hl >> 8); l = (uint8_t)hl;
            break;
        case 5: // EX DE,HL (4T)
            { uint8_t th=h, tl=l; h=d; l=e; d=th; e=tl; }
            break;
        case 6: // ADD HL,DE (11T)
            hl = ((uint16_t)h << 8) | l;
            de = ((uint16_t)d << 8) | e;
            r = hl + de;
            h = (uint8_t)(r >> 8); l = (uint8_t)r;
            break;
        case 7: // SUB HL,DE (15T)
            hl = ((uint16_t)h << 8) | l;
            de = ((uint16_t)d << 8) | e;
            hl = hl - de;
            h = (uint8_t)(hl >> 8); l = (uint8_t)hl;
            break;
        case 8: // SRL H / RR L = 16-bit shift right (16T)
            { uint8_t hbit = h & 1;
              h = h >> 1;
              uint8_t lbit = l & 1;
              l = (l >> 1) | (hbit << 7);
              carry = lbit; }
            break;
        // --- Per-byte ops (9-20) ---
        case 9:  // XOR A (4T) — A=0, carry=0
            a = 0; carry = 0; break;
        case 10: // SUB L (4T) — A = A - L
            carry = (a < l) ? 1 : 0; a = a - l; break;
        case 11: // SUB H (4T) — A = A - H
            carry = (a < h) ? 1 : 0; a = a - h; break;
        case 12: // ADD A,L (4T)
            { uint16_t r2 = (uint16_t)a + l; carry = r2 > 0xFF; a = (uint8_t)r2; } break;
        case 13: // ADD A,H (4T)
            { uint16_t r2 = (uint16_t)a + h; carry = r2 > 0xFF; a = (uint8_t)r2; } break;
        case 14: // SBC A,A (4T) — A = -carry (0xFF if carry, 0x00 if not)
            { int cc = carry; carry = cc ? 1 : 0; a = cc ? 0xFF : 0x00; } break;
        case 15: // LD L,A (4T)
            l = a; break;
        case 16: // LD H,A (4T)
            h = a; break;
        case 17: // LD A,L (4T)
            a = l; break;
        case 18: // LD A,H (4T)
            a = h; break;
        case 19: // OR L (4T) — A |= L
            a |= l; carry = 0; break;
        case 20: // NEG (8T) — A = -A
            carry = (a != 0) ? 1 : 0; a = (uint8_t)(0 - a); break;
        // --- Full ALU per-byte (21-32) ---
        case 21: // ADC A,L (4T)
            { int cc=carry?1:0; uint16_t r2=a+l+cc; carry=r2>0xFF; a=(uint8_t)r2; } break;
        case 22: // ADC A,H (4T)
            { int cc=carry?1:0; uint16_t r2=a+h+cc; carry=r2>0xFF; a=(uint8_t)r2; } break;
        case 23: // SBC A,L (4T)
            { int cc=carry?1:0; carry=((int)a-(int)l-cc)<0; a=a-l-(uint8_t)cc; } break;
        case 24: // SBC A,H (4T)
            { int cc=carry?1:0; carry=((int)a-(int)h-cc)<0; a=a-h-(uint8_t)cc; } break;
        case 25: // INC L (4T) — NO CARRY CHANGE!
            l++; break;
        case 26: // INC H (4T) — NO CARRY CHANGE!
            h++; break;
        case 27: // DEC L (4T) — NO CARRY CHANGE!
            l--; break;
        case 28: // DEC H (4T) — NO CARRY CHANGE!
            h--; break;
        case 29: // AND L (4T)
            a &= l; carry = 0; break;
        case 30: // XOR L (4T)
            a ^= l; carry = 0; break;
        case 31: // XOR H (4T)
            a ^= h; carry = 0; break;
        case 32: // OR H (4T)
            a |= h; carry = 0; break;
        }
    }
    return ((uint16_t)h << 8) | l;
}

// 512 representative input values in constant memory
__constant__ uint16_t d_inputs[NUM_INPUTS];

// Target output for each of the 512 inputs
__constant__ uint16_t d_target[NUM_INPUTS];

// QuickCheck indices (8 representative inputs to test first)
__constant__ int d_qc_idx[NUM_QC];

__constant__ uint8_t opCost[] = {
    11, 11, 4, 11, 15, 4, 11, 15, 16,  // 16-bit ops (0-8)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8,  // per-byte ops (9-20)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4   // ALU per-byte (21-32)
};

static const char *opNames[] = {
    "ADD HL,HL", "ADD HL,BC", "LD C,A", "SWAP_HL", "SUB HL,BC",
    "EX DE,HL", "ADD HL,DE", "SUB HL,DE", "SHR_HL",
    "XOR A", "SUB L", "SUB H", "ADD A,L", "ADD A,H",
    "SBC A,A", "LD L,A", "LD H,A", "LD A,L", "LD A,H", "OR L", "NEG",
    "ADC A,L", "ADC A,H", "SBC A,L", "SBC A,H",
    "INC L", "INC H", "DEC L", "DEC H", "AND L", "XOR L", "XOR H", "OR H"
};

__global__ void fp16_kernel(int seqLen, uint64_t offset, uint64_t count,
                             uint32_t *bestScore, uint64_t *bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[20];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % NUM_OPS);
        tmp /= NUM_OPS;
    }

    // QuickCheck: test 8 representative values first
    for (int q = 0; q < NUM_QC; q++) {
        int idx = d_qc_idx[q];
        if (run_seq(ops, seqLen, d_inputs[idx]) != d_target[idx]) return;
    }

    // Full verify: all 512 inputs
    for (int i = 0; i < NUM_INPUTS; i++) {
        if (run_seq(ops, seqLen, d_inputs[i]) != d_target[i]) return;
    }

    uint16_t cost = 0;
    for (int i = 0; i < seqLen; i++) cost += opCost[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;

    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) atomicExch((unsigned long long*)bestIdx, (unsigned long long)seqIdx);
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

// Generate 512 representative FP16 input values
static void gen_inputs(uint16_t *inputs) {
    int idx = 0;

    // Block 1: 256 values — 16 exponents x 16 mantissa patterns
    // Covers the structured FP16 space uniformly
    for (int e = 0; e < 16; e++) {
        for (int m = 0; m < 16; m++) {
            uint8_t h_byte = (uint8_t)(e * 16);   // exponent spread: 0,16,32,...,240
            uint8_t l_byte = (uint8_t)(m * 8);     // mantissa spread: 0,8,16,...,120
            inputs[idx++] = ((uint16_t)h_byte << 8) | l_byte;
        }
    }

    // Block 2: 256 edge cases
    // Sub-block A: special IEEE half values (sign, exp, mantissa combos)
    // Zero
    inputs[idx++] = 0x0000;  // +0.0
    inputs[idx++] = 0x0080;  // -0.0 (sign bit in L.7 for IEEE half: H=0x00, L=0x80)
    // Infinity patterns
    inputs[idx++] = 0x7C00;  // +inf (exp=11111, mant=0)
    inputs[idx++] = 0xFC00;  // -inf
    // NaN patterns
    inputs[idx++] = 0x7C01;  // NaN (smallest)
    inputs[idx++] = 0x7FFF;  // NaN (largest)
    // Subnormals
    inputs[idx++] = 0x0001;  // smallest subnormal
    inputs[idx++] = 0x03FF;  // largest subnormal
    // Normalized boundary
    inputs[idx++] = 0x0400;  // smallest normalized
    inputs[idx++] = 0x7BFF;  // largest finite positive
    // One and nearby
    inputs[idx++] = 0x3C00;  // 1.0
    inputs[idx++] = 0xBC00;  // -1.0
    inputs[idx++] = 0x4000;  // 2.0
    inputs[idx++] = 0x3800;  // 0.5
    inputs[idx++] = 0x3555;  // ~1/3
    inputs[idx++] = 0x4200;  // 3.0

    // Sub-block B: all-bits patterns and sign variants
    inputs[idx++] = 0xFFFF;
    inputs[idx++] = 0xFF00;
    inputs[idx++] = 0x00FF;
    inputs[idx++] = 0x8000;
    inputs[idx++] = 0x0100;
    inputs[idx++] = 0x807F;
    inputs[idx++] = 0x5555;
    inputs[idx++] = 0xAAAA;

    // Sub-block C: systematic sign-flipped versions (high bit of H toggled)
    // For each of 16 exponents, 2 mantissa values, with sign bit set
    for (int e = 0; e < 16; e++) {
        uint8_t h_byte = (uint8_t)(e * 16) | 0x80;  // sign bit in H.7
        inputs[idx++] = ((uint16_t)h_byte << 8) | 0x00;
        inputs[idx++] = ((uint16_t)h_byte << 8) | 0x3F;
    }

    // Sub-block D: dense low-exponent coverage (exponents 0-7, fine mantissa)
    for (int e = 0; e < 8; e++) {
        for (int m = 0; m < 8; m++) {
            uint8_t h_byte = (uint8_t)(e * 4);      // fine exponent: 0,4,8,...,28
            uint8_t l_byte = (uint8_t)(m * 17);     // mantissa: 0,17,34,...,119
            inputs[idx++] = ((uint16_t)h_byte << 8) | l_byte;
        }
    }

    // Sub-block E: dense high-exponent coverage
    for (int e = 0; e < 8; e++) {
        for (int m = 0; m < 8; m++) {
            uint8_t h_byte = (uint8_t)(128 + e * 16); // exponents 128,144,...,240
            uint8_t l_byte = (uint8_t)(m * 18);       // mantissa spread
            inputs[idx++] = ((uint16_t)h_byte << 8) | l_byte;
        }
    }

    // Sub-block F: fill remaining with linear spread if needed
    uint16_t step = 0xFFFF / (NUM_INPUTS - idx + 1);
    uint16_t val = 1;
    while (idx < NUM_INPUTS) {
        inputs[idx++] = val;
        val += step;
    }
}

// QuickCheck indices: pick 8 diverse samples from the 512 inputs
static void gen_qc_indices(int *qc) {
    qc[0] = 0;     // 0x0000 — zero
    qc[1] = 1;     // 0x0008
    qc[2] = 128;   // mid block 1
    qc[3] = 255;   // end block 1
    qc[4] = 256;   // +0.0 edge case
    qc[5] = 258;   // +inf
    qc[6] = 266;   // 1.0
    qc[7] = 511;   // last input
}

// IEEE half-precision: H=[SEEEEE.MM] L=[MMMMMMMM]
//   sign    = H bit 7
//   exp5    = H bits [6:2]
//   mant10  = H bits [1:0] : L bits [7:0]
//
// Z80-FP16: H=[EEEEEEEE] L=[SMMMMMMM]
//   exp8    = H (exp5 == 0 ? 0 : exp5 + 112)
//   sign    = L bit 7
//   mant7   = L bits [6:0] = mant10 >> 3

static void gen_target_ieee_to_z80(const uint16_t *inputs, uint16_t *tgt, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t val = inputs[i];
        uint8_t hi = (uint8_t)(val >> 8);
        uint8_t lo = (uint8_t)(val & 0xFF);

        uint8_t sign  = (hi >> 7) & 1;
        uint8_t exp5  = (hi >> 2) & 0x1F;
        uint16_t mant10 = ((uint16_t)(hi & 0x03) << 8) | lo;

        uint8_t exp8  = (exp5 == 0) ? 0 : (uint8_t)(exp5 + 112);
        uint8_t mant7 = (uint8_t)(mant10 >> 3);

        uint8_t out_h = exp8;
        uint8_t out_l = (sign << 7) | (mant7 & 0x7F);
        tgt[i] = ((uint16_t)out_h << 8) | out_l;
    }
}

static void gen_target_z80_to_ieee(const uint16_t *inputs, uint16_t *tgt, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t val = inputs[i];
        uint8_t exp8  = (uint8_t)(val >> 8);
        uint8_t lo    = (uint8_t)(val & 0xFF);

        uint8_t sign  = (lo >> 7) & 1;
        uint8_t mant7 = lo & 0x7F;

        uint8_t exp5  = (exp8 == 0) ? 0 : (uint8_t)(exp8 - 112);
        // Clamp: if exp8 < 113 and exp8 != 0, map to 1 (smallest normalized)
        if (exp8 != 0 && exp8 < 113) exp5 = 1;
        // If exp8 > 142, clamp to 31 (inf/NaN territory)
        if (exp8 > 142) exp5 = 31;

        uint16_t mant10 = (uint16_t)mant7 << 3;

        uint8_t out_h = (sign << 7) | (exp5 << 2) | (uint8_t)((mant10 >> 8) & 0x03);
        uint8_t out_l = (uint8_t)(mant10 & 0xFF);
        tgt[i] = ((uint16_t)out_h << 8) | out_l;
    }
}

int main(int argc, char *argv[]) {
    int maxLen = 10;
    const char *target = "ieee_to_z80";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1 < argc) target = argv[++i];
        else if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
    }

    // Generate inputs
    uint16_t inputs[NUM_INPUTS];
    gen_inputs(inputs);

    // Generate targets
    uint16_t tgt[NUM_INPUTS];
    if (!strcmp(target, "ieee_to_z80")) {
        gen_target_ieee_to_z80(inputs, tgt, NUM_INPUTS);
    } else if (!strcmp(target, "z80_to_ieee")) {
        gen_target_z80_to_ieee(inputs, tgt, NUM_INPUTS);
    } else {
        fprintf(stderr, "Unknown target: %s\n", target);
        fprintf(stderr, "Available: ieee_to_z80, z80_to_ieee\n");
        return 1;
    }

    // Print a few sample I/O pairs for verification
    fprintf(stderr, "Target: %s (%d inputs, max-len %d, %d ops)\n", target, NUM_INPUTS, maxLen, NUM_OPS);
    fprintf(stderr, "Sample I/O pairs:\n");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "  input=0x%04X → output=0x%04X\n", inputs[i], tgt[i]);
    }

    // QuickCheck indices
    int qc[NUM_QC];
    gen_qc_indices(qc);

    // CUDA setup
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);

    cudaMemcpyToSymbol(d_inputs, inputs, NUM_INPUTS * sizeof(uint16_t));
    cudaMemcpyToSymbol(d_target, tgt, NUM_INPUTS * sizeof(uint16_t));
    cudaMemcpyToSymbol(d_qc_idx, qc, NUM_QC * sizeof(int));

    uint32_t initScore = 0xFFFFFFFF;
    uint64_t initIdx = 0;
    int found = 0;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);
        if (total > 50000000000000ULL) {
            fprintf(stderr, "len=%d: search space %.2e exceeds 50T limit, stopping\n",
                    len, (double)total);
            break;
        }

        cudaMemcpy(d_best, &initScore, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx, &initIdx, 8, cudaMemcpyHostToDevice);

        fprintf(stderr, "len=%d: searching %.2e sequences...\n", len, (double)total);

        int bs = 256;
        uint64_t batch = (uint64_t)bs * 65535;
        for (uint64_t off = 0; off < total; off += batch) {
            uint64_t cnt = total - off;
            if (cnt > batch) cnt = batch;
            fp16_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt, d_best, d_idx);
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
    if (!found) printf("%s: NOT FOUND at max-len %d\n", target, maxLen);

    cudaFree(d_best); cudaFree(d_idx);
    return 0;
}

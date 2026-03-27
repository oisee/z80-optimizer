// z80_universal21.cu — Universal 21-op kernel for Z80 brute-force search
// The MINIMUM instruction set that generates ALL known optimal arithmetic.
// 21 ops = union of mul8(14) + mul16(5) + div8(6) pools (with overlaps).
// Covers: 8-bit mul/div, 16-bit mul, shifts, rotates, carry tricks.
//
// Search space: 21^9 = 7 min, 21^10 = 2.3h, 21^11 = 2d on 1 GPU.
// Build: nvcc -O3 -o cuda/z80_universal21 cuda/z80_universal21.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_OPS 21

// State: A, B (save reg), C (save reg), H, L, carry
// Input: A = input, B = 0, C = 0, H = 0, L = input
// Output: depends on target (A, HL, or flags)

__device__ uint32_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0, c = 0, h = 0, l = input;
    int carry = 0;

    for (int i = 0; i < len; i++) {
        uint16_t hl, bc, r16;
        uint16_t r;
        switch (ops[i]) {
        // --- 8-bit arithmetic (from mul8 pool) ---
        case 0:  // ADD A,A (4T)
            r = (uint16_t)a + a; carry = r > 0xFF; a = (uint8_t)r; break;
        case 1:  // ADD A,B (4T)
            r = (uint16_t)a + b; carry = r > 0xFF; a = (uint8_t)r; break;
        case 2:  // LD B,A (4T)
            b = a; break;
        case 3:  // RLA (4T) — rotate left through carry
            { uint8_t bit = carry ? 1 : 0; carry = (a >> 7) & 1; a = (a << 1) | bit; } break;
        case 4:  // SBC A,B (4T) — subtract with carry
            { int cc = carry ? 1 : 0; carry = ((int)a - (int)b - cc) < 0; a = a - b - (uint8_t)cc; } break;
        case 5:  // RLCA (4T) — rotate left circular
            carry = (a >> 7) & 1; a = ((a << 1) | (a >> 7)) & 0xFF; break;
        case 6:  // NEG (8T) — negate A
            carry = (a != 0) ? 1 : 0; a = (uint8_t)(0 - a); break;
        case 7:  // ADC A,B (4T) — add with carry
            { int cc = carry ? 1 : 0; r = (uint16_t)a + b + cc; carry = r > 0xFF; a = (uint8_t)r; } break;
        case 8:  // RRCA (4T) — rotate right circular
            carry = a & 1; a = ((a >> 1) | (a << 7)) & 0xFF; break;
        case 9:  // SUB B (4T)
            carry = (a < b) ? 1 : 0; a = a - b; break;
        case 10: // RRA (4T) — rotate right through carry
            { uint8_t bit = carry ? 0x80 : 0; carry = a & 1; a = (a >> 1) | bit; } break;
        case 11: // SBC A,A (4T) — carry to mask (0xFF if carry, 0x00 if not)
            { int cc = carry ? 1 : 0; carry = cc; a = cc ? 0xFF : 0x00; } break;
        case 12: // SRL A (8T) — logical shift right
            carry = a & 1; a = a >> 1; break;
        case 13: // ADC A,A (4T) — double + carry
            { int cc = carry ? 1 : 0; r = (uint16_t)a + a + cc; carry = r > 0xFF; a = (uint8_t)r; } break;

        // --- 16-bit arithmetic (from mul16 + div8 pools) ---
        case 14: // ADD HL,HL (11T) — double HL
            hl = ((uint16_t)h << 8) | l;
            r16 = hl + hl;
            h = (uint8_t)(r16 >> 8); l = (uint8_t)r16; break;
        case 15: // ADD HL,BC (11T)
            hl = ((uint16_t)h << 8) | l;
            bc = ((uint16_t)b << 8) | c;
            r16 = hl + bc;
            h = (uint8_t)(r16 >> 8); l = (uint8_t)r16; break;
        case 16: // LD C,A (4T)
            c = a; break;
        case 17: // SUB HL,BC (15T) — virtual: OR A / SBC HL,BC
            hl = ((uint16_t)h << 8) | l;
            bc = ((uint16_t)b << 8) | c;
            hl = hl - bc;
            h = (uint8_t)(hl >> 8); l = (uint8_t)hl; break;
        case 18: // SWAP_HL (11T) — virtual: LD H,L / LD L,0 (= ×256)
            h = l; l = 0; break;
        case 19: // LD A,H (4T) — read high byte of HL result
            a = h; break;
        case 20: // SHR_HL (16T) — virtual: SRL H / RR L
            { uint8_t hbit = h & 1;
              h = h >> 1;
              l = (l >> 1) | (hbit << 7);
              carry = l & 1; /* actually the old L bit 0 */ } break;
        }
    }
    // Return packed state: H:L:A (24 bits in uint32)
    return ((uint32_t)h << 16) | ((uint32_t)l << 8) | a;
}

// Target tables
#define MAX_INPUTS 256
__constant__ uint32_t d_target[MAX_INPUTS];
__constant__ int d_num_inputs;
__constant__ int d_target_mode;  // 0=A only, 1=HL only, 2=HLA full

__constant__ uint8_t opCost[] = {
    4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 4,  // 8-bit (0-13)
    11, 11, 4, 15, 11, 4, 16                       // 16-bit (14-20)
};

static const char *opNames[] = {
    "ADD A,A", "ADD A,B", "LD B,A", "RLA", "SBC A,B", "RLCA", "NEG",
    "ADC A,B", "RRCA", "SUB B", "RRA", "SBC A,A", "SRL A", "ADC A,A",
    "ADD HL,HL", "ADD HL,BC", "LD C,A", "SUB HL,BC", "SWAP_HL",
    "LD A,H", "SHR_HL"
};

__global__ void search_kernel(int seqLen, uint64_t offset, uint64_t count,
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

    // QuickCheck: 4 inputs
    uint32_t r;
    r = run_seq(ops, seqLen, 0);
    if (d_target_mode == 0) { if ((r & 0xFF) != (d_target[0] & 0xFF)) return; }
    else if (d_target_mode == 1) { if ((r >> 8) != (d_target[0] >> 8)) return; }
    else { if (r != d_target[0]) return; }

    r = run_seq(ops, seqLen, 1);
    if (d_target_mode == 0) { if ((r & 0xFF) != (d_target[1] & 0xFF)) return; }
    else if (d_target_mode == 1) { if ((r >> 8) != (d_target[1] >> 8)) return; }
    else { if (r != d_target[1]) return; }

    r = run_seq(ops, seqLen, 127);
    if (d_target_mode == 0) { if ((r & 0xFF) != (d_target[127] & 0xFF)) return; }
    else if (d_target_mode == 1) { if ((r >> 8) != (d_target[127] >> 8)) return; }
    else { if (r != d_target[127]) return; }

    r = run_seq(ops, seqLen, 255);
    if (d_target_mode == 0) { if ((r & 0xFF) != (d_target[255] & 0xFF)) return; }
    else if (d_target_mode == 1) { if ((r >> 8) != (d_target[255] >> 8)) return; }
    else { if (r != d_target[255]) return; }

    // Full verify
    for (int i = 0; i < 256; i++) {
        r = run_seq(ops, seqLen, (uint8_t)i);
        if (d_target_mode == 0) { if ((r & 0xFF) != (d_target[i] & 0xFF)) return; }
        else if (d_target_mode == 1) { if ((r >> 8) != (d_target[i] >> 8)) return; }
        else { if (r != d_target[i]) return; }
    }

    uint16_t cost = 0;
    for (int i = 0; i < seqLen; i++) cost += opCost[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;

    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) atomicExch((unsigned long long*)bestIdx, (unsigned long long)seqIdx);
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

// Target generators
static void gen_target_mul8(int k, uint32_t *tgt) {
    for (int i = 0; i < 256; i++) tgt[i] = ((uint8_t)(i * k)) & 0xFF;
}
static void gen_target_mul16(int k, uint32_t *tgt) {
    for (int i = 0; i < 256; i++) {
        uint16_t hl = (uint16_t)i * k;
        tgt[i] = ((uint32_t)(hl >> 8) << 16) | ((uint32_t)(hl & 0xFF) << 8);
    }
}
static void gen_target_div8(int k, uint32_t *tgt) {
    for (int i = 0; i < 256; i++) tgt[i] = (uint8_t)(i / k);
}
static void gen_target_mod8(int k, uint32_t *tgt) {
    for (int i = 0; i < 256; i++) tgt[i] = (uint8_t)(i % k);
}
static void gen_target_custom(const char *name, uint32_t *tgt) {
    for (int i = 0; i < 256; i++) {
        uint8_t a = (uint8_t)i;
        if (!strcmp(name, "neg_hl"))     { uint16_t hl = i; hl = -hl; tgt[i] = ((uint32_t)(hl>>8)<<16)|((hl&0xFF)<<8); }
        else if (!strcmp(name, "sext"))  { tgt[i] = (a >= 128) ? (0xFF0000 | (a << 8) | a) : (a << 8) | a; }
        else if (!strcmp(name, "bool"))  { tgt[i] = (a != 0) ? 1 : 0; }
        else if (!strcmp(name, "not"))   { tgt[i] = (a != 0) ? 0 : 1; }
        else if (!strcmp(name, "abs"))   { tgt[i] = (a >= 128) ? (uint8_t)(-a) : a; }
        else if (!strcmp(name, "clz"))   { uint8_t v=a; int c=0; if(!v){c=8;} else{while(!(v&0x80)){v<<=1;c++;}} tgt[i]=c; }
        else if (!strcmp(name, "popcount")) { uint8_t v=a; int c=0; while(v){c+=v&1;v>>=1;} tgt[i]=c; }
        else if (!strcmp(name, "reverse")) { uint8_t v=0; for(int b=0;b<8;b++) v|=((a>>b)&1)<<(7-b); tgt[i]=v; }
        else if (!strcmp(name, "sqrt"))  { uint8_t v=0; while((v+1)*(v+1)<=a)v++; tgt[i]=v; }
        else if (!strcmp(name, "log2"))  { if(a==0)tgt[i]=0; else{int l=0;uint8_t v=a;while(v>>=1)l++;tgt[i]=l;} }
        else if (!strcmp(name, "is_pow2")) { tgt[i] = (a && !(a&(a-1))) ? 1 : 0; }
        else if (!strcmp(name, "max_a_b")) { tgt[i] = a; /* A=max(A,B) where B=0 initially = A */ }
        else if (!strcmp(name, "swap_nibble")) { tgt[i] = ((a << 4) | (a >> 4)) & 0xFF; }
        else if (!strcmp(name, "gray"))  { tgt[i] = a ^ (a >> 1); }
        else if (!strcmp(name, "int2fp16")) {
            if (a == 0) { tgt[i] = 0; }
            else {
                int exp = 134; uint8_t v = a;
                while (!(v & 0x80)) { v <<= 1; exp--; }
                uint8_t mant = v & 0x7F;
                tgt[i] = ((uint32_t)(exp & 0xFF) << 16) | ((uint32_t)mant << 8);
            }
        }
        else tgt[i] = a;  // identity
    }
}

int main(int argc, char *argv[]) {
    int maxLen = 12;
    const char *target = "mul8";
    int k = 0;
    int mode = -1;  // auto-detect
    int gpuId = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1 < argc) target = argv[++i];
        else if (!strcmp(argv[i], "--k") && i+1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);
    }

    cudaSetDevice(gpuId);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    uint32_t tgt[256];
    int tgt_mode;

    if (!strcmp(target, "mul8"))       { gen_target_mul8(k, tgt); tgt_mode = 0; }
    else if (!strcmp(target, "mul16")) { gen_target_mul16(k, tgt); tgt_mode = 1; }
    else if (!strcmp(target, "div8"))  { gen_target_div8(k, tgt); tgt_mode = 0; }
    else if (!strcmp(target, "mod8"))  { gen_target_mod8(k, tgt); tgt_mode = 0; }
    else {
        gen_target_custom(target, tgt);
        tgt_mode = (mode >= 0) ? mode : 0;  // default to A-only for custom
        // Auto-detect: if any target has non-zero H or L, use mode 1 or 2
        if (mode < 0) {
            for (int i = 0; i < 256; i++) {
                if (tgt[i] & 0xFFFF00) { tgt_mode = 2; break; }  // has H or L component
            }
        }
    }
    if (mode >= 0) tgt_mode = mode;

    fprintf(stderr, "Target: %s (k=%d, mode=%d [%s], max-len %d, %d ops, GPU %d)\n",
            target, k, tgt_mode,
            tgt_mode == 0 ? "A-only" : tgt_mode == 1 ? "HL-only" : "HLA-full",
            maxLen, NUM_OPS, gpuId);

    // Upload targets
    cudaMemcpyToSymbol(d_target, tgt, 256 * sizeof(uint32_t));
    int nm = 256;
    cudaMemcpyToSymbol(d_num_inputs, &nm, sizeof(int));
    cudaMemcpyToSymbol(d_target_mode, &tgt_mode, sizeof(int));

    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);

    uint32_t initScore = 0xFFFFFFFF;
    uint64_t initIdx = 0;
    int found = 0;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);
        if (total > 500000000000000ULL) {  // 500T limit (~3 days on 1 GPU)
            fprintf(stderr, "len=%d: %.2e exceeds limit, stopping\n", len, (double)total);
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
            search_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt, d_best, d_idx);
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

            printf("%s(k=%d):", target, k);
            for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
            printf(" (%d ops, %dT)\n", rlen, rcost);
            fflush(stdout);
            break;
        }
    }
    if (!found) printf("%s(k=%d): NOT FOUND at len %d\n", target, k, maxLen);

    cudaFree(d_best); cudaFree(d_idx);
    return 0;
}

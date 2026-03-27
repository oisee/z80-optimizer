// z80_arith16.cu — GPU brute-force 16-bit arithmetic idioms for Z80
// Uses mulopt16_mini's 8-op pool. Target: HL = f(input) for all 256 8-bit inputs.
// Build: nvcc -O3 -o z80_arith16 z80_arith16.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 21-op pool: 16-bit ops + per-byte ops (for Alf-style patterns)
// 0-8: 16-bit level ops
// 9-20: per-byte ops (A↔H, A↔L, cross-register arithmetic)
#define NUM_OPS 46

__device__ uint16_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0, c = 0, d = 0, e = 0, h = 0, l = input;
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
        // --- Immediate ops + shifts (33-42) for bit-field manipulation ---
        case 33: // AND 0x0F (7T)
            a &= 0x0F; carry = 0; break;
        case 34: // AND 0xF0 (7T)
            a &= 0xF0; carry = 0; break;
        case 35: // AND 0x7F (7T)
            a &= 0x7F; carry = 0; break;
        case 36: // AND 0x80 (7T)
            a &= 0x80; carry = 0; break;
        case 37: // OR 0x80 (7T)
            a |= 0x80; carry = 0; break;
        case 38: // XOR 0x80 (7T) — flip sign bit
            a ^= 0x80; carry = 0; break;
        case 39: // SRL A (8T) — logical shift right
            carry = a & 1; a = a >> 1; break;
        case 40: // RLCA (4T) — rotate left circular
            carry = (a >> 7) & 1; a = ((a << 1) | (a >> 7)) & 0xFF; break;
        case 41: // RRCA (4T) — rotate right circular
            carry = a & 1; a = ((a >> 1) | (a << 7)) & 0xFF; break;
        case 42: // CPL (4T) — complement A
            a ^= 0xFF; break;
        case 43: // ADD A,A (4T) — double
            { uint16_t r2 = (uint16_t)a + a; carry = r2 > 0xFF; a = (uint8_t)r2; } break;
        case 44: // RES 7,L (8T) — clear sign bit of L directly
            l &= 0x7F; break;
        case 45: // SET 7,L (8T) — set sign bit of L
            l |= 0x80; break;
        }
    }
    return ((uint16_t)h << 8) | l;
}

// Target function table
__constant__ uint16_t d_target[256];

#define NUM_OPS_EXT 46

__constant__ uint8_t opCost[] = {
    11, 11, 4, 11, 15, 4, 11, 15, 16,  // 16-bit ops (0-8)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8,  // per-byte ops (9-20)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // ALU per-byte (21-32)
    7, 7, 7, 7, 7, 7, 8, 4, 4, 4, 4, 8, 8  // imm+shift ops (33-45)
};

static const char *opNames[] = {
    "ADD HL,HL", "ADD HL,BC", "LD C,A", "SWAP_HL", "SUB HL,BC",
    "EX DE,HL", "ADD HL,DE", "SUB HL,DE", "SHR_HL",
    "XOR A", "SUB L", "SUB H", "ADD A,L", "ADD A,H",
    "SBC A,A", "LD L,A", "LD H,A", "LD A,L", "LD A,H", "OR L", "NEG",
    "ADC A,L", "ADC A,H", "SBC A,L", "SBC A,H",
    "INC L", "INC H", "DEC L", "DEC H", "AND L", "XOR L", "XOR H", "OR H",
    "AND 0x0F", "AND 0xF0", "AND 0x7F", "AND 0x80", "OR 0x80", "XOR 0x80",
    "SRL A", "RLCA", "RRCA", "CPL", "ADD A,A", "RES 7,L", "SET 7,L"
};

__global__ void arith_kernel(int seqLen, uint64_t offset, uint64_t count,
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
    
    // QuickCheck
    if (run_seq(ops, seqLen, 0) != d_target[0]) return;
    if (run_seq(ops, seqLen, 1) != d_target[1]) return;
    if (run_seq(ops, seqLen, 127) != d_target[127]) return;
    if (run_seq(ops, seqLen, 255) != d_target[255]) return;
    
    // Full verify
    for (int i = 0; i < 256; i++) {
        if (run_seq(ops, seqLen, (uint8_t)i) != d_target[i]) return;
    }
    
    uint16_t cost = 0;
    for (int i = 0; i < seqLen; i++) cost += opCost[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;
    
    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) atomicExch((unsigned long long*)bestIdx, (unsigned long long)seqIdx);
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

static void gen_target(const char *name, uint16_t *tgt) {
    for (int i = 0; i < 256; i++) {
        uint8_t l_in = (uint8_t)i;
        uint16_t hl = (uint16_t)i; // initial: H=0, L=input → HL = input
        if (!strcmp(name, "neg"))       tgt[i] = (-hl) & 0xFFFF;
        else if (!strcmp(name, "shr1")) tgt[i] = hl >> 1;
        else if (!strcmp(name, "shr4")) tgt[i] = hl >> 4;
        else if (!strcmp(name, "shl4")) tgt[i] = (hl << 4) & 0xFFFF;
        else if (!strcmp(name, "swap")) tgt[i] = ((hl & 0xFF) << 8) | ((hl >> 8) & 0xFF);
        else if (!strcmp(name, "x256")) tgt[i] = hl << 8;
        else if (!strcmp(name, "abs"))  tgt[i] = (hl & 0x8000) ? (-hl) & 0xFFFF : hl;
        else if (!strcmp(name, "sqr"))  tgt[i] = (hl * hl) & 0xFFFF;
        else if (!strcmp(name, "not_hl"))  tgt[i] = (~hl) & 0xFFFF;
        else if (!strcmp(name, "shr8"))    tgt[i] = hl >> 8;
        else if (!strcmp(name, "mul3_hl")) tgt[i] = (hl * 3) & 0xFFFF;
        else if (!strcmp(name, "mul5_hl")) tgt[i] = (hl * 5) & 0xFFFF;
        else if (!strcmp(name, "mul10_hl"))tgt[i] = (hl * 10) & 0xFFFF;
        else if (!strcmp(name, "inc_hl"))  tgt[i] = (hl + 1) & 0xFFFF;
        else if (!strcmp(name, "dec_hl"))  tgt[i] = (hl - 1) & 0xFFFF;
        else if (!strcmp(name, "and_7f"))  tgt[i] = hl & 0x007F;
        else if (!strcmp(name, "or_80"))   tgt[i] = hl | 0x0080;
        else if (!strcmp(name, "hi_byte")) tgt[i] = (hl >> 8) & 0xFF;
        else if (!strcmp(name, "int2fp16")) {
            // int (in L) → Z80-FP16 (in HL): H=exp, L=mant
            uint8_t inp = hl & 0xFF;
            if (inp == 0) { tgt[i] = 0; }
            else {
                int exp = 127;
                uint8_t v = inp;
                while (v >= 2) { v >>= 1; exp++; }
                // mant = ((inp / 2^(exp-127)) - 1) * 128
                uint8_t mant = ((inp << 7) >> (exp - 127)) & 0x7F;
                tgt[i] = ((uint16_t)(exp & 0xFF) << 8) | mant;
            }
        }
        else if (!strcmp(name, "fp16_normalize")) {
            // Normalize FP16: input H=exp, L=0|mantissa (with implicit leading 1 missing)
            // Shift mant left until bit 6 set, decrement exp per shift
            uint8_t exp = (hl >> 8) & 0xFF;
            uint8_t mant = hl & 0x7F;
            if (mant == 0) { tgt[i] = 0; }
            else {
                while (!(mant & 0x40) && exp > 0) { mant <<= 1; exp--; }
                tgt[i] = ((uint16_t)exp << 8) | (mant & 0x7F);
            }
        }
        else if (!strcmp(name, "fp16_to_int")) {
            // FP16 → integer: H=exp, L=[S]MMMMMMM → A in L
            // Integer = mant >> (bias+6 - exp), with implicit 1
            uint8_t exp = (hl >> 8) & 0xFF;
            uint8_t mant = hl & 0x7F;
            if (exp < 127) { tgt[i] = 0; }  // < 1.0
            else {
                int shift = exp - 127;
                uint8_t val = (0x80 | mant) >> (7 - shift);
                if (shift > 7) val = 0xFF;
                tgt[i] = val & 0xFF;
            }
        }
        else if (!strcmp(name, "fp16_abs")) {
            // Clear sign bit: L &= 0x7F
            tgt[i] = (hl & 0xFF00) | (hl & 0x7F);
        }
        else if (!strcmp(name, "fp16_neg")) {
            // Flip sign bit: L ^= 0x80
            tgt[i] = (hl & 0xFF00) | ((hl ^ 0x80) & 0xFF);
        }
        else if (!strcmp(name, "fp16_is_zero")) {
            // H==0 && (L&0x7F)==0 → 1 in L, else 0
            uint8_t exp = (hl >> 8) & 0xFF;
            uint8_t mant = hl & 0x7F;
            tgt[i] = (exp == 0 && mant == 0) ? 1 : 0;
        }
        else if (!strcmp(name, "fp16_x2")) {
            // ×2: increment exponent. H++
            tgt[i] = (((hl >> 8) + 1) << 8) | (hl & 0xFF);
        }
        else if (!strcmp(name, "fp16_half")) {
            // ÷2: decrement exponent. H--
            tgt[i] = (((hl >> 8) - 1) << 8) | (hl & 0xFF);
        }
        else if (!strcmp(name, "byteswap")) tgt[i] = (l_in << 8) | 0;
        else if (!strcmp(name, "realswap")) {
            tgt[i] = (hl & 0xFF) << 8;
        }
        else if (!strcmp(name, "zext")) tgt[i] = hl & 0xFF;
        else if (!strcmp(name, "sext")) {
            uint8_t inp = hl & 0xFF;
            tgt[i] = (inp >= 128) ? (0xFF00 | inp) : inp;
        }
        else if (!strcmp(name, "clamp127")) { // clamp to 0-127 (clear bit 7)
            tgt[i] = hl & 0x7F;
        }
        else if (!strcmp(name, "hl_eq_0")) { // HL==0 ? 1 : 0 (result in L, H=0)
            tgt[i] = (hl == 0) ? 1 : 0;
        }
        else if (!strcmp(name, "hl_ne_0")) { // HL!=0 ? 1 : 0
            tgt[i] = (hl != 0) ? 1 : 0;
        }
        else if (!strcmp(name, "double_lo")) { // HL = (0, L*2) — double low byte only
            tgt[i] = ((hl & 0xFF) * 2) & 0xFF;
        }
        else if (!strcmp(name, "neg8_in_hl")) { // HL = (0xFF if input>0 else 0, -input)
            uint8_t inp = hl & 0xFF;
            uint16_t neg = (-inp) & 0xFF;
            uint8_t hi = (inp > 0) ? 0xFF : 0;
            tgt[i] = (hi << 8) | neg;
        }
        else tgt[i] = hl;
    }
}

int main(int argc, char *argv[]) {
    int maxLen = 12;
    const char *idiom = "neg";
    int runAll = 0;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--idiom") && i+1 < argc) idiom = argv[++i];
        else if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--all")) runAll = 1;
    }
    
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);
    uint32_t dummy = 0;
    cudaMemcpy(d_best, &dummy, 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    const char *all[] = {"neg","shr1","shr4","shl4","swap","x256","sext","clamp127","hl_ne_0","neg8_in_hl","not_hl","shr8","mul3_hl","mul5_hl","mul10_hl","inc_hl","dec_hl","int2fp16","fp16_normalize","fp16_to_int","fp16_abs","fp16_neg","fp16_is_zero","fp16_x2","fp16_half",NULL};
    
    for (int ii = 0; all[ii]; ii++) {
        if (!runAll && strcmp(idiom, all[ii]) != 0) continue;
        
        const char *name = all[ii];
        uint16_t tgt[256];
        gen_target(name, tgt);
        cudaMemcpyToSymbol(d_target, tgt, 512);
        
        fprintf(stderr, "Searching: %s (max-len %d, %d ops)\n", name, maxLen, NUM_OPS);
        
        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initIdx = 0;
        int found = 0;
        
        for (int len = 1; len <= maxLen; len++) {
            uint64_t total = ipow(NUM_OPS, len);
            if (total > 50000000000000ULL) break; // ~50T max (len-9 at 33 ops = 5.5T)
            
            cudaMemcpy(d_best, &initScore, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, &initIdx, 8, cudaMemcpyHostToDevice);
            
            int bs = 256;
            uint64_t batch = (uint64_t)bs * 65535;
            for (uint64_t off = 0; off < total; off += batch) {
                uint64_t cnt = total - off;
                if (cnt > batch) cnt = batch;
                arith_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt, d_best, d_idx);
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
                
                printf("%s:", name);
                for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
                printf(" (%d virtual, %dT)\n", rlen, rcost);
                fflush(stdout);
                break;
            }
        }
        if (!found) printf("%s: NOT FOUND at len %d\n", name, maxLen);
    }
    
    cudaFree(d_best); cudaFree(d_idx);
    return 0;
}

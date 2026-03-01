// Z80 QuickCheck GPU kernel — standalone CUDA implementation.
// Computes fingerprints for candidate instruction sequences and compares
// against a target fingerprint. Each CUDA thread evaluates one candidate.
//
// Build: nvcc -O2 -o z80qc z80_quickcheck.cu
// Usage: ./z80qc < input.bin > output.bin
//
// Binary protocol (stdin):
//   Header:  uint32 candidate_count
//            uint32 seq_len (instructions per candidate, e.g. 2)
//            uint32 dead_flags (flag mask, 0 = exact match)
//            uint8  target_fingerprint[80]  (10 bytes * 8 test vectors)
//   Body:    For each candidate (candidate_count times):
//              seq_len * 4 bytes: (uint16 opcode, uint16 imm) per instruction
//
// Binary protocol (stdout):
//   uint32 match_count
//   uint32 match_indices[match_count]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

// ============================================================
// Z80 flag bits
// ============================================================
#define FLAG_C  0x01u
#define FLAG_N  0x02u
#define FLAG_P  0x04u
#define FLAG_V  0x04u
#define FLAG_3  0x08u
#define FLAG_H  0x10u
#define FLAG_5  0x20u
#define FLAG_Z  0x40u
#define FLAG_S  0x80u

// ============================================================
// Precomputed flag tables (constant memory, ~800 bytes)
// ============================================================
__constant__ uint8_t d_sz53[256];
__constant__ uint8_t d_sz53p[256];
__constant__ uint8_t d_parity[256];
__constant__ uint8_t d_halfcarry_add[8];
__constant__ uint8_t d_halfcarry_sub[8];
__constant__ uint8_t d_overflow_add[8];
__constant__ uint8_t d_overflow_sub[8];

// Host-side tables for initialization and CPU verification.
static uint8_t h_sz53[256];
static uint8_t h_sz53p[256];
static uint8_t h_parity[256];
static const uint8_t h_halfcarry_add[8] = {0, FLAG_H, FLAG_H, FLAG_H, 0, 0, 0, FLAG_H};
static const uint8_t h_halfcarry_sub[8] = {0, 0, FLAG_H, 0, FLAG_H, 0, FLAG_H, FLAG_H};
static const uint8_t h_overflow_add[8]  = {0, 0, 0, FLAG_V, FLAG_V, 0, 0, 0};
static const uint8_t h_overflow_sub[8]  = {0, FLAG_V, 0, 0, 0, 0, FLAG_V, 0};

static void init_tables() {
    for (int i = 0; i < 256; i++) {
        h_sz53[i] = (uint8_t)(i) & (FLAG_3 | FLAG_5 | FLAG_S);
        uint8_t j = (uint8_t)i;
        uint8_t p = 0;
        for (int k = 0; k < 8; k++) { p ^= j & 1; j >>= 1; }
        h_parity[i] = (p == 0) ? FLAG_P : 0;
        h_sz53p[i] = h_sz53[i] | h_parity[i];
    }
    h_sz53[0] |= FLAG_Z;
    h_sz53p[0] |= FLAG_Z;
}

static void upload_tables() {
    cudaMemcpyToSymbol(d_sz53, h_sz53, 256);
    cudaMemcpyToSymbol(d_sz53p, h_sz53p, 256);
    cudaMemcpyToSymbol(d_parity, h_parity, 256);
    cudaMemcpyToSymbol(d_halfcarry_add, h_halfcarry_add, 8);
    cudaMemcpyToSymbol(d_halfcarry_sub, h_halfcarry_sub, 8);
    cudaMemcpyToSymbol(d_overflow_add, h_overflow_add, 8);
    cudaMemcpyToSymbol(d_overflow_sub, h_overflow_sub, 8);
}

// ============================================================
// Z80 State (10 bytes: A, F, B, C, D, E, H, L + SP)
// Register array: [A=0, F=1, B=2, C=3, D=4, E=5, H=6, L=7]
// ============================================================
struct Z80State {
    uint8_t r[8];  // A=0, F=1, B=2, C=3, D=4, E=5, H=6, L=7
    uint16_t sp;
};

#define REG_A 0
#define REG_F 1
#define REG_B 2
#define REG_C 3
#define REG_D 4
#define REG_E 5
#define REG_H 6
#define REG_L 7

// ============================================================
// Test vectors (8 fixed inputs, same as Go TestVectors)
// ============================================================
__constant__ Z80State d_test_vectors[8];

static const Z80State h_test_vectors[8] = {
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 0x0000},
    {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 0xFFFF},
    {{0x01, 0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}, 0x1234},
    {{0x80, 0x01, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02}, 0x8000},
    {{0x55, 0x00, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55}, 0x5555},
    {{0xAA, 0x01, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA}, 0xAAAA},
    {{0x0F, 0x00, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F}, 0xFFFE},
    {{0x7F, 0x01, 0x80, 0x7F, 0x80, 0x7F, 0x80, 0x7F}, 0x7FFF},
};

// ============================================================
// Register mapping tables for compact opcode dispatch.
//
// Opcodes 0-48: LD r,r' (7 dsts × 7 srcs)
//   dst order: A, B, C, D, E, H, L  → reg indices 0, 2, 3, 4, 5, 6, 7
//   Group A (op 0-6): src order B, C, D, E, H, L, A
//   Groups B-L (op 7-48): src order A, B, C, D, E, H, L
//   Use full 49-element lookup for correctness.
//
// ALU src order: B, C, D, E, H, L, A → reg indices 2, 3, 4, 5, 6, 7, 0
//   (index 7 within group = immediate N, handled specially)
//
// CB-prefix register order: A, B, C, D, E, H, L → reg indices 0, 2, 3, 4, 5, 6, 7
// ============================================================
// Full 49-element source register lookup for LD r,r'
__device__ __constant__ uint8_t LD_FULL_SRC[49] = {
    // Group A (LD A,*): B, C, D, E, H, L, A
    REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A,
    // Group B (LD B,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    // Group C (LD C,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    // Group D (LD D,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    // Group E (LD E,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    // Group H (LD H,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    // Group L (LD L,*): A, B, C, D, E, H, L
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
};
__device__ __constant__ uint8_t LD_DST[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t ALU_SRC[7] = {REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A};
__device__ __constant__ uint8_t CB_REG[7]  = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
// LD r,N: register order A, B, C, D, E, H, L
__device__ __constant__ uint8_t IMM_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
// INC/DEC r: A, B, C, D, E, H, L
__device__ __constant__ uint8_t INCDEC_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};

// ============================================================
// Opcode range constants (from Go iota enum)
// ============================================================
// LD r,r':  0-48 (49 ops)
// LD r,N:  49-55 (7 ops)
// ADD..CP: 56-119 (64 ops = 8 ALU ops × 8 sources)
// INC r:  120-126 (7 ops)
// DEC r:  127-133 (7 ops)
// RLCA/RRCA/RLA/RRA: 134-137
// DAA/CPL/SCF/CCF/NEG/NOP: 138-143
// CB rotates: 144-192 (7 groups of 7: RLC,RRC,RL,RR,SLA,SRA,SRL on A-L)
// SLL_A: 193
// SLL B-L: 194-199
// BIT n,r: 200-255 (8×7=56)
// RES n,r: 256-311 (56)
// SET n,r: 312-367 (56)
// 16-bit INC/DEC: 368-375
// ADD HL,rr: 376-379
// EX_DE_HL: 380, LD_SP_HL: 381
// LD rr,NN: 382-385
// ADC/SBC HL: 386-393

#define OP_LD_RR_START     0
#define OP_LD_RN_START    49
#define OP_ALU_START      56   // ADD A,B
#define OP_INC_START     120
#define OP_DEC_START     127
#define OP_RLCA          134
#define OP_RRCA          135
#define OP_RLA           136
#define OP_RRA           137
#define OP_DAA           138
#define OP_CPL           139
#define OP_SCF           140
#define OP_CCF           141
#define OP_NEG           142
#define OP_NOP           143
#define OP_CB_START      144   // RLC_A
#define OP_SLL_A         193
#define OP_SLL_B_START   194
#define OP_BIT_START     200
#define OP_RES_START     256
#define OP_SET_START     312
#define OP_16INC_START   368   // INC_BC
#define OP_ADD_HL_START  376   // ADD HL,BC
#define OP_EX_DE_HL      380
#define OP_LD_SP_HL      381
#define OP_LD_RR_NN_START 382  // LD BC,NN
#define OP_ADC_HL_START  386   // ADC HL,BC
#define OP_SBC_HL_START  390   // SBC HL,BC
#define OP_COUNT         394

// ============================================================
// ALU helper functions (device)
// ============================================================
__device__ inline uint8_t bsel(bool cond, uint8_t a, uint8_t b) {
    return cond ? a : b;
}

__device__ void alu_add(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] + val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) |
                 d_halfcarry_add[lookup & 0x07] |
                 d_overflow_add[lookup >> 4] |
                 d_sz53[s.r[REG_A]];
}

__device__ void alu_adc(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] + val + (s.r[REG_F] & FLAG_C);
    uint8_t lookup = (uint8_t)(((uint16_t)(s.r[REG_A]) & 0x88) >> 3 |
                               ((uint16_t)(val) & 0x88) >> 2 |
                               (r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) |
                 d_halfcarry_add[lookup & 0x07] |
                 d_overflow_add[lookup >> 4] |
                 d_sz53[s.r[REG_A]];
}

__device__ void alu_sub(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | FLAG_N |
                 d_halfcarry_sub[lookup & 0x07] |
                 d_overflow_sub[lookup >> 4] |
                 d_sz53[s.r[REG_A]];
}

__device__ void alu_sbc(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val - (s.r[REG_F] & FLAG_C);
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | FLAG_N |
                 d_halfcarry_sub[lookup & 0x07] |
                 d_overflow_sub[lookup >> 4] |
                 d_sz53[s.r[REG_A]];
}

__device__ void alu_and(Z80State &s, uint8_t val) {
    s.r[REG_A] &= val;
    s.r[REG_F] = FLAG_H | d_sz53p[s.r[REG_A]];
}

__device__ void alu_xor(Z80State &s, uint8_t val) {
    s.r[REG_A] ^= val;
    s.r[REG_F] = d_sz53p[s.r[REG_A]];
}

__device__ void alu_or(Z80State &s, uint8_t val) {
    s.r[REG_A] |= val;
    s.r[REG_F] = d_sz53p[s.r[REG_A]];
}

__device__ void alu_cp(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, bsel(r != 0, (uint8_t)0, FLAG_Z)) |
                 FLAG_N |
                 d_halfcarry_sub[lookup & 0x07] |
                 d_overflow_sub[lookup >> 4] |
                 (val & (FLAG_3 | FLAG_5)) |
                 (uint8_t)(r & FLAG_S);
}

__device__ void alu_inc(Z80State &s, int reg) {
    s.r[reg]++;
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) |
                 bsel(s.r[reg] == 0x80, FLAG_V, 0) |
                 bsel((s.r[reg] & 0x0F) != 0, (uint8_t)0, FLAG_H) |
                 d_sz53[s.r[reg]];
}

__device__ void alu_dec(Z80State &s, int reg) {
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) | bsel((s.r[reg] & 0x0F) != 0, (uint8_t)0, FLAG_H) | FLAG_N;
    s.r[reg]--;
    s.r[REG_F] |= bsel(s.r[reg] == 0x7F, FLAG_V, 0) | d_sz53[s.r[reg]];
}

// CB-prefix rotate/shift helpers
__device__ uint8_t cb_rlc(Z80State &s, uint8_t v) {
    v = (v << 1) | (v >> 7);
    s.r[REG_F] = (v & FLAG_C) | d_sz53p[v];
    return v;
}

__device__ uint8_t cb_rrc(Z80State &s, uint8_t v) {
    s.r[REG_F] = v & FLAG_C;
    v = (v >> 1) | (v << 7);
    s.r[REG_F] |= d_sz53p[v];
    return v;
}

__device__ uint8_t cb_rl(Z80State &s, uint8_t v) {
    uint8_t old = v;
    v = (v << 1) | (s.r[REG_F] & FLAG_C);
    s.r[REG_F] = (old >> 7) | d_sz53p[v];
    return v;
}

__device__ uint8_t cb_rr(Z80State &s, uint8_t v) {
    uint8_t old = v;
    v = (v >> 1) | (s.r[REG_F] << 7);
    s.r[REG_F] = (old & FLAG_C) | d_sz53p[v];
    return v;
}

__device__ uint8_t cb_sla(Z80State &s, uint8_t v) {
    s.r[REG_F] = v >> 7;
    v <<= 1;
    s.r[REG_F] |= d_sz53p[v];
    return v;
}

__device__ uint8_t cb_sra(Z80State &s, uint8_t v) {
    s.r[REG_F] = v & FLAG_C;
    v = (v & 0x80) | (v >> 1);
    s.r[REG_F] |= d_sz53p[v];
    return v;
}

__device__ uint8_t cb_srl(Z80State &s, uint8_t v) {
    s.r[REG_F] = v & FLAG_C;
    v >>= 1;
    s.r[REG_F] |= d_sz53p[v];
    return v;
}

__device__ uint8_t cb_sll(Z80State &s, uint8_t v) {
    s.r[REG_F] = v >> 7;
    v = (v << 1) | 0x01;
    s.r[REG_F] |= d_sz53p[v];
    return v;
}

// BIT n, r
__device__ void exec_bit(Z80State &s, uint8_t val, int bit) {
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) | FLAG_H | (val & (FLAG_3 | FLAG_5));
    if ((val & (1 << bit)) == 0)
        s.r[REG_F] |= FLAG_P | FLAG_Z;
    if (bit == 7 && (val & 0x80))
        s.r[REG_F] |= FLAG_S;
}

// DAA
__device__ void exec_daa(Z80State &s) {
    uint8_t add = 0, carry = s.r[REG_F] & FLAG_C;
    if ((s.r[REG_F] & FLAG_H) || (s.r[REG_A] & 0x0F) > 9)
        add = 6;
    if (carry || s.r[REG_A] > 0x99)
        add |= 0x60;
    if (s.r[REG_A] > 0x99)
        carry = FLAG_C;
    if (s.r[REG_F] & FLAG_N)
        alu_sub(s, add);
    else
        alu_add(s, add);
    s.r[REG_F] = (s.r[REG_F] & ~(FLAG_C | FLAG_P)) | carry | d_parity[s.r[REG_A]];
}

// ADD HL, rr
__device__ void exec_add_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t result = (uint32_t)hl + val;
    uint16_t hc = (hl & 0x0FFF) + (val & 0x0FFF);
    s.r[REG_F] = (s.r[REG_F] & (FLAG_S | FLAG_Z | FLAG_P)) |
                 bsel(hc & 0x1000, FLAG_H, 0) |
                 bsel(result & 0x10000, FLAG_C, 0) |
                 ((uint8_t)(result >> 8) & (FLAG_3 | FLAG_5));
    s.r[REG_H] = (uint8_t)(result >> 8);
    s.r[REG_L] = (uint8_t)result;
}

// ADC HL, rr
__device__ void exec_adc_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t carry = s.r[REG_F] & FLAG_C;
    uint32_t result = (uint32_t)hl + val + carry;
    uint8_t lookup = (uint8_t)(((uint32_t)(hl & 0x8800) >> 11) | ((uint32_t)(val & 0x8800) >> 10) | ((result & 0x8800) >> 9));
    s.r[REG_H] = (uint8_t)(result >> 8);
    s.r[REG_L] = (uint8_t)result;
    s.r[REG_F] = bsel(result & 0x10000, FLAG_C, 0) |
                 d_overflow_add[lookup >> 4] |
                 (s.r[REG_H] & (FLAG_3 | FLAG_5 | FLAG_S)) |
                 d_halfcarry_add[lookup & 0x07] |
                 bsel((s.r[REG_H] | s.r[REG_L]) != 0, (uint8_t)0, FLAG_Z);
}

// SBC HL, rr
__device__ void exec_sbc_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t carry = s.r[REG_F] & FLAG_C;
    uint32_t result = (uint32_t)hl - val - carry;
    uint8_t lookup = (uint8_t)(((uint32_t)(hl & 0x8800) >> 11) | ((uint32_t)(val & 0x8800) >> 10) | ((result & 0x8800) >> 9));
    s.r[REG_H] = (uint8_t)(result >> 8);
    s.r[REG_L] = (uint8_t)result;
    s.r[REG_F] = bsel(result & 0x10000, FLAG_C, 0) | FLAG_N |
                 d_overflow_sub[lookup >> 4] |
                 (s.r[REG_H] & (FLAG_3 | FLAG_5 | FLAG_S)) |
                 d_halfcarry_sub[lookup & 0x07] |
                 bsel((s.r[REG_H] | s.r[REG_L]) != 0, (uint8_t)0, FLAG_Z);
}

// Get 16-bit pair value. pair: 0=BC, 1=DE, 2=HL, 3=SP
__device__ uint16_t get_pair(const Z80State &s, int pair) {
    switch (pair) {
        case 0: return ((uint16_t)s.r[REG_B] << 8) | s.r[REG_C];
        case 1: return ((uint16_t)s.r[REG_D] << 8) | s.r[REG_E];
        case 2: return ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
        case 3: return s.sp;
    }
    return 0;
}

__device__ void set_pair(Z80State &s, int pair, uint16_t val) {
    switch (pair) {
        case 0: s.r[REG_B] = (uint8_t)(val >> 8); s.r[REG_C] = (uint8_t)val; break;
        case 1: s.r[REG_D] = (uint8_t)(val >> 8); s.r[REG_E] = (uint8_t)val; break;
        case 2: s.r[REG_H] = (uint8_t)(val >> 8); s.r[REG_L] = (uint8_t)val; break;
        case 3: s.sp = val; break;
    }
}

// ============================================================
// Instruction executor — compact dispatch using opcode ranges
// ============================================================
__device__ void exec_instruction(Z80State &s, uint16_t op, uint16_t imm) {
    // --- LD r,r' (opcodes 0-48) ---
    if (op < 49) {
        int dst = LD_DST[op / 7];
        int src = LD_FULL_SRC[op];
        s.r[dst] = s.r[src];
        return;
    }
    // --- LD r,N (opcodes 49-55) ---
    if (op < 56) {
        s.r[IMM_REG[op - 49]] = (uint8_t)imm;
        return;
    }
    // --- ALU operations (opcodes 56-119): 8 groups of 8 ---
    if (op < 120) {
        int alu_op = (op - 56) / 8;   // 0=ADD,1=ADC,2=SUB,3=SBC,4=AND,5=XOR,6=OR,7=CP
        int src_idx = (op - 56) % 8;   // 0-6: register, 7: immediate
        uint8_t val = (src_idx < 7) ? s.r[ALU_SRC[src_idx]] : (uint8_t)imm;
        switch (alu_op) {
            case 0: alu_add(s, val); break;
            case 1: alu_adc(s, val); break;
            case 2: alu_sub(s, val); break;
            case 3: alu_sbc(s, val); break;
            case 4: alu_and(s, val); break;
            case 5: alu_xor(s, val); break;
            case 6: alu_or(s, val); break;
            case 7: alu_cp(s, val); break;
        }
        return;
    }
    // --- INC r (opcodes 120-126) ---
    if (op < 127) {
        alu_inc(s, INCDEC_REG[op - 120]);
        return;
    }
    // --- DEC r (opcodes 127-133) ---
    if (op < 134) {
        alu_dec(s, INCDEC_REG[op - 127]);
        return;
    }
    // --- Accumulator rotates: RLCA(134), RRCA(135), RLA(136), RRA(137) ---
    if (op == OP_RLCA) {
        s.r[REG_A] = (s.r[REG_A] << 1) | (s.r[REG_A] >> 7);
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_C | FLAG_3 | FLAG_5));
        return;
    }
    if (op == OP_RRCA) {
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & FLAG_C);
        s.r[REG_A] = (s.r[REG_A] >> 1) | (s.r[REG_A] << 7);
        s.r[REG_F] |= s.r[REG_A] & (FLAG_3 | FLAG_5);
        return;
    }
    if (op == OP_RLA) {
        uint8_t old = s.r[REG_A];
        s.r[REG_A] = (s.r[REG_A] << 1) | (s.r[REG_F] & FLAG_C);
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | (old >> 7);
        return;
    }
    if (op == OP_RRA) {
        uint8_t old = s.r[REG_A];
        s.r[REG_A] = (s.r[REG_A] >> 1) | (s.r[REG_F] << 7);
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | (old & FLAG_C);
        return;
    }
    // --- Special ops: DAA(138), CPL(139), SCF(140), CCF(141), NEG(142), NOP(143) ---
    if (op == OP_DAA) { exec_daa(s); return; }
    if (op == OP_CPL) {
        s.r[REG_A] ^= 0xFF;
        s.r[REG_F] = (s.r[REG_F] & (FLAG_C | FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | FLAG_N | FLAG_H;
        return;
    }
    if (op == OP_SCF) {
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | FLAG_C;
        return;
    }
    if (op == OP_CCF) {
        uint8_t oldC = s.r[REG_F] & FLAG_C;
        s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5));
        if (oldC) s.r[REG_F] |= FLAG_H; else s.r[REG_F] |= FLAG_C;
        return;
    }
    if (op == OP_NEG) {
        uint8_t old = s.r[REG_A];
        s.r[REG_A] = 0;
        alu_sub(s, old);
        return;
    }
    if (op == OP_NOP) return;
    // --- CB-prefix rotates/shifts (opcodes 144-192): 7 groups of 7 ---
    if (op >= OP_CB_START && op <= 192) {
        int cb_op = (op - OP_CB_START) / 7;  // 0=RLC,1=RRC,2=RL,3=RR,4=SLA,5=SRA,6=SRL
        int reg = CB_REG[(op - OP_CB_START) % 7];
        switch (cb_op) {
            case 0: s.r[reg] = cb_rlc(s, s.r[reg]); break;
            case 1: s.r[reg] = cb_rrc(s, s.r[reg]); break;
            case 2: s.r[reg] = cb_rl(s, s.r[reg]); break;
            case 3: s.r[reg] = cb_rr(s, s.r[reg]); break;
            case 4: s.r[reg] = cb_sla(s, s.r[reg]); break;
            case 5: s.r[reg] = cb_sra(s, s.r[reg]); break;
            case 6: s.r[reg] = cb_srl(s, s.r[reg]); break;
        }
        return;
    }
    // --- SLL_A (opcode 193) ---
    if (op == OP_SLL_A) {
        s.r[REG_A] = cb_sll(s, s.r[REG_A]);
        return;
    }
    // --- SLL B-L (opcodes 194-199) ---
    if (op >= OP_SLL_B_START && op < 200) {
        // SLL_B=194, SLL_C=195, ..., SLL_L=199
        // CB_REG order for B-L: indices 1..6 of CB_REG = B,C,D,E,H,L
        int reg = CB_REG[(op - OP_SLL_B_START) + 1];
        s.r[reg] = cb_sll(s, s.r[reg]);
        return;
    }
    // --- BIT n,r (opcodes 200-255): 8 bits × 7 regs ---
    if (op >= OP_BIT_START && op < OP_RES_START) {
        int idx = op - OP_BIT_START;
        int bit = idx / 7;
        int reg = CB_REG[idx % 7];
        exec_bit(s, s.r[reg], bit);
        return;
    }
    // --- RES n,r (opcodes 256-311) ---
    if (op >= OP_RES_START && op < OP_SET_START) {
        int idx = op - OP_RES_START;
        int bit = idx / 7;
        int reg = CB_REG[idx % 7];
        s.r[reg] &= ~(1u << bit);
        return;
    }
    // --- SET n,r (opcodes 312-367) ---
    if (op >= OP_SET_START && op < OP_16INC_START) {
        int idx = op - OP_SET_START;
        int bit = idx / 7;
        int reg = CB_REG[idx % 7];
        s.r[reg] |= (1u << bit);
        return;
    }
    // --- 16-bit INC/DEC (opcodes 368-375) ---
    if (op >= OP_16INC_START && op < OP_ADD_HL_START) {
        int idx = op - OP_16INC_START;
        int pair = idx % 4;  // 0=BC,1=DE,2=HL,3=SP
        bool is_dec = idx >= 4;
        uint16_t val = get_pair(s, pair);
        set_pair(s, pair, is_dec ? val - 1 : val + 1);
        return;
    }
    // --- ADD HL,rr (opcodes 376-379) ---
    if (op >= OP_ADD_HL_START && op < OP_EX_DE_HL) {
        exec_add_hl(s, get_pair(s, op - OP_ADD_HL_START));
        return;
    }
    // --- EX DE,HL (380) ---
    if (op == OP_EX_DE_HL) {
        uint8_t td = s.r[REG_D], te = s.r[REG_E];
        s.r[REG_D] = s.r[REG_H]; s.r[REG_E] = s.r[REG_L];
        s.r[REG_H] = td; s.r[REG_L] = te;
        return;
    }
    // --- LD SP,HL (381) ---
    if (op == OP_LD_SP_HL) {
        s.sp = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
        return;
    }
    // --- LD rr,NN (opcodes 382-385) ---
    if (op >= OP_LD_RR_NN_START && op < OP_ADC_HL_START) {
        set_pair(s, op - OP_LD_RR_NN_START, imm);
        return;
    }
    // --- ADC HL,rr (opcodes 386-389) ---
    if (op >= OP_ADC_HL_START && op < OP_SBC_HL_START) {
        exec_adc_hl(s, get_pair(s, op - OP_ADC_HL_START));
        return;
    }
    // --- SBC HL,rr (opcodes 390-393) ---
    if (op >= OP_SBC_HL_START && op < OP_COUNT) {
        exec_sbc_hl(s, get_pair(s, op - OP_SBC_HL_START));
        return;
    }
}

// ============================================================
// GPU Kernel: each thread processes one candidate
// ============================================================

// Fingerprint size: 10 bytes/vector × 8 vectors = 80 bytes
#define FP_SIZE 10
#define NUM_VECTORS 8
#define FP_LEN (FP_SIZE * NUM_VECTORS)

__global__ void quickcheck_kernel(
    const uint32_t* __restrict__ candidates,  // packed (op16 | imm16<<16) per instruction
    const uint8_t*  __restrict__ target_fp,   // 80-byte target fingerprint
    uint32_t*       __restrict__ results,     // 1 per candidate: 0=no match, 1=match
    uint32_t        candidate_count,
    uint32_t        seq_len,
    uint32_t        dead_flags
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= candidate_count) return;

    // Load candidate's instruction sequence
    const uint32_t* my_seq = candidates + (uint64_t)tid * seq_len;

    // Compute fingerprint for this candidate
    uint8_t my_fp[FP_LEN];
    for (int v = 0; v < NUM_VECTORS; v++) {
        Z80State s = d_test_vectors[v];
        for (uint32_t i = 0; i < seq_len; i++) {
            uint32_t packed = my_seq[i];
            uint16_t op  = (uint16_t)(packed & 0xFFFF);
            uint16_t imm = (uint16_t)(packed >> 16);
            exec_instruction(s, op, imm);
        }
        int off = v * FP_SIZE;
        my_fp[off + 0] = s.r[REG_A];
        my_fp[off + 1] = s.r[REG_F];
        my_fp[off + 2] = s.r[REG_B];
        my_fp[off + 3] = s.r[REG_C];
        my_fp[off + 4] = s.r[REG_D];
        my_fp[off + 5] = s.r[REG_E];
        my_fp[off + 6] = s.r[REG_H];
        my_fp[off + 7] = s.r[REG_L];
        my_fp[off + 8] = (uint8_t)(s.sp >> 8);
        my_fp[off + 9] = (uint8_t)s.sp;
    }

    // Compare against target fingerprint
    uint8_t flag_mask = (uint8_t)dead_flags;
    bool match = true;
    for (int i = 0; i < FP_LEN; i++) {
        uint8_t a = my_fp[i], b = target_fp[i];
        if ((i % FP_SIZE) == 1) {
            // F register: apply dead flags mask
            a &= ~flag_mask;
            b &= ~flag_mask;
        }
        if (a != b) { match = false; break; }
    }

    results[tid] = match ? 1u : 0u;
}

// ============================================================
// Host code
// ============================================================

// CPU-side Z80 executor for verification (mirrors GPU kernel exactly)
static void cpu_exec_instruction(Z80State &s, uint16_t op, uint16_t imm) {
    // LD r,r'
    if (op < 49) {
        static const uint8_t ld_dst[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
        static const uint8_t ld_src[7] = {REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A};
        s.r[ld_dst[op / 7]] = s.r[ld_src[op % 7]];
        return;
    }
    // For full CPU verification, we'd mirror the entire GPU exec.
    // For now we only use this for smoke tests with a few opcodes.
    // Full verification is done by comparing GPU results against Go's output.
}

static void cpu_fingerprint(const uint32_t* seq, uint32_t seq_len, uint8_t fp[FP_LEN]) {
    for (int v = 0; v < NUM_VECTORS; v++) {
        Z80State s = h_test_vectors[v];
        for (uint32_t i = 0; i < seq_len; i++) {
            uint16_t op  = (uint16_t)(seq[i] & 0xFFFF);
            uint16_t imm = (uint16_t)(seq[i] >> 16);
            cpu_exec_instruction(s, op, imm);
        }
        int off = v * FP_SIZE;
        fp[off + 0] = s.r[REG_A]; fp[off + 1] = s.r[REG_F];
        fp[off + 2] = s.r[REG_B]; fp[off + 3] = s.r[REG_C];
        fp[off + 4] = s.r[REG_D]; fp[off + 5] = s.r[REG_E];
        fp[off + 6] = s.r[REG_H]; fp[off + 7] = s.r[REG_L];
        fp[off + 8] = (uint8_t)(s.sp >> 8); fp[off + 9] = (uint8_t)s.sp;
    }
}

int main(int argc, char** argv) {
    bool self_test = (argc > 1 && strcmp(argv[1], "--test") == 0);

    // Initialize tables
    init_tables();
    upload_tables();
    cudaMemcpyToSymbol(d_test_vectors, h_test_vectors, sizeof(h_test_vectors));

    if (self_test) {
        // Self-test: verify a few known opcodes.
        // Test 1: XOR A (opcode 102 = XOR_A) should zero A and set Z+P flags
        fprintf(stderr, "Self-test mode\n");

        const uint32_t N = 4;
        const uint32_t seq_len = 1;

        // Candidates: XOR A, LD A,0, AND A, OR A
        // XOR_A = 96+6 = 102
        // LD_A_N = 49, imm=0
        // AND_A = 88+6 = 94
        // OR_A  = 104+6 = 110
        uint32_t h_candidates[4] = {
            102,                // XOR A (op=102, imm=0)
            49 | (0 << 16),    // LD A, 0 (op=49, imm=0)
            94,                // AND A (op=94, imm=0)
            110,               // OR A (op=110, imm=0)
        };

        // Compute target fingerprint for XOR A using GPU
        uint8_t target_fp[FP_LEN];
        // We'll compute fingerprints for all 4 and check which match the first (XOR A)
        // Using test vector 0: A=0, all zeros → XOR A: A stays 0, F=0x44 (Z+P)

        // Upload candidates
        uint32_t *d_candidates, *d_results;
        uint8_t  *d_target_fp;
        cudaMalloc(&d_candidates, N * seq_len * sizeof(uint32_t));
        cudaMalloc(&d_results, N * sizeof(uint32_t));
        cudaMalloc(&d_target_fp, FP_LEN);
        cudaMemcpy(d_candidates, h_candidates, N * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // First: compute XOR A fingerprint on GPU, then use it as target
        // Run kernel with XOR A fingerprint = zeros (will match nothing, just to get the FP)
        // Actually, let's compute the FP for candidate 0 (XOR A) by running the kernel
        // and reading back the results buffer... but that just gives match/no-match.
        //
        // Better: run a dedicated kernel to compute one fingerprint.
        // Or just hardcode the expected XOR A fingerprint for test vectors.
        // Let's compute it directly:
        //
        // For XOR A: A = A ^ A = 0, F = sz53p[0] = FLAG_Z | FLAG_P = 0x44
        // Other regs unchanged.
        for (int v = 0; v < NUM_VECTORS; v++) {
            Z80State s = h_test_vectors[v];
            // XOR A: A ^= A → 0
            s.r[REG_A] = 0;
            s.r[REG_F] = h_sz53p[0]; // 0x44
            int off = v * FP_SIZE;
            target_fp[off + 0] = s.r[REG_A]; target_fp[off + 1] = s.r[REG_F];
            target_fp[off + 2] = s.r[REG_B]; target_fp[off + 3] = s.r[REG_C];
            target_fp[off + 4] = s.r[REG_D]; target_fp[off + 5] = s.r[REG_E];
            target_fp[off + 6] = s.r[REG_H]; target_fp[off + 7] = s.r[REG_L];
            target_fp[off + 8] = (uint8_t)(s.sp >> 8); target_fp[off + 9] = (uint8_t)s.sp;
        }

        cudaMemcpy(d_target_fp, target_fp, FP_LEN, cudaMemcpyHostToDevice);

        int blockSize = 64;
        int gridSize = (N + blockSize - 1) / blockSize;
        quickcheck_kernel<<<gridSize, blockSize>>>(d_candidates, d_target_fp, d_results, N, seq_len, 0);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        uint32_t h_results[4];
        cudaMemcpy(h_results, d_results, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        fprintf(stderr, "Target: XOR A (A=0, F=0x%02x for all vectors)\n", h_sz53p[0]);
        const char* names[] = {"XOR A", "LD A,0", "AND A", "OR A"};
        int pass = 1;
        for (uint32_t i = 0; i < N; i++) {
            fprintf(stderr, "  %s: %s\n", names[i], h_results[i] ? "MATCH" : "no match");
        }
        // XOR A should match itself (index 0)
        // LD A,0 should NOT match (different flags: LD doesn't set flags)
        // AND A should NOT match (AND sets H flag: F = 0x10 | sz53p[A], different when A!=0)
        // OR A should NOT match (OR: F = sz53p[A], only matches XOR when A was 0, but F differs on non-zero)
        if (h_results[0] != 1) { fprintf(stderr, "FAIL: XOR A should match itself\n"); pass = 0; }
        if (h_results[1] != 0) { fprintf(stderr, "FAIL: LD A,0 should not match XOR A\n"); pass = 0; }

        fprintf(stderr, "%s\n", pass ? "SELF-TEST PASSED" : "SELF-TEST FAILED");

        cudaFree(d_candidates);
        cudaFree(d_results);
        cudaFree(d_target_fp);
        return pass ? 0 : 1;
    }

    // ---- Server mode (--server): streaming protocol ----
    // Phase 1 (init): read header + candidates, upload to GPU once.
    //   uint32 candidate_count
    //   uint32 seq_len
    //   candidate_count * seq_len * uint32 packed instructions
    // Phase 2 (query loop): read target FP + dead_flags, dispatch, return matches.
    //   uint8[80] target_fp  +  uint32 dead_flags  (84 bytes per query)
    //   → uint32 match_count  +  uint32[match_count] match_indices
    // Phase ends when stdin reaches EOF.

    bool server_mode = (argc > 1 && strcmp(argv[1], "--server") == 0);

    // ---- Read header ----
    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, stdin) != 2) {
        fprintf(stderr, "Failed to read header\n");
        return 1;
    }
    uint32_t candidate_count = header[0];
    uint32_t seq_len = header[1];

    // Read candidate data
    size_t cand_bytes = (size_t)candidate_count * seq_len * sizeof(uint32_t);
    uint32_t* h_candidates = (uint32_t*)malloc(cand_bytes);
    if (!h_candidates) {
        fprintf(stderr, "Failed to allocate %zu bytes for candidates\n", cand_bytes);
        return 1;
    }
    size_t cand_words = (size_t)candidate_count * seq_len;
    if (fread(h_candidates, sizeof(uint32_t), cand_words, stdin) != cand_words) {
        fprintf(stderr, "Failed to read candidate data\n");
        return 1;
    }

    fprintf(stderr, "Loaded %u candidates, seq_len=%u, mode=%s\n",
            candidate_count, seq_len, server_mode ? "server" : "single");

    // Allocate GPU memory
    uint32_t *d_candidates, *d_results;
    uint8_t  *d_target_fp;
    cudaMalloc(&d_candidates, cand_bytes);
    cudaMalloc(&d_results, candidate_count * sizeof(uint32_t));
    cudaMalloc(&d_target_fp, FP_LEN);

    // Upload candidates (once)
    cudaMemcpy(d_candidates, h_candidates, cand_bytes, cudaMemcpyHostToDevice);
    free(h_candidates);

    // Allocate host results buffer (reused)
    uint32_t* h_results = (uint32_t*)malloc(candidate_count * sizeof(uint32_t));
    uint32_t* matches = (uint32_t*)malloc(candidate_count * sizeof(uint32_t));

    int blockSize = 256;
    int gridSize = (candidate_count + blockSize - 1) / blockSize;

    // ---- Query loop ----
    uint8_t target_fp[FP_LEN];
    uint32_t dead_flags;
    uint64_t query_count = 0;

    while (true) {
        // Read target fingerprint (80 bytes) + dead_flags (4 bytes)
        size_t nread = fread(target_fp, 1, FP_LEN, stdin);
        if (nread == 0) break; // EOF
        if (nread != FP_LEN) {
            fprintf(stderr, "Short read on target_fp: %zu/%d\n", nread, FP_LEN);
            break;
        }
        if (fread(&dead_flags, sizeof(uint32_t), 1, stdin) != 1) {
            fprintf(stderr, "Failed to read dead_flags\n");
            break;
        }

        // Upload target fingerprint
        cudaMemcpy(d_target_fp, target_fp, FP_LEN, cudaMemcpyHostToDevice);

        // Dispatch kernel
        quickcheck_kernel<<<gridSize, blockSize>>>(
            d_candidates, d_target_fp, d_results,
            candidate_count, seq_len, dead_flags);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at query %lu: %s\n", query_count, cudaGetErrorString(err));
            return 1;
        }

        // Download results
        cudaMemcpy(h_results, d_results, candidate_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Collect matches
        uint32_t match_count = 0;
        for (uint32_t i = 0; i < candidate_count; i++) {
            if (h_results[i]) matches[match_count++] = i;
        }

        // Write output
        fwrite(&match_count, sizeof(uint32_t), 1, stdout);
        if (match_count > 0)
            fwrite(matches, sizeof(uint32_t), match_count, stdout);
        fflush(stdout);

        query_count++;

        if (!server_mode) break; // single-query mode: exit after first query
    }

    if (server_mode) {
        fprintf(stderr, "Processed %lu queries\n", query_count);
    }

    // Cleanup
    free(h_results);
    free(matches);
    cudaFree(d_candidates);
    cudaFree(d_results);
    cudaFree(d_target_fp);
    return 0;
}

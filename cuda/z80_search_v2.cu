// Z80 Standalone GPU Superoptimizer Search — v2 Batched Pipeline
//
// 3-stage GPU pipeline:
//   Stage 1: Batched QuickCheck (512 targets x N candidates, 8 test vectors)
//   Stage 2: MidCheck (survivors only, 24 additional test vectors)
//   Stage 3: GPU ExhaustiveCheck (256 threads/block, full input sweep)
//
// Build: nvcc -O2 -o z80search_v2 z80_search_v2.cu
// Usage: ./z80search_v2 --max-target 2 [--dead-flags 0x28] [--gpu-id N]
//                       [--first-op-start M] [--first-op-end N]
//
// Output: JSONL to stdout (one result per line)
// Progress: stderr

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

#include "z80_common.h"

// ============================================================
// Pipeline tuning constants
// ============================================================
#define BATCH_SIZE     512    // targets per batch
#define BLOCK_SIZE     256    // threads per CUDA block
#define EXHAUST_BLOCK  256    // threads per ExhaustiveCheck block (one per A value)

// Bitmap: ceil(max_candidates / 32) words per target
// max_candidates ~ 4215 for 8-bit instructions -> 132 words
#define MAX_CAND_WORDS 132

// ============================================================
// CUDA constants (device memory)
// ============================================================
__constant__ uint8_t d_sz53[256];
__constant__ uint8_t d_sz53p[256];
__constant__ uint8_t d_parity[256];
__constant__ uint8_t d_halfcarry_add[8];
__constant__ uint8_t d_halfcarry_sub[8];
__constant__ uint8_t d_overflow_add[8];
__constant__ uint8_t d_overflow_sub[8];
__constant__ Z80State d_test_vectors[8];
__constant__ Z80State d_mid_vectors[MID_VECTORS];

__device__ __constant__ uint8_t LD_FULL_SRC[49] = {
    REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
};
__device__ __constant__ uint8_t LD_DST[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t ALU_SRC[7] = {REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A};
__device__ __constant__ uint8_t CB_REG[7]  = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t IMM_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t INCDEC_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};

// Representative values for GPU exhaustive reduced sweep
__device__ __constant__ uint8_t d_rep_values[32] = {
    0x00, 0x01, 0x02, 0x0F, 0x10, 0x1F, 0x20, 0x3F,
    0x40, 0x55, 0x7E, 0x7F, 0x80, 0x81, 0xAA, 0xBF,
    0xC0, 0xD5, 0xE0, 0xEF, 0xF0, 0xF7, 0xFE, 0xFF,
    0x03, 0x07, 0x11, 0x33, 0x77, 0xBB, 0xDD, 0xEE,
};
__device__ __constant__ uint16_t d_rep_sp[16] = {
    0x0000, 0x0001, 0x00FF, 0x0100, 0x7FFE, 0x7FFF, 0x8000, 0x8001,
    0xFFFE, 0xFFFF, 0x1234, 0x5678, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE,
};

static void upload_tables_cuda() {
    cudaMemcpyToSymbol(d_sz53, h_sz53, 256);
    cudaMemcpyToSymbol(d_sz53p, h_sz53p, 256);
    cudaMemcpyToSymbol(d_parity, h_parity, 256);
    cudaMemcpyToSymbol(d_halfcarry_add, h_halfcarry_add, 8);
    cudaMemcpyToSymbol(d_halfcarry_sub, h_halfcarry_sub, 8);
    cudaMemcpyToSymbol(d_overflow_add, h_overflow_add, 8);
    cudaMemcpyToSymbol(d_overflow_sub, h_overflow_sub, 8);
    cudaMemcpyToSymbol(d_test_vectors, h_test_vectors, sizeof(h_test_vectors));
    cudaMemcpyToSymbol(d_mid_vectors, h_mid_vectors, sizeof(h_mid_vectors));
}

// ============================================================
// GPU device-side ALU helpers (verbatim from v1)
// ============================================================
__device__ inline uint8_t bsel(bool cond, uint8_t a, uint8_t b) { return cond ? a : b; }

__device__ void alu_add(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] + val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | d_halfcarry_add[lookup & 0x07] | d_overflow_add[lookup >> 4] | d_sz53[s.r[REG_A]];
}
__device__ void alu_adc(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] + val + (s.r[REG_F] & FLAG_C);
    uint8_t lookup = (uint8_t)(((uint16_t)(s.r[REG_A]) & 0x88) >> 3 | ((uint16_t)(val) & 0x88) >> 2 | (r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | d_halfcarry_add[lookup & 0x07] | d_overflow_add[lookup >> 4] | d_sz53[s.r[REG_A]];
}
__device__ void alu_sub(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | FLAG_N | d_halfcarry_sub[lookup & 0x07] | d_overflow_sub[lookup >> 4] | d_sz53[s.r[REG_A]];
}
__device__ void alu_sbc(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val - (s.r[REG_F] & FLAG_C);
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_A] = (uint8_t)r;
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, 0) | FLAG_N | d_halfcarry_sub[lookup & 0x07] | d_overflow_sub[lookup >> 4] | d_sz53[s.r[REG_A]];
}
__device__ void alu_and(Z80State &s, uint8_t val) { s.r[REG_A] &= val; s.r[REG_F] = FLAG_H | d_sz53p[s.r[REG_A]]; }
__device__ void alu_xor(Z80State &s, uint8_t val) { s.r[REG_A] ^= val; s.r[REG_F] = d_sz53p[s.r[REG_A]]; }
__device__ void alu_or(Z80State &s, uint8_t val) { s.r[REG_A] |= val; s.r[REG_F] = d_sz53p[s.r[REG_A]]; }
__device__ void alu_cp(Z80State &s, uint8_t val) {
    uint16_t r = (uint16_t)s.r[REG_A] - val;
    uint8_t lookup = ((s.r[REG_A] & 0x88) >> 3) | ((val & 0x88) >> 2) | (uint8_t)((r & 0x88) >> 1);
    s.r[REG_F] = bsel(r & 0x100, FLAG_C, bsel(r != 0, (uint8_t)0, FLAG_Z)) | FLAG_N |
                 d_halfcarry_sub[lookup & 0x07] | d_overflow_sub[lookup >> 4] | (val & (FLAG_3 | FLAG_5)) | (uint8_t)(r & FLAG_S);
}
__device__ void alu_inc(Z80State &s, int reg) {
    s.r[reg]++;
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) | bsel(s.r[reg] == 0x80, FLAG_V, 0) | bsel((s.r[reg] & 0x0F) != 0, (uint8_t)0, FLAG_H) | d_sz53[s.r[reg]];
}
__device__ void alu_dec(Z80State &s, int reg) {
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) | bsel((s.r[reg] & 0x0F) != 0, (uint8_t)0, FLAG_H) | FLAG_N;
    s.r[reg]--;
    s.r[REG_F] |= bsel(s.r[reg] == 0x7F, FLAG_V, 0) | d_sz53[s.r[reg]];
}

__device__ uint8_t cb_rlc(Z80State &s, uint8_t v) { v = (v << 1) | (v >> 7); s.r[REG_F] = (v & FLAG_C) | d_sz53p[v]; return v; }
__device__ uint8_t cb_rrc(Z80State &s, uint8_t v) { s.r[REG_F] = v & FLAG_C; v = (v >> 1) | (v << 7); s.r[REG_F] |= d_sz53p[v]; return v; }
__device__ uint8_t cb_rl(Z80State &s, uint8_t v) { uint8_t o = v; v = (v << 1) | (s.r[REG_F] & FLAG_C); s.r[REG_F] = (o >> 7) | d_sz53p[v]; return v; }
__device__ uint8_t cb_rr(Z80State &s, uint8_t v) { uint8_t o = v; v = (v >> 1) | (s.r[REG_F] << 7); s.r[REG_F] = (o & FLAG_C) | d_sz53p[v]; return v; }
__device__ uint8_t cb_sla(Z80State &s, uint8_t v) { s.r[REG_F] = v >> 7; v <<= 1; s.r[REG_F] |= d_sz53p[v]; return v; }
__device__ uint8_t cb_sra(Z80State &s, uint8_t v) { s.r[REG_F] = v & FLAG_C; v = (v & 0x80) | (v >> 1); s.r[REG_F] |= d_sz53p[v]; return v; }
__device__ uint8_t cb_srl(Z80State &s, uint8_t v) { s.r[REG_F] = v & FLAG_C; v >>= 1; s.r[REG_F] |= d_sz53p[v]; return v; }
__device__ uint8_t cb_sll(Z80State &s, uint8_t v) { s.r[REG_F] = v >> 7; v = (v << 1) | 0x01; s.r[REG_F] |= d_sz53p[v]; return v; }

__device__ void exec_bit(Z80State &s, uint8_t val, int bit) {
    s.r[REG_F] = (s.r[REG_F] & FLAG_C) | FLAG_H | (val & (FLAG_3 | FLAG_5));
    if ((val & (1 << bit)) == 0) s.r[REG_F] |= FLAG_P | FLAG_Z;
    if (bit == 7 && (val & 0x80)) s.r[REG_F] |= FLAG_S;
}

__device__ void exec_daa(Z80State &s) {
    uint8_t add = 0, carry = s.r[REG_F] & FLAG_C;
    if ((s.r[REG_F] & FLAG_H) || (s.r[REG_A] & 0x0F) > 9) add = 6;
    if (carry || s.r[REG_A] > 0x99) add |= 0x60;
    if (s.r[REG_A] > 0x99) carry = FLAG_C;
    if (s.r[REG_F] & FLAG_N) alu_sub(s, add); else alu_add(s, add);
    s.r[REG_F] = (s.r[REG_F] & ~(FLAG_C | FLAG_P)) | carry | d_parity[s.r[REG_A]];
}

__device__ void exec_add_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t result = (uint32_t)hl + val;
    uint16_t hc = (hl & 0x0FFF) + (val & 0x0FFF);
    s.r[REG_F] = (s.r[REG_F] & (FLAG_S | FLAG_Z | FLAG_P)) | bsel(hc & 0x1000, FLAG_H, 0) | bsel(result & 0x10000, FLAG_C, 0) | ((uint8_t)(result >> 8) & (FLAG_3 | FLAG_5));
    s.r[REG_H] = (uint8_t)(result >> 8); s.r[REG_L] = (uint8_t)result;
}

__device__ void exec_adc_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t carry = s.r[REG_F] & FLAG_C;
    uint32_t result = (uint32_t)hl + val + carry;
    uint8_t lookup = (uint8_t)(((uint32_t)(hl & 0x8800) >> 11) | ((uint32_t)(val & 0x8800) >> 10) | ((result & 0x8800) >> 9));
    s.r[REG_H] = (uint8_t)(result >> 8); s.r[REG_L] = (uint8_t)result;
    s.r[REG_F] = bsel(result & 0x10000, FLAG_C, 0) | d_overflow_add[lookup >> 4] | (s.r[REG_H] & (FLAG_3 | FLAG_5 | FLAG_S)) | d_halfcarry_add[lookup & 0x07] | bsel((s.r[REG_H] | s.r[REG_L]) != 0, (uint8_t)0, FLAG_Z);
}

__device__ void exec_sbc_hl(Z80State &s, uint16_t val) {
    uint16_t hl = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L];
    uint32_t carry = s.r[REG_F] & FLAG_C;
    uint32_t result = (uint32_t)hl - val - carry;
    uint8_t lookup = (uint8_t)(((uint32_t)(hl & 0x8800) >> 11) | ((uint32_t)(val & 0x8800) >> 10) | ((result & 0x8800) >> 9));
    s.r[REG_H] = (uint8_t)(result >> 8); s.r[REG_L] = (uint8_t)result;
    s.r[REG_F] = bsel(result & 0x10000, FLAG_C, 0) | FLAG_N | d_overflow_sub[lookup >> 4] | (s.r[REG_H] & (FLAG_3 | FLAG_5 | FLAG_S)) | d_halfcarry_sub[lookup & 0x07] | bsel((s.r[REG_H] | s.r[REG_L]) != 0, (uint8_t)0, FLAG_Z);
}

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

__device__ void exec_instruction(Z80State &s, uint16_t op, uint16_t imm) {
    if (op < 49) { s.r[LD_DST[op / 7]] = s.r[LD_FULL_SRC[op]]; return; }
    if (op < 56) { s.r[IMM_REG[op - 49]] = (uint8_t)imm; return; }
    if (op < 120) {
        int alu_op = (op - 56) / 8, src_idx = (op - 56) % 8;
        uint8_t val = (src_idx < 7) ? s.r[ALU_SRC[src_idx]] : (uint8_t)imm;
        switch (alu_op) {
            case 0: alu_add(s, val); break; case 1: alu_adc(s, val); break;
            case 2: alu_sub(s, val); break; case 3: alu_sbc(s, val); break;
            case 4: alu_and(s, val); break; case 5: alu_xor(s, val); break;
            case 6: alu_or(s, val); break;  case 7: alu_cp(s, val); break;
        }
        return;
    }
    if (op < 127) { alu_inc(s, INCDEC_REG[op - 120]); return; }
    if (op < 134) { alu_dec(s, INCDEC_REG[op - 127]); return; }
    if (op == OP_RLCA) { s.r[REG_A] = (s.r[REG_A] << 1) | (s.r[REG_A] >> 7); s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_C | FLAG_3 | FLAG_5)); return; }
    if (op == OP_RRCA) { s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & FLAG_C); s.r[REG_A] = (s.r[REG_A] >> 1) | (s.r[REG_A] << 7); s.r[REG_F] |= s.r[REG_A] & (FLAG_3 | FLAG_5); return; }
    if (op == OP_RLA) { uint8_t o = s.r[REG_A]; s.r[REG_A] = (s.r[REG_A] << 1) | (s.r[REG_F] & FLAG_C); s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | (o >> 7); return; }
    if (op == OP_RRA) { uint8_t o = s.r[REG_A]; s.r[REG_A] = (s.r[REG_A] >> 1) | (s.r[REG_F] << 7); s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | (o & FLAG_C); return; }
    if (op == OP_DAA) { exec_daa(s); return; }
    if (op == OP_CPL) { s.r[REG_A] ^= 0xFF; s.r[REG_F] = (s.r[REG_F] & (FLAG_C | FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | FLAG_N | FLAG_H; return; }
    if (op == OP_SCF) { s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)) | FLAG_C; return; }
    if (op == OP_CCF) { uint8_t c = s.r[REG_F] & FLAG_C; s.r[REG_F] = (s.r[REG_F] & (FLAG_P | FLAG_Z | FLAG_S)) | (s.r[REG_A] & (FLAG_3 | FLAG_5)); if (c) s.r[REG_F] |= FLAG_H; else s.r[REG_F] |= FLAG_C; return; }
    if (op == OP_NEG) { uint8_t o = s.r[REG_A]; s.r[REG_A] = 0; alu_sub(s, o); return; }
    if (op == OP_NOP) return;
    if (op >= OP_CB_START && op <= 192) { int cb_op = (op - OP_CB_START) / 7; int reg = CB_REG[(op - OP_CB_START) % 7]; switch (cb_op) { case 0: s.r[reg] = cb_rlc(s, s.r[reg]); break; case 1: s.r[reg] = cb_rrc(s, s.r[reg]); break; case 2: s.r[reg] = cb_rl(s, s.r[reg]); break; case 3: s.r[reg] = cb_rr(s, s.r[reg]); break; case 4: s.r[reg] = cb_sla(s, s.r[reg]); break; case 5: s.r[reg] = cb_sra(s, s.r[reg]); break; case 6: s.r[reg] = cb_srl(s, s.r[reg]); break; } return; }
    if (op == OP_SLL_A) { s.r[REG_A] = cb_sll(s, s.r[REG_A]); return; }
    if (op >= OP_SLL_B_START && op < 200) { int reg = CB_REG[(op - OP_SLL_B_START) + 1]; s.r[reg] = cb_sll(s, s.r[reg]); return; }
    if (op >= OP_BIT_START && op < OP_RES_START) { int idx = op - OP_BIT_START; exec_bit(s, s.r[CB_REG[idx % 7]], idx / 7); return; }
    if (op >= OP_RES_START && op < OP_SET_START) { int idx = op - OP_RES_START; s.r[CB_REG[idx % 7]] &= ~(1u << (idx / 7)); return; }
    if (op >= OP_SET_START && op < OP_16INC_START) { int idx = op - OP_SET_START; s.r[CB_REG[idx % 7]] |= (1u << (idx / 7)); return; }
    if (op >= OP_16INC_START && op < OP_ADD_HL_START) { int idx = op - OP_16INC_START; int pair = idx % 4; bool dec = idx >= 4; uint16_t v = get_pair(s, pair); set_pair(s, pair, dec ? v - 1 : v + 1); return; }
    if (op >= OP_ADD_HL_START && op < OP_EX_DE_HL) { exec_add_hl(s, get_pair(s, op - OP_ADD_HL_START)); return; }
    if (op == OP_EX_DE_HL) { uint8_t td = s.r[REG_D], te = s.r[REG_E]; s.r[REG_D] = s.r[REG_H]; s.r[REG_E] = s.r[REG_L]; s.r[REG_H] = td; s.r[REG_L] = te; return; }
    if (op == OP_LD_SP_HL) { s.sp = ((uint16_t)s.r[REG_H] << 8) | s.r[REG_L]; return; }
    if (op >= OP_LD_RR_NN_START && op < OP_ADC_HL_START) { set_pair(s, op - OP_LD_RR_NN_START, imm); return; }
    if (op >= OP_ADC_HL_START && op < OP_SBC_HL_START) { exec_adc_hl(s, get_pair(s, op - OP_ADC_HL_START)); return; }
    if (op >= OP_SBC_HL_START && op < OP_COUNT) { exec_sbc_hl(s, get_pair(s, op - OP_SBC_HL_START)); return; }
}

// Device helpers
__device__ bool d_states_equal(const Z80State &a, const Z80State &b, uint8_t dead_flags) {
    return a.r[REG_A] == b.r[REG_A] &&
           ((a.r[REG_F] & ~dead_flags) == (b.r[REG_F] & ~dead_flags)) &&
           a.r[REG_B] == b.r[REG_B] && a.r[REG_C] == b.r[REG_C] &&
           a.r[REG_D] == b.r[REG_D] && a.r[REG_E] == b.r[REG_E] &&
           a.r[REG_H] == b.r[REG_H] && a.r[REG_L] == b.r[REG_L] &&
           a.sp == b.sp;
}

__device__ void d_set_reg_by_offset(Z80State &s, int offset, uint8_t val) {
    switch (offset) {
        case 2: s.r[REG_B] = val; break; case 3: s.r[REG_C] = val; break;
        case 4: s.r[REG_D] = val; break; case 5: s.r[REG_E] = val; break;
        case 6: s.r[REG_H] = val; break; case 7: s.r[REG_L] = val; break;
    }
}

// ============================================================
// KERNEL 1: Batched QuickCheck
// One thread per (target, candidate) pair.
// Output: bitmap with atomicOr, one bit per candidate per target.
// ============================================================
__global__ void quickcheck_batched(
    const uint32_t* __restrict__ candidates,
    const uint8_t*  __restrict__ target_fps,    // batch_count * FP_LEN
    uint32_t*       __restrict__ hit_bitmap,    // batch_count * bitmap_words
    uint32_t cand_count,
    uint32_t batch_count,
    uint32_t bitmap_words,
    uint32_t dead_flags
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = batch_count * cand_count;
    if (tid >= total_threads) return;

    uint32_t target_idx = tid / cand_count;
    uint32_t cand_idx   = tid % cand_count;

    uint32_t packed = candidates[cand_idx];
    uint16_t c_op  = (uint16_t)(packed & 0xFFFF);
    uint16_t c_imm = (uint16_t)(packed >> 16);

    const uint8_t* tfp = target_fps + target_idx * FP_LEN;
    uint8_t fm = (uint8_t)dead_flags;

    for (int v = 0; v < NUM_VECTORS; v++) {
        Z80State s = d_test_vectors[v];
        exec_instruction(s, c_op, c_imm);
        int off = v * FP_SIZE;
        uint8_t cf = s.r[REG_F], tf = tfp[off+1];
        if (fm) { cf &= ~fm; tf &= ~fm; }
        if (s.r[REG_A] != tfp[off+0] || cf != tf ||
            s.r[REG_B] != tfp[off+2] || s.r[REG_C] != tfp[off+3] ||
            s.r[REG_D] != tfp[off+4] || s.r[REG_E] != tfp[off+5] ||
            s.r[REG_H] != tfp[off+6] || s.r[REG_L] != tfp[off+7] ||
            (uint8_t)(s.sp >> 8) != tfp[off+8] || (uint8_t)s.sp != tfp[off+9])
            return;
    }
    // All vectors match
    uint32_t word = cand_idx / 32;
    uint32_t bit  = cand_idx % 32;
    atomicOr(&hit_bitmap[target_idx * bitmap_words + word], 1u << bit);
}

// ============================================================
// KERNEL 2: MidCheck — 24 additional test vectors
// ============================================================
struct MidPair {
    uint16_t target_idx;
    uint16_t cand_idx;
};

__global__ void midcheck_kernel(
    const MidPair*  __restrict__ pairs,
    uint32_t pair_count,
    const uint32_t* __restrict__ candidates,
    const uint8_t*  __restrict__ target_mid_fps,
    uint32_t*       __restrict__ survived,
    uint32_t dead_flags
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    MidPair p = pairs[tid];
    uint32_t packed = candidates[p.cand_idx];
    uint16_t c_op  = (uint16_t)(packed & 0xFFFF);
    uint16_t c_imm = (uint16_t)(packed >> 16);

    const uint8_t* tmfp = target_mid_fps + p.target_idx * MID_FP_LEN;
    uint8_t fm = (uint8_t)dead_flags;

    for (int v = 0; v < MID_VECTORS; v++) {
        Z80State s = d_mid_vectors[v];
        exec_instruction(s, c_op, c_imm);
        int off = v * MID_FP_SIZE;
        uint8_t cf = s.r[REG_F], tf = tmfp[off+1];
        if (fm) { cf &= ~fm; tf &= ~fm; }
        if (s.r[REG_A] != tmfp[off+0] || cf != tf ||
            s.r[REG_B] != tmfp[off+2] || s.r[REG_C] != tmfp[off+3] ||
            s.r[REG_D] != tmfp[off+4] || s.r[REG_E] != tmfp[off+5] ||
            s.r[REG_H] != tmfp[off+6] || s.r[REG_L] != tmfp[off+7] ||
            (uint8_t)(s.sp >> 8) != tmfp[off+8] || (uint8_t)s.sp != tmfp[off+9]) {
            survived[tid] = 0;
            return;
        }
    }
    survived[tid] = 1;
}

// ============================================================
// KERNEL 3: GPU ExhaustiveCheck
// One thread-block (256 threads) per (target, candidate) pair.
// Each thread handles one value of A (0-255).
// ============================================================
struct ExhaustPair {
    uint16_t t_ops[3];
    uint16_t t_imms[3];
    uint16_t c_ops[1];
    uint16_t c_imms[1];
    uint8_t  t_len;
    uint8_t  c_len;
    uint8_t  nextra;
    uint8_t  extra[6];
    uint8_t  sweep_sp;
    uint8_t  use_full;
    uint8_t  dead_flags;
    uint8_t  _pad;
};

__global__ void exhaustive_check_gpu(
    const ExhaustPair* __restrict__ pairs,
    uint32_t* __restrict__ results
) {
    uint32_t pair_idx = blockIdx.x;
    uint32_t a_val = threadIdx.x;  // 0..255

    __shared__ uint32_t mismatch;
    if (threadIdx.x == 0) mismatch = 0;
    __syncthreads();

    ExhaustPair ep = pairs[pair_idx];
    uint8_t dfm = ep.dead_flags;

    // Macros for executing target/candidate sequences
    #define EXEC_TARGET(st) do { \
        for (int _i = 0; _i < ep.t_len; _i++) \
            exec_instruction(st, ep.t_ops[_i], ep.t_imms[_i]); \
    } while(0)

    #define EXEC_CAND(sc) exec_instruction(sc, ep.c_ops[0], ep.c_imms[0])

    if (ep.nextra == 0 && !ep.sweep_sp) {
        for (int carry = 0; carry <= 1; carry++) {
            if (mismatch) goto done;
            Z80State st = {}, sc = {};
            st.r[REG_A] = sc.r[REG_A] = (uint8_t)a_val;
            st.r[REG_F] = sc.r[REG_F] = (uint8_t)carry;
            EXEC_TARGET(st); EXEC_CAND(sc);
            if (!d_states_equal(st, sc, dfm)) { atomicOr(&mismatch, 1); goto done; }
        }
    } else if (ep.use_full && ep.nextra == 1) {
        for (int r = 0; r < 256; r++) {
            for (int carry = 0; carry <= 1; carry++) {
                if (mismatch) goto done;
                Z80State st = {}, sc = {};
                st.r[REG_A] = sc.r[REG_A] = (uint8_t)a_val;
                st.r[REG_F] = sc.r[REG_F] = (uint8_t)carry;
                d_set_reg_by_offset(st, ep.extra[0], (uint8_t)r);
                d_set_reg_by_offset(sc, ep.extra[0], (uint8_t)r);
                EXEC_TARGET(st); EXEC_CAND(sc);
                if (!d_states_equal(st, sc, dfm)) { atomicOr(&mismatch, 1); goto done; }
            }
        }
    } else if (ep.use_full && ep.nextra == 2) {
        for (int r1 = 0; r1 < 256; r1++) {
            for (int r2 = 0; r2 < 256; r2++) {
                for (int carry = 0; carry <= 1; carry++) {
                    if (mismatch) goto done;
                    Z80State st = {}, sc = {};
                    st.r[REG_A] = sc.r[REG_A] = (uint8_t)a_val;
                    st.r[REG_F] = sc.r[REG_F] = (uint8_t)carry;
                    d_set_reg_by_offset(st, ep.extra[0], (uint8_t)r1);
                    d_set_reg_by_offset(sc, ep.extra[0], (uint8_t)r1);
                    d_set_reg_by_offset(st, ep.extra[1], (uint8_t)r2);
                    d_set_reg_by_offset(sc, ep.extra[1], (uint8_t)r2);
                    EXEC_TARGET(st); EXEC_CAND(sc);
                    if (!d_states_equal(st, sc, dfm)) { atomicOr(&mismatch, 1); goto done; }
                }
            }
        }
    } else {
        // Reduced sweep: rep_values for extra regs, rep_sp for SP
        int indices[6] = {};
        bool iter_done = false;
        while (!iter_done) {
            if (mismatch) goto done;
            Z80State base = {};
            base.r[REG_A] = (uint8_t)a_val;
            for (int i = 0; i < ep.nextra; i++)
                d_set_reg_by_offset(base, ep.extra[i], d_rep_values[indices[i]]);

            if (ep.sweep_sp) {
                for (int si = 0; si < 16; si++) {
                    for (int carry = 0; carry <= 1; carry++) {
                        if (mismatch) goto done;
                        Z80State st = base, sc = base;
                        st.r[REG_F] = sc.r[REG_F] = (uint8_t)carry;
                        st.sp = sc.sp = d_rep_sp[si];
                        EXEC_TARGET(st); EXEC_CAND(sc);
                        if (!d_states_equal(st, sc, dfm)) { atomicOr(&mismatch, 1); goto done; }
                    }
                }
            } else {
                for (int carry = 0; carry <= 1; carry++) {
                    if (mismatch) goto done;
                    Z80State st = base, sc = base;
                    st.r[REG_F] = sc.r[REG_F] = (uint8_t)carry;
                    EXEC_TARGET(st); EXEC_CAND(sc);
                    if (!d_states_equal(st, sc, dfm)) { atomicOr(&mismatch, 1); goto done; }
                }
            }

            int k = ep.nextra - 1;
            while (k >= 0) {
                indices[k]++;
                if (indices[k] < 32) break;
                indices[k] = 0;
                k--;
            }
            if (k < 0) iter_done = true;
        }
    }

    #undef EXEC_TARGET
    #undef EXEC_CAND

done:
    __syncthreads();
    if (threadIdx.x == 0) {
        results[pair_idx] = (mismatch == 0) ? 1u : 0u;
    }
}

// ============================================================
// Host-side tables (verbatim from v1)
// ============================================================
static bool is_imm8(uint16_t op) {
    if (op >= 49 && op <= 55) return true;
    if (op >= 56 && op < 120 && ((op - 56) % 8) == 7) return true;
    return false;
}
static bool is_imm16(uint16_t op) { return op >= 382 && op <= 385; }

static int byte_size(uint16_t op) {
    if (op >= 144 && op <= 367) return 2;
    if (op == 142) return 2;
    if (op >= 386 && op <= 393) return 2;
    if (op >= 382 && op <= 385) return 3;
    if (is_imm8(op)) return 2;
    return 1;
}

static int tstates(uint16_t op) {
    if (op < 49) return 4;
    if (op < 56) return 7;
    if (op < 120) { if ((op - 56) % 8 == 7) return 7; return 4; }
    if (op < 134) return 4;
    if (op <= 141) return 4;
    if (op == 142) return 8;
    if (op == 143) return 4;
    if (op < 200) return 8;
    if (op < 368) return 8;
    if (op < 376) return 6;
    if (op < 380) return 11;
    if (op == 380) return 4;
    if (op == 381) return 6;
    if (op < 386) return 10;
    return 15;
}

static const char* reg_names[8] = {"A", "F", "B", "C", "D", "E", "H", "L"};
static const char* pair_names[4] = {"BC", "DE", "HL", "SP"};
static const char* alu_names[8] = {"ADD A,", "ADC A,", "SUB", "SBC A,", "AND", "XOR", "OR", "CP"};
static const char* cb_names[7] = {"RLC", "RRC", "RL", "RR", "SLA", "SRA", "SRL"};

static void disasm(uint16_t op, uint16_t imm, char* buf, int bufsz) {
    if (op < 49) { snprintf(buf, bufsz, "LD %s, %s", reg_names[LD_DST_H[op/7]], reg_names[LD_FULL_SRC_H[op]]); return; }
    if (op < 56) { snprintf(buf, bufsz, "LD %s, 0x%02X", reg_names[IMM_REG_H[op-49]], imm & 0xFF); return; }
    if (op < 120) { int a=(op-56)/8, s=(op-56)%8; if(s<7) snprintf(buf,bufsz,"%s %s",alu_names[a],reg_names[ALU_SRC_H[s]]); else snprintf(buf,bufsz,"%s 0x%02X",alu_names[a],imm&0xFF); return; }
    if (op < 127) { snprintf(buf, bufsz, "INC %s", reg_names[INCDEC_REG_H[op-120]]); return; }
    if (op < 134) { snprintf(buf, bufsz, "DEC %s", reg_names[INCDEC_REG_H[op-127]]); return; }
    if (op==134) { snprintf(buf,bufsz,"RLCA"); return; } if (op==135) { snprintf(buf,bufsz,"RRCA"); return; }
    if (op==136) { snprintf(buf,bufsz,"RLA"); return; }  if (op==137) { snprintf(buf,bufsz,"RRA"); return; }
    if (op==138) { snprintf(buf,bufsz,"DAA"); return; }  if (op==139) { snprintf(buf,bufsz,"CPL"); return; }
    if (op==140) { snprintf(buf,bufsz,"SCF"); return; }  if (op==141) { snprintf(buf,bufsz,"CCF"); return; }
    if (op==142) { snprintf(buf,bufsz,"NEG"); return; }  if (op==143) { snprintf(buf,bufsz,"NOP"); return; }
    if (op>=144&&op<=192) { int c=(op-144)/7,r=CB_REG_H[(op-144)%7]; snprintf(buf,bufsz,"%s %s",cb_names[c],reg_names[r]); return; }
    if (op==193) { snprintf(buf,bufsz,"SLL A"); return; }
    if (op>=194&&op<200) { snprintf(buf,bufsz,"SLL %s",reg_names[CB_REG_H[(op-194)+1]]); return; }
    if (op>=200&&op<256) { int i=op-200; snprintf(buf,bufsz,"BIT %d, %s",i/7,reg_names[CB_REG_H[i%7]]); return; }
    if (op>=256&&op<312) { int i=op-256; snprintf(buf,bufsz,"RES %d, %s",i/7,reg_names[CB_REG_H[i%7]]); return; }
    if (op>=312&&op<368) { int i=op-312; snprintf(buf,bufsz,"SET %d, %s",i/7,reg_names[CB_REG_H[i%7]]); return; }
    if (op>=368&&op<372) { snprintf(buf,bufsz,"INC %s",pair_names[op-368]); return; }
    if (op>=372&&op<376) { snprintf(buf,bufsz,"DEC %s",pair_names[op-372]); return; }
    if (op>=376&&op<380) { snprintf(buf,bufsz,"ADD HL, %s",pair_names[op-376]); return; }
    if (op==380) { snprintf(buf,bufsz,"EX DE, HL"); return; }
    if (op==381) { snprintf(buf,bufsz,"LD SP, HL"); return; }
    if (op>=382&&op<386) { snprintf(buf,bufsz,"LD %s, 0x%04X",pair_names[op-382],imm); return; }
    if (op>=386&&op<390) { snprintf(buf,bufsz,"ADC HL, %s",pair_names[op-386]); return; }
    if (op>=390&&op<394) { snprintf(buf,bufsz,"SBC HL, %s",pair_names[op-390]); return; }
    snprintf(buf,bufsz,"OP(%d)",op);
}

// ============================================================
// Register dependency (verbatim from v1)
// ============================================================
#define RMASK_A 0x001u
#define RMASK_F 0x002u
#define RMASK_B 0x004u
#define RMASK_C 0x008u
#define RMASK_D 0x010u
#define RMASK_E 0x020u
#define RMASK_H 0x040u
#define RMASK_L 0x080u
#define RMASK_SP 0x100u

static uint16_t op_reads(uint16_t op) {
    if (op<49) { return 1u<<LD_FULL_SRC_H[op]; }
    if (op<56) return 0;
    if (op<120) { int ao=(op-56)/8,si=(op-56)%8; uint16_t m=0; if(si<7)m|=(1u<<ALU_SRC_H[si]); if(ao==1||ao==3)m|=RMASK_F; m|=RMASK_A; return m; }
    if (op<127) return 1u<<INCDEC_REG_H[op-120];
    if (op<134) return 1u<<INCDEC_REG_H[op-127];
    if (op==OP_RLCA||op==OP_RRCA) return RMASK_A;
    if (op==OP_RLA||op==OP_RRA) return RMASK_A|RMASK_F;
    if (op==OP_DAA) return RMASK_A|RMASK_F;
    if (op==OP_CPL||op==OP_NEG) return RMASK_A;
    if (op==OP_SCF||op==OP_CCF) return RMASK_F;
    if (op==OP_NOP) return 0;
    if (op>=144&&op<=192) { int co=(op-144)/7; uint8_t rg=CB_REG_H[(op-144)%7]; uint16_t m=1u<<rg; if(co==2||co==3)m|=RMASK_F; return m; }
    if (op==OP_SLL_A) return RMASK_A;
    if (op>=194&&op<200) return 1u<<CB_REG_H[(op-194)+1];
    if (op>=200&&op<368) { int idx=(op>=312)?op-312:(op>=256)?op-256:op-200; return 1u<<CB_REG_H[idx%7]; }
    if (op>=368&&op<376) { int p=(op-368)%4; static const uint16_t pm[4]={RMASK_B|RMASK_C,RMASK_D|RMASK_E,RMASK_H|RMASK_L,RMASK_SP}; return pm[p]; }
    if (op>=376&&op<380) { int p=op-376; static const uint16_t pm[4]={RMASK_B|RMASK_C,RMASK_D|RMASK_E,RMASK_H|RMASK_L,RMASK_SP}; return RMASK_H|RMASK_L|pm[p]; }
    if (op==380) return RMASK_D|RMASK_E|RMASK_H|RMASK_L;
    if (op==381) return RMASK_H|RMASK_L;
    if (op>=382&&op<386) return 0;
    if (op>=386&&op<394) { int p=(op>=390)?op-390:op-386; static const uint16_t pm[4]={RMASK_B|RMASK_C,RMASK_D|RMASK_E,RMASK_H|RMASK_L,RMASK_SP}; return RMASK_H|RMASK_L|RMASK_F|pm[p]; }
    return 0;
}

static uint16_t op_writes(uint16_t op) {
    if (op<49) return 1u<<LD_DST_H[op/7];
    if (op<56) return 1u<<IMM_REG_H[op-49];
    if (op<120) { int ao=(op-56)/8; if(ao==7)return RMASK_F; return RMASK_A|RMASK_F; }
    if (op<134) return (1u<<INCDEC_REG_H[op<127?op-120:op-127])|RMASK_F;
    if (op>=134&&op<=137) return RMASK_A|RMASK_F;
    if (op==OP_DAA||op==OP_CPL||op==OP_NEG) return RMASK_A|RMASK_F;
    if (op==OP_SCF||op==OP_CCF) return RMASK_F;
    if (op==OP_NOP) return 0;
    if (op>=144&&op<=199) { uint8_t rg; if(op<=192)rg=CB_REG_H[(op-144)%7]; else if(op==193)rg=REG_A; else rg=CB_REG_H[(op-194)+1]; return(1u<<rg)|RMASK_F; }
    if (op>=200&&op<256) return RMASK_F;
    if (op>=256&&op<368) { int idx=(op>=312)?op-312:op-256; return 1u<<CB_REG_H[idx%7]; }
    if (op>=368&&op<376) { int p=(op-368)%4; static const uint16_t pm[4]={RMASK_B|RMASK_C,RMASK_D|RMASK_E,RMASK_H|RMASK_L,RMASK_SP}; return pm[p]; }
    if (op>=376&&op<380) return RMASK_H|RMASK_L|RMASK_F;
    if (op==380) return RMASK_D|RMASK_E|RMASK_H|RMASK_L;
    if (op==381) return RMASK_SP;
    if (op>=382&&op<386) { int p=op-382; static const uint16_t pm[4]={RMASK_B|RMASK_C,RMASK_D|RMASK_E,RMASK_H|RMASK_L,RMASK_SP}; return pm[p]; }
    if (op>=386&&op<394) return RMASK_H|RMASK_L|RMASK_F;
    return 0;
}

// Pruning
static bool is_self_load(uint16_t op) { return op==6||op==8||op==16||op==24||op==32||op==40||op==48; }
static uint32_t inst_key(uint16_t op, uint16_t imm) { return ((uint32_t)op<<16)|imm; }
static bool are_independent(uint16_t op1, uint16_t op2) {
    uint16_t aR=op_reads(op1),aW=op_writes(op1),bR=op_reads(op2),bW=op_writes(op2);
    return (aW&bR)==0&&(aR&bW)==0&&(aW&bW)==0;
}
static bool should_prune(const uint16_t* ops, const uint16_t* imms, int n) {
    for (int i=0;i<n;i++) {
        if (ops[i]==OP_NOP) return true;
        if (is_self_load(ops[i])) return true;
        if (i+1<n) { uint16_t w1=op_writes(ops[i]); if(w1){uint16_t r2=op_reads(ops[i+1]),w2=op_writes(ops[i+1]),dead=w1&w2&~RMASK_F&~r2; if(dead)return true;} }
    }
    for (int i=0;i+1<n;i++) { if(are_independent(ops[i],ops[i+1])&&inst_key(ops[i],imms[i])>inst_key(ops[i+1],imms[i+1])) return true; }
    return false;
}

// Register reads helper
static uint16_t regs_read(const uint16_t* ops, int n) { uint16_t m=0; for(int i=0;i<n;i++)m|=op_reads(ops[i]); return m; }

// Build ExhaustPair struct for GPU kernel 3
static ExhaustPair build_exhaust_pair(
    const uint16_t* t_ops, const uint16_t* t_imms, int t_n,
    const uint16_t* c_ops, const uint16_t* c_imms, int c_n,
    uint8_t dead_flags
) {
    ExhaustPair ep = {};
    for (int i=0; i<t_n && i<3; i++) { ep.t_ops[i]=t_ops[i]; ep.t_imms[i]=t_imms[i]; }
    ep.c_ops[0]=c_ops[0]; ep.c_imms[0]=c_imms[0];
    ep.t_len=(uint8_t)t_n; ep.c_len=(uint8_t)c_n; ep.dead_flags=dead_flags;
    uint16_t reads = regs_read(t_ops,t_n)|regs_read(c_ops,c_n);
    ep.nextra=0;
    if (reads&RMASK_B) ep.extra[ep.nextra++]=2;
    if (reads&RMASK_C) ep.extra[ep.nextra++]=3;
    if (reads&RMASK_D) ep.extra[ep.nextra++]=4;
    if (reads&RMASK_E) ep.extra[ep.nextra++]=5;
    if (reads&RMASK_H) ep.extra[ep.nextra++]=6;
    if (reads&RMASK_L) ep.extra[ep.nextra++]=7;
    ep.sweep_sp=(reads&RMASK_SP)?1:0;
    ep.use_full=(ep.nextra<=2&&!ep.sweep_sp)?1:0;
    return ep;
}

// Host-side helpers for CPU ExhaustiveCheck fallback
static void h_set_reg_by_offset(Z80State &s, int offset, uint8_t val) {
    switch (offset) {
        case 2: s.r[REG_B]=val; break; case 3: s.r[REG_C]=val; break;
        case 4: s.r[REG_D]=val; break; case 5: s.r[REG_E]=val; break;
        case 6: s.r[REG_H]=val; break; case 7: s.r[REG_L]=val; break;
    }
}

static const uint8_t h_rep_values[32] = {
    0x00,0x01,0x02,0x0F,0x10,0x1F,0x20,0x3F,
    0x40,0x55,0x7E,0x7F,0x80,0x81,0xAA,0xBF,
    0xC0,0xD5,0xE0,0xEF,0xF0,0xF7,0xFE,0xFF,
    0x03,0x07,0x11,0x33,0x77,0xBB,0xDD,0xEE,
};
static const uint16_t h_rep_sp[16] = {
    0x0000,0x0001,0x00FF,0x0100,0x7FFE,0x7FFF,0x8000,0x8001,
    0xFFFE,0xFFFF,0x1234,0x5678,0xABCD,0xDEAD,0xBEEF,0xCAFE,
};

// CPU ExhaustiveCheck — used for reduced-sweep pairs (nextra>2 or sweep_sp)
// where CPU is faster than GPU due to early-exit and less launch overhead.
static bool cpu_exhaustive_check(
    const uint16_t* t_ops, const uint16_t* t_imms, int t_n,
    const uint16_t* c_ops, const uint16_t* c_imms, int c_n,
    uint8_t dead_flags
) {
    uint16_t reads = regs_read(t_ops,t_n)|regs_read(c_ops,c_n);
    int extra[6]; int nextra=0;
    if (reads&RMASK_B) extra[nextra++]=2;
    if (reads&RMASK_C) extra[nextra++]=3;
    if (reads&RMASK_D) extra[nextra++]=4;
    if (reads&RMASK_E) extra[nextra++]=5;
    if (reads&RMASK_H) extra[nextra++]=6;
    if (reads&RMASK_L) extra[nextra++]=7;
    bool sweep_sp = (reads&RMASK_SP)!=0;

    for (int a=0;a<256;a++) {
        for (int carry=0;carry<=1;carry++) {
            if (nextra==0 && !sweep_sp) {
                Z80State s={}; s.r[REG_A]=(uint8_t)a; s.r[REG_F]=(uint8_t)carry;
                Z80State st=s, sc=s;
                h_exec_seq(st,t_ops,t_imms,t_n); h_exec_seq(sc,c_ops,c_imms,c_n);
                if (!h_states_equal(st,sc,dead_flags)) return false;
                continue;
            }
            if (nextra==1 && !sweep_sp) {
                for (int r=0;r<256;r++) {
                    Z80State s={}; s.r[REG_A]=(uint8_t)a; s.r[REG_F]=(uint8_t)carry;
                    h_set_reg_by_offset(s,extra[0],(uint8_t)r);
                    Z80State st=s, sc=s;
                    h_exec_seq(st,t_ops,t_imms,t_n); h_exec_seq(sc,c_ops,c_imms,c_n);
                    if (!h_states_equal(st,sc,dead_flags)) return false;
                }
                continue;
            }
            if (nextra==2 && !sweep_sp) {
                for (int r1=0;r1<256;r1++) {
                    for (int r2=0;r2<256;r2++) {
                        Z80State s={}; s.r[REG_A]=(uint8_t)a; s.r[REG_F]=(uint8_t)carry;
                        h_set_reg_by_offset(s,extra[0],(uint8_t)r1);
                        h_set_reg_by_offset(s,extra[1],(uint8_t)r2);
                        Z80State st=s, sc=s;
                        h_exec_seq(st,t_ops,t_imms,t_n); h_exec_seq(sc,c_ops,c_imms,c_n);
                        if (!h_states_equal(st,sc,dead_flags)) return false;
                    }
                }
                continue;
            }
            // Reduced sweep (3+ regs or SP): use rep_values
            auto do_sweep = [&](auto& self, Z80State s, int ri) -> bool {
                if (ri>=nextra) {
                    if (sweep_sp) {
                        for (int si=0;si<16;si++) {
                            Z80State s2=s; s2.sp=h_rep_sp[si];
                            Z80State st=s2, sc=s2;
                            h_exec_seq(st,t_ops,t_imms,t_n); h_exec_seq(sc,c_ops,c_imms,c_n);
                            if (!h_states_equal(st,sc,dead_flags)) return false;
                        }
                        return true;
                    }
                    Z80State st=s, sc=s;
                    h_exec_seq(st,t_ops,t_imms,t_n); h_exec_seq(sc,c_ops,c_imms,c_n);
                    return h_states_equal(st,sc,dead_flags);
                }
                for (int vi=0;vi<32;vi++) {
                    Z80State s2=s;
                    h_set_reg_by_offset(s2,extra[ri],h_rep_values[vi]);
                    if (!self(self,s2,ri+1)) return false;
                }
                return true;
            };
            Z80State base={}; base.r[REG_A]=(uint8_t)a; base.r[REG_F]=(uint8_t)carry;
            if (!do_sweep(do_sweep,base,0)) return false;
        }
    }
    return true;
}

// Instruction enumeration
struct Inst { uint16_t op, imm; };
static std::vector<Inst> enumerate_instructions_8() {
    std::vector<Inst> result;
    for (uint16_t op=0; op<OP_COUNT; op++) {
        if (is_imm16(op)) continue;
        if (is_imm8(op)) { for(int imm=0;imm<256;imm++) result.push_back({op,(uint16_t)imm}); }
        else result.push_back({op,0});
    }
    return result;
}

// Batch target
struct BatchTarget {
    uint16_t ops[3], imms[3];
    int len, bytes;
};

// ============================================================
// Main — 3-stage batched pipeline
// ============================================================
int main(int argc, char** argv) {
    int max_target=2; uint8_t dead_flags=0; int gpu_id=0;
    int first_op_start=0, first_op_end=-1;
    bool no_exhaust=false;

    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--max-target")&&i+1<argc) max_target=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--dead-flags")&&i+1<argc) dead_flags=(uint8_t)strtoul(argv[++i],NULL,0);
        else if (!strcmp(argv[i],"--gpu-id")&&i+1<argc) gpu_id=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--first-op-start")&&i+1<argc) first_op_start=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--first-op-end")&&i+1<argc) first_op_end=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--no-exhaust")) no_exhaust=true;
        else if (!strcmp(argv[i],"--help")) {
            fprintf(stderr,"Usage: z80search_v2 [OPTIONS]\n"
                "  --max-target N        Max target sequence length (default: 2)\n"
                "  --dead-flags 0xNN     Flag mask for dead flags\n"
                "  --gpu-id N            CUDA device ID (default: 0)\n"
                "  --first-op-start M    Start outer loop at instruction index M\n"
                "  --first-op-end N      End outer loop at instruction index N\n"
                "  --no-exhaust          Skip ExhaustiveCheck, output MidCheck survivors\n");
            return 0;
        }
    }

    cudaSetDevice(gpu_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_id);
    fprintf(stderr,"GPU %d: %s (%.1f GB, SM %d.%d)\n", gpu_id, prop.name, prop.totalGlobalMem/1e9, prop.major, prop.minor);

    init_tables();
    upload_tables_cuda();

    std::vector<Inst> all_insts = enumerate_instructions_8();
    uint32_t cand_count = (uint32_t)all_insts.size();
    fprintf(stderr,"Instruction set: %u instructions (8-bit)\n", cand_count);
    if (first_op_end<0) first_op_end=(int)all_insts.size();

    std::vector<uint32_t> cand_packed(cand_count);
    for (size_t i=0; i<all_insts.size(); i++)
        cand_packed[i] = (uint32_t)all_insts[i].op | ((uint32_t)all_insts[i].imm<<16);

    // GPU allocations
    uint32_t bitmap_words = (cand_count+31)/32;
    uint32_t *d_candidates, *d_hit_bitmap;
    uint8_t *d_target_fps, *d_target_mid_fps;
    MidPair *d_mid_pairs; uint32_t *d_mid_survived;
    ExhaustPair *d_exhaust_pairs; uint32_t *d_exhaust_results;

    uint32_t max_mid_pairs = BATCH_SIZE * 256;  // 128K generous
    uint32_t max_exhaust_pairs = 16384;

    cudaMalloc(&d_candidates, cand_count*sizeof(uint32_t));
    cudaMalloc(&d_target_fps, BATCH_SIZE*FP_LEN);
    cudaMalloc(&d_target_mid_fps, BATCH_SIZE*MID_FP_LEN);
    cudaMalloc(&d_hit_bitmap, BATCH_SIZE*bitmap_words*sizeof(uint32_t));
    cudaMalloc(&d_mid_pairs, max_mid_pairs*sizeof(MidPair));
    cudaMalloc(&d_mid_survived, max_mid_pairs*sizeof(uint32_t));
    cudaMalloc(&d_exhaust_pairs, max_exhaust_pairs*sizeof(ExhaustPair));
    cudaMalloc(&d_exhaust_results, max_exhaust_pairs*sizeof(uint32_t));
    cudaMemcpy(d_candidates, cand_packed.data(), cand_count*sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Host buffers
    uint8_t *h_fps = (uint8_t*)malloc(BATCH_SIZE*FP_LEN);
    uint8_t *h_mfps = (uint8_t*)malloc(BATCH_SIZE*MID_FP_LEN);
    uint32_t *h_bitmap = (uint32_t*)malloc(BATCH_SIZE*bitmap_words*sizeof(uint32_t));
    MidPair *h_mpairs = (MidPair*)malloc(max_mid_pairs*sizeof(MidPair));
    uint32_t *h_msurv = (uint32_t*)malloc(max_mid_pairs*sizeof(uint32_t));
    ExhaustPair *h_epairs = (ExhaustPair*)malloc(max_exhaust_pairs*sizeof(ExhaustPair));
    uint32_t *h_eresults = (uint32_t*)malloc(max_exhaust_pairs*sizeof(uint32_t));

    uint64_t total_found=0, total_targets=0, total_qc_hits=0, total_mid_hits=0, total_exhaust=0, total_cpu_exhaust=0, total_batches=0;
    time_t start_time = time(NULL);

    fprintf(stderr,"Starting v2 search: max_target=%d, dead_flags=0x%02X, gpu=%d, ops=[%d,%d)\n",
            max_target, dead_flags, gpu_id, first_op_start, first_op_end);

    std::vector<BatchTarget> batch;
    batch.reserve(BATCH_SIZE);

    // Flush one batch through 3-stage pipeline
    auto flush_batch = [&]() {
        if (batch.empty()) return;
        uint32_t bc = (uint32_t)batch.size();
        total_batches++;

        // CPU: compute fingerprints
        for (uint32_t bi=0; bi<bc; bi++) {
            h_fingerprint(batch[bi].ops, batch[bi].imms, batch[bi].len, h_fps+bi*FP_LEN);
            h_mid_fingerprint(batch[bi].ops, batch[bi].imms, batch[bi].len, h_mfps+bi*MID_FP_LEN);
        }
        cudaMemcpy(d_target_fps, h_fps, bc*FP_LEN, cudaMemcpyHostToDevice);
        cudaMemcpy(d_target_mid_fps, h_mfps, bc*MID_FP_LEN, cudaMemcpyHostToDevice);
        cudaMemset(d_hit_bitmap, 0, bc*bitmap_words*sizeof(uint32_t));

        // Stage 1: Batched QuickCheck
        uint32_t total_threads = bc * cand_count;
        int grid1 = (total_threads+BLOCK_SIZE-1)/BLOCK_SIZE;
        quickcheck_batched<<<grid1, BLOCK_SIZE>>>(d_candidates, d_target_fps, d_hit_bitmap,
            cand_count, bc, bitmap_words, dead_flags);
        cudaDeviceSynchronize();
        cudaMemcpy(h_bitmap, d_hit_bitmap, bc*bitmap_words*sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Collect QC hits
        uint32_t mid_count=0;
        for (uint32_t bi=0; bi<bc; bi++) {
            int tbytes = batch[bi].bytes;
            for (uint32_t w=0; w<bitmap_words && mid_count<max_mid_pairs; w++) {
                uint32_t bits = h_bitmap[bi*bitmap_words+w];
                while (bits && mid_count<max_mid_pairs) {
                    int bit = __builtin_ctz(bits);
                    bits &= bits-1;
                    uint32_t ci = w*32+bit;
                    if (ci>=cand_count) break;
                    int cb = byte_size(all_insts[ci].op);
                    if (cb>=tbytes) continue;
                    uint16_t co[1]={all_insts[ci].op}, cm[1]={all_insts[ci].imm};
                    if (should_prune(co,cm,1)) continue;
                    h_mpairs[mid_count++] = {(uint16_t)bi, (uint16_t)ci};
                }
            }
        }
        total_qc_hits += mid_count;
        if (mid_count==0) { batch.clear(); return; }

        // Stage 2: MidCheck
        cudaMemcpy(d_mid_pairs, h_mpairs, mid_count*sizeof(MidPair), cudaMemcpyHostToDevice);
        int grid2 = (mid_count+BLOCK_SIZE-1)/BLOCK_SIZE;
        midcheck_kernel<<<grid2, BLOCK_SIZE>>>(d_mid_pairs, mid_count, d_candidates,
            d_target_mid_fps, d_mid_survived, dead_flags);
        cudaDeviceSynchronize();
        cudaMemcpy(h_msurv, d_mid_survived, mid_count*sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Collect MidCheck survivors
        struct EInfo { uint32_t bi, ci; };
        std::vector<EInfo> mid_survivors;
        for (uint32_t mi=0; mi<mid_count; mi++) {
            if (!h_msurv[mi]) continue;
            MidPair mp = h_mpairs[mi];
            mid_survivors.push_back({mp.target_idx, mp.cand_idx});
        }
        total_mid_hits += (uint64_t)mid_survivors.size();
        if (mid_survivors.empty()) { batch.clear(); return; }

        // Helper: output one result as JSONL
        auto emit_jsonl = [&](BatchTarget &bt, uint16_t cop, uint16_t cimm) {
            int cb=byte_size(cop);
            total_found++;
            char sbuf[256], rbuf[64], p[3][64];
            for (int j=0;j<bt.len;j++) disasm(bt.ops[j],bt.imms[j],p[j],sizeof(p[j]));
            disasm(cop,cimm,rbuf,sizeof(rbuf));
            if (bt.len==2) snprintf(sbuf,sizeof(sbuf),"%s : %s",p[0],p[1]);
            else if (bt.len==3) snprintf(sbuf,sizeof(sbuf),"%s : %s : %s",p[0],p[1],p[2]);
            else snprintf(sbuf,sizeof(sbuf),"%s",p[0]);
            int bsaved=bt.bytes-cb, csaved=0;
            for (int j=0;j<bt.len;j++) csaved+=tstates(bt.ops[j]);
            csaved-=tstates(cop);
            printf("{\"source_asm\":\"%s\",\"replacement_asm\":\"%s\","
                   "\"source_bytes\":%d,\"replacement_bytes\":%d,"
                   "\"bytes_saved\":%d,\"cycles_saved\":%d",
                   sbuf, rbuf, bt.bytes, cb, bsaved, csaved);
            if (dead_flags) printf(",\"dead_flags\":\"0x%02X\"",dead_flags);
            printf("}\n"); fflush(stdout);
        };

        if (no_exhaust) {
            // --no-exhaust: output all MidCheck survivors without ExhaustiveCheck
            for (auto &inf : mid_survivors) {
                emit_jsonl(batch[inf.bi], all_insts[inf.ci].op, all_insts[inf.ci].imm);
            }
        } else {
            // Stage 3: split into GPU (full sweep) and CPU (reduced sweep)
            uint32_t exhaust_count=0;
            std::vector<EInfo> gpu_einfo;
            std::vector<EInfo> cpu_einfo;
            for (auto &inf : mid_survivors) {
                uint16_t co[1]={all_insts[inf.ci].op}, cm[1]={all_insts[inf.ci].imm};
                ExhaustPair ep = build_exhaust_pair(
                    batch[inf.bi].ops, batch[inf.bi].imms, batch[inf.bi].len,
                    co, cm, 1, dead_flags);
                if (ep.use_full && exhaust_count<max_exhaust_pairs) {
                    h_epairs[exhaust_count] = ep;
                    gpu_einfo.push_back(inf);
                    exhaust_count++;
                } else {
                    cpu_einfo.push_back(inf);
                }
            }

            // Stage 3a: GPU ExhaustiveCheck (full-sweep pairs)
            if (exhaust_count>0) {
                cudaMemcpy(d_exhaust_pairs, h_epairs, exhaust_count*sizeof(ExhaustPair), cudaMemcpyHostToDevice);
                exhaustive_check_gpu<<<exhaust_count, EXHAUST_BLOCK>>>(d_exhaust_pairs, d_exhaust_results);
                cudaDeviceSynchronize();
                cudaMemcpy(h_eresults, d_exhaust_results, exhaust_count*sizeof(uint32_t), cudaMemcpyDeviceToHost);
                total_exhaust += exhaust_count;
                for (uint32_t ei=0; ei<exhaust_count; ei++) {
                    if (!h_eresults[ei]) continue;
                    emit_jsonl(batch[gpu_einfo[ei].bi], all_insts[gpu_einfo[ei].ci].op, all_insts[gpu_einfo[ei].ci].imm);
                }
            }

            // Stage 3b: CPU ExhaustiveCheck (reduced-sweep pairs)
            for (auto &inf : cpu_einfo) {
                BatchTarget &bt = batch[inf.bi];
                uint16_t co[1]={all_insts[inf.ci].op}, cm[1]={all_insts[inf.ci].imm};
                total_cpu_exhaust++;
                if (cpu_exhaustive_check(bt.ops, bt.imms, bt.len, co, cm, 1, dead_flags))
                    emit_jsonl(bt, co[0], cm[0]);
            }
        }
        batch.clear();
    };

    // Enumerate targets
    for (int target_len=2; target_len<=max_target; target_len++) {
        fprintf(stderr,"=== Target length %d ===\n", target_len);
        uint64_t targets_this=0, found_before=total_found;
        time_t len_start=time(NULL), last_report=len_start;

        if (target_len==2) {
            for (int i0=first_op_start; i0<first_op_end && i0<(int)all_insts.size(); i0++) {
                time_t now=time(NULL);
                if (now-last_report>=10) {
                    last_report=now;
                    double pct=100.0*(i0-first_op_start)/(first_op_end-first_op_start);
                    double el=difftime(now,start_time), eta=(pct>0.1)?el*(100.0/pct-1.0):0;
                    fprintf(stderr,"  [%.1f%%] op %d/%d | targets:%lu | QC:%lu Mid:%lu Ex:%lu | found:%lu | %lds, ETA %lds\n",
                        pct,i0,first_op_end,(unsigned long)total_targets,
                        (unsigned long)total_qc_hits,(unsigned long)total_mid_hits,
                        (unsigned long)total_exhaust,(unsigned long)total_found,
                        (long)el,(long)eta);
                }
                for (size_t i1=0; i1<all_insts.size(); i1++) {
                    uint16_t to[2]={all_insts[i0].op,all_insts[i1].op};
                    uint16_t ti[2]={all_insts[i0].imm,all_insts[i1].imm};
                    if (should_prune(to,ti,2)) continue;
                    targets_this++; total_targets++;
                    BatchTarget bt; bt.ops[0]=to[0]; bt.ops[1]=to[1]; bt.ops[2]=0;
                    bt.imms[0]=ti[0]; bt.imms[1]=ti[1]; bt.imms[2]=0;
                    bt.len=2; bt.bytes=byte_size(to[0])+byte_size(to[1]);
                    batch.push_back(bt);
                    if ((int)batch.size()>=BATCH_SIZE) flush_batch();
                }
            }
        } else if (target_len==3) {
            for (int i0=first_op_start; i0<first_op_end && i0<(int)all_insts.size(); i0++) {
                time_t now=time(NULL);
                if (now-last_report>=10) {
                    last_report=now;
                    double pct=100.0*(i0-first_op_start)/(first_op_end-first_op_start);
                    double el=difftime(now,start_time), eta=(pct>0.1)?el*(100.0/pct-1.0):0;
                    fprintf(stderr,"  [%.1f%%] op %d/%d | targets:%lu | found:%lu | %lds, ETA %lds\n",
                        pct,i0,first_op_end,(unsigned long)total_targets,
                        (unsigned long)total_found,(long)el,(long)eta);
                }
                for (size_t i1=0; i1<all_insts.size(); i1++) {
                    for (size_t i2=0; i2<all_insts.size(); i2++) {
                        uint16_t to[3]={all_insts[i0].op,all_insts[i1].op,all_insts[i2].op};
                        uint16_t ti[3]={all_insts[i0].imm,all_insts[i1].imm,all_insts[i2].imm};
                        if (should_prune(to,ti,3)) continue;
                        targets_this++; total_targets++;
                        BatchTarget bt;
                        bt.ops[0]=to[0]; bt.ops[1]=to[1]; bt.ops[2]=to[2];
                        bt.imms[0]=ti[0]; bt.imms[1]=ti[1]; bt.imms[2]=ti[2];
                        bt.len=3; bt.bytes=byte_size(to[0])+byte_size(to[1])+byte_size(to[2]);
                        batch.push_back(bt);
                        if ((int)batch.size()>=BATCH_SIZE) flush_batch();
                    }
                }
            }
        }
        flush_batch();
        time_t len_end=time(NULL);
        fprintf(stderr,"  Length %d done: %lu targets, %lu found (%lds)\n",
            target_len,(unsigned long)targets_this,(unsigned long)(total_found-found_before),(long)(len_end-len_start));
    }

    time_t end_time=time(NULL);
    fprintf(stderr,"\n=== DONE (v2 batched pipeline) ===\n");
    fprintf(stderr,"Targets tested:     %lu\n",(unsigned long)total_targets);
    fprintf(stderr,"Batches processed:  %lu\n",(unsigned long)total_batches);
    fprintf(stderr,"QuickCheck hits:    %lu\n",(unsigned long)total_qc_hits);
    fprintf(stderr,"MidCheck survivors: %lu\n",(unsigned long)total_mid_hits);
    fprintf(stderr,"ExhaustiveCheck:    %lu (GPU:%lu CPU:%lu)\n",(unsigned long)(total_exhaust+total_cpu_exhaust),(unsigned long)total_exhaust,(unsigned long)total_cpu_exhaust);
    fprintf(stderr,"Results found:      %lu\n",(unsigned long)total_found);
    fprintf(stderr,"Total time:         %lds\n",(long)(end_time-start_time));
    if (total_qc_hits>0) fprintf(stderr,"False positive rate: %.1f%% (QC->confirmed)\n",100.0*(1.0-(double)total_found/total_qc_hits));

    free(h_fps); free(h_mfps); free(h_bitmap);
    free(h_mpairs); free(h_msurv); free(h_epairs); free(h_eresults);
    cudaFree(d_candidates); cudaFree(d_target_fps); cudaFree(d_target_mid_fps);
    cudaFree(d_hit_bitmap); cudaFree(d_mid_pairs); cudaFree(d_mid_survived);
    cudaFree(d_exhaust_pairs); cudaFree(d_exhaust_results);
    return 0;
}

// Z80 Standalone GPU Superoptimizer Search
//
// Build: nvcc -O2 -o z80search z80_search.cu
// Usage: ./z80search --max-target 3 [--dead-flags 0x28] [--gpu-only]
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

static void upload_tables_cuda() {
    cudaMemcpyToSymbol(d_sz53, h_sz53, 256);
    cudaMemcpyToSymbol(d_sz53p, h_sz53p, 256);
    cudaMemcpyToSymbol(d_parity, h_parity, 256);
    cudaMemcpyToSymbol(d_halfcarry_add, h_halfcarry_add, 8);
    cudaMemcpyToSymbol(d_halfcarry_sub, h_halfcarry_sub, 8);
    cudaMemcpyToSymbol(d_overflow_add, h_overflow_add, 8);
    cudaMemcpyToSymbol(d_overflow_sub, h_overflow_sub, 8);
    cudaMemcpyToSymbol(d_test_vectors, h_test_vectors, sizeof(h_test_vectors));
}

// ============================================================
// GPU device-side ALU helpers (needed by kernel)
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

// ============================================================
// GPU QuickCheck kernel
// ============================================================
__global__ void quickcheck_kernel(
    const uint32_t* __restrict__ candidates,
    const uint8_t*  __restrict__ target_fp,
    uint32_t*       __restrict__ results,
    uint32_t candidate_count,
    uint32_t seq_len,
    uint32_t dead_flags
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= candidate_count) return;

    const uint32_t* my_seq = candidates + (uint64_t)tid * seq_len;
    uint8_t my_fp[FP_LEN];
    for (int v = 0; v < NUM_VECTORS; v++) {
        Z80State s = d_test_vectors[v];
        for (uint32_t i = 0; i < seq_len; i++) {
            uint32_t packed = my_seq[i];
            exec_instruction(s, (uint16_t)(packed & 0xFFFF), (uint16_t)(packed >> 16));
        }
        int off = v * FP_SIZE;
        my_fp[off+0] = s.r[REG_A]; my_fp[off+1] = s.r[REG_F];
        my_fp[off+2] = s.r[REG_B]; my_fp[off+3] = s.r[REG_C];
        my_fp[off+4] = s.r[REG_D]; my_fp[off+5] = s.r[REG_E];
        my_fp[off+6] = s.r[REG_H]; my_fp[off+7] = s.r[REG_L];
        my_fp[off+8] = (uint8_t)(s.sp >> 8); my_fp[off+9] = (uint8_t)s.sp;
    }

    uint8_t fm = (uint8_t)dead_flags;
    bool match = true;
    for (int i = 0; i < FP_LEN; i++) {
        uint8_t a = my_fp[i], b = target_fp[i];
        if ((i % FP_SIZE) == 1) { a &= ~fm; b &= ~fm; }
        if (a != b) { match = false; break; }
    }
    results[tid] = match ? 1u : 0u;
}

// ============================================================
// Opcode classification tables (mirrors Go inst package)
// ============================================================

// Immediate-8 opcodes (15 total: 7 LD + 8 ALU)
static bool is_imm8(uint16_t op) {
    // LD_r_N: 49-55
    if (op >= 49 && op <= 55) return true;
    // ALU immediate: index 7 within each 8-group (ADD_A_N=63, ADC_A_N=71, ..., CP_N=119)
    if (op >= 56 && op < 120 && ((op - 56) % 8) == 7) return true;
    return false;
}

// Immediate-16 opcodes (4 total: LD_BC_NN=382..LD_SP_NN=385)
static bool is_imm16(uint16_t op) {
    return op >= 382 && op <= 385;
}

// Byte size of instruction encoding (1, 2, or 3 bytes)
static int byte_size(uint16_t op) {
    // 2-byte CB prefix ops: 144-199, 200-367
    if (op >= 144 && op <= 367) return 2;
    // ED prefix: NEG(142), ADC/SBC HL (386-393)
    if (op == 142) return 2;
    if (op >= 386 && op <= 393) return 2;
    // LD rr,NN: 3 bytes
    if (op >= 382 && op <= 385) return 3;
    // Imm8 ops: 2 bytes
    if (is_imm8(op)) return 2;
    // Everything else: 1 byte
    return 1;
}

// T-states per opcode
static int tstates(uint16_t op) {
    if (op < 49) return 4;          // LD r,r'
    if (op < 56) return 7;          // LD r,N
    if (op < 120) {
        if ((op - 56) % 8 == 7) return 7; // ALU imm
        return 4;                    // ALU reg
    }
    if (op < 134) return 4;         // INC/DEC r
    if (op <= 141) return 4;        // RLCA..CCF
    if (op == 142) return 8;        // NEG
    if (op == 143) return 4;        // NOP
    if (op < 200) return 8;         // CB rotate/shift
    if (op < 368) return 8;         // BIT/RES/SET
    if (op < 376) return 6;         // INC/DEC rr
    if (op < 380) return 11;        // ADD HL,rr
    if (op == 380) return 4;        // EX DE,HL
    if (op == 381) return 6;        // LD SP,HL
    if (op < 386) return 10;        // LD rr,NN
    return 15;                       // ADC/SBC HL,rr
}

// ============================================================
// Disassembly (minimal, for JSONL output)
// ============================================================
static const char* reg_names[8] = {"A", "F", "B", "C", "D", "E", "H", "L"};
static const char* pair_names[4] = {"BC", "DE", "HL", "SP"};
static const char* alu_names[8] = {"ADD A,", "ADC A,", "SUB", "SBC A,", "AND", "XOR", "OR", "CP"};
static const char* cb_names[7] = {"RLC", "RRC", "RL", "RR", "SLA", "SRA", "SRL"};

static void disasm(uint16_t op, uint16_t imm, char* buf, int bufsz) {
    if (op < 49) { snprintf(buf, bufsz, "LD %s, %s", reg_names[LD_DST_H[op/7]], reg_names[LD_FULL_SRC_H[op]]); return; }
    if (op < 56) { snprintf(buf, bufsz, "LD %s, 0x%02X", reg_names[IMM_REG_H[op-49]], imm & 0xFF); return; }
    if (op < 120) {
        int a = (op-56)/8, s = (op-56)%8;
        if (s < 7) snprintf(buf, bufsz, "%s %s", alu_names[a], reg_names[ALU_SRC_H[s]]);
        else snprintf(buf, bufsz, "%s 0x%02X", alu_names[a], imm & 0xFF);
        return;
    }
    if (op < 127) { snprintf(buf, bufsz, "INC %s", reg_names[INCDEC_REG_H[op-120]]); return; }
    if (op < 134) { snprintf(buf, bufsz, "DEC %s", reg_names[INCDEC_REG_H[op-127]]); return; }
    if (op == 134) { snprintf(buf, bufsz, "RLCA"); return; }
    if (op == 135) { snprintf(buf, bufsz, "RRCA"); return; }
    if (op == 136) { snprintf(buf, bufsz, "RLA"); return; }
    if (op == 137) { snprintf(buf, bufsz, "RRA"); return; }
    if (op == 138) { snprintf(buf, bufsz, "DAA"); return; }
    if (op == 139) { snprintf(buf, bufsz, "CPL"); return; }
    if (op == 140) { snprintf(buf, bufsz, "SCF"); return; }
    if (op == 141) { snprintf(buf, bufsz, "CCF"); return; }
    if (op == 142) { snprintf(buf, bufsz, "NEG"); return; }
    if (op == 143) { snprintf(buf, bufsz, "NOP"); return; }
    if (op >= 144 && op <= 192) {
        int c = (op-144)/7, r = CB_REG_H[(op-144)%7];
        snprintf(buf, bufsz, "%s %s", cb_names[c], reg_names[r]); return;
    }
    if (op == 193) { snprintf(buf, bufsz, "SLL A"); return; }
    if (op >= 194 && op < 200) { snprintf(buf, bufsz, "SLL %s", reg_names[CB_REG_H[(op-194)+1]]); return; }
    if (op >= 200 && op < 256) { int i = op-200; snprintf(buf, bufsz, "BIT %d, %s", i/7, reg_names[CB_REG_H[i%7]]); return; }
    if (op >= 256 && op < 312) { int i = op-256; snprintf(buf, bufsz, "RES %d, %s", i/7, reg_names[CB_REG_H[i%7]]); return; }
    if (op >= 312 && op < 368) { int i = op-312; snprintf(buf, bufsz, "SET %d, %s", i/7, reg_names[CB_REG_H[i%7]]); return; }
    if (op >= 368 && op < 372) { snprintf(buf, bufsz, "INC %s", pair_names[op-368]); return; }
    if (op >= 372 && op < 376) { snprintf(buf, bufsz, "DEC %s", pair_names[op-372]); return; }
    if (op >= 376 && op < 380) { snprintf(buf, bufsz, "ADD HL, %s", pair_names[op-376]); return; }
    if (op == 380) { snprintf(buf, bufsz, "EX DE, HL"); return; }
    if (op == 381) { snprintf(buf, bufsz, "LD SP, HL"); return; }
    if (op >= 382 && op < 386) { snprintf(buf, bufsz, "LD %s, 0x%04X", pair_names[op-382], imm); return; }
    if (op >= 386 && op < 390) { snprintf(buf, bufsz, "ADC HL, %s", pair_names[op-386]); return; }
    if (op >= 390 && op < 394) { snprintf(buf, bufsz, "SBC HL, %s", pair_names[op-390]); return; }
    snprintf(buf, bufsz, "???(%d)", op);
}

// ============================================================
// opReads / opWrites — register dependency bitmasks
// ============================================================
#define RMASK_A  0x001u
#define RMASK_F  0x002u
#define RMASK_B  0x004u
#define RMASK_C  0x008u
#define RMASK_D  0x010u
#define RMASK_E  0x020u
#define RMASK_H  0x040u
#define RMASK_L  0x080u
#define RMASK_SP 0x100u

static uint16_t op_reads(uint16_t op) {
    if (op < 49) {
        // LD r,r': reads source register
        uint8_t src = LD_FULL_SRC_H[op];
        return 1u << src; // REG_A=0 → bit 0, etc.
    }
    if (op < 56) return 0; // LD r,N: reads nothing
    if (op < 120) {
        int alu_op = (op - 56) / 8, src_idx = (op - 56) % 8;
        uint16_t mask = 0;
        if (src_idx < 7) mask |= (1u << ALU_SRC_H[src_idx]);
        // ADC/SBC read carry flag
        if (alu_op == 1 || alu_op == 3) mask |= RMASK_F;
        // All ALU ops read A (for ADD/ADC/SUB/SBC/AND/XOR/OR/CP)
        // Actually: ADD A,B reads A and B; CP B reads A and B; etc.
        // But for immediate ops (src_idx==7), only A is read (plus the immediate)
        mask |= RMASK_A;
        return mask;
    }
    if (op < 127) return 1u << INCDEC_REG_H[op - 120];  // INC r reads r
    if (op < 134) return 1u << INCDEC_REG_H[op - 127];  // DEC r reads r
    // Accumulator rotates
    if (op == OP_RLCA || op == OP_RRCA) return RMASK_A;
    if (op == OP_RLA || op == OP_RRA) return RMASK_A | RMASK_F;
    if (op == OP_DAA) return RMASK_A | RMASK_F;
    if (op == OP_CPL || op == OP_NEG) return RMASK_A;
    if (op == OP_SCF || op == OP_CCF) return RMASK_F;
    if (op == OP_NOP) return 0;
    // CB prefix rotate/shift
    if (op >= 144 && op <= 192) {
        int cb_op = (op - 144) / 7;
        uint8_t reg = CB_REG_H[(op - 144) % 7];
        uint16_t mask = 1u << reg;
        if (cb_op == 2 || cb_op == 3) mask |= RMASK_F; // RL/RR read carry
        return mask;
    }
    if (op == OP_SLL_A) return RMASK_A;
    if (op >= 194 && op < 200) return 1u << CB_REG_H[(op - 194) + 1];
    // BIT/RES/SET read the register
    if (op >= 200 && op < 368) { int idx = (op >= 312) ? op - 312 : (op >= 256) ? op - 256 : op - 200; return 1u << CB_REG_H[idx % 7]; }
    // 16-bit INC/DEC
    if (op >= 368 && op < 376) { int pair = (op - 368) % 4; static const uint16_t pm[4] = {RMASK_B|RMASK_C, RMASK_D|RMASK_E, RMASK_H|RMASK_L, RMASK_SP}; return pm[pair]; }
    // ADD HL,rr
    if (op >= 376 && op < 380) { int pair = op - 376; static const uint16_t pm[4] = {RMASK_B|RMASK_C, RMASK_D|RMASK_E, RMASK_H|RMASK_L, RMASK_SP}; return RMASK_H | RMASK_L | pm[pair]; }
    if (op == 380) return RMASK_D | RMASK_E | RMASK_H | RMASK_L; // EX DE,HL
    if (op == 381) return RMASK_H | RMASK_L; // LD SP,HL
    if (op >= 382 && op < 386) return 0; // LD rr,NN
    // ADC/SBC HL,rr
    if (op >= 386 && op < 394) { int pair = (op >= 390) ? op - 390 : op - 386; static const uint16_t pm[4] = {RMASK_B|RMASK_C, RMASK_D|RMASK_E, RMASK_H|RMASK_L, RMASK_SP}; return RMASK_H | RMASK_L | RMASK_F | pm[pair]; }
    return 0;
}

static uint16_t op_writes(uint16_t op) {
    if (op < 49) return 1u << LD_DST_H[op / 7];  // LD r,r': writes dst
    if (op < 56) return 1u << IMM_REG_H[op - 49]; // LD r,N
    if (op < 120) {
        int alu_op = (op - 56) / 8;
        if (alu_op == 7) return RMASK_F; // CP only writes F
        return RMASK_A | RMASK_F;         // Other ALU write A+F
    }
    if (op < 134) return (1u << INCDEC_REG_H[op < 127 ? op - 120 : op - 127]) | RMASK_F;
    if (op >= 134 && op <= 137) return RMASK_A | RMASK_F; // RLCA..RRA
    if (op == OP_DAA || op == OP_CPL || op == OP_NEG) return RMASK_A | RMASK_F;
    if (op == OP_SCF || op == OP_CCF) return RMASK_F;
    if (op == OP_NOP) return 0;
    if (op >= 144 && op <= 199) { // CB rotate/shift + SLL
        uint8_t reg;
        if (op <= 192) reg = CB_REG_H[(op - 144) % 7];
        else if (op == 193) reg = REG_A;
        else reg = CB_REG_H[(op - 194) + 1];
        return (1u << reg) | RMASK_F;
    }
    if (op >= 200 && op < 256) return RMASK_F; // BIT writes only F
    if (op >= 256 && op < 368) { // RES/SET write only register
        int idx = (op >= 312) ? op - 312 : op - 256;
        return 1u << CB_REG_H[idx % 7];
    }
    if (op >= 368 && op < 376) { int pair = (op - 368) % 4; static const uint16_t pm[4] = {RMASK_B|RMASK_C, RMASK_D|RMASK_E, RMASK_H|RMASK_L, RMASK_SP}; return pm[pair]; }
    if (op >= 376 && op < 380) return RMASK_H | RMASK_L | RMASK_F;
    if (op == 380) return RMASK_D | RMASK_E | RMASK_H | RMASK_L;
    if (op == 381) return RMASK_SP;
    if (op >= 382 && op < 386) { int pair = op - 382; static const uint16_t pm[4] = {RMASK_B|RMASK_C, RMASK_D|RMASK_E, RMASK_H|RMASK_L, RMASK_SP}; return pm[pair]; }
    if (op >= 386 && op < 394) return RMASK_H | RMASK_L | RMASK_F;
    return 0;
}

// ============================================================
// Pruning (mirrors Go ShouldPrune)
// ============================================================
static bool is_self_load(uint16_t op) {
    // LD_A_A=6, LD_B_B=8, LD_C_C=16, LD_D_D=24, LD_E_E=32, LD_H_H=40, LD_L_L=48
    return op == 6 || op == 8 || op == 16 || op == 24 || op == 32 || op == 40 || op == 48;
}

static uint32_t inst_key(uint16_t op, uint16_t imm) {
    return ((uint32_t)op << 16) | imm;
}

static bool are_independent(uint16_t op1, uint16_t op2) {
    uint16_t aR = op_reads(op1), aW = op_writes(op1);
    uint16_t bR = op_reads(op2), bW = op_writes(op2);
    return (aW & bR) == 0 && (aR & bW) == 0 && (aW & bW) == 0;
}

static bool should_prune(const uint16_t* ops, const uint16_t* imms, int n) {
    for (int i = 0; i < n; i++) {
        if (ops[i] == OP_NOP) return true;
        if (is_self_load(ops[i])) return true;
        if (i + 1 < n) {
            // Dead write check
            uint16_t w1 = op_writes(ops[i]);
            if (w1) {
                uint16_t r2 = op_reads(ops[i+1]);
                uint16_t w2 = op_writes(ops[i+1]);
                uint16_t dead = w1 & w2 & ~RMASK_F & ~r2;
                if (dead) return true;
            }
        }
    }
    // Canonical ordering
    for (int i = 0; i + 1 < n; i++) {
        if (are_independent(ops[i], ops[i+1]) &&
            inst_key(ops[i], imms[i]) > inst_key(ops[i+1], imms[i+1]))
            return true;
    }
    return false;
}

// ============================================================
// ExhaustiveCheck (CPU, mirrors Go verifier.go)
// ============================================================
static const uint8_t rep_values[32] = {
    0x00, 0x01, 0x02, 0x0F, 0x10, 0x1F, 0x20, 0x3F,
    0x40, 0x55, 0x7E, 0x7F, 0x80, 0x81, 0xAA, 0xBF,
    0xC0, 0xD5, 0xE0, 0xEF, 0xF0, 0xF7, 0xFE, 0xFF,
    0x03, 0x07, 0x11, 0x33, 0x77, 0xBB, 0xDD, 0xEE,
};
static const uint16_t rep_sp[16] = {
    0x0000, 0x0001, 0x00FF, 0x0100, 0x7FFE, 0x7FFF, 0x8000, 0x8001,
    0xFFFE, 0xFFFF, 0x1234, 0x5678, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE,
};

static uint16_t regs_read(const uint16_t* ops, int n) {
    uint16_t mask = 0;
    for (int i = 0; i < n; i++) mask |= op_reads(ops[i]);
    return mask;
}

static void set_reg_by_offset(Z80State &s, int offset, uint8_t val) {
    switch (offset) {
        case 2: s.r[REG_B] = val; break; case 3: s.r[REG_C] = val; break;
        case 4: s.r[REG_D] = val; break; case 5: s.r[REG_E] = val; break;
        case 6: s.r[REG_H] = val; break; case 7: s.r[REG_L] = val; break;
    }
}

static bool exhaustive_check(
    const uint16_t* t_ops, const uint16_t* t_imms, int t_n,
    const uint16_t* c_ops, const uint16_t* c_imms, int c_n,
    uint8_t dead_flags
) {
    uint16_t reads = regs_read(t_ops, t_n) | regs_read(c_ops, c_n);

    // Collect extra registers
    int extra[6]; int nextra = 0;
    if (reads & RMASK_B) extra[nextra++] = 2;
    if (reads & RMASK_C) extra[nextra++] = 3;
    if (reads & RMASK_D) extra[nextra++] = 4;
    if (reads & RMASK_E) extra[nextra++] = 5;
    if (reads & RMASK_H) extra[nextra++] = 6;
    if (reads & RMASK_L) extra[nextra++] = 7;
    bool sweep_sp = (reads & RMASK_SP) != 0;

    // Full sweep for 0-2 extra regs (no SP)
    bool use_full = (nextra <= 2 && !sweep_sp);
    int nvals = use_full ? 256 : 32;
    const uint8_t* vals = use_full ? nullptr : rep_values;

    for (int a = 0; a < 256; a++) {
        for (int carry = 0; carry <= 1; carry++) {
            // For 0 extra regs
            if (nextra == 0 && !sweep_sp) {
                Z80State s = {}; s.r[REG_A] = (uint8_t)a; s.r[REG_F] = (uint8_t)carry;
                Z80State st = s, sc = s;
                h_exec_seq(st, t_ops, t_imms, t_n);
                h_exec_seq(sc, c_ops, c_imms, c_n);
                if (!h_states_equal(st, sc, dead_flags)) return false;
                continue;
            }
            // For 1 extra reg (full sweep)
            if (nextra == 1 && !sweep_sp) {
                for (int r = 0; r < 256; r++) {
                    Z80State s = {}; s.r[REG_A] = (uint8_t)a; s.r[REG_F] = (uint8_t)carry;
                    set_reg_by_offset(s, extra[0], (uint8_t)r);
                    Z80State st = s, sc = s;
                    h_exec_seq(st, t_ops, t_imms, t_n);
                    h_exec_seq(sc, c_ops, c_imms, c_n);
                    if (!h_states_equal(st, sc, dead_flags)) return false;
                }
                continue;
            }
            // For 2 extra regs (full sweep)
            if (nextra == 2 && !sweep_sp) {
                for (int r1 = 0; r1 < 256; r1++) {
                    for (int r2 = 0; r2 < 256; r2++) {
                        Z80State s = {}; s.r[REG_A] = (uint8_t)a; s.r[REG_F] = (uint8_t)carry;
                        set_reg_by_offset(s, extra[0], (uint8_t)r1);
                        set_reg_by_offset(s, extra[1], (uint8_t)r2);
                        Z80State st = s, sc = s;
                        h_exec_seq(st, t_ops, t_imms, t_n);
                        h_exec_seq(sc, c_ops, c_imms, c_n);
                        if (!h_states_equal(st, sc, dead_flags)) return false;
                    }
                }
                continue;
            }
            // Reduced sweep (3+ regs or SP)
            // Use recursive reduced sweep
            // For simplicity, just do nested loops with rep_values
            // This handles up to 6 extra regs + SP
            auto do_sweep = [&](auto& self, Z80State s, int ri) -> bool {
                if (ri >= nextra) {
                    if (sweep_sp) {
                        for (int si = 0; si < 16; si++) {
                            Z80State s2 = s; s2.sp = rep_sp[si];
                            Z80State st = s2, sc = s2;
                            h_exec_seq(st, t_ops, t_imms, t_n);
                            h_exec_seq(sc, c_ops, c_imms, c_n);
                            if (!h_states_equal(st, sc, dead_flags)) return false;
                        }
                        return true;
                    }
                    Z80State st = s, sc = s;
                    h_exec_seq(st, t_ops, t_imms, t_n);
                    h_exec_seq(sc, c_ops, c_imms, c_n);
                    return h_states_equal(st, sc, dead_flags);
                }
                for (int vi = 0; vi < 32; vi++) {
                    Z80State s2 = s;
                    set_reg_by_offset(s2, extra[ri], rep_values[vi]);
                    if (!self(self, s2, ri + 1)) return false;
                }
                return true;
            };
            Z80State base = {}; base.r[REG_A] = (uint8_t)a; base.r[REG_F] = (uint8_t)carry;
            if (!do_sweep(do_sweep, base, 0)) return false;
        }
    }
    return true;
}

// ============================================================
// Instruction enumeration
// ============================================================
struct Inst {
    uint16_t op;
    uint16_t imm;
};

// Build list of all 8-bit instructions (non-imm + imm8×256)
static std::vector<Inst> enumerate_instructions_8() {
    std::vector<Inst> result;
    for (uint16_t op = 0; op < OP_COUNT; op++) {
        if (is_imm16(op)) continue;
        if (is_imm8(op)) {
            for (int imm = 0; imm < 256; imm++)
                result.push_back({op, (uint16_t)imm});
        } else {
            result.push_back({op, 0});
        }
    }
    return result;
}

// ============================================================
// Main search
// ============================================================
int main(int argc, char** argv) {
    int max_target = 2;
    uint8_t dead_flags = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-target") == 0 && i+1 < argc) max_target = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dead-flags") == 0 && i+1 < argc) dead_flags = (uint8_t)strtoul(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: z80search [--max-target N] [--dead-flags 0xNN]\n");
            fprintf(stderr, "  Output: JSONL to stdout, progress to stderr\n");
            return 0;
        }
    }

    init_tables();
    upload_tables_cuda();

    // Build candidate list (all 8-bit instructions)
    std::vector<Inst> all_insts = enumerate_instructions_8();
    fprintf(stderr, "Instruction set: %zu instructions (8-bit)\n", all_insts.size());

    // Pack candidates for GPU upload: (op | imm<<16) per candidate
    std::vector<uint32_t> cand_packed(all_insts.size());
    for (size_t i = 0; i < all_insts.size(); i++)
        cand_packed[i] = (uint32_t)all_insts[i].op | ((uint32_t)all_insts[i].imm << 16);

    // Upload candidates to GPU
    uint32_t cand_count = (uint32_t)all_insts.size();
    uint32_t *d_candidates, *d_results;
    uint8_t *d_target_fp;
    cudaMalloc(&d_candidates, cand_count * sizeof(uint32_t));
    cudaMalloc(&d_results, cand_count * sizeof(uint32_t));
    cudaMalloc(&d_target_fp, FP_LEN);
    cudaMemcpy(d_candidates, cand_packed.data(), cand_count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t* h_results = (uint32_t*)malloc(cand_count * sizeof(uint32_t));
    uint32_t* matches = (uint32_t*)malloc(cand_count * sizeof(uint32_t));

    int blockSize = 256;
    int gridSize = (cand_count + blockSize - 1) / blockSize;

    uint64_t total_found = 0;
    uint64_t total_targets = 0;
    uint64_t total_gpu_hits = 0;
    uint64_t total_cpu_checks = 0;
    time_t start_time = time(NULL);

    fprintf(stderr, "Starting search: max_target=%d, dead_flags=0x%02X\n", max_target, dead_flags);

    for (int target_len = 2; target_len <= max_target; target_len++) {
        // For target_len N, candidates must have fewer bytes
        // We search length 1..target_len-1 candidates (currently just length 1)
        fprintf(stderr, "=== Target length %d ===\n", target_len);

        uint64_t targets_this_len = 0;
        uint64_t found_this_len = 0;
        time_t len_start = time(NULL);
        time_t last_report = len_start;

        // Enumerate all target sequences of this length
        // For length 2: nested loop over all_insts × all_insts
        // For length 3: triple nested (large!)
        // We'll use a flat array approach

        if (target_len == 2) {
            for (size_t i0 = 0; i0 < all_insts.size(); i0++) {
                // Progress every 10 seconds
                time_t now = time(NULL);
                if (now - last_report >= 10) {
                    last_report = now;
                    double pct = 100.0 * i0 / all_insts.size();
                    fprintf(stderr, "  [%.1f%%] %zu/%zu first ops | %lu targets | %lu GPU hits | %lu found | %lds\n",
                            pct, i0, all_insts.size(), (unsigned long)targets_this_len,
                            (unsigned long)total_gpu_hits, (unsigned long)total_found,
                            (long)(now - start_time));
                }

                for (size_t i1 = 0; i1 < all_insts.size(); i1++) {
                    uint16_t t_ops[2] = {all_insts[i0].op, all_insts[i1].op};
                    uint16_t t_imms[2] = {all_insts[i0].imm, all_insts[i1].imm};

                    if (should_prune(t_ops, t_imms, 2)) continue;
                    targets_this_len++;
                    total_targets++;

                    int target_bytes = byte_size(t_ops[0]) + byte_size(t_ops[1]);

                    // Compute target fingerprint
                    uint8_t fp[FP_LEN];
                    h_fingerprint(t_ops, t_imms, 2, fp);

                    // GPU QuickCheck
                    cudaMemcpy(d_target_fp, fp, FP_LEN, cudaMemcpyHostToDevice);
                    quickcheck_kernel<<<gridSize, blockSize>>>(d_candidates, d_target_fp, d_results, cand_count, 1, dead_flags);
                    cudaDeviceSynchronize();
                    cudaMemcpy(h_results, d_results, cand_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                    // Collect matches
                    uint32_t match_count = 0;
                    for (uint32_t k = 0; k < cand_count; k++)
                        if (h_results[k]) matches[match_count++] = k;

                    total_gpu_hits += match_count;

                    // CPU ExhaustiveCheck on hits
                    for (uint32_t mi = 0; mi < match_count; mi++) {
                        uint32_t ci = matches[mi];
                        uint16_t c_ops[1] = {all_insts[ci].op};
                        uint16_t c_imms[1] = {all_insts[ci].imm};

                        int cand_bytes = byte_size(c_ops[0]);
                        if (cand_bytes >= target_bytes) continue;
                        if (should_prune(c_ops, c_imms, 1)) continue;

                        total_cpu_checks++;

                        if (!exhaustive_check(t_ops, t_imms, 2, c_ops, c_imms, 1, dead_flags))
                            continue;

                        // Found! Output JSONL
                        total_found++;
                        found_this_len++;

                        char src_buf[128], repl_buf[64];
                        char s0[64], s1[64], r0[64];
                        disasm(t_ops[0], t_imms[0], s0, sizeof(s0));
                        disasm(t_ops[1], t_imms[1], s1, sizeof(s1));
                        disasm(c_ops[0], c_imms[0], r0, sizeof(r0));
                        snprintf(src_buf, sizeof(src_buf), "%s : %s", s0, s1);
                        snprintf(repl_buf, sizeof(repl_buf), "%s", r0);

                        int bytes_saved = target_bytes - cand_bytes;
                        int cycles_saved = (tstates(t_ops[0]) + tstates(t_ops[1])) - tstates(c_ops[0]);

                        printf("{\"source_asm\":\"%s\",\"replacement_asm\":\"%s\","
                               "\"source_bytes\":%d,\"replacement_bytes\":%d,"
                               "\"bytes_saved\":%d,\"cycles_saved\":%d",
                               src_buf, repl_buf, target_bytes, cand_bytes,
                               bytes_saved, cycles_saved);
                        if (dead_flags) printf(",\"dead_flags\":\"0x%02X\"", dead_flags);
                        printf("}\n");
                        fflush(stdout);
                    }
                }
            }
        } else if (target_len == 3) {
            for (size_t i0 = 0; i0 < all_insts.size(); i0++) {
                time_t now = time(NULL);
                if (now - last_report >= 10) {
                    last_report = now;
                    double pct = 100.0 * i0 / all_insts.size();
                    fprintf(stderr, "  [%.1f%%] %zu/%zu first ops | %lu targets | %lu found | %lds\n",
                            pct, i0, all_insts.size(), (unsigned long)targets_this_len,
                            (unsigned long)total_found, (long)(now - start_time));
                }
                for (size_t i1 = 0; i1 < all_insts.size(); i1++) {
                    for (size_t i2 = 0; i2 < all_insts.size(); i2++) {
                        uint16_t t_ops[3] = {all_insts[i0].op, all_insts[i1].op, all_insts[i2].op};
                        uint16_t t_imms[3] = {all_insts[i0].imm, all_insts[i1].imm, all_insts[i2].imm};

                        if (should_prune(t_ops, t_imms, 3)) continue;
                        targets_this_len++;
                        total_targets++;

                        int target_bytes = byte_size(t_ops[0]) + byte_size(t_ops[1]) + byte_size(t_ops[2]);

                        uint8_t fp[FP_LEN];
                        h_fingerprint(t_ops, t_imms, 3, fp);

                        // GPU QuickCheck against length-1 candidates
                        cudaMemcpy(d_target_fp, fp, FP_LEN, cudaMemcpyHostToDevice);
                        quickcheck_kernel<<<gridSize, blockSize>>>(d_candidates, d_target_fp, d_results, cand_count, 1, dead_flags);
                        cudaDeviceSynchronize();
                        cudaMemcpy(h_results, d_results, cand_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                        uint32_t match_count = 0;
                        for (uint32_t k = 0; k < cand_count; k++)
                            if (h_results[k]) matches[match_count++] = k;
                        total_gpu_hits += match_count;

                        for (uint32_t mi = 0; mi < match_count; mi++) {
                            uint32_t ci = matches[mi];
                            uint16_t c_ops[1] = {all_insts[ci].op};
                            uint16_t c_imms[1] = {all_insts[ci].imm};
                            int cand_bytes = byte_size(c_ops[0]);
                            if (cand_bytes >= target_bytes) continue;
                            if (should_prune(c_ops, c_imms, 1)) continue;
                            total_cpu_checks++;
                            if (!exhaustive_check(t_ops, t_imms, 3, c_ops, c_imms, 1, dead_flags)) continue;

                            total_found++; found_this_len++;
                            char s0[64], s1[64], s2[64], r0[64], src_buf[192];
                            disasm(t_ops[0], t_imms[0], s0, sizeof(s0));
                            disasm(t_ops[1], t_imms[1], s1, sizeof(s1));
                            disasm(t_ops[2], t_imms[2], s2, sizeof(s2));
                            disasm(c_ops[0], c_imms[0], r0, sizeof(r0));
                            snprintf(src_buf, sizeof(src_buf), "%s : %s : %s", s0, s1, s2);
                            int bytes_saved = target_bytes - cand_bytes;
                            int cycles_saved = (tstates(t_ops[0]) + tstates(t_ops[1]) + tstates(t_ops[2])) - tstates(c_ops[0]);
                            printf("{\"source_asm\":\"%s\",\"replacement_asm\":\"%s\","
                                   "\"source_bytes\":%d,\"replacement_bytes\":%d,"
                                   "\"bytes_saved\":%d,\"cycles_saved\":%d",
                                   src_buf, r0, target_bytes, cand_bytes, bytes_saved, cycles_saved);
                            if (dead_flags) printf(",\"dead_flags\":\"0x%02X\"", dead_flags);
                            printf("}\n"); fflush(stdout);
                        }
                    }
                }
            }
        }

        time_t len_end = time(NULL);
        fprintf(stderr, "  Length %d done: %lu targets, %lu found (%lds)\n",
                target_len, (unsigned long)targets_this_len, (unsigned long)found_this_len,
                (long)(len_end - len_start));
    }

    time_t end_time = time(NULL);
    fprintf(stderr, "\n=== DONE ===\n");
    fprintf(stderr, "Targets tested:  %lu\n", (unsigned long)total_targets);
    fprintf(stderr, "GPU QuickCheck:  %lu hits\n", (unsigned long)total_gpu_hits);
    fprintf(stderr, "CPU Exhaustive:  %lu checks\n", (unsigned long)total_cpu_checks);
    fprintf(stderr, "Results found:   %lu\n", (unsigned long)total_found);
    fprintf(stderr, "Total time:      %lds\n", (long)(end_time - start_time));

    free(h_results);
    free(matches);
    cudaFree(d_candidates);
    cudaFree(d_results);
    cudaFree(d_target_fp);
    return 0;
}

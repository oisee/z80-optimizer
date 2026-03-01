// Z80 QuickCheck GPU Compute Shader
// Tests candidate instruction sequences against a target's fingerprint using 8 test vectors.
// Each GPU thread tests ONE candidate instruction sequence.

// ============================================================================
// Flag constants (matching Go cpu.FlagX values)
// ============================================================================

const FLAG_C: u32 = 0x01u;  // Carry
const FLAG_N: u32 = 0x02u;  // Subtract
const FLAG_P: u32 = 0x04u;  // Parity/Overflow
const FLAG_V: u32 = 0x04u;  // Overflow (same bit as Parity)
const FLAG_3: u32 = 0x08u;  // Undocumented bit 3
const FLAG_H: u32 = 0x10u;  // Half-carry
const FLAG_5: u32 = 0x20u;  // Undocumented bit 5
const FLAG_Z: u32 = 0x40u;  // Zero
const FLAG_S: u32 = 0x80u;  // Sign

// ============================================================================
// Small lookup tables (constant arrays, 8 entries each)
// ============================================================================

// Note: gogpu/naga has a bug with const arrays (scalar width 0 in SPIR-V
// validation), so we use var<private> instead. These are per-invocation but
// the compiler should optimize them to constants.
var<private> HALFCARRY_ADD: array<u32, 8> = array<u32, 8>(
    0u, 0x10u, 0x10u, 0x10u, 0u, 0u, 0u, 0x10u  // FLAG_H = 0x10u
);
var<private> HALFCARRY_SUB: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0x10u, 0u, 0x10u, 0u, 0x10u, 0x10u
);
var<private> OVERFLOW_ADD: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0x04u, 0x04u, 0u, 0u, 0u  // FLAG_V = 0x04u
);
var<private> OVERFLOW_SUB: array<u32, 8> = array<u32, 8>(
    0u, 0x04u, 0u, 0u, 0u, 0u, 0x04u, 0u
);

// ============================================================================
// Z80 State struct
// ============================================================================

struct Z80State {
    a: u32,
    f: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    h: u32,
    l: u32,
    sp: u32,
}

// Per-invocation global state. Using var<private> instead of ptr<function>
// parameters because gogpu/naga generates invalid SPIR-V for function pointers.
var<private> state: Z80State;

// ============================================================================
// Params struct
// ============================================================================

struct Params {
    candidate_count: u32,
    seq_len: u32,
    num_candidates_per_pos: u32,
    dead_flags: u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lookup_tables: array<u32>;
@group(0) @binding(1) var<storage, read> candidates: array<u32>;
@group(0) @binding(2) var<storage, read> target_fp: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

// ============================================================================
// Lookup table accessors
// ============================================================================

// Read a single byte from a packed u32 table.
// table_offset is the u32 offset of the table start in lookup_tables.
// index is the byte index (0..255).
fn lut_byte(table_offset: u32, index: u32) -> u32 {
    let word = lookup_tables[table_offset + (index >> 2u)];
    return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

// sz53_table: offset 0, 64 u32s (256 bytes packed)
fn lut_sz53(index: u32) -> u32 {
    return lut_byte(0u, index);
}

// sz53p_table: offset 64, 64 u32s (256 bytes packed)
fn lut_sz53p(index: u32) -> u32 {
    return lut_byte(64u, index);
}

// parity_table: offset 128, 64 u32s (256 bytes packed)
fn lut_parity(index: u32) -> u32 {
    return lut_byte(128u, index);
}

// ============================================================================
// Test vector loading (8 hardcoded test vectors)
// ============================================================================

fn load_test_vector(tv: u32) -> Z80State {
    switch tv {
        case 0u {
            return Z80State(0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x0000u);
        }
        case 1u {
            return Z80State(0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFFFu);
        }
        case 2u {
            return Z80State(0x01u, 0x00u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x1234u);
        }
        case 3u {
            return Z80State(0x80u, 0x01u, 0x40u, 0x20u, 0x10u, 0x08u, 0x04u, 0x02u, 0x8000u);
        }
        case 4u {
            return Z80State(0x55u, 0x00u, 0xAAu, 0x55u, 0xAAu, 0x55u, 0xAAu, 0x55u, 0x5555u);
        }
        case 5u {
            return Z80State(0xAAu, 0x01u, 0x55u, 0xAAu, 0x55u, 0xAAu, 0x55u, 0xAAu, 0xAAAAu);
        }
        case 6u {
            return Z80State(0x0Fu, 0x00u, 0xF0u, 0x0Fu, 0xF0u, 0x0Fu, 0xF0u, 0x0Fu, 0xFFFEu);
        }
        case 7u {
            return Z80State(0x7Fu, 0x01u, 0x80u, 0x7Fu, 0x80u, 0x7Fu, 0x80u, 0x7Fu, 0x7FFFu);
        }
        default {
            return Z80State(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        }
    }
}

// ============================================================================
// Fingerprint comparison
// ============================================================================

// The target_fp buffer has 24 u32s (3 per test vector x 8 test vectors).
// Per test vector:
//   word0: A(byte3) | F(byte2) | B(byte1) | C(byte0)
//   word1: D(byte3) | E(byte2) | H(byte1) | L(byte0)
//   word2: SP_hi(byte3) | SP_lo(byte2) | 0 | 0
fn compare_state(state: Z80State, tv_idx: u32, dead_flags: u32) -> bool {
    let off = tv_idx * 3u;

    let w0 = (state.a << 24u) | (state.f << 16u) | (state.b << 8u) | state.c;
    let w1 = (state.d << 24u) | (state.e << 16u) | (state.h << 8u) | state.l;
    let w2 = ((state.sp >> 8u) << 24u) | ((state.sp & 0xFFu) << 16u);

    // Build mask for dead flags: F is in bits[23:16] of w0
    let f_mask = (0xFFu ^ dead_flags) << 16u;
    let w0_mask = 0xFF000000u | f_mask | 0x0000FFFFu;

    if ((w0 ^ target_fp[off]) & w0_mask) != 0u {
        return false;
    }
    if (w1 ^ target_fp[off + 1u]) != 0u {
        return false;
    }
    if ((w2 ^ target_fp[off + 2u]) & 0xFFFF0000u) != 0u {
        return false;
    }
    return true;
}

// ============================================================================
// ALU helper functions (ported exactly from Go cpu/exec.go)
// ============================================================================

// --- 8-bit ADD ---
fn exec_add(value: u32) {
    let a = state.a;
    let addtemp = a + value;
    let lookup = ((a & 0x88u) >> 3u) | ((value & 0x88u) >> 2u) | ((addtemp & 0x88u) >> 1u);
    state.a = addtemp & 0xFFu;
    state.f = select(0u, FLAG_C, (addtemp & 0x100u) != 0u) |
        HALFCARRY_ADD[lookup & 0x07u] |
        OVERFLOW_ADD[lookup >> 4u] |
        lut_sz53(state.a);
}

// --- 8-bit ADC ---
fn exec_adc(value: u32) {
    let a = state.a;
    let carry_in = state.f & FLAG_C;
    let adctemp = a + value + carry_in;
    let lookup = ((a & 0x88u) >> 3u) | ((value & 0x88u) >> 2u) | ((adctemp & 0x88u) >> 1u);
    state.a = adctemp & 0xFFu;
    state.f = select(0u, FLAG_C, (adctemp & 0x100u) != 0u) |
        HALFCARRY_ADD[lookup & 0x07u] |
        OVERFLOW_ADD[lookup >> 4u] |
        lut_sz53(state.a);
}

// --- 8-bit SUB ---
// In Go: subtemp := uint16(A) - uint16(value)
// In WGSL u32: wrapping subtraction preserves bit 8 semantics correctly
// because both operands are 0-255, so the u32 result & 0xFFFF matches Go uint16.
fn exec_sub(value: u32) {
    let a = state.a;
    let subtemp = (a - value) & 0xFFFFu;
    let result = subtemp & 0xFFu;
    let lookup = ((a & 0x88u) >> 3u) | ((value & 0x88u) >> 2u) | ((result & 0x88u) >> 1u);
    state.a = result;
    state.f = select(0u, FLAG_C, (subtemp & 0x100u) != 0u) | FLAG_N |
        HALFCARRY_SUB[lookup & 0x07u] |
        OVERFLOW_SUB[lookup >> 4u] |
        lut_sz53(result);
}

// --- 8-bit SBC ---
fn exec_sbc(value: u32) {
    let a = state.a;
    let carry_in = state.f & FLAG_C;
    let sbctemp = (a - value - carry_in) & 0xFFFFu;
    let result = sbctemp & 0xFFu;
    let lookup = ((a & 0x88u) >> 3u) | ((value & 0x88u) >> 2u) | ((result & 0x88u) >> 1u);
    state.a = result;
    state.f = select(0u, FLAG_C, (sbctemp & 0x100u) != 0u) | FLAG_N |
        HALFCARRY_SUB[lookup & 0x07u] |
        OVERFLOW_SUB[lookup >> 4u] |
        lut_sz53(result);
}

// --- 8-bit AND ---
fn exec_and(value: u32) {
    state.a &= value;
    state.f = FLAG_H | lut_sz53p(state.a);
}

// --- 8-bit OR ---
fn exec_or(value: u32) {
    state.a |= value;
    state.f = lut_sz53p(state.a);
}

// --- 8-bit XOR ---
fn exec_xor(value: u32) {
    state.a ^= value;
    state.f = lut_sz53p(state.a);
}

// --- 8-bit CP (compare, does not modify A) ---
// Go: s.F = bsel(cptemp&0x100!=0, FlagC, bsel(cptemp!=0, 0, FlagZ)) | FlagN |
//          HalfcarrySubTable[lookup&0x07] | OverflowSubTable[lookup>>4] |
//          (value & (Flag3|Flag5)) | uint8(cptemp & uint16(FlagS))
fn exec_cp(value: u32) {
    let a = state.a;
    let cptemp = (a - value) & 0xFFFFu;
    let result = cptemp & 0xFFu;
    let lookup = ((a & 0x88u) >> 3u) | ((value & 0x88u) >> 2u) | ((result & 0x88u) >> 1u);
    let carry = (cptemp & 0x100u) != 0u;
    let zero = cptemp == 0u;
    // Go: bsel(carry, FlagC, bsel(cptemp!=0, 0, FlagZ))
    // If carry set: base = FlagC. Else if cptemp==0: base = FlagZ. Else: base = 0.
    let base = select(select(0u, FLAG_Z, zero), FLAG_C, carry);
    state.f = base | FLAG_N |
        HALFCARRY_SUB[lookup & 0x07u] |
        OVERFLOW_SUB[lookup >> 4u] |
        (value & (FLAG_3 | FLAG_5)) |
        (result & FLAG_S);
}

// --- 8-bit INC (returns new value) ---
// Go: *reg++; F = (F & FlagC) | bsel(*reg==0x80, FlagV, 0) | bsel(*reg&0x0F!=0, 0, FlagH) | Sz53Table[*reg]
fn exec_inc(val: u32) -> u32 {
    let result = (val + 1u) & 0xFFu;
    state.f = (state.f & FLAG_C) |
        select(0u, FLAG_V, result == 0x80u) |
        select(0u, FLAG_H, (result & 0x0Fu) == 0u) |
        lut_sz53(result);
    return result;
}

// --- 8-bit DEC (returns new value) ---
// Go: F = (F & FlagC) | bsel(*reg&0x0F!=0, 0, FlagH) | FlagN; *reg--; F |= bsel(*reg==0x7F, FlagV, 0) | Sz53Table[*reg]
fn exec_dec(val: u32) -> u32 {
    state.f = (state.f & FLAG_C) | select(0u, FLAG_H, (val & 0x0Fu) == 0u) | FLAG_N;
    let result = (val - 1u) & 0xFFu;
    state.f |= select(0u, FLAG_V, result == 0x7Fu) | lut_sz53(result);
    return result;
}

// --- DAA ---
fn exec_daa() {
    var add_val: u32 = 0u;
    var carry: u32 = state.f & FLAG_C;
    if (state.f & FLAG_H) != 0u || (state.a & 0x0Fu) > 9u {
        add_val = 6u;
    }
    if carry != 0u || state.a > 0x99u {
        add_val |= 0x60u;
    }
    if state.a > 0x99u {
        carry = FLAG_C;
    }
    if (state.f & FLAG_N) != 0u {
        exec_sub(add_val);
    } else {
        exec_add(add_val);
    }
    // Go: s.F = (s.F & ^(FlagC | FlagP)) | carry | ParityTable[s.A]
    // ^(FlagC | FlagP) = ^0x05 = 0xFA in 8-bit
    state.f = (state.f & 0xFAu) | carry | lut_parity(state.a);
}

// --- CB-prefix RLC (rotate left circular) ---
fn exec_rlc(v: u32) -> u32 {
    let result = ((v << 1u) | (v >> 7u)) & 0xFFu;
    state.f = (result & FLAG_C) | lut_sz53p(result);
    return result;
}

// --- CB-prefix RRC (rotate right circular) ---
fn exec_rrc(v: u32) -> u32 {
    state.f = v & FLAG_C;
    let result = ((v >> 1u) | (v << 7u)) & 0xFFu;
    state.f |= lut_sz53p(result);
    return result;
}

// --- CB-prefix RL (rotate left through carry) ---
fn exec_rl(v: u32) -> u32 {
    let result = ((v << 1u) | (state.f & FLAG_C)) & 0xFFu;
    state.f = (v >> 7u) | lut_sz53p(result);
    return result;
}

// --- CB-prefix RR (rotate right through carry) ---
fn exec_rr(v: u32) -> u32 {
    let old_carry = v & FLAG_C;
    let result = ((v >> 1u) | (state.f << 7u)) & 0xFFu;
    state.f = old_carry | lut_sz53p(result);
    return result;
}

// --- CB-prefix SLA (shift left arithmetic) ---
fn exec_sla(v: u32) -> u32 {
    state.f = v >> 7u;
    let result = (v << 1u) & 0xFFu;
    state.f |= lut_sz53p(result);
    return result;
}

// --- CB-prefix SRA (shift right arithmetic, preserves sign bit) ---
fn exec_sra(v: u32) -> u32 {
    state.f = v & FLAG_C;
    let result = (v & 0x80u) | (v >> 1u);
    state.f |= lut_sz53p(result);
    return result;
}

// --- CB-prefix SRL (shift right logical) ---
fn exec_srl(v: u32) -> u32 {
    state.f = v & FLAG_C;
    let result = v >> 1u;
    state.f |= lut_sz53p(result);
    return result;
}

// --- CB-prefix SLL (undocumented: shift left, set bit 0) ---
fn exec_sll(v: u32) -> u32 {
    state.f = v >> 7u;
    let result = ((v << 1u) | 0x01u) & 0xFFu;
    state.f |= lut_sz53p(result);
    return result;
}

// --- BIT n, r (test bit, flags only) ---
// Go: F = (F & FlagC) | FlagH | (r & (Flag3|Flag5)); if r&(1<<bit)==0 { F |= FlagP|FlagZ }; if bit==7 && r&0x80!=0 { F |= FlagS }
fn exec_bit(r: u32, bit: u32) {
    state.f = (state.f & FLAG_C) | FLAG_H | (r & (FLAG_3 | FLAG_5));
    if (r & (1u << bit)) == 0u {
        state.f |= FLAG_P | FLAG_Z;
    }
    if bit == 7u && (r & 0x80u) != 0u {
        state.f |= FLAG_S;
    }
}

// --- ADD HL, rr (16-bit add, preserves S/Z/P flags) ---
fn exec_add_hl(value: u32) {
    let hl = (state.h << 8u) | state.l;
    let result = hl + value;
    let hc = (hl & 0x0FFFu) + (value & 0x0FFFu);
    state.f = (state.f & (FLAG_S | FLAG_Z | FLAG_P)) |
        select(0u, FLAG_H, (hc & 0x1000u) != 0u) |
        select(0u, FLAG_C, (result & 0x10000u) != 0u) |
        ((result >> 8u) & (FLAG_3 | FLAG_5));
    state.h = (result >> 8u) & 0xFFu;
    state.l = result & 0xFFu;
}

// --- ADC HL, rr (16-bit add with carry, full flag computation) ---
fn exec_adc_hl(value: u32) {
    let hl = (state.h << 8u) | state.l;
    let carry = state.f & FLAG_C;
    let result = hl + value + carry;
    let lookup = ((hl & 0x8800u) >> 11u) | ((value & 0x8800u) >> 10u) | ((result & 0x8800u) >> 9u);
    state.h = (result >> 8u) & 0xFFu;
    state.l = result & 0xFFu;
    state.f = select(0u, FLAG_C, (result & 0x10000u) != 0u) |
        OVERFLOW_ADD[lookup >> 4u] |
        (state.h & (FLAG_3 | FLAG_5 | FLAG_S)) |
        HALFCARRY_ADD[lookup & 0x07u] |
        select(0u, FLAG_Z, (state.h | state.l) == 0u);
}

// --- SBC HL, rr (16-bit subtract with carry, full flag computation) ---
// In Go: result := uint(hl) - uint(value) - carry (64-bit uint, no wrap issues)
// In WGSL u32: wrapping subtraction is fine; bit 16 check works identically.
fn exec_sbc_hl(value: u32) {
    let hl = (state.h << 8u) | state.l;
    let carry = state.f & FLAG_C;
    let result = hl - value - carry;
    let lookup = ((hl & 0x8800u) >> 11u) | ((value & 0x8800u) >> 10u) | ((result & 0x8800u) >> 9u);
    state.h = (result >> 8u) & 0xFFu;
    state.l = result & 0xFFu;
    state.f = select(0u, FLAG_C, (result & 0x10000u) != 0u) | FLAG_N |
        OVERFLOW_SUB[lookup >> 4u] |
        (state.h & (FLAG_3 | FLAG_5 | FLAG_S)) |
        HALFCARRY_SUB[lookup & 0x07u] |
        select(0u, FLAG_Z, (state.h | state.l) == 0u);
}

// ============================================================================
// Instruction execution (all 394 opcodes: 0-393)
// ============================================================================

fn exec_instruction(op: u32, imm: u32) {
    switch op {
        // ================================================================
        // 8-bit register loads (0-48)
        // ================================================================
        case 0u { state.a = state.b; }        // LD A, B
        case 1u { state.a = state.c; }        // LD A, C
        case 2u { state.a = state.d; }        // LD A, D
        case 3u { state.a = state.e; }        // LD A, E
        case 4u { state.a = state.h; }        // LD A, H
        case 5u { state.a = state.l; }        // LD A, L
        case 6u { }                                  // LD A, A (nop)
        case 7u { state.b = state.a; }        // LD B, A
        case 8u { }                                  // LD B, B (nop)
        case 9u { state.b = state.c; }        // LD B, C
        case 10u { state.b = state.d; }       // LD B, D
        case 11u { state.b = state.e; }       // LD B, E
        case 12u { state.b = state.h; }       // LD B, H
        case 13u { state.b = state.l; }       // LD B, L
        case 14u { state.c = state.a; }       // LD C, A
        case 15u { state.c = state.b; }       // LD C, B
        case 16u { }                                 // LD C, C (nop)
        case 17u { state.c = state.d; }       // LD C, D
        case 18u { state.c = state.e; }       // LD C, E
        case 19u { state.c = state.h; }       // LD C, H
        case 20u { state.c = state.l; }       // LD C, L
        case 21u { state.d = state.a; }       // LD D, A
        case 22u { state.d = state.b; }       // LD D, B
        case 23u { state.d = state.c; }       // LD D, C
        case 24u { }                                 // LD D, D (nop)
        case 25u { state.d = state.e; }       // LD D, E
        case 26u { state.d = state.h; }       // LD D, H
        case 27u { state.d = state.l; }       // LD D, L
        case 28u { state.e = state.a; }       // LD E, A
        case 29u { state.e = state.b; }       // LD E, B
        case 30u { state.e = state.c; }       // LD E, C
        case 31u { state.e = state.d; }       // LD E, D
        case 32u { }                                 // LD E, E (nop)
        case 33u { state.e = state.h; }       // LD E, H
        case 34u { state.e = state.l; }       // LD E, L
        case 35u { state.h = state.a; }       // LD H, A
        case 36u { state.h = state.b; }       // LD H, B
        case 37u { state.h = state.c; }       // LD H, C
        case 38u { state.h = state.d; }       // LD H, D
        case 39u { state.h = state.e; }       // LD H, E
        case 40u { }                                 // LD H, H (nop)
        case 41u { state.h = state.l; }       // LD H, L
        case 42u { state.l = state.a; }       // LD L, A
        case 43u { state.l = state.b; }       // LD L, B
        case 44u { state.l = state.c; }       // LD L, C
        case 45u { state.l = state.d; }       // LD L, D
        case 46u { state.l = state.e; }       // LD L, E
        case 47u { state.l = state.h; }       // LD L, H
        case 48u { }                                 // LD L, L (nop)

        // ================================================================
        // Immediate loads (49-55)
        // ================================================================
        case 49u { state.a = imm & 0xFFu; }     // LD A, n
        case 50u { state.b = imm & 0xFFu; }     // LD B, n
        case 51u { state.c = imm & 0xFFu; }     // LD C, n
        case 52u { state.d = imm & 0xFFu; }     // LD D, n
        case 53u { state.e = imm & 0xFFu; }     // LD E, n
        case 54u { state.h = imm & 0xFFu; }     // LD H, n
        case 55u { state.l = imm & 0xFFu; }     // LD L, n

        // ================================================================
        // ADD A, r (56-63)
        // ================================================================
        case 56u { exec_add(state.b); }
        case 57u { exec_add(state.c); }
        case 58u { exec_add(state.d); }
        case 59u { exec_add(state.e); }
        case 60u { exec_add(state.h); }
        case 61u { exec_add(state.l); }
        case 62u { exec_add(state.a); }
        case 63u { exec_add(imm & 0xFFu); }

        // ================================================================
        // ADC A, r (64-71)
        // ================================================================
        case 64u { exec_adc(state.b); }
        case 65u { exec_adc(state.c); }
        case 66u { exec_adc(state.d); }
        case 67u { exec_adc(state.e); }
        case 68u { exec_adc(state.h); }
        case 69u { exec_adc(state.l); }
        case 70u { exec_adc(state.a); }
        case 71u { exec_adc(imm & 0xFFu); }

        // ================================================================
        // SUB r (72-79)
        // ================================================================
        case 72u { exec_sub(state.b); }
        case 73u { exec_sub(state.c); }
        case 74u { exec_sub(state.d); }
        case 75u { exec_sub(state.e); }
        case 76u { exec_sub(state.h); }
        case 77u { exec_sub(state.l); }
        case 78u { exec_sub(state.a); }
        case 79u { exec_sub(imm & 0xFFu); }

        // ================================================================
        // SBC A, r (80-87)
        // ================================================================
        case 80u { exec_sbc(state.b); }
        case 81u { exec_sbc(state.c); }
        case 82u { exec_sbc(state.d); }
        case 83u { exec_sbc(state.e); }
        case 84u { exec_sbc(state.h); }
        case 85u { exec_sbc(state.l); }
        case 86u { exec_sbc(state.a); }
        case 87u { exec_sbc(imm & 0xFFu); }

        // ================================================================
        // AND r (88-95)
        // ================================================================
        case 88u { exec_and(state.b); }
        case 89u { exec_and(state.c); }
        case 90u { exec_and(state.d); }
        case 91u { exec_and(state.e); }
        case 92u { exec_and(state.h); }
        case 93u { exec_and(state.l); }
        case 94u { exec_and(state.a); }
        case 95u { exec_and(imm & 0xFFu); }

        // ================================================================
        // XOR r (96-103)
        // ================================================================
        case 96u { exec_xor(state.b); }
        case 97u { exec_xor(state.c); }
        case 98u { exec_xor(state.d); }
        case 99u { exec_xor(state.e); }
        case 100u { exec_xor(state.h); }
        case 101u { exec_xor(state.l); }
        case 102u { exec_xor(state.a); }
        case 103u { exec_xor(imm & 0xFFu); }

        // ================================================================
        // OR r (104-111)
        // ================================================================
        case 104u { exec_or(state.b); }
        case 105u { exec_or(state.c); }
        case 106u { exec_or(state.d); }
        case 107u { exec_or(state.e); }
        case 108u { exec_or(state.h); }
        case 109u { exec_or(state.l); }
        case 110u { exec_or(state.a); }
        case 111u { exec_or(imm & 0xFFu); }

        // ================================================================
        // CP r (112-119)
        // ================================================================
        case 112u { exec_cp(state.b); }
        case 113u { exec_cp(state.c); }
        case 114u { exec_cp(state.d); }
        case 115u { exec_cp(state.e); }
        case 116u { exec_cp(state.h); }
        case 117u { exec_cp(state.l); }
        case 118u { exec_cp(state.a); }
        case 119u { exec_cp(imm & 0xFFu); }

        // ================================================================
        // INC r (120-126)
        // ================================================================
        case 120u { state.a = exec_inc(state.a); }
        case 121u { state.b = exec_inc(state.b); }
        case 122u { state.c = exec_inc(state.c); }
        case 123u { state.d = exec_inc(state.d); }
        case 124u { state.e = exec_inc(state.e); }
        case 125u { state.h = exec_inc(state.h); }
        case 126u { state.l = exec_inc(state.l); }

        // ================================================================
        // DEC r (127-133)
        // ================================================================
        case 127u { state.a = exec_dec(state.a); }
        case 128u { state.b = exec_dec(state.b); }
        case 129u { state.c = exec_dec(state.c); }
        case 130u { state.d = exec_dec(state.d); }
        case 131u { state.e = exec_dec(state.e); }
        case 132u { state.h = exec_dec(state.h); }
        case 133u { state.l = exec_dec(state.l); }

        // ================================================================
        // Accumulator rotates (non-CB prefix, different flag behavior) (134-137)
        // ================================================================
        case 134u { // RLCA
            state.a = ((state.a << 1u) | (state.a >> 7u)) & 0xFFu;
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_C | FLAG_3 | FLAG_5));
        }
        case 135u { // RRCA
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & FLAG_C);
            state.a = ((state.a >> 1u) | (state.a << 7u)) & 0xFFu;
            state.f |= state.a & (FLAG_3 | FLAG_5);
        }
        case 136u { // RLA
            let old_a = state.a;
            state.a = ((state.a << 1u) | (state.f & FLAG_C)) & 0xFFu;
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_3 | FLAG_5)) | (old_a >> 7u);
        }
        case 137u { // RRA
            let old_a = state.a;
            state.a = ((state.a >> 1u) | (state.f << 7u)) & 0xFFu;
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_3 | FLAG_5)) | (old_a & FLAG_C);
        }

        // ================================================================
        // Special A operations (138-143)
        // ================================================================
        case 138u { // DAA
            exec_daa();
        }
        case 139u { // CPL
            state.a = state.a ^ 0xFFu;
            state.f = (state.f & (FLAG_C | FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_3 | FLAG_5)) | FLAG_N | FLAG_H;
        }
        case 140u { // SCF
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_3 | FLAG_5)) | FLAG_C;
        }
        case 141u { // CCF
            let old_c = state.f & FLAG_C;
            state.f = (state.f & (FLAG_P | FLAG_Z | FLAG_S)) | (state.a & (FLAG_3 | FLAG_5));
            if old_c != 0u {
                state.f |= FLAG_H;
            } else {
                state.f |= FLAG_C;
            }
        }
        case 142u { // NEG
            let old_a = state.a;
            state.a = 0u;
            exec_sub(old_a);
        }
        case 143u { } // NOP

        // ================================================================
        // CB-prefix: RLC r (144-150)
        // ================================================================
        case 144u { state.a = exec_rlc(state.a); }
        case 145u { state.b = exec_rlc(state.b); }
        case 146u { state.c = exec_rlc(state.c); }
        case 147u { state.d = exec_rlc(state.d); }
        case 148u { state.e = exec_rlc(state.e); }
        case 149u { state.h = exec_rlc(state.h); }
        case 150u { state.l = exec_rlc(state.l); }

        // ================================================================
        // CB-prefix: RRC r (151-157)
        // ================================================================
        case 151u { state.a = exec_rrc(state.a); }
        case 152u { state.b = exec_rrc(state.b); }
        case 153u { state.c = exec_rrc(state.c); }
        case 154u { state.d = exec_rrc(state.d); }
        case 155u { state.e = exec_rrc(state.e); }
        case 156u { state.h = exec_rrc(state.h); }
        case 157u { state.l = exec_rrc(state.l); }

        // ================================================================
        // CB-prefix: RL r (158-164)
        // ================================================================
        case 158u { state.a = exec_rl(state.a); }
        case 159u { state.b = exec_rl(state.b); }
        case 160u { state.c = exec_rl(state.c); }
        case 161u { state.d = exec_rl(state.d); }
        case 162u { state.e = exec_rl(state.e); }
        case 163u { state.h = exec_rl(state.h); }
        case 164u { state.l = exec_rl(state.l); }

        // ================================================================
        // CB-prefix: RR r (165-171)
        // ================================================================
        case 165u { state.a = exec_rr(state.a); }
        case 166u { state.b = exec_rr(state.b); }
        case 167u { state.c = exec_rr(state.c); }
        case 168u { state.d = exec_rr(state.d); }
        case 169u { state.e = exec_rr(state.e); }
        case 170u { state.h = exec_rr(state.h); }
        case 171u { state.l = exec_rr(state.l); }

        // ================================================================
        // CB-prefix: SLA r (172-178)
        // ================================================================
        case 172u { state.a = exec_sla(state.a); }
        case 173u { state.b = exec_sla(state.b); }
        case 174u { state.c = exec_sla(state.c); }
        case 175u { state.d = exec_sla(state.d); }
        case 176u { state.e = exec_sla(state.e); }
        case 177u { state.h = exec_sla(state.h); }
        case 178u { state.l = exec_sla(state.l); }

        // ================================================================
        // CB-prefix: SRA r (179-185)
        // ================================================================
        case 179u { state.a = exec_sra(state.a); }
        case 180u { state.b = exec_sra(state.b); }
        case 181u { state.c = exec_sra(state.c); }
        case 182u { state.d = exec_sra(state.d); }
        case 183u { state.e = exec_sra(state.e); }
        case 184u { state.h = exec_sra(state.h); }
        case 185u { state.l = exec_sra(state.l); }

        // ================================================================
        // CB-prefix: SRL r (186-192)
        // ================================================================
        case 186u { state.a = exec_srl(state.a); }
        case 187u { state.b = exec_srl(state.b); }
        case 188u { state.c = exec_srl(state.c); }
        case 189u { state.d = exec_srl(state.d); }
        case 190u { state.e = exec_srl(state.e); }
        case 191u { state.h = exec_srl(state.h); }
        case 192u { state.l = exec_srl(state.l); }

        // ================================================================
        // CB-prefix: SLL r (undocumented) (193-199)
        // ================================================================
        case 193u { state.a = exec_sll(state.a); }
        case 194u { state.b = exec_sll(state.b); }
        case 195u { state.c = exec_sll(state.c); }
        case 196u { state.d = exec_sll(state.d); }
        case 197u { state.e = exec_sll(state.e); }
        case 198u { state.h = exec_sll(state.h); }
        case 199u { state.l = exec_sll(state.l); }

        // ================================================================
        // BIT 0, r (200-206)
        // ================================================================
        case 200u { exec_bit(state.a, 0u); }
        case 201u { exec_bit(state.b, 0u); }
        case 202u { exec_bit(state.c, 0u); }
        case 203u { exec_bit(state.d, 0u); }
        case 204u { exec_bit(state.e, 0u); }
        case 205u { exec_bit(state.h, 0u); }
        case 206u { exec_bit(state.l, 0u); }

        // ================================================================
        // BIT 1, r (207-213)
        // ================================================================
        case 207u { exec_bit(state.a, 1u); }
        case 208u { exec_bit(state.b, 1u); }
        case 209u { exec_bit(state.c, 1u); }
        case 210u { exec_bit(state.d, 1u); }
        case 211u { exec_bit(state.e, 1u); }
        case 212u { exec_bit(state.h, 1u); }
        case 213u { exec_bit(state.l, 1u); }

        // ================================================================
        // BIT 2, r (214-220)
        // ================================================================
        case 214u { exec_bit(state.a, 2u); }
        case 215u { exec_bit(state.b, 2u); }
        case 216u { exec_bit(state.c, 2u); }
        case 217u { exec_bit(state.d, 2u); }
        case 218u { exec_bit(state.e, 2u); }
        case 219u { exec_bit(state.h, 2u); }
        case 220u { exec_bit(state.l, 2u); }

        // ================================================================
        // BIT 3, r (221-227)
        // ================================================================
        case 221u { exec_bit(state.a, 3u); }
        case 222u { exec_bit(state.b, 3u); }
        case 223u { exec_bit(state.c, 3u); }
        case 224u { exec_bit(state.d, 3u); }
        case 225u { exec_bit(state.e, 3u); }
        case 226u { exec_bit(state.h, 3u); }
        case 227u { exec_bit(state.l, 3u); }

        // ================================================================
        // BIT 4, r (228-234)
        // ================================================================
        case 228u { exec_bit(state.a, 4u); }
        case 229u { exec_bit(state.b, 4u); }
        case 230u { exec_bit(state.c, 4u); }
        case 231u { exec_bit(state.d, 4u); }
        case 232u { exec_bit(state.e, 4u); }
        case 233u { exec_bit(state.h, 4u); }
        case 234u { exec_bit(state.l, 4u); }

        // ================================================================
        // BIT 5, r (235-241)
        // ================================================================
        case 235u { exec_bit(state.a, 5u); }
        case 236u { exec_bit(state.b, 5u); }
        case 237u { exec_bit(state.c, 5u); }
        case 238u { exec_bit(state.d, 5u); }
        case 239u { exec_bit(state.e, 5u); }
        case 240u { exec_bit(state.h, 5u); }
        case 241u { exec_bit(state.l, 5u); }

        // ================================================================
        // BIT 6, r (242-248)
        // ================================================================
        case 242u { exec_bit(state.a, 6u); }
        case 243u { exec_bit(state.b, 6u); }
        case 244u { exec_bit(state.c, 6u); }
        case 245u { exec_bit(state.d, 6u); }
        case 246u { exec_bit(state.e, 6u); }
        case 247u { exec_bit(state.h, 6u); }
        case 248u { exec_bit(state.l, 6u); }

        // ================================================================
        // BIT 7, r (249-255)
        // ================================================================
        case 249u { exec_bit(state.a, 7u); }
        case 250u { exec_bit(state.b, 7u); }
        case 251u { exec_bit(state.c, 7u); }
        case 252u { exec_bit(state.d, 7u); }
        case 253u { exec_bit(state.e, 7u); }
        case 254u { exec_bit(state.h, 7u); }
        case 255u { exec_bit(state.l, 7u); }

        // ================================================================
        // RES 0, r (256-262)
        // ================================================================
        case 256u { state.a &= 0xFEu; }
        case 257u { state.b &= 0xFEu; }
        case 258u { state.c &= 0xFEu; }
        case 259u { state.d &= 0xFEu; }
        case 260u { state.e &= 0xFEu; }
        case 261u { state.h &= 0xFEu; }
        case 262u { state.l &= 0xFEu; }

        // ================================================================
        // RES 1, r (263-269)
        // ================================================================
        case 263u { state.a &= 0xFDu; }
        case 264u { state.b &= 0xFDu; }
        case 265u { state.c &= 0xFDu; }
        case 266u { state.d &= 0xFDu; }
        case 267u { state.e &= 0xFDu; }
        case 268u { state.h &= 0xFDu; }
        case 269u { state.l &= 0xFDu; }

        // ================================================================
        // RES 2, r (270-276)
        // ================================================================
        case 270u { state.a &= 0xFBu; }
        case 271u { state.b &= 0xFBu; }
        case 272u { state.c &= 0xFBu; }
        case 273u { state.d &= 0xFBu; }
        case 274u { state.e &= 0xFBu; }
        case 275u { state.h &= 0xFBu; }
        case 276u { state.l &= 0xFBu; }

        // ================================================================
        // RES 3, r (277-283)
        // ================================================================
        case 277u { state.a &= 0xF7u; }
        case 278u { state.b &= 0xF7u; }
        case 279u { state.c &= 0xF7u; }
        case 280u { state.d &= 0xF7u; }
        case 281u { state.e &= 0xF7u; }
        case 282u { state.h &= 0xF7u; }
        case 283u { state.l &= 0xF7u; }

        // ================================================================
        // RES 4, r (284-290)
        // ================================================================
        case 284u { state.a &= 0xEFu; }
        case 285u { state.b &= 0xEFu; }
        case 286u { state.c &= 0xEFu; }
        case 287u { state.d &= 0xEFu; }
        case 288u { state.e &= 0xEFu; }
        case 289u { state.h &= 0xEFu; }
        case 290u { state.l &= 0xEFu; }

        // ================================================================
        // RES 5, r (291-297)
        // ================================================================
        case 291u { state.a &= 0xDFu; }
        case 292u { state.b &= 0xDFu; }
        case 293u { state.c &= 0xDFu; }
        case 294u { state.d &= 0xDFu; }
        case 295u { state.e &= 0xDFu; }
        case 296u { state.h &= 0xDFu; }
        case 297u { state.l &= 0xDFu; }

        // ================================================================
        // RES 6, r (298-304)
        // ================================================================
        case 298u { state.a &= 0xBFu; }
        case 299u { state.b &= 0xBFu; }
        case 300u { state.c &= 0xBFu; }
        case 301u { state.d &= 0xBFu; }
        case 302u { state.e &= 0xBFu; }
        case 303u { state.h &= 0xBFu; }
        case 304u { state.l &= 0xBFu; }

        // ================================================================
        // RES 7, r (305-311)
        // ================================================================
        case 305u { state.a &= 0x7Fu; }
        case 306u { state.b &= 0x7Fu; }
        case 307u { state.c &= 0x7Fu; }
        case 308u { state.d &= 0x7Fu; }
        case 309u { state.e &= 0x7Fu; }
        case 310u { state.h &= 0x7Fu; }
        case 311u { state.l &= 0x7Fu; }

        // ================================================================
        // SET 0, r (312-318)
        // ================================================================
        case 312u { state.a |= 0x01u; }
        case 313u { state.b |= 0x01u; }
        case 314u { state.c |= 0x01u; }
        case 315u { state.d |= 0x01u; }
        case 316u { state.e |= 0x01u; }
        case 317u { state.h |= 0x01u; }
        case 318u { state.l |= 0x01u; }

        // ================================================================
        // SET 1, r (319-325)
        // ================================================================
        case 319u { state.a |= 0x02u; }
        case 320u { state.b |= 0x02u; }
        case 321u { state.c |= 0x02u; }
        case 322u { state.d |= 0x02u; }
        case 323u { state.e |= 0x02u; }
        case 324u { state.h |= 0x02u; }
        case 325u { state.l |= 0x02u; }

        // ================================================================
        // SET 2, r (326-332)
        // ================================================================
        case 326u { state.a |= 0x04u; }
        case 327u { state.b |= 0x04u; }
        case 328u { state.c |= 0x04u; }
        case 329u { state.d |= 0x04u; }
        case 330u { state.e |= 0x04u; }
        case 331u { state.h |= 0x04u; }
        case 332u { state.l |= 0x04u; }

        // ================================================================
        // SET 3, r (333-339)
        // ================================================================
        case 333u { state.a |= 0x08u; }
        case 334u { state.b |= 0x08u; }
        case 335u { state.c |= 0x08u; }
        case 336u { state.d |= 0x08u; }
        case 337u { state.e |= 0x08u; }
        case 338u { state.h |= 0x08u; }
        case 339u { state.l |= 0x08u; }

        // ================================================================
        // SET 4, r (340-346)
        // ================================================================
        case 340u { state.a |= 0x10u; }
        case 341u { state.b |= 0x10u; }
        case 342u { state.c |= 0x10u; }
        case 343u { state.d |= 0x10u; }
        case 344u { state.e |= 0x10u; }
        case 345u { state.h |= 0x10u; }
        case 346u { state.l |= 0x10u; }

        // ================================================================
        // SET 5, r (347-353)
        // ================================================================
        case 347u { state.a |= 0x20u; }
        case 348u { state.b |= 0x20u; }
        case 349u { state.c |= 0x20u; }
        case 350u { state.d |= 0x20u; }
        case 351u { state.e |= 0x20u; }
        case 352u { state.h |= 0x20u; }
        case 353u { state.l |= 0x20u; }

        // ================================================================
        // SET 6, r (354-360)
        // ================================================================
        case 354u { state.a |= 0x40u; }
        case 355u { state.b |= 0x40u; }
        case 356u { state.c |= 0x40u; }
        case 357u { state.d |= 0x40u; }
        case 358u { state.e |= 0x40u; }
        case 359u { state.h |= 0x40u; }
        case 360u { state.l |= 0x40u; }

        // ================================================================
        // SET 7, r (361-367)
        // ================================================================
        case 361u { state.a |= 0x80u; }
        case 362u { state.b |= 0x80u; }
        case 363u { state.c |= 0x80u; }
        case 364u { state.d |= 0x80u; }
        case 365u { state.e |= 0x80u; }
        case 366u { state.h |= 0x80u; }
        case 367u { state.l |= 0x80u; }

        // ================================================================
        // 16-bit INC (no flag changes) (368-371)
        // ================================================================
        case 368u { // INC BC
            let bc = (((state.b << 8u) | state.c) + 1u) & 0xFFFFu;
            state.b = (bc >> 8u) & 0xFFu;
            state.c = bc & 0xFFu;
        }
        case 369u { // INC DE
            let de = (((state.d << 8u) | state.e) + 1u) & 0xFFFFu;
            state.d = (de >> 8u) & 0xFFu;
            state.e = de & 0xFFu;
        }
        case 370u { // INC HL
            let hl = (((state.h << 8u) | state.l) + 1u) & 0xFFFFu;
            state.h = (hl >> 8u) & 0xFFu;
            state.l = hl & 0xFFu;
        }
        case 371u { // INC SP
            state.sp = (state.sp + 1u) & 0xFFFFu;
        }

        // ================================================================
        // 16-bit DEC (no flag changes) (372-375)
        // ================================================================
        case 372u { // DEC BC
            let bc = (((state.b << 8u) | state.c) - 1u) & 0xFFFFu;
            state.b = (bc >> 8u) & 0xFFu;
            state.c = bc & 0xFFu;
        }
        case 373u { // DEC DE
            let de = (((state.d << 8u) | state.e) - 1u) & 0xFFFFu;
            state.d = (de >> 8u) & 0xFFu;
            state.e = de & 0xFFu;
        }
        case 374u { // DEC HL
            let hl = (((state.h << 8u) | state.l) - 1u) & 0xFFFFu;
            state.h = (hl >> 8u) & 0xFFu;
            state.l = hl & 0xFFu;
        }
        case 375u { // DEC SP
            state.sp = (state.sp - 1u) & 0xFFFFu;
        }

        // ================================================================
        // ADD HL, rr (376-379)
        // ================================================================
        case 376u { exec_add_hl((state.b << 8u) | state.c); }   // ADD HL, BC
        case 377u { exec_add_hl((state.d << 8u) | state.e); }   // ADD HL, DE
        case 378u { exec_add_hl((state.h << 8u) | state.l); }   // ADD HL, HL
        case 379u { exec_add_hl(state.sp); }                        // ADD HL, SP

        // ================================================================
        // EX DE, HL (380)
        // ================================================================
        case 380u {
            let tmp_d = state.d;
            let tmp_e = state.e;
            state.d = state.h;
            state.e = state.l;
            state.h = tmp_d;
            state.l = tmp_e;
        }

        // ================================================================
        // LD SP, HL (381)
        // ================================================================
        case 381u {
            state.sp = (state.h << 8u) | state.l;
        }

        // ================================================================
        // LD rr, nn (382-385)
        // ================================================================
        case 382u { // LD BC, nn
            state.b = (imm >> 8u) & 0xFFu;
            state.c = imm & 0xFFu;
        }
        case 383u { // LD DE, nn
            state.d = (imm >> 8u) & 0xFFu;
            state.e = imm & 0xFFu;
        }
        case 384u { // LD HL, nn
            state.h = (imm >> 8u) & 0xFFu;
            state.l = imm & 0xFFu;
        }
        case 385u { // LD SP, nn
            state.sp = imm & 0xFFFFu;
        }

        // ================================================================
        // ADC HL, rr (386-389)
        // ================================================================
        case 386u { exec_adc_hl((state.b << 8u) | state.c); }   // ADC HL, BC
        case 387u { exec_adc_hl((state.d << 8u) | state.e); }   // ADC HL, DE
        case 388u { exec_adc_hl((state.h << 8u) | state.l); }   // ADC HL, HL
        case 389u { exec_adc_hl(state.sp); }                        // ADC HL, SP

        // ================================================================
        // SBC HL, rr (390-393)
        // ================================================================
        case 390u { exec_sbc_hl((state.b << 8u) | state.c); }   // SBC HL, BC
        case 391u { exec_sbc_hl((state.d << 8u) | state.e); }   // SBC HL, DE
        case 392u { exec_sbc_hl((state.h << 8u) | state.l); }   // SBC HL, HL
        case 393u { exec_sbc_hl(state.sp); }                        // SBC HL, SP

        default { } // Unknown opcode (should not happen)
    }
}

// ============================================================================
// Main compute shader entry point
// ============================================================================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.candidate_count {
        return;
    }

    // For each of the 8 test vectors
    for (var tv = 0u; tv < 8u; tv++) {
        state = load_test_vector(tv);

        // Execute candidate sequence
        if params.seq_len == 1u {
            let packed = candidates[idx];
            let op = packed >> 16u;
            let imm = packed & 0xFFFFu;
            exec_instruction(op, imm);
        } else {
            // Length 2: decode two instructions from flat index
            let i0 = idx / params.num_candidates_per_pos;
            let i1 = idx % params.num_candidates_per_pos;
            let packed0 = candidates[i0];
            let packed1 = candidates[i1];
            exec_instruction(packed0 >> 16u, packed0 & 0xFFFFu);
            exec_instruction(packed1 >> 16u, packed1 & 0xFFFFu);
        }

        // Compare with target fingerprint
        if !compare_state(state, tv, params.dead_flags) {
            return; // mismatch on this test vector
        }
    }

    // All 8 test vectors matched — mark this candidate as a hit.
    results[idx] = 1u;
}

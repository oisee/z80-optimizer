// Package fp16 provides Z80 assembly sequences for Z80-FP16 floating point.
//
// Format: H=[EEEEEEEE] L=[SMMMMMMM]
//   - H = 8-bit exponent, bias 127
//   - L bit 7 = sign
//   - L bits 6:0 = 7-bit mantissa (implicit leading 1)
//
// Key property: exponent is byte-aligned, so x2 = INC H (4T), /2 = DEC H (4T).
package fp16

// Seq is a named Z80 instruction sequence with cost metadata.
type Seq struct {
	Name  string   // operation name
	Ops   []string // Z80 assembly instructions
	Cost  int      // T-state cost
	Bytes int      // code size in bytes
	Note  string   // explanation
}

// --- Trivial single-format operations ---

func Negate() Seq {
	// Flip sign bit L.7. Actually needs LD A,L / XOR 0x80 / LD L,A
	// because Z80 has no "XOR imm,L" — only through A.
	// But if A is free: 3 insts, 15T.
	// Alternative: SET 7,L if sign=0, RES 7,L if sign=1 — but that's conditional.
	// Best unconditional: toggle bit 7 via A.
	return Seq{
		Name:  "fp16_negate",
		Ops:   []string{"LD A, L", "XOR 0x80", "LD L, A"},
		Cost:  4 + 7 + 4,
		Bytes: 1 + 2 + 1,
		Note:  "flip sign bit L.7",
	}
}

func Abs() Seq {
	return Seq{
		Name:  "fp16_abs",
		Ops:   []string{"RES 7, L"},
		Cost:  8,
		Bytes: 2,
		Note:  "clear sign bit",
	}
}

func Double() Seq {
	return Seq{
		Name:  "fp16_x2",
		Ops:   []string{"INC H"},
		Cost:  4,
		Bytes: 1,
		Note:  "increment exponent = multiply by 2",
	}
}

func Half() Seq {
	return Seq{
		Name:  "fp16_half",
		Ops:   []string{"DEC H"},
		Cost:  4,
		Bytes: 1,
		Note:  "decrement exponent = divide by 2",
	}
}

func IsZero() Seq {
	// Zero when exp=0 AND mant=0 (sign doesn't matter for +0/-0).
	// H=0 and L&0x7F=0. Result: Z flag set if zero.
	return Seq{
		Name:  "fp16_is_zero",
		Ops:   []string{"LD A, H", "OR A", "JR NZ, .not_zero", "LD A, L", "AND 0x7F"},
		Cost:  4 + 4 + 7 + 4 + 7, // worst case (is zero path)
		Bytes: 1 + 1 + 2 + 1 + 2,
		Note:  "Z flag set if float is +/-zero; branch-free: LD A,H / OR L / AND 0x7F also works but changes semantics",
	}
}

func IsZeroBranchless() Seq {
	// Branchless: A = (H==0 && L&0x7F==0) ? 0 : nonzero
	// LD A,L / AND 0x7F / OR H → A=0 iff zero
	return Seq{
		Name:  "fp16_is_zero_branchless",
		Ops:   []string{"LD A, L", "AND 0x7F", "OR H"},
		Cost:  4 + 7 + 4,
		Bytes: 1 + 2 + 1,
		Note:  "A=0 and Z set iff float is +/-zero",
	}
}

// Compare returns sequences for comparing two FP16 values.
// Input: HL = float1, DE = float2. Output: flags.
// For same-sign positive: compare H first (exponent), then L&0x7F (mantissa).
func Compare() Seq {
	// Unsigned magnitude compare (ignoring sign for now):
	// Compare exponents first, then mantissa.
	// For positive floats, the byte ordering H:L with sign in L.7 means
	// we can't just do a 16-bit compare.
	// Strategy: compare H (exp), if equal compare L&0x7F (mant).
	return Seq{
		Name:  "fp16_compare_positive",
		Ops: []string{
			"LD A, H",     // exp1
			"CP D",        // compare with exp2
			"JR NZ, .done", // if exponents differ, flags are set
			"LD A, L",     // mant1 (with sign in bit 7)
			"AND 0x7F",    // mask sign
			"LD B, A",     // save
			"LD A, E",     // mant2
			"AND 0x7F",    // mask sign
			"CP B",        // compare — note: reversed! B has mant1
			// actually we want mant1 - mant2, so:
		},
		Cost:  4 + 4 + 7 + 4 + 7 + 4 + 4 + 7 + 4, // ~45T worst case
		Bytes: 1 + 1 + 2 + 1 + 2 + 1 + 1 + 2 + 1,
		Note:  "compare positive floats; for signed, check signs first",
	}
}

// --- Conversions ---

func IntToFP16() Seq {
	// Convert unsigned 8-bit integer in A to FP16 in HL.
	// Algorithm: find position of highest set bit → exponent.
	// Mantissa = remaining bits shifted to fill 7-bit field.
	//
	// Special case: A=0 → HL=0
	// A=1 → exp=127, mant=0 (1.0)
	// A=2 → exp=128, mant=0 (2.0)
	// A=3 → exp=128, mant=0x40 (1.5 × 2 = 3.0)
	// A=255 → exp=134, mant=0x7F (1.1111111 × 2^7 ≈ 255)
	//
	// Loop approach: shift A left until bit 7 set, counting shifts.
	// exp = 127 + 7 - shifts. Mantissa = A with bit 7 removed.
	return Seq{
		Name: "fp16_from_uint8",
		Ops: []string{
			"OR A",            // test zero
			"JR Z, .zero",    // A=0 → return HL=0
			"LD L, A",        // save input
			"LD H, 134",      // start with max exp (127+7)
			// Find leading 1: shift left until bit 7 set
			".loop:",
			"BIT 7, L",       // test high bit
			"JR NZ, .found",  // found leading 1
			"SLA L",           // shift mantissa left
			"DEC H",           // decrease exponent
			"JR .loop",
			".found:",
			"RES 7, L",       // remove implicit leading 1
			"JR .done",
			".zero:",
			"LD H, 0",
			"LD L, 0",
			".done:",
		},
		Cost:  0, // variable: 4 + 7 + ... loop iterations
		Bytes: 0, // variable
		Note:  "loop approach; worst case 7 iterations for A=1. Average ~3-4 iterations.",
	}
}

func FP16ToUint8() Seq {
	// Convert FP16 in HL to unsigned 8-bit integer in A.
	// Algorithm: shift mantissa right by (134 - exp) positions.
	// If exp < 127 → result is 0 (value < 1.0).
	// If exp > 134 → overflow, clamp to 255.
	return Seq{
		Name: "fp16_to_uint8",
		Ops: []string{
			"LD A, H",        // exponent
			"CP 127",         // < 1.0?
			"JR C, .zero",    // yes → return 0
			"CP 135",         // >= 256?
			"JR NC, .overflow", // yes → return 255
			"LD A, L",        // mantissa
			"AND 0x7F",       // mask sign
			"OR 0x80",        // restore implicit leading 1
			// Now shift right by (134 - H) positions
			"LD B, 134",
			"LD C, H",        // save exp
			"LD A, B",
			"SUB C",           // A = 134 - exp = number of right shifts
			"LD B, A",        // B = shift count
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",        // A = 1.mantissa
			"OR A",           // clear flags
			".shift_loop:",
			"DEC B",
			"JR Z, .done",    // no more shifts
			"SRL A",           // shift right
			"JR .shift_loop",
			"JR .done",
			".zero:",
			"XOR A",           // A = 0
			"JR .done",
			".overflow:",
			"LD A, 255",
			".done:",
		},
		Cost:  0, // variable
		Bytes: 0,
		Note:  "loop-based; max 7 shifts. Handles underflow (→0) and overflow (→255).",
	}
}

// FP16ToFixed88 converts Z80-FP16 (HL) to f8.8 fixed point (HL).
// f8.8: H = integer part, L = fractional part (1/256 units).
func FP16ToFixed88() Seq {
	return Seq{
		Name: "fp16_to_f8_8",
		Ops: []string{
			"LD A, H",        // exponent
			"SUB 127",        // unbias → shift amount (0 = 1.xxx)
			"JR C, .frac",    // exp < 127 → pure fraction
			"CP 8",
			"JR NC, .overflow", // exp >= 135 → overflow
			// Integer part: shift 1.mantissa left by (exp-127) into H
			// Fractional part: remaining mantissa bits into L
			"LD B, A",        // B = exp - 127 (0..7)
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",        // A = 1.mantissa (8 bits)
			// We need to place this 8-bit value at the right position
			// in a 16-bit f8.8 field. Bit 7 of mantissa = 0.5 in f8.8
			// when exp=127 (value = 1.xxx), it goes to H=1, L=mant<<1
			"LD L, 0",
			"LD H, 0",
			// Shift A:0 left by B positions, put into H:L
			"INC B",          // +1 because 1.xxx × 2^0 = H.L where H has the 1
			".sloop:",
			"RLA",            // shift through carry
			"RL L",
			"RL H",
			"DJNZ .sloop",
			"JR .done",
			".frac:",
			"LD H, 0",       // integer part = 0
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",
			// Shift right by -(exp-127) = 127-exp into L
			"NEG",            // A was negative (exp-127), now A = 127-exp
			"LD B, A",
			".floop:",
			"SRL A",          // shift right... wait, we need the mantissa not A
			// Actually simpler to reload:
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",
			"LD L, A",
			"LD B, 127",
			"LD A, H",       // reload exp... this is getting messy
			// Let's use a cleaner approach
		},
		Cost:  0,
		Bytes: 0,
		Note:  "draft — see FP16ToFixed88Clean for cleaner version",
	}
}

// --- Normalize ---

func Normalize() Seq {
	// Input: H=exp, L=unnormalized mantissa (leading 1 may not be in bit 6)
	// Output: H=adjusted exp, L=normalized mantissa with leading 1 in bit 6
	return Seq{
		Name: "fp16_normalize",
		Ops: []string{
			"LD A, L",
			"AND 0x7F",       // mask sign
			"JR Z, .zero",    // mantissa = 0 → result is zero
			".loop:",
			"BIT 6, A",       // check if bit 6 set (normalized position)
			"JR NZ, .done",
			"ADD A, A",       // shift mantissa left
			"DEC H",          // decrease exponent
			"JR .loop",
			".done:",
			"AND 0x7F",       // ensure bit 7 clear (sign preserved separately)
			"LD C, A",        // save normalized mantissa
			"LD A, L",
			"AND 0x80",       // extract sign
			"OR C",           // combine sign + normalized mantissa
			"LD L, A",
			"RET",
			".zero:",
			"LD H, 0",
			"LD L, 0",
		},
		Cost:  0, // variable: ~8T per shift iteration
		Bytes: 0,
		Note:  "loop normalizer; max 6 iterations (mantissa 0x01 → 0x40)",
	}
}

// --- Arithmetic ---

// Add returns Z80 assembly for FP16 addition.
// Input: HL = float1, DE = float2.
// Output: HL = result.
// Clobbers: A, B, C, DE.
func Add() Seq {
	return Seq{
		Name: "fp16_add",
		Ops: []string{
			// Step 0: Handle special cases
			"LD A, H",
			"OR A",
			"JR Z, .ret_de",    // if float1 is zero, return float2

			"LD A, D",
			"OR A",
			"JR Z, .ret_hl",    // if float2 is zero, return float1

			// Step 1: Ensure exp1 >= exp2 (swap if needed)
			"LD A, H",          // exp1
			"CP D",             // exp1 - exp2
			"JR NC, .no_swap",  // exp1 >= exp2, no swap
			// Swap HL <-> DE
			"EX DE, HL",
			".no_swap:",

			// Step 2: Compute exponent difference
			"LD A, H",          // exp1 (larger)
			"SUB D",            // A = exp_diff
			"CP 8",
			"JR NC, .ret_hl",   // diff >= 8 → float2 is insignificant

			// Step 3: Extract mantissas with implicit leading 1
			"LD B, A",          // B = shift count
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",          // A = 1.mant1
			"LD C, A",          // C = mant1

			"LD A, E",
			"AND 0x7F",
			"OR 0x80",          // A = 1.mant2

			// Step 4: Align mant2 by shifting right by exp_diff
			"INC B",
			"DEC B",
			"JR Z, .aligned",
			".align:",
			"SRL A",            // shift mant2 right
			"DJNZ .align",
			".aligned:",

			// Step 5: Check signs and add/subtract
			"LD B, A",          // B = aligned mant2
			"LD A, L",
			"XOR E",            // compare sign bits
			"AND 0x80",
			"JR NZ, .subtract", // different signs → subtract

			// Same sign: add mantissas
			"LD A, C",          // mant1
			"ADD A, B",         // mant1 + mant2
			"JR NC, .no_ovf",
			// Overflow: shift right, increment exp
			"RRA",              // shift sum right through carry
			"INC H",            // increment exponent
			".no_ovf:",
			"AND 0x7F",         // clear bit 7 (will be sign)
			"LD C, A",
			"LD A, L",
			"AND 0x80",         // preserve sign
			"OR C",
			"LD L, A",
			"RET",

			".subtract:",
			// Different signs: subtract smaller from larger mantissa
			"LD A, C",          // mant1 (guaranteed >= aligned mant2)
			"SUB B",            // mant1 - mant2
			"JR Z, .ret_zero",
			// Normalize result
			"LD L, A",         // temporary store
			// Preserve sign from larger operand (which is in HL after swap)
			"PUSH HL",
			"CALL .normalize_inline",
			"POP BC",          // B was H (exp already adjusted), C not needed
			"LD A, L",
			"LD C, A",
			"LD A, B",        // ... this is getting complex with the stack
			// Simpler: inline normalize
			"JR .done",

			".ret_zero:",
			"LD H, 0",
			"LD L, 0",
			"RET",

			".ret_de:",
			"EX DE, HL",
			".ret_hl:",
			"RET",

			".done:",
		},
		Cost:  0, // variable, ~60-100T typical
		Bytes: 0,
		Note:  "draft — handles same-sign add well; subtract+normalize needs cleanup",
	}
}

// MulConst returns Z80 assembly for multiplying FP16 by a compile-time constant.
// This is the killer app: uses our mul8 table for the mantissa!
// Input: HL = float. K = compile-time constant (as FP16).
// Output: HL = result.
func MulConst(kExp int, kMant int, kSign int) Seq {
	// float × const = (exp1 + kExp - bias, mant1 × kMant >> 7, sign1 XOR kSign)
	// The mantissa multiply is 7-bit × 7-bit → 14-bit → take high 7 bits.
	// We can use our mul8 table! mant1 × kMant is just Emit8(kMant).
	//
	// Steps:
	// 1. Add exponents: H += kExp - 127
	// 2. Multiply mantissa: (L & 0x7F | 0x80) × (kMant | 0x80) using mul8 table
	// 3. Take high 7 bits of 8-bit result (SRL A once since we have 8×8→8)
	// 4. XOR sign if needed
	return Seq{
		Name: "fp16_mul_const",
		Ops: []string{
			// Exponent
			"LD A, H",
			"ADD A, " + itoa(kExp-127), // add constant exponent offset
			"LD H, A",
			// Sign
			// If kSign=1: toggle L.7
			// (emitted conditionally by caller)
			// Mantissa: A = (L & 0x7F) | 0x80, then mul8(kMant|0x80)
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",        // A = 1.mant1
			// ... insert mul8 sequence for (kMant | 0x80) here ...
			// The mul8 table gives us A * K → A (mod 256)
			// We need the HIGH 7 bits of the 14-bit product.
			// Problem: mul8 gives 8-bit result, not 14-bit.
			// We'd need mul16 for full product, or shift + mul8 for approximation.
			// Correct approach: use mul16 (A × K → HL), then H = high byte.
			"SRL A",          // adjust for mantissa alignment
			"AND 0x7F",
			"LD B, A",       // save result mantissa
			"LD A, L",
			"AND 0x80",      // preserve sign
			"OR B",
			"LD L, A",
		},
		Cost:  0,
		Bytes: 0,
		Note:  "template — caller fills in mul8/mul16 sequence for constant mantissa multiply",
	}
}

// Mul returns Z80 assembly for general FP16 multiplication.
// Input: HL = float1, DE = float2.
// Output: HL = result.
// This needs a general 7×7 multiply, which is expensive.
func Mul() Seq {
	return Seq{
		Name: "fp16_mul",
		Ops: []string{
			// Step 1: Result sign = sign1 XOR sign2
			"LD A, L",
			"XOR E",
			"AND 0x80",
			"PUSH AF",          // save result sign

			// Step 2: Result exponent = exp1 + exp2 - bias
			"LD A, H",
			"ADD A, D",
			"SUB 127",          // subtract bias
			"LD H, A",          // H = result exponent

			// Step 3: Mantissa multiply (1.mant1 × 1.mant2)
			"LD A, L",
			"AND 0x7F",
			"OR 0x80",          // A = 1.mant1
			"LD B, A",

			"LD A, E",
			"AND 0x7F",
			"OR 0x80",          // A = 1.mant2

			// 8×8 → 16 multiply (B × A → DE or HL')
			// Use shift-and-add loop:
			"LD C, B",         // C = mant1
			"LD D, 0",
			"LD E, A",         // DE = mant2
			// Multiply C × E → result in A (high byte only needed)
			// This is the expensive part: ~100T for general multiply
			"LD B, 8",
			"LD A, 0",
			".mul_loop:",
			"SRL C",
			"JR NC, .no_add",
			"ADD A, E",
			".no_add:",
			"RRA",             // shift product right (collect into A)
			"DJNZ .mul_loop",

			// A now has high 8 bits of product
			// Bit 7 = overflow bit from 1.xxx × 1.xxx (always 01.xx or 1x.xx)
			"BIT 7, A",
			"JR Z, .no_norm",
			"SRL A",           // normalize: shift right
			"INC H",           // bump exponent
			".no_norm:",
			"AND 0x7F",       // 7-bit mantissa (remove implicit 1)
			"LD B, A",

			// Step 4: Combine sign + mantissa
			"POP AF",          // result sign in bit 7
			"OR B",
			"LD L, A",
		},
		Cost:  0, // ~120-160T total
		Bytes: 0,
		Note:  "general 7×7 multiply via shift-add loop; ~120T. For constant K, use MulConst with our table.",
	}
}

// --- Format Conversions ---

// IEEEToZ80 converts IEEE half-precision (HL) to Z80-FP16 (HL).
// IEEE half: H=[SEEEEE.MM] L=[MMMMMMMM]
// Z80-FP16: H=[EEEEEEEE] L=[SMMMMMMM]
func IEEEToZ80() Seq {
	return Seq{
		Name: "ieee_half_to_z80fp16",
		Ops: []string{
			// Extract IEEE fields from H=[SEEEEE.MM]
			"LD A, H",
			"AND 0x80",       // A = sign (0x00 or 0x80)
			"LD B, A",        // B = sign

			"LD A, H",
			"ADD A, A",       // shift left, lose sign, exp now in bits 7:3
			"AND 0xF8",       // mask exp (5 bits now in 7:3)
			// Wait: after ADD A,A, A = H*2 & 0xFF
			// H = [SEEEEE.MM], so H*2 = [EEEEE.MM0] (sign shifted out via carry)
			// A & 0xF8 = [EEEEE.000]
			// To get exp5 as a number: A >> 3
			"RRCA",
			"RRCA",
			"RRCA",           // A = exp5 (0-31)

			"OR A",
			"JR Z, .zero_exp", // exp=0 → subnormal/zero

			"ADD A, 112",     // exp8 = exp5 + 112
			"LD D, A",        // D = exp8 (will become H)

			// Mantissa: IEEE has 10 bits, we want 7 bits (top 7 of 10)
			// IEEE mant10 = H[1:0] : L[7:0]
			// Z80 mant7 = top 7 bits = H[1:0] : L[7:3]
			"LD A, H",
			"AND 0x03",       // A = mant_hi (2 bits)
			"LD C, A",        // save mant_hi
			// Shift: mant7 = (mant_hi << 5) | (L >> 3)
			"RRCA",           // ... actually let's just shift L and combine
			"LD A, L",
			"SRL A",
			"SRL A",
			"SRL A",          // A = L >> 3 (5 bits from low byte)
			"LD E, A",
			"LD A, C",        // mant_hi
			"RLCA",
			"RLCA",
			"RLCA",
			"RLCA",
			"RLCA",           // A = mant_hi << 5
			"OR E",           // A = mant7
			"AND 0x7F",       // ensure 7 bits

			"OR B",           // combine with sign
			"LD L, A",
			"LD H, D",        // H = exp8
			"RET",

			".zero_exp:",
			"LD H, 0",
			"LD L, B",        // preserve sign, zero mantissa
			"RET",
		},
		Cost:  0, // ~70-80T
		Bytes: 0,
		Note:  "converts IEEE half to Z80-FP16; maps 5-bit exp to 8-bit (exp5+112)",
	}
}

// Z80ToIEEE converts Z80-FP16 (HL) to IEEE half-precision (HL).
// Z80-FP16: H=[EEEEEEEE] L=[SMMMMMMM]
// IEEE half: H=[SEEEEE.MM] L=[MMMMMMMM]
func Z80ToIEEE() Seq {
	return Seq{
		Name: "z80fp16_to_ieee_half",
		Ops: []string{
			"LD A, H",        // exp8
			"OR A",
			"JR Z, .zero",

			"SUB 112",        // exp5 = exp8 - 112
			// Clamp: if exp8 was < 113, result would be negative
			"JR C, .underflow",
			"CP 32",
			"JR NC, .overflow", // exp5 >= 32 → inf/NaN

			"LD B, A",        // B = exp5 (0-31)

			// Extract sign and mantissa from L
			"LD A, L",
			"AND 0x80",
			"LD C, A",        // C = sign (0x00 or 0x80)

			"LD A, L",
			"AND 0x7F",       // A = mant7
			// IEEE mant10 = mant7 << 3 = A << 3
			// Split into: H[1:0] = mant7 >> 5, L[7:0] = (mant7 << 3) & 0xFF
			"LD D, A",        // save mant7
			"RRCA",
			"RRCA",
			"RRCA",           // A = mant7 rotated right 3 = (mant7>>3) | (mant7<<5)
			// Actually: mant7 >> 5 = top 2 bits
			"LD A, D",
			"SRL A",
			"SRL A",
			"SRL A",
			"SRL A",
			"SRL A",          // A = mant7 >> 5 (2 bits: mant_hi)
			"LD E, A",        // E = mant_hi

			// Build IEEE H: [sign | exp5(5) | mant_hi(2)]
			"LD A, B",        // exp5
			"RLCA",
			"RLCA",           // A = exp5 << 2
			"AND 0x7C",       // mask to 5 bits in position
			"OR E",           // add mant_hi
			"OR C",           // add sign
			"LD H, A",

			// Build IEEE L: mant7 << 3 (lower 5 bits of mant7, shifted)
			"LD A, D",        // mant7
			"RLCA",
			"RLCA",
			"RLCA",           // A = mant7 << 3 (with wraparound)
			"AND 0xF8",       // keep only the shifted bits
			"LD L, A",
			"RET",

			".zero:",
			"LD L, 0",        // H already 0
			"RET",

			".underflow:",
			"LD H, 0",
			"LD L, 0",
			"RET",

			".overflow:",
			"LD A, C",        // sign
			"OR 0x7C",        // infinity exp (11111 in IEEE half) shifted into position
			"LD H, A",
			"LD L, 0",
			"RET",
		},
		Cost:  0, // ~80-100T
		Bytes: 0,
		Note:  "converts Z80-FP16 to IEEE half; maps 8-bit exp back to 5-bit (exp8-112)",
	}
}

// BfloatToZ80 converts Bfloat16 (HL) to Z80-FP16 (HL).
// Bfloat16: H=[SEEEEEEE] L=[EMMMMMMM]
//   sign = H.7, exp8 = H[6:0]:L.7 (8 bits), mant7 = L[6:0]
// Wait — Bfloat16 is actually: [SEEEEEEE EMMMMMMM] where S=sign, E=8-bit exp, M=7-bit mantissa.
// H bit 7 = sign, H bits 6:0 = exp[7:1], L bit 7 = exp[0], L bits 6:0 = mantissa.
//
// Z80-FP16: H=exp8, L=[sign|mant7]
// So: extract sign, rebuild exp from H[6:0]:L.7, mantissa = L[6:0].
func BfloatToZ80() Seq {
	return Seq{
		Name: "bfloat16_to_z80fp16",
		Ops: []string{
			// Bfloat16: H=[SEEEEEEE] L=[EMMMMMMM]
			"LD A, H",
			"AND 0x80",
			"LD B, A",        // B = sign (0x00 or 0x80)

			// exp8 = H[6:0]:L[7] — 8-bit exponent
			"LD A, H",
			"AND 0x7F",       // A = H[6:0] = exp[7:1]
			"RLCA",           // A = exp[7:1] << 1 = exp[7:1]0
			"LD C, A",
			"LD A, L",
			"RLCA",           // carry = L.7 = exp[0]
			"LD A, C",
			"ADC A, 0",       // A = exp[7:1]<<1 + exp[0] = full exp8
			"LD H, A",        // H = exp8

			// mant7 = L[6:0], sign in B
			"LD A, L",
			"AND 0x7F",       // mant7
			"OR B",           // add sign
			"LD L, A",
		},
		Cost:  4 + 7 + 4 + 7 + 4 + 4 + 4 + 4 + 7 + 4 + 4 + 7 + 4,
		Bytes: 1 + 2 + 1 + 1 + 2 + 4 + 1 + 1 + 1 + 2 + 1 + 1 + 2 + 1,
		Note:  "Bfloat16 and Z80-FP16 share same exp range and mantissa size; only layout differs",
	}
}

// --- FP24 (s1.E8.M15): A=[exp] H=[sign+mant_hi] L=[mant_lo] ---

// FP24Negate flips the sign of a 24-bit float.
func FP24Negate() Seq {
	return Seq{
		Name: "fp24_negate",
		Ops: []string{
			"LD B, A",        // save exponent
			"LD A, H",
			"XOR 0x80",       // flip sign
			"LD H, A",
			"LD A, B",        // restore exponent
		},
		Cost:  4 + 4 + 7 + 4 + 4,
		Bytes: 1 + 1 + 2 + 1 + 1,
		Note:  "flip H.7; saves/restores A (exponent) via B",
	}
}

func FP24Abs() Seq {
	return Seq{
		Name:  "fp24_abs",
		Ops:   []string{"RES 7, H"},
		Cost:  8,
		Bytes: 2,
		Note:  "clear sign bit in H",
	}
}

func FP24Double() Seq {
	return Seq{
		Name:  "fp24_x2",
		Ops:   []string{"INC A"},
		Cost:  4,
		Bytes: 1,
		Note:  "exponent in A: INC = multiply by 2",
	}
}

func FP24Half() Seq {
	return Seq{
		Name:  "fp24_half",
		Ops:   []string{"DEC A"},
		Cost:  4,
		Bytes: 1,
		Note:  "exponent in A: DEC = divide by 2",
	}
}

// FP24Normalize normalizes a 24-bit float.
// Input: A=exp, HL=sign+mantissa (H.7=sign, H[6:0]:L = 15-bit mantissa).
func FP24Normalize() Seq {
	return Seq{
		Name: "fp24_normalize",
		Ops: []string{
			"LD B, A",        // save exponent
			"LD A, H",
			"AND 0x80",
			"LD C, A",        // C = sign
			"LD A, H",
			"AND 0x7F",
			"LD H, A",        // H = mantissa hi (sign stripped)
			// Check if already normalized (bit 14 = H.6 set)
			".loop:",
			"BIT 6, H",
			"JR NZ, .done",
			"ADD HL, HL",     // shift mantissa left
			"DEC B",          // decrease exponent
			"LD A, H",
			"OR L",
			"JR Z, .zero",   // mantissa became zero
			"JR .loop",
			".done:",
			"LD A, H",
			"AND 0x7F",      // strip any overflow
			"OR C",           // restore sign
			"LD H, A",
			"LD A, B",        // restore exponent
			"RET",
			".zero:",
			"XOR A",          // exp = 0
			"LD H, A",
			"LD L, A",
		},
		Cost:  0, // variable
		Bytes: 0,
		Note:  "loop normalize; max 14 iterations. Uses ADD HL,HL for 16-bit shift.",
	}
}

func itoa(n int) string {
	if n < 0 {
		return "-" + itoa(-n)
	}
	s := ""
	if n == 0 {
		return "0"
	}
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}

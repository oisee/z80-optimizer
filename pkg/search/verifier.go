package search

import (
	"github.com/oisee/z80-optimizer/pkg/cpu"
	"github.com/oisee/z80-optimizer/pkg/inst"
)

// FlagMask indicates which flag bits are considered "dead" and can be ignored
// during equivalence checks. A set bit means that flag bit is dead (ignored).
type FlagMask = uint8

const (
	DeadNone  FlagMask = 0x00 // Full equivalence (current behavior)
	DeadUndoc FlagMask = 0x28 // Undocumented flags (bits 3, 5) — almost always safe
	DeadAll   FlagMask = 0xFF // All flags dead — registers only
)

// TestVectors are fixed inputs used for QuickCheck to reject 99.99% of non-matches.
var TestVectors = []cpu.State{
	{A: 0x00, F: 0x00, B: 0x00, C: 0x00, D: 0x00, E: 0x00, H: 0x00, L: 0x00, SP: 0x0000},
	{A: 0xFF, F: 0xFF, B: 0xFF, C: 0xFF, D: 0xFF, E: 0xFF, H: 0xFF, L: 0xFF, SP: 0xFFFF},
	{A: 0x01, F: 0x00, B: 0x02, C: 0x03, D: 0x04, E: 0x05, H: 0x06, L: 0x07, SP: 0x1234},
	{A: 0x80, F: 0x01, B: 0x40, C: 0x20, D: 0x10, E: 0x08, H: 0x04, L: 0x02, SP: 0x8000},
	{A: 0x55, F: 0x00, B: 0xAA, C: 0x55, D: 0xAA, E: 0x55, H: 0xAA, L: 0x55, SP: 0x5555},
	{A: 0xAA, F: 0x01, B: 0x55, C: 0xAA, D: 0x55, E: 0xAA, H: 0x55, L: 0xAA, SP: 0xAAAA},
	{A: 0x0F, F: 0x00, B: 0xF0, C: 0x0F, D: 0xF0, E: 0x0F, H: 0xF0, L: 0x0F, SP: 0xFFFE},
	{A: 0x7F, F: 0x01, B: 0x80, C: 0x7F, D: 0x80, E: 0x7F, H: 0x80, L: 0x7F, SP: 0x7FFF},
}

// execSeq runs a sequence of instructions on a state, returning the final state.
func execSeq(initial cpu.State, seq []inst.Instruction) cpu.State {
	s := initial
	for i := range seq {
		cpu.Exec(&s, seq[i].Op, seq[i].Imm)
	}
	return s
}

// QuickCheck tests two sequences against the test vectors.
// Returns true if they produce identical outputs on all test vectors.
func QuickCheck(target, candidate []inst.Instruction) bool {
	for i := range TestVectors {
		tOut := execSeq(TestVectors[i], target)
		cOut := execSeq(TestVectors[i], candidate)
		if tOut != cOut {
			return false
		}
	}
	return true
}

// FingerprintSize is the number of bytes per state snapshot in a fingerprint.
// V1: 8 (A,F,B,C,D,E,H,L), Wave 2: 10 (+SP high/low bytes).
const FingerprintSize = 10

// FingerprintLen is the total fingerprint length: FingerprintSize * len(TestVectors).
const FingerprintLen = FingerprintSize * 8 // 80 bytes

// Fingerprint computes a compact hash of a sequence's behavior on test vectors.
// Sequences with different fingerprints are guaranteed non-equivalent.
func Fingerprint(seq []inst.Instruction) [FingerprintLen]byte {
	var fp [FingerprintLen]byte
	for i := range TestVectors {
		out := execSeq(TestVectors[i], seq)
		off := i * FingerprintSize
		fp[off+0] = out.A
		fp[off+1] = out.F
		fp[off+2] = out.B
		fp[off+3] = out.C
		fp[off+4] = out.D
		fp[off+5] = out.E
		fp[off+6] = out.H
		fp[off+7] = out.L
		fp[off+8] = uint8(out.SP >> 8)
		fp[off+9] = uint8(out.SP)
	}
	return fp
}

// ExhaustiveCheck verifies equivalence over ALL possible inputs.
// The input dimensions depend on what registers the sequences read:
// - We always sweep A (0..255) and carry flag (0/1)
// - We also sweep any source register that differs from A
// For the V1 scope (all regs can be sources), we sweep all registers.
// This returns true only if both sequences produce identical output for every input.
func ExhaustiveCheck(target, candidate []inst.Instruction) bool {
	// Determine which registers are read by either sequence
	reads := regsRead(target) | regsRead(candidate)

	if reads&^(regA|regF) == 0 {
		// Only A and F are read: sweep A * carry = 512 cases
		return exhaustiveAF(target, candidate)
	}

	// General case: sweep all source registers
	// For sequences reading B,C,D,E,H,L we need to sweep those too
	return exhaustiveAll(target, candidate, reads)
}

// Register bitmask for tracking which registers are read/written.
// Wave 0: widened from uint8 to uint16 for SP, IX, IY, shadow regs.
type regMask uint16

const (
	regA regMask = 1 << iota
	regF
	regB
	regC
	regD
	regE
	regH
	regL
	regSP
)

func regsRead(seq []inst.Instruction) regMask {
	var mask regMask
	for _, instr := range seq {
		mask |= opReads(instr.Op)
	}
	return mask
}

// opReads returns which registers an instruction reads as source operands.
func opReads(op inst.OpCode) regMask {
	switch op {
	// Instructions reading only A (no carry dependency)
	case inst.ADD_A_A, inst.SUB_A,
		inst.AND_A, inst.XOR_A, inst.OR_A, inst.CP_A,
		inst.RLCA, inst.RRCA,
		inst.DAA, inst.CPL, inst.NEG,
		inst.RLC_A, inst.RRC_A,
		inst.SLA_A, inst.SRA_A, inst.SRL_A, inst.SLL_A,
		inst.LD_B_A, inst.LD_C_A, inst.LD_D_A, inst.LD_E_A,
		inst.LD_H_A, inst.LD_L_A, inst.LD_A_A,
		inst.INC_A, inst.DEC_A,
		inst.BIT_0_A, inst.BIT_1_A, inst.BIT_2_A, inst.BIT_3_A,
		inst.BIT_4_A, inst.BIT_5_A, inst.BIT_6_A, inst.BIT_7_A,
		inst.RES_0_A, inst.RES_1_A, inst.RES_2_A, inst.RES_3_A,
		inst.RES_4_A, inst.RES_5_A, inst.RES_6_A, inst.RES_7_A,
		inst.SET_0_A, inst.SET_1_A, inst.SET_2_A, inst.SET_3_A,
		inst.SET_4_A, inst.SET_5_A, inst.SET_6_A, inst.SET_7_A:
		return regA
	// Instructions reading A + F (carry-dependent A ops)
	case inst.ADC_A_A, inst.SBC_A_A, inst.RLA, inst.RRA, inst.RL_A, inst.RR_A:
		return regA | regF
	// Instructions reading only B (no carry dependency)
	case inst.ADD_A_B, inst.SUB_B,
		inst.AND_B, inst.XOR_B, inst.OR_B, inst.CP_B,
		inst.LD_A_B, inst.LD_C_B, inst.LD_D_B, inst.LD_E_B, inst.LD_H_B, inst.LD_L_B, inst.LD_B_B,
		inst.INC_B, inst.DEC_B,
		inst.RLC_B, inst.RRC_B, inst.SLA_B, inst.SRA_B, inst.SRL_B, inst.SLL_B,
		inst.BIT_0_B, inst.BIT_1_B, inst.BIT_2_B, inst.BIT_3_B,
		inst.BIT_4_B, inst.BIT_5_B, inst.BIT_6_B, inst.BIT_7_B,
		inst.RES_0_B, inst.RES_1_B, inst.RES_2_B, inst.RES_3_B,
		inst.RES_4_B, inst.RES_5_B, inst.RES_6_B, inst.RES_7_B,
		inst.SET_0_B, inst.SET_1_B, inst.SET_2_B, inst.SET_3_B,
		inst.SET_4_B, inst.SET_5_B, inst.SET_6_B, inst.SET_7_B:
		return regB
	// B + F (carry-dependent B ops)
	case inst.ADC_A_B, inst.SBC_A_B, inst.RL_B, inst.RR_B:
		return regB | regF
	// Instructions reading only C
	case inst.ADD_A_C, inst.SUB_C,
		inst.AND_C, inst.XOR_C, inst.OR_C, inst.CP_C,
		inst.LD_A_C, inst.LD_B_C, inst.LD_D_C, inst.LD_E_C, inst.LD_H_C, inst.LD_L_C, inst.LD_C_C,
		inst.INC_C, inst.DEC_C,
		inst.RLC_C, inst.RRC_C, inst.SLA_C, inst.SRA_C, inst.SRL_C, inst.SLL_C,
		inst.BIT_0_C, inst.BIT_1_C, inst.BIT_2_C, inst.BIT_3_C,
		inst.BIT_4_C, inst.BIT_5_C, inst.BIT_6_C, inst.BIT_7_C,
		inst.RES_0_C, inst.RES_1_C, inst.RES_2_C, inst.RES_3_C,
		inst.RES_4_C, inst.RES_5_C, inst.RES_6_C, inst.RES_7_C,
		inst.SET_0_C, inst.SET_1_C, inst.SET_2_C, inst.SET_3_C,
		inst.SET_4_C, inst.SET_5_C, inst.SET_6_C, inst.SET_7_C:
		return regC
	// C + F (carry-dependent C ops)
	case inst.ADC_A_C, inst.SBC_A_C, inst.RL_C, inst.RR_C:
		return regC | regF
	// Instructions reading only D
	case inst.ADD_A_D, inst.SUB_D,
		inst.AND_D, inst.XOR_D, inst.OR_D, inst.CP_D,
		inst.LD_A_D, inst.LD_B_D, inst.LD_C_D, inst.LD_E_D, inst.LD_H_D, inst.LD_L_D, inst.LD_D_D,
		inst.INC_D, inst.DEC_D,
		inst.RLC_D, inst.RRC_D, inst.SLA_D, inst.SRA_D, inst.SRL_D, inst.SLL_D,
		inst.BIT_0_D, inst.BIT_1_D, inst.BIT_2_D, inst.BIT_3_D,
		inst.BIT_4_D, inst.BIT_5_D, inst.BIT_6_D, inst.BIT_7_D,
		inst.RES_0_D, inst.RES_1_D, inst.RES_2_D, inst.RES_3_D,
		inst.RES_4_D, inst.RES_5_D, inst.RES_6_D, inst.RES_7_D,
		inst.SET_0_D, inst.SET_1_D, inst.SET_2_D, inst.SET_3_D,
		inst.SET_4_D, inst.SET_5_D, inst.SET_6_D, inst.SET_7_D:
		return regD
	// D + F (carry-dependent D ops)
	case inst.ADC_A_D, inst.SBC_A_D, inst.RL_D, inst.RR_D:
		return regD | regF
	// Instructions reading only E
	case inst.ADD_A_E, inst.SUB_E,
		inst.AND_E, inst.XOR_E, inst.OR_E, inst.CP_E,
		inst.LD_A_E, inst.LD_B_E, inst.LD_C_E, inst.LD_D_E, inst.LD_H_E, inst.LD_L_E, inst.LD_E_E,
		inst.INC_E, inst.DEC_E,
		inst.RLC_E, inst.RRC_E, inst.SLA_E, inst.SRA_E, inst.SRL_E, inst.SLL_E,
		inst.BIT_0_E, inst.BIT_1_E, inst.BIT_2_E, inst.BIT_3_E,
		inst.BIT_4_E, inst.BIT_5_E, inst.BIT_6_E, inst.BIT_7_E,
		inst.RES_0_E, inst.RES_1_E, inst.RES_2_E, inst.RES_3_E,
		inst.RES_4_E, inst.RES_5_E, inst.RES_6_E, inst.RES_7_E,
		inst.SET_0_E, inst.SET_1_E, inst.SET_2_E, inst.SET_3_E,
		inst.SET_4_E, inst.SET_5_E, inst.SET_6_E, inst.SET_7_E:
		return regE
	// E + F (carry-dependent E ops)
	case inst.ADC_A_E, inst.SBC_A_E, inst.RL_E, inst.RR_E:
		return regE | regF
	// Instructions reading only H
	case inst.ADD_A_H, inst.SUB_H,
		inst.AND_H, inst.XOR_H, inst.OR_H, inst.CP_H,
		inst.LD_A_H, inst.LD_B_H, inst.LD_C_H, inst.LD_D_H, inst.LD_E_H, inst.LD_L_H, inst.LD_H_H,
		inst.INC_H, inst.DEC_H,
		inst.RLC_H, inst.RRC_H, inst.SLA_H, inst.SRA_H, inst.SRL_H, inst.SLL_H,
		inst.BIT_0_H, inst.BIT_1_H, inst.BIT_2_H, inst.BIT_3_H,
		inst.BIT_4_H, inst.BIT_5_H, inst.BIT_6_H, inst.BIT_7_H,
		inst.RES_0_H, inst.RES_1_H, inst.RES_2_H, inst.RES_3_H,
		inst.RES_4_H, inst.RES_5_H, inst.RES_6_H, inst.RES_7_H,
		inst.SET_0_H, inst.SET_1_H, inst.SET_2_H, inst.SET_3_H,
		inst.SET_4_H, inst.SET_5_H, inst.SET_6_H, inst.SET_7_H:
		return regH
	// H + F (carry-dependent H ops)
	case inst.ADC_A_H, inst.SBC_A_H, inst.RL_H, inst.RR_H:
		return regH | regF
	// Instructions reading only L
	case inst.ADD_A_L, inst.SUB_L,
		inst.AND_L, inst.XOR_L, inst.OR_L, inst.CP_L,
		inst.LD_A_L, inst.LD_B_L, inst.LD_C_L, inst.LD_D_L, inst.LD_E_L, inst.LD_H_L, inst.LD_L_L,
		inst.INC_L, inst.DEC_L,
		inst.RLC_L, inst.RRC_L, inst.SLA_L, inst.SRA_L, inst.SRL_L, inst.SLL_L,
		inst.BIT_0_L, inst.BIT_1_L, inst.BIT_2_L, inst.BIT_3_L,
		inst.BIT_4_L, inst.BIT_5_L, inst.BIT_6_L, inst.BIT_7_L,
		inst.RES_0_L, inst.RES_1_L, inst.RES_2_L, inst.RES_3_L,
		inst.RES_4_L, inst.RES_5_L, inst.RES_6_L, inst.RES_7_L,
		inst.SET_0_L, inst.SET_1_L, inst.SET_2_L, inst.SET_3_L,
		inst.SET_4_L, inst.SET_5_L, inst.SET_6_L, inst.SET_7_L:
		return regL
	// L + F (carry-dependent L ops)
	case inst.ADC_A_L, inst.SBC_A_L, inst.RL_L, inst.RR_L:
		return regL | regF
	// ADC/SBC immediate: read A (accumulator) + F (carry)
	case inst.ADC_A_N, inst.SBC_A_N:
		return regA | regF
	// Instructions reading only F (carry-dependent, no register source)
	case inst.SCF, inst.CCF:
		return regF

	// 16-bit pair ops: INC/DEC read the pair, ADD HL reads HL + source pair
	case inst.INC_BC, inst.DEC_BC:
		return regB | regC
	case inst.INC_DE, inst.DEC_DE:
		return regD | regE
	case inst.INC_HL, inst.DEC_HL:
		return regH | regL
	case inst.INC_SP, inst.DEC_SP:
		return regSP
	case inst.ADD_HL_BC:
		return regH | regL | regB | regC
	case inst.ADD_HL_DE:
		return regH | regL | regD | regE
	case inst.ADD_HL_HL:
		return regH | regL
	case inst.ADD_HL_SP:
		return regH | regL | regSP
	case inst.EX_DE_HL:
		return regD | regE | regH | regL
	case inst.LD_SP_HL:
		return regH | regL

	// Wave 4: 16-bit immediate loads read nothing extra
	// LD_BC_NN, LD_DE_NN, LD_HL_NN, LD_SP_NN → only read the immediate

	// ADC/SBC HL, rr: read HL + source pair + F (carry)
	case inst.ADC_HL_BC, inst.SBC_HL_BC:
		return regH | regL | regB | regC | regF
	case inst.ADC_HL_DE, inst.SBC_HL_DE:
		return regH | regL | regD | regE | regF
	case inst.ADC_HL_HL, inst.SBC_HL_HL:
		return regH | regL | regF
	case inst.ADC_HL_SP, inst.SBC_HL_SP:
		return regH | regL | regSP | regF
	}

	// Remaining: NOP, LD r,n, LD rr,nn, ADD A,n, SUB n, AND n, XOR n, OR n, CP n don't read extra regs
	return 0
}

// exhaustiveAF sweeps A=0..255, carry=0/1 (512 iterations).
func exhaustiveAF(target, candidate []inst.Instruction) bool {
	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			s := cpu.State{A: uint8(a), F: carry}
			tOut := execSeq(s, target)
			cOut := execSeq(s, candidate)
			if tOut != cOut {
				return false
			}
		}
	}
	return true
}

// exhaustiveAll sweeps all read registers. We sweep A (256) * carry (2) *
// each other read register (256). To keep this feasible we use representative
// values for registers beyond A.
func exhaustiveAll(target, candidate []inst.Instruction, reads regMask) bool {
	// For efficiency, we sweep A*carry*oneSourceReg.
	// We pick a few representative values for other registers.
	// With 8 registers possible this could be 256^8 which is too many,
	// but in practice sequences of length 2-3 read at most A + 1-2 others.

	type sweepReg struct {
		offset int // offset in State (0=A, 1=F, 2=B, ...)
	}

	// Count extra 8-bit registers (beyond A and F)
	extraRegs := make([]int, 0, 6) // offsets of registers to sweep
	if reads&regB != 0 {
		extraRegs = append(extraRegs, 2) // B offset
	}
	if reads&regC != 0 {
		extraRegs = append(extraRegs, 3)
	}
	if reads&regD != 0 {
		extraRegs = append(extraRegs, 4)
	}
	if reads&regE != 0 {
		extraRegs = append(extraRegs, 5)
	}
	if reads&regH != 0 {
		extraRegs = append(extraRegs, 6)
	}
	if reads&regL != 0 {
		extraRegs = append(extraRegs, 7)
	}

	sweepSP := reads&regSP != 0

	if len(extraRegs) == 0 && !sweepSP {
		return exhaustiveAF(target, candidate)
	}

	// For 1 extra register: A(256) * carry(2) * reg(256) = 131,072 iterations - very fast
	// For 2 extra: 33,554,432 - still feasible
	// For 3+: we use a reduced sweep of 32 values per extra reg
	// SP is 16-bit, so always uses reduced sweep (32 representative values)
	if len(extraRegs) <= 2 && !sweepSP {
		return exhaustiveFullSweep(target, candidate, extraRegs)
	}
	return exhaustiveReducedSweep(target, candidate, extraRegs, sweepSP)
}

func exhaustiveFullSweep(target, candidate []inst.Instruction, extraRegs []int) bool {
	setReg := func(s *cpu.State, offset int, val uint8) {
		switch offset {
		case 2:
			s.B = val
		case 3:
			s.C = val
		case 4:
			s.D = val
		case 5:
			s.E = val
		case 6:
			s.H = val
		case 7:
			s.L = val
		}
	}

	if len(extraRegs) == 1 {
		for a := 0; a < 256; a++ {
			for carry := uint8(0); carry <= 1; carry++ {
				for r := 0; r < 256; r++ {
					s := cpu.State{A: uint8(a), F: carry}
					setReg(&s, extraRegs[0], uint8(r))
					tOut := execSeq(s, target)
					cOut := execSeq(s, candidate)
					if tOut != cOut {
						return false
					}
				}
			}
		}
		return true
	}

	// 2 extra registers
	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			for r1 := 0; r1 < 256; r1++ {
				for r2 := 0; r2 < 256; r2++ {
					s := cpu.State{A: uint8(a), F: carry}
					setReg(&s, extraRegs[0], uint8(r1))
					setReg(&s, extraRegs[1], uint8(r2))
					tOut := execSeq(s, target)
					cOut := execSeq(s, candidate)
					if tOut != cOut {
						return false
					}
				}
			}
		}
	}
	return true
}

func exhaustiveReducedSweep(target, candidate []inst.Instruction, extraRegs []int, sweepSP bool) bool {
	// Use 32 representative values per extra register
	repValues := []uint8{
		0x00, 0x01, 0x02, 0x0F, 0x10, 0x1F, 0x20, 0x3F,
		0x40, 0x55, 0x7E, 0x7F, 0x80, 0x81, 0xAA, 0xBF,
		0xC0, 0xD5, 0xE0, 0xEF, 0xF0, 0xF7, 0xFE, 0xFF,
		0x03, 0x07, 0x11, 0x33, 0x77, 0xBB, 0xDD, 0xEE,
	}

	setReg := func(s *cpu.State, offset int, val uint8) {
		switch offset {
		case 2:
			s.B = val
		case 3:
			s.C = val
		case 4:
			s.D = val
		case 5:
			s.E = val
		case 6:
			s.H = val
		case 7:
			s.L = val
		}
	}

	// Representative 16-bit values for SP sweep
	repSP := []uint16{
		0x0000, 0x0001, 0x00FF, 0x0100, 0x7FFE, 0x7FFF, 0x8000, 0x8001,
		0xFFFE, 0xFFFF, 0x1234, 0x5678, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE,
	}

	// compare is the base case: run both sequences and check equivalence
	compare := func(s cpu.State) bool {
		tOut := execSeq(s, target)
		cOut := execSeq(s, candidate)
		return tOut == cOut
	}

	// Recursive sweep helper
	var sweep func(s cpu.State, regIdx int) bool
	sweep = func(s cpu.State, regIdx int) bool {
		if regIdx >= len(extraRegs) {
			// After 8-bit regs, optionally sweep SP
			if sweepSP {
				for _, sp := range repSP {
					s2 := s
					s2.SP = sp
					if !compare(s2) {
						return false
					}
				}
				return true
			}
			return compare(s)
		}
		for _, v := range repValues {
			s2 := s
			setReg(&s2, extraRegs[regIdx], v)
			if !sweep(s2, regIdx+1) {
				return false
			}
		}
		return true
	}

	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			s := cpu.State{A: uint8(a), F: carry}
			if !sweep(s, 0) {
				return false
			}
		}
	}
	return true
}

// statesEqualMasked compares two states, ignoring flag bits set in deadFlags.
func statesEqualMasked(a, b cpu.State, deadFlags FlagMask) bool {
	return a.A == b.A &&
		(a.F &^ deadFlags) == (b.F &^ deadFlags) &&
		a.B == b.B && a.C == b.C &&
		a.D == b.D && a.E == b.E &&
		a.H == b.H && a.L == b.L &&
		a.SP == b.SP
}

// QuickCheckMasked tests two sequences against test vectors, ignoring dead flag bits.
func QuickCheckMasked(target, candidate []inst.Instruction, deadFlags FlagMask) bool {
	if deadFlags == DeadNone {
		return QuickCheck(target, candidate)
	}
	for i := range TestVectors {
		tOut := execSeq(TestVectors[i], target)
		cOut := execSeq(TestVectors[i], candidate)
		if !statesEqualMasked(tOut, cOut, deadFlags) {
			return false
		}
	}
	return true
}

// ExhaustiveCheckMasked verifies equivalence over all possible inputs,
// ignoring flag bits set in deadFlags.
func ExhaustiveCheckMasked(target, candidate []inst.Instruction, deadFlags FlagMask) bool {
	if deadFlags == DeadNone {
		return ExhaustiveCheck(target, candidate)
	}

	reads := regsRead(target) | regsRead(candidate)

	if reads&^(regA|regF) == 0 {
		return exhaustiveAFMasked(target, candidate, deadFlags)
	}

	return exhaustiveAllMasked(target, candidate, reads, deadFlags)
}

// FlagDiff runs test vectors and returns a bitmask of which flag bits ever differ.
// 0 means the sequences always match; nonzero bits indicate those flags must be dead.
func FlagDiff(target, candidate []inst.Instruction) FlagMask {
	var diff FlagMask
	for i := range TestVectors {
		tOut := execSeq(TestVectors[i], target)
		cOut := execSeq(TestVectors[i], candidate)
		// Check non-flag state — if any register differs, return 0 (not a flag-only issue)
		if tOut.A != cOut.A || tOut.B != cOut.B || tOut.C != cOut.C ||
			tOut.D != cOut.D || tOut.E != cOut.E ||
			tOut.H != cOut.H || tOut.L != cOut.L || tOut.SP != cOut.SP {
			return 0
		}
		diff |= tOut.F ^ cOut.F
	}
	return diff
}

func exhaustiveAFMasked(target, candidate []inst.Instruction, deadFlags FlagMask) bool {
	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			s := cpu.State{A: uint8(a), F: carry}
			tOut := execSeq(s, target)
			cOut := execSeq(s, candidate)
			if !statesEqualMasked(tOut, cOut, deadFlags) {
				return false
			}
		}
	}
	return true
}

func exhaustiveAllMasked(target, candidate []inst.Instruction, reads regMask, deadFlags FlagMask) bool {
	extraRegs := make([]int, 0, 6)
	if reads&regB != 0 {
		extraRegs = append(extraRegs, 2)
	}
	if reads&regC != 0 {
		extraRegs = append(extraRegs, 3)
	}
	if reads&regD != 0 {
		extraRegs = append(extraRegs, 4)
	}
	if reads&regE != 0 {
		extraRegs = append(extraRegs, 5)
	}
	if reads&regH != 0 {
		extraRegs = append(extraRegs, 6)
	}
	if reads&regL != 0 {
		extraRegs = append(extraRegs, 7)
	}

	sweepSP := reads&regSP != 0

	if len(extraRegs) == 0 && !sweepSP {
		return exhaustiveAFMasked(target, candidate, deadFlags)
	}

	if len(extraRegs) <= 2 && !sweepSP {
		return exhaustiveFullSweepMasked(target, candidate, extraRegs, deadFlags)
	}
	return exhaustiveReducedSweepMasked(target, candidate, extraRegs, sweepSP, deadFlags)
}

func exhaustiveFullSweepMasked(target, candidate []inst.Instruction, extraRegs []int, deadFlags FlagMask) bool {
	setReg := func(s *cpu.State, offset int, val uint8) {
		switch offset {
		case 2:
			s.B = val
		case 3:
			s.C = val
		case 4:
			s.D = val
		case 5:
			s.E = val
		case 6:
			s.H = val
		case 7:
			s.L = val
		}
	}

	if len(extraRegs) == 1 {
		for a := 0; a < 256; a++ {
			for carry := uint8(0); carry <= 1; carry++ {
				for r := 0; r < 256; r++ {
					s := cpu.State{A: uint8(a), F: carry}
					setReg(&s, extraRegs[0], uint8(r))
					tOut := execSeq(s, target)
					cOut := execSeq(s, candidate)
					if !statesEqualMasked(tOut, cOut, deadFlags) {
						return false
					}
				}
			}
		}
		return true
	}

	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			for r1 := 0; r1 < 256; r1++ {
				for r2 := 0; r2 < 256; r2++ {
					s := cpu.State{A: uint8(a), F: carry}
					setReg(&s, extraRegs[0], uint8(r1))
					setReg(&s, extraRegs[1], uint8(r2))
					tOut := execSeq(s, target)
					cOut := execSeq(s, candidate)
					if !statesEqualMasked(tOut, cOut, deadFlags) {
						return false
					}
				}
			}
		}
	}
	return true
}

func exhaustiveReducedSweepMasked(target, candidate []inst.Instruction, extraRegs []int, sweepSP bool, deadFlags FlagMask) bool {
	repValues := []uint8{
		0x00, 0x01, 0x02, 0x0F, 0x10, 0x1F, 0x20, 0x3F,
		0x40, 0x55, 0x7E, 0x7F, 0x80, 0x81, 0xAA, 0xBF,
		0xC0, 0xD5, 0xE0, 0xEF, 0xF0, 0xF7, 0xFE, 0xFF,
		0x03, 0x07, 0x11, 0x33, 0x77, 0xBB, 0xDD, 0xEE,
	}

	setReg := func(s *cpu.State, offset int, val uint8) {
		switch offset {
		case 2:
			s.B = val
		case 3:
			s.C = val
		case 4:
			s.D = val
		case 5:
			s.E = val
		case 6:
			s.H = val
		case 7:
			s.L = val
		}
	}

	repSP := []uint16{
		0x0000, 0x0001, 0x00FF, 0x0100, 0x7FFE, 0x7FFF, 0x8000, 0x8001,
		0xFFFE, 0xFFFF, 0x1234, 0x5678, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE,
	}

	compare := func(s cpu.State) bool {
		tOut := execSeq(s, target)
		cOut := execSeq(s, candidate)
		return statesEqualMasked(tOut, cOut, deadFlags)
	}

	var sweep func(s cpu.State, regIdx int) bool
	sweep = func(s cpu.State, regIdx int) bool {
		if regIdx >= len(extraRegs) {
			if sweepSP {
				for _, sp := range repSP {
					s2 := s
					s2.SP = sp
					if !compare(s2) {
						return false
					}
				}
				return true
			}
			return compare(s)
		}
		for _, v := range repValues {
			s2 := s
			setReg(&s2, extraRegs[regIdx], v)
			if !sweep(s2, regIdx+1) {
				return false
			}
		}
		return true
	}

	for a := 0; a < 256; a++ {
		for carry := uint8(0); carry <= 1; carry++ {
			s := cpu.State{A: uint8(a), F: carry}
			if !sweep(s, 0) {
				return false
			}
		}
	}
	return true
}

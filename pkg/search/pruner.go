package search

import "github.com/oisee/z80-optimizer/pkg/inst"

// ShouldPrune returns true if the sequence can be skipped.
// Pruning rules eliminate provably redundant sequences.
func ShouldPrune(seq []inst.Instruction) bool {
	for i := 0; i < len(seq); i++ {
		// NOP elimination: skip sequences containing NOP
		if seq[i].Op == inst.NOP {
			return true
		}

		// Self-load elimination: LD X,X is a NOP
		if isSelfLoad(seq[i].Op) {
			return true
		}

		// Dead write: if instruction at i writes a register that is
		// immediately overwritten at i+1 without being read
		if i+1 < len(seq) && isDeadWrite(seq[i], seq[i+1]) {
			return true
		}
	}

	// Canonical ordering: for independent adjacent instructions,
	// force opcode order to eliminate permutation duplicates
	for i := 0; i+1 < len(seq); i++ {
		if areIndependent(seq[i], seq[i+1]) && instKey(seq[i]) > instKey(seq[i+1]) {
			return true
		}
	}

	return false
}

// isSelfLoad returns true for LD X,X instructions (which are NOPs).
func isSelfLoad(op inst.OpCode) bool {
	switch op {
	case inst.LD_A_A, inst.LD_B_B, inst.LD_C_C, inst.LD_D_D,
		inst.LD_E_E, inst.LD_H_H, inst.LD_L_L:
		return true
	}
	return false
}

// isDeadWrite returns true if 'first' writes a register that 'second'
// overwrites without reading first.
func isDeadWrite(first, second inst.Instruction) bool {
	written := opWrites(first.Op)
	if written == 0 {
		return false
	}
	// Check if second also writes the same register without reading it
	read := opReads(second.Op)
	written2 := opWrites(second.Op)

	// For each register written by first:
	// if second writes it too AND doesn't read it, it's dead
	// Only apply to non-flag registers
	dead := written & written2 & ^regF & ^(read)
	return dead != 0
}

// opWrites returns which registers an instruction modifies.
func opWrites(op inst.OpCode) regMask {
	switch op {
	// ALU ops always write A and F
	case inst.ADD_A_B, inst.ADD_A_C, inst.ADD_A_D, inst.ADD_A_E, inst.ADD_A_H, inst.ADD_A_L, inst.ADD_A_A, inst.ADD_A_N,
		inst.ADC_A_B, inst.ADC_A_C, inst.ADC_A_D, inst.ADC_A_E, inst.ADC_A_H, inst.ADC_A_L, inst.ADC_A_A, inst.ADC_A_N,
		inst.SUB_B, inst.SUB_C, inst.SUB_D, inst.SUB_E, inst.SUB_H, inst.SUB_L, inst.SUB_A, inst.SUB_N,
		inst.SBC_A_B, inst.SBC_A_C, inst.SBC_A_D, inst.SBC_A_E, inst.SBC_A_H, inst.SBC_A_L, inst.SBC_A_A, inst.SBC_A_N,
		inst.AND_B, inst.AND_C, inst.AND_D, inst.AND_E, inst.AND_H, inst.AND_L, inst.AND_A, inst.AND_N,
		inst.XOR_B, inst.XOR_C, inst.XOR_D, inst.XOR_E, inst.XOR_H, inst.XOR_L, inst.XOR_A, inst.XOR_N,
		inst.OR_B, inst.OR_C, inst.OR_D, inst.OR_E, inst.OR_H, inst.OR_L, inst.OR_A, inst.OR_N,
		inst.RLCA, inst.RRCA, inst.RLA, inst.RRA, inst.DAA, inst.CPL, inst.NEG:
		return regA | regF

	// CP only writes F (not A)
	case inst.CP_B, inst.CP_C, inst.CP_D, inst.CP_E, inst.CP_H, inst.CP_L, inst.CP_A, inst.CP_N:
		return regF

	// SCF/CCF write F
	case inst.SCF, inst.CCF:
		return regF

	// INC/DEC write the target register + F
	case inst.INC_A, inst.DEC_A:
		return regA | regF
	case inst.INC_B, inst.DEC_B:
		return regB | regF
	case inst.INC_C, inst.DEC_C:
		return regC | regF
	case inst.INC_D, inst.DEC_D:
		return regD | regF
	case inst.INC_E, inst.DEC_E:
		return regE | regF
	case inst.INC_H, inst.DEC_H:
		return regH | regF
	case inst.INC_L, inst.DEC_L:
		return regL | regF

	// LD A, r writes A
	case inst.LD_A_B, inst.LD_A_C, inst.LD_A_D, inst.LD_A_E, inst.LD_A_H, inst.LD_A_L, inst.LD_A_A, inst.LD_A_N:
		return regA
	// LD B, r writes B
	case inst.LD_B_A, inst.LD_B_B, inst.LD_B_C, inst.LD_B_D, inst.LD_B_E, inst.LD_B_H, inst.LD_B_L, inst.LD_B_N:
		return regB
	// LD C, r writes C
	case inst.LD_C_A, inst.LD_C_B, inst.LD_C_C, inst.LD_C_D, inst.LD_C_E, inst.LD_C_H, inst.LD_C_L, inst.LD_C_N:
		return regC
	// LD D, r writes D
	case inst.LD_D_A, inst.LD_D_B, inst.LD_D_C, inst.LD_D_D, inst.LD_D_E, inst.LD_D_H, inst.LD_D_L, inst.LD_D_N:
		return regD
	// LD E, r writes E
	case inst.LD_E_A, inst.LD_E_B, inst.LD_E_C, inst.LD_E_D, inst.LD_E_E, inst.LD_E_H, inst.LD_E_L, inst.LD_E_N:
		return regE
	// LD H, r writes H
	case inst.LD_H_A, inst.LD_H_B, inst.LD_H_C, inst.LD_H_D, inst.LD_H_E, inst.LD_H_H, inst.LD_H_L, inst.LD_H_N:
		return regH
	// LD L, r writes L
	case inst.LD_L_A, inst.LD_L_B, inst.LD_L_C, inst.LD_L_D, inst.LD_L_E, inst.LD_L_H, inst.LD_L_L, inst.LD_L_N:
		return regL

	// CB prefix rotates/shifts write the target register + F
	case inst.RLC_A, inst.RRC_A, inst.RL_A, inst.RR_A, inst.SLA_A, inst.SRA_A, inst.SRL_A, inst.SLL_A:
		return regA | regF
	case inst.RLC_B, inst.RRC_B, inst.RL_B, inst.RR_B, inst.SLA_B, inst.SRA_B, inst.SRL_B, inst.SLL_B:
		return regB | regF
	case inst.RLC_C, inst.RRC_C, inst.RL_C, inst.RR_C, inst.SLA_C, inst.SRA_C, inst.SRL_C, inst.SLL_C:
		return regC | regF
	case inst.RLC_D, inst.RRC_D, inst.RL_D, inst.RR_D, inst.SLA_D, inst.SRA_D, inst.SRL_D, inst.SLL_D:
		return regD | regF
	case inst.RLC_E, inst.RRC_E, inst.RL_E, inst.RR_E, inst.SLA_E, inst.SRA_E, inst.SRL_E, inst.SLL_E:
		return regE | regF
	case inst.RLC_H, inst.RRC_H, inst.RL_H, inst.RR_H, inst.SLA_H, inst.SRA_H, inst.SRL_H, inst.SLL_H:
		return regH | regF
	case inst.RLC_L, inst.RRC_L, inst.RL_L, inst.RR_L, inst.SLA_L, inst.SRA_L, inst.SRL_L, inst.SLL_L:
		return regL | regF

	// BIT n, r: writes only F (does not modify register)
	case inst.BIT_0_A, inst.BIT_1_A, inst.BIT_2_A, inst.BIT_3_A,
		inst.BIT_4_A, inst.BIT_5_A, inst.BIT_6_A, inst.BIT_7_A,
		inst.BIT_0_B, inst.BIT_1_B, inst.BIT_2_B, inst.BIT_3_B,
		inst.BIT_4_B, inst.BIT_5_B, inst.BIT_6_B, inst.BIT_7_B,
		inst.BIT_0_C, inst.BIT_1_C, inst.BIT_2_C, inst.BIT_3_C,
		inst.BIT_4_C, inst.BIT_5_C, inst.BIT_6_C, inst.BIT_7_C,
		inst.BIT_0_D, inst.BIT_1_D, inst.BIT_2_D, inst.BIT_3_D,
		inst.BIT_4_D, inst.BIT_5_D, inst.BIT_6_D, inst.BIT_7_D,
		inst.BIT_0_E, inst.BIT_1_E, inst.BIT_2_E, inst.BIT_3_E,
		inst.BIT_4_E, inst.BIT_5_E, inst.BIT_6_E, inst.BIT_7_E,
		inst.BIT_0_H, inst.BIT_1_H, inst.BIT_2_H, inst.BIT_3_H,
		inst.BIT_4_H, inst.BIT_5_H, inst.BIT_6_H, inst.BIT_7_H,
		inst.BIT_0_L, inst.BIT_1_L, inst.BIT_2_L, inst.BIT_3_L,
		inst.BIT_4_L, inst.BIT_5_L, inst.BIT_6_L, inst.BIT_7_L:
		return regF

	// RES n, r: writes only the register (no flag changes)
	case inst.RES_0_A, inst.RES_1_A, inst.RES_2_A, inst.RES_3_A,
		inst.RES_4_A, inst.RES_5_A, inst.RES_6_A, inst.RES_7_A,
		inst.SET_0_A, inst.SET_1_A, inst.SET_2_A, inst.SET_3_A,
		inst.SET_4_A, inst.SET_5_A, inst.SET_6_A, inst.SET_7_A:
		return regA
	case inst.RES_0_B, inst.RES_1_B, inst.RES_2_B, inst.RES_3_B,
		inst.RES_4_B, inst.RES_5_B, inst.RES_6_B, inst.RES_7_B,
		inst.SET_0_B, inst.SET_1_B, inst.SET_2_B, inst.SET_3_B,
		inst.SET_4_B, inst.SET_5_B, inst.SET_6_B, inst.SET_7_B:
		return regB
	case inst.RES_0_C, inst.RES_1_C, inst.RES_2_C, inst.RES_3_C,
		inst.RES_4_C, inst.RES_5_C, inst.RES_6_C, inst.RES_7_C,
		inst.SET_0_C, inst.SET_1_C, inst.SET_2_C, inst.SET_3_C,
		inst.SET_4_C, inst.SET_5_C, inst.SET_6_C, inst.SET_7_C:
		return regC
	case inst.RES_0_D, inst.RES_1_D, inst.RES_2_D, inst.RES_3_D,
		inst.RES_4_D, inst.RES_5_D, inst.RES_6_D, inst.RES_7_D,
		inst.SET_0_D, inst.SET_1_D, inst.SET_2_D, inst.SET_3_D,
		inst.SET_4_D, inst.SET_5_D, inst.SET_6_D, inst.SET_7_D:
		return regD
	case inst.RES_0_E, inst.RES_1_E, inst.RES_2_E, inst.RES_3_E,
		inst.RES_4_E, inst.RES_5_E, inst.RES_6_E, inst.RES_7_E,
		inst.SET_0_E, inst.SET_1_E, inst.SET_2_E, inst.SET_3_E,
		inst.SET_4_E, inst.SET_5_E, inst.SET_6_E, inst.SET_7_E:
		return regE
	case inst.RES_0_H, inst.RES_1_H, inst.RES_2_H, inst.RES_3_H,
		inst.RES_4_H, inst.RES_5_H, inst.RES_6_H, inst.RES_7_H,
		inst.SET_0_H, inst.SET_1_H, inst.SET_2_H, inst.SET_3_H,
		inst.SET_4_H, inst.SET_5_H, inst.SET_6_H, inst.SET_7_H:
		return regH
	case inst.RES_0_L, inst.RES_1_L, inst.RES_2_L, inst.RES_3_L,
		inst.RES_4_L, inst.RES_5_L, inst.RES_6_L, inst.RES_7_L,
		inst.SET_0_L, inst.SET_1_L, inst.SET_2_L, inst.SET_3_L,
		inst.SET_4_L, inst.SET_5_L, inst.SET_6_L, inst.SET_7_L:
		return regL

	// 16-bit INC/DEC: write the pair, NO flag changes
	case inst.INC_BC, inst.DEC_BC:
		return regB | regC
	case inst.INC_DE, inst.DEC_DE:
		return regD | regE
	case inst.INC_HL, inst.DEC_HL:
		return regH | regL
	case inst.INC_SP, inst.DEC_SP:
		return regSP

	// ADD HL, rr: writes H, L, and flags (H, N=0, C; preserves S,Z,P/V)
	case inst.ADD_HL_BC, inst.ADD_HL_DE, inst.ADD_HL_HL, inst.ADD_HL_SP:
		return regH | regL | regF

	// EX DE, HL: writes D, E, H, L
	case inst.EX_DE_HL:
		return regD | regE | regH | regL

	// LD SP, HL: writes SP
	case inst.LD_SP_HL:
		return regSP

	// Wave 4: 16-bit immediate loads
	case inst.LD_BC_NN:
		return regB | regC
	case inst.LD_DE_NN:
		return regD | regE
	case inst.LD_HL_NN:
		return regH | regL
	case inst.LD_SP_NN:
		return regSP

	// ADC/SBC HL, rr: writes H, L, and all flags (S, Z, H, P/V, N, C)
	case inst.ADC_HL_BC, inst.ADC_HL_DE, inst.ADC_HL_HL, inst.ADC_HL_SP,
		inst.SBC_HL_BC, inst.SBC_HL_DE, inst.SBC_HL_HL, inst.SBC_HL_SP:
		return regH | regL | regF

	case inst.NOP:
		return 0
	}
	return 0
}

// areIndependent returns true if swapping two instructions produces the same result.
// Conservatively: they must not write anything the other reads or writes.
func areIndependent(a, b inst.Instruction) bool {
	aR := opReads(a.Op)
	aW := opWrites(a.Op)
	bR := opReads(b.Op)
	bW := opWrites(b.Op)

	// No WAR, RAW, or WAW dependencies
	if aW&bR != 0 || aR&bW != 0 || aW&bW != 0 {
		return false
	}
	return true
}

// instKey returns a sortable key for canonical ordering.
func instKey(i inst.Instruction) uint32 {
	return uint32(i.Op)<<16 | uint32(i.Imm)
}

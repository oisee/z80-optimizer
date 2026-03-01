package inst

// OpCode is a compact identifier for a Z80 instruction (not the raw byte encoding).
// We use our own enum because some instructions share raw opcodes but differ
// by prefix (e.g., CB-prefix rotates vs base opcodes).
type OpCode uint16

// Instruction is a compact representation of one Z80 instruction.
// 6 bytes: Op (uint16) + Imm (uint16) + padding. Still trivially copyable.
type Instruction struct {
	Op  OpCode
	Imm uint16 // Immediate value (8-bit for most ops, 16-bit for LD rr,nn)
}

// HasImmediate returns true if this opcode uses an immediate operand (8 or 16-bit).
func HasImmediate(op OpCode) bool {
	switch op {
	case LD_A_N, LD_B_N, LD_C_N, LD_D_N, LD_E_N, LD_H_N, LD_L_N,
		ADD_A_N, ADC_A_N, SUB_N, SBC_A_N, AND_N, XOR_N, OR_N, CP_N,
		LD_HLI_N:
		return true
	}
	return HasImm16(op)
}

// UsesMemory returns true if this opcode accesses the virtual memory byte (State.M).
func UsesMemory(op OpCode) bool {
	return op >= LD_A_HLI && op < OpCodeCount
}

// HasImm16 returns true if this opcode uses a 16-bit immediate operand.
func HasImm16(op OpCode) bool {
	switch op {
	case LD_BC_NN, LD_DE_NN, LD_HL_NN, LD_SP_NN:
		return true
	}
	return false
}

// OpCode constants for the Z80 superoptimizer.
// Organized by implementation wave:
//
//   V1 (206 ops):  8-bit register loads, ALU, shifts, rotates, specials
//   Wave 0:        Structural fixes (OpCode uint8→uint16, carry-flag bug, regMask widening)
//   Wave 1 (+174): BIT/RES/SET n,r and SLL r (CB-prefix register ops)
//   Wave 2 (+14):  16-bit pair ops (INC/DEC rr, ADD HL,rr, EX DE,HL, LD SP,HL)
//   Wave 4 (+12):  16-bit immediates (LD rr,nn) and ED arithmetic (ADC/SBC HL,rr)
//
// Total: 406 opcodes, 266,359 distinct instructions per search position.
const (
	// === V1: 8-bit register operations ===

	// Register-to-register loads (49 instructions: 7x7, includes self-loads which are NOPs)
	LD_A_B OpCode = iota
	LD_A_C
	LD_A_D
	LD_A_E
	LD_A_H
	LD_A_L
	LD_A_A
	LD_B_A
	LD_B_B
	LD_B_C
	LD_B_D
	LD_B_E
	LD_B_H
	LD_B_L
	LD_C_A
	LD_C_B
	LD_C_C
	LD_C_D
	LD_C_E
	LD_C_H
	LD_C_L
	LD_D_A
	LD_D_B
	LD_D_C
	LD_D_D
	LD_D_E
	LD_D_H
	LD_D_L
	LD_E_A
	LD_E_B
	LD_E_C
	LD_E_D
	LD_E_E
	LD_E_H
	LD_E_L
	LD_H_A
	LD_H_B
	LD_H_C
	LD_H_D
	LD_H_E
	LD_H_H
	LD_H_L
	LD_L_A
	LD_L_B
	LD_L_C
	LD_L_D
	LD_L_E
	LD_L_H
	LD_L_L

	// Immediate loads (7 registers)
	LD_A_N
	LD_B_N
	LD_C_N
	LD_D_N
	LD_E_N
	LD_H_N
	LD_L_N

	// 8-bit arithmetic: ADD A, r
	ADD_A_B
	ADD_A_C
	ADD_A_D
	ADD_A_E
	ADD_A_H
	ADD_A_L
	ADD_A_A
	ADD_A_N

	// ADC A, r
	ADC_A_B
	ADC_A_C
	ADC_A_D
	ADC_A_E
	ADC_A_H
	ADC_A_L
	ADC_A_A
	ADC_A_N

	// SUB r
	SUB_B
	SUB_C
	SUB_D
	SUB_E
	SUB_H
	SUB_L
	SUB_A
	SUB_N

	// SBC A, r
	SBC_A_B
	SBC_A_C
	SBC_A_D
	SBC_A_E
	SBC_A_H
	SBC_A_L
	SBC_A_A
	SBC_A_N

	// AND r
	AND_B
	AND_C
	AND_D
	AND_E
	AND_H
	AND_L
	AND_A
	AND_N

	// XOR r
	XOR_B
	XOR_C
	XOR_D
	XOR_E
	XOR_H
	XOR_L
	XOR_A
	XOR_N

	// OR r
	OR_B
	OR_C
	OR_D
	OR_E
	OR_H
	OR_L
	OR_A
	OR_N

	// CP r
	CP_B
	CP_C
	CP_D
	CP_E
	CP_H
	CP_L
	CP_A
	CP_N

	// INC r
	INC_A
	INC_B
	INC_C
	INC_D
	INC_E
	INC_H
	INC_L

	// DEC r
	DEC_A
	DEC_B
	DEC_C
	DEC_D
	DEC_E
	DEC_H
	DEC_L

	// Accumulator rotates (non-CB prefix)
	RLCA
	RRCA
	RLA
	RRA

	// Special
	DAA
	CPL
	SCF
	CCF
	NEG
	NOP

	// CB-prefix rotate/shift on all registers
	RLC_A
	RLC_B
	RLC_C
	RLC_D
	RLC_E
	RLC_H
	RLC_L

	RRC_A
	RRC_B
	RRC_C
	RRC_D
	RRC_E
	RRC_H
	RRC_L

	RL_A
	RL_B
	RL_C
	RL_D
	RL_E
	RL_H
	RL_L

	RR_A
	RR_B
	RR_C
	RR_D
	RR_E
	RR_H
	RR_L

	SLA_A
	SLA_B
	SLA_C
	SLA_D
	SLA_E
	SLA_H
	SLA_L

	SRA_A
	SRA_B
	SRA_C
	SRA_D
	SRA_E
	SRA_H
	SRA_L

	SRL_A
	SRL_B
	SRL_C
	SRL_D
	SRL_E
	SRL_H
	SRL_L

	SLL_A // Undocumented

	// === Wave 1: BIT / RES / SET / SLL on registers (174 opcodes) ===
	// Pure register operations, no memory, no state struct changes.
	// BIT n,r: test bit, only modifies F. RES/SET n,r: clear/set bit, no flag changes.
	// SLL r: undocumented shift left setting bit 0 to 1.

	// CB-prefix SLL on B-L (undocumented)
	SLL_B
	SLL_C
	SLL_D
	SLL_E
	SLL_H
	SLL_L

	// CB-prefix BIT n, r (n=0..7, r=A,B,C,D,E,H,L) — tests bit, only modifies F
	BIT_0_A
	BIT_0_B
	BIT_0_C
	BIT_0_D
	BIT_0_E
	BIT_0_H
	BIT_0_L
	BIT_1_A
	BIT_1_B
	BIT_1_C
	BIT_1_D
	BIT_1_E
	BIT_1_H
	BIT_1_L
	BIT_2_A
	BIT_2_B
	BIT_2_C
	BIT_2_D
	BIT_2_E
	BIT_2_H
	BIT_2_L
	BIT_3_A
	BIT_3_B
	BIT_3_C
	BIT_3_D
	BIT_3_E
	BIT_3_H
	BIT_3_L
	BIT_4_A
	BIT_4_B
	BIT_4_C
	BIT_4_D
	BIT_4_E
	BIT_4_H
	BIT_4_L
	BIT_5_A
	BIT_5_B
	BIT_5_C
	BIT_5_D
	BIT_5_E
	BIT_5_H
	BIT_5_L
	BIT_6_A
	BIT_6_B
	BIT_6_C
	BIT_6_D
	BIT_6_E
	BIT_6_H
	BIT_6_L
	BIT_7_A
	BIT_7_B
	BIT_7_C
	BIT_7_D
	BIT_7_E
	BIT_7_H
	BIT_7_L

	// CB-prefix RES n, r — clears bit, no flag changes
	RES_0_A
	RES_0_B
	RES_0_C
	RES_0_D
	RES_0_E
	RES_0_H
	RES_0_L
	RES_1_A
	RES_1_B
	RES_1_C
	RES_1_D
	RES_1_E
	RES_1_H
	RES_1_L
	RES_2_A
	RES_2_B
	RES_2_C
	RES_2_D
	RES_2_E
	RES_2_H
	RES_2_L
	RES_3_A
	RES_3_B
	RES_3_C
	RES_3_D
	RES_3_E
	RES_3_H
	RES_3_L
	RES_4_A
	RES_4_B
	RES_4_C
	RES_4_D
	RES_4_E
	RES_4_H
	RES_4_L
	RES_5_A
	RES_5_B
	RES_5_C
	RES_5_D
	RES_5_E
	RES_5_H
	RES_5_L
	RES_6_A
	RES_6_B
	RES_6_C
	RES_6_D
	RES_6_E
	RES_6_H
	RES_6_L
	RES_7_A
	RES_7_B
	RES_7_C
	RES_7_D
	RES_7_E
	RES_7_H
	RES_7_L

	// CB-prefix SET n, r — sets bit, no flag changes
	SET_0_A
	SET_0_B
	SET_0_C
	SET_0_D
	SET_0_E
	SET_0_H
	SET_0_L
	SET_1_A
	SET_1_B
	SET_1_C
	SET_1_D
	SET_1_E
	SET_1_H
	SET_1_L
	SET_2_A
	SET_2_B
	SET_2_C
	SET_2_D
	SET_2_E
	SET_2_H
	SET_2_L
	SET_3_A
	SET_3_B
	SET_3_C
	SET_3_D
	SET_3_E
	SET_3_H
	SET_3_L
	SET_4_A
	SET_4_B
	SET_4_C
	SET_4_D
	SET_4_E
	SET_4_H
	SET_4_L
	SET_5_A
	SET_5_B
	SET_5_C
	SET_5_D
	SET_5_E
	SET_5_H
	SET_5_L
	SET_6_A
	SET_6_B
	SET_6_C
	SET_6_D
	SET_6_E
	SET_6_H
	SET_6_L
	SET_7_A
	SET_7_B
	SET_7_C
	SET_7_D
	SET_7_E
	SET_7_H
	SET_7_L

	// === Wave 2: 16-bit register pair ops (14 opcodes) ===
	// Added SP to State. 16-bit INC/DEC do NOT affect flags.
	// ADD HL,rr: sets H (bit 11 carry), N=0, C; preserves S,Z,P/V.
	INC_BC
	INC_DE
	INC_HL
	INC_SP
	DEC_BC
	DEC_DE
	DEC_HL
	DEC_SP
	ADD_HL_BC
	ADD_HL_DE
	ADD_HL_HL
	ADD_HL_SP
	EX_DE_HL
	LD_SP_HL

	// === Wave 4: 16-bit immediates + ED arithmetic (12 opcodes) ===
	// Widened Instruction.Imm from uint8 to uint16.
	// LD rr,nn: 3-byte, no flag changes. ADC/SBC HL,rr: full S,Z,H,P/V,N,C computation.
	LD_BC_NN
	LD_DE_NN
	LD_HL_NN
	LD_SP_NN

	// ED-prefix 16-bit arithmetic
	ADC_HL_BC
	ADC_HL_DE
	ADC_HL_HL
	ADC_HL_SP
	SBC_HL_BC
	SBC_HL_DE
	SBC_HL_HL
	SBC_HL_SP

	// === Wave 5: Memory ops — (HL)/(BC)/(DE) indirect (61 opcodes) ===
	// All memory-accessing instructions use State.M as the virtual memory byte.
	// Prerequisite: all memory ops in a sequence must target the same address.

	// LD r, (HL): r = M (7 ops, 7 T-states, 1 byte)
	LD_A_HLI
	LD_B_HLI
	LD_C_HLI
	LD_D_HLI
	LD_E_HLI
	LD_H_HLI
	LD_L_HLI

	// LD (HL), r: M = r (7 ops, 7 T-states, 1 byte)
	LD_HLI_A
	LD_HLI_B
	LD_HLI_C
	LD_HLI_D
	LD_HLI_E
	LD_HLI_H
	LD_HLI_L

	// LD (HL), n: M = imm8 (1 op, 10 T-states, 2 bytes)
	LD_HLI_N

	// LD A, (BC) / LD A, (DE): A = M (2 ops, 7 T-states, 1 byte)
	LD_A_BCI
	LD_A_DEI

	// LD (BC), A / LD (DE), A: M = A (2 ops, 7 T-states, 1 byte)
	LD_BCI_A
	LD_DEI_A

	// ALU A, (HL): 8 ops, 7 T-states, 1 byte
	ADD_A_HLI
	ADC_A_HLI
	SUB_HLI
	SBC_A_HLI
	AND_HLI
	XOR_HLI
	OR_HLI
	CP_HLI

	// INC/DEC (HL): 2 ops, 11 T-states, 1 byte
	INC_HLI
	DEC_HLI

	// CB-prefix rotate/shift (HL): 8 ops, 15 T-states, 2 bytes
	RLC_HLI
	RRC_HLI
	RL_HLI
	RR_HLI
	SLA_HLI
	SRA_HLI
	SRL_HLI
	SLL_HLI

	// CB-prefix BIT n, (HL): 8 ops, 12 T-states, 2 bytes
	BIT_0_HLI
	BIT_1_HLI
	BIT_2_HLI
	BIT_3_HLI
	BIT_4_HLI
	BIT_5_HLI
	BIT_6_HLI
	BIT_7_HLI

	// CB-prefix RES n, (HL): 8 ops, 15 T-states, 2 bytes
	RES_0_HLI
	RES_1_HLI
	RES_2_HLI
	RES_3_HLI
	RES_4_HLI
	RES_5_HLI
	RES_6_HLI
	RES_7_HLI

	// CB-prefix SET n, (HL): 8 ops, 15 T-states, 2 bytes
	SET_0_HLI
	SET_1_HLI
	SET_2_HLI
	SET_3_HLI
	SET_4_HLI
	SET_5_HLI
	SET_6_HLI
	SET_7_HLI

	OpCodeCount // sentinel
)

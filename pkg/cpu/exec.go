package cpu

import "github.com/oisee/z80-optimizer/pkg/inst"

// Exec executes a single instruction on the given state.
// Returns the T-state cost. The state is modified in place.
// imm is uint16 to support 16-bit immediates (LD rr,nn); 8-bit ops use low byte.
func Exec(s *State, op inst.OpCode, imm uint16) int {
	switch op {
	// === 8-bit register loads ===
	case inst.LD_A_B:
		s.A = s.B
	case inst.LD_A_C:
		s.A = s.C
	case inst.LD_A_D:
		s.A = s.D
	case inst.LD_A_E:
		s.A = s.E
	case inst.LD_A_H:
		s.A = s.H
	case inst.LD_A_L:
		s.A = s.L
	case inst.LD_A_A:
		// nop
	case inst.LD_B_A:
		s.B = s.A
	case inst.LD_B_B:
		// nop
	case inst.LD_B_C:
		s.B = s.C
	case inst.LD_B_D:
		s.B = s.D
	case inst.LD_B_E:
		s.B = s.E
	case inst.LD_B_H:
		s.B = s.H
	case inst.LD_B_L:
		s.B = s.L
	case inst.LD_C_A:
		s.C = s.A
	case inst.LD_C_B:
		s.C = s.B
	case inst.LD_C_C:
		// nop
	case inst.LD_C_D:
		s.C = s.D
	case inst.LD_C_E:
		s.C = s.E
	case inst.LD_C_H:
		s.C = s.H
	case inst.LD_C_L:
		s.C = s.L
	case inst.LD_D_A:
		s.D = s.A
	case inst.LD_D_B:
		s.D = s.B
	case inst.LD_D_C:
		s.D = s.C
	case inst.LD_D_D:
		// nop
	case inst.LD_D_E:
		s.D = s.E
	case inst.LD_D_H:
		s.D = s.H
	case inst.LD_D_L:
		s.D = s.L
	case inst.LD_E_A:
		s.E = s.A
	case inst.LD_E_B:
		s.E = s.B
	case inst.LD_E_C:
		s.E = s.C
	case inst.LD_E_D:
		s.E = s.D
	case inst.LD_E_E:
		// nop
	case inst.LD_E_H:
		s.E = s.H
	case inst.LD_E_L:
		s.E = s.L
	case inst.LD_H_A:
		s.H = s.A
	case inst.LD_H_B:
		s.H = s.B
	case inst.LD_H_C:
		s.H = s.C
	case inst.LD_H_D:
		s.H = s.D
	case inst.LD_H_E:
		s.H = s.E
	case inst.LD_H_H:
		// nop
	case inst.LD_H_L:
		s.H = s.L
	case inst.LD_L_A:
		s.L = s.A
	case inst.LD_L_B:
		s.L = s.B
	case inst.LD_L_C:
		s.L = s.C
	case inst.LD_L_D:
		s.L = s.D
	case inst.LD_L_E:
		s.L = s.E
	case inst.LD_L_H:
		s.L = s.H
	case inst.LD_L_L:
		// nop

	// === Immediate loads ===
	case inst.LD_A_N:
		s.A = uint8(imm)
	case inst.LD_B_N:
		s.B = uint8(imm)
	case inst.LD_C_N:
		s.C = uint8(imm)
	case inst.LD_D_N:
		s.D = uint8(imm)
	case inst.LD_E_N:
		s.E = uint8(imm)
	case inst.LD_H_N:
		s.H = uint8(imm)
	case inst.LD_L_N:
		s.L = uint8(imm)

	// === 8-bit arithmetic: ADD ===
	case inst.ADD_A_B:
		execAdd(s, s.B)
	case inst.ADD_A_C:
		execAdd(s, s.C)
	case inst.ADD_A_D:
		execAdd(s, s.D)
	case inst.ADD_A_E:
		execAdd(s, s.E)
	case inst.ADD_A_H:
		execAdd(s, s.H)
	case inst.ADD_A_L:
		execAdd(s, s.L)
	case inst.ADD_A_A:
		execAdd(s, s.A)
	case inst.ADD_A_N:
		execAdd(s, uint8(imm))

	// === 8-bit arithmetic: ADC ===
	case inst.ADC_A_B:
		execAdc(s, s.B)
	case inst.ADC_A_C:
		execAdc(s, s.C)
	case inst.ADC_A_D:
		execAdc(s, s.D)
	case inst.ADC_A_E:
		execAdc(s, s.E)
	case inst.ADC_A_H:
		execAdc(s, s.H)
	case inst.ADC_A_L:
		execAdc(s, s.L)
	case inst.ADC_A_A:
		execAdc(s, s.A)
	case inst.ADC_A_N:
		execAdc(s, uint8(imm))

	// === 8-bit arithmetic: SUB ===
	case inst.SUB_B:
		execSub(s, s.B)
	case inst.SUB_C:
		execSub(s, s.C)
	case inst.SUB_D:
		execSub(s, s.D)
	case inst.SUB_E:
		execSub(s, s.E)
	case inst.SUB_H:
		execSub(s, s.H)
	case inst.SUB_L:
		execSub(s, s.L)
	case inst.SUB_A:
		execSub(s, s.A)
	case inst.SUB_N:
		execSub(s, uint8(imm))

	// === 8-bit arithmetic: SBC ===
	case inst.SBC_A_B:
		execSbc(s, s.B)
	case inst.SBC_A_C:
		execSbc(s, s.C)
	case inst.SBC_A_D:
		execSbc(s, s.D)
	case inst.SBC_A_E:
		execSbc(s, s.E)
	case inst.SBC_A_H:
		execSbc(s, s.H)
	case inst.SBC_A_L:
		execSbc(s, s.L)
	case inst.SBC_A_A:
		execSbc(s, s.A)
	case inst.SBC_A_N:
		execSbc(s, uint8(imm))

	// === 8-bit logic: AND ===
	case inst.AND_B:
		execAnd(s, s.B)
	case inst.AND_C:
		execAnd(s, s.C)
	case inst.AND_D:
		execAnd(s, s.D)
	case inst.AND_E:
		execAnd(s, s.E)
	case inst.AND_H:
		execAnd(s, s.H)
	case inst.AND_L:
		execAnd(s, s.L)
	case inst.AND_A:
		execAnd(s, s.A)
	case inst.AND_N:
		execAnd(s, uint8(imm))

	// === 8-bit logic: XOR ===
	case inst.XOR_B:
		execXor(s, s.B)
	case inst.XOR_C:
		execXor(s, s.C)
	case inst.XOR_D:
		execXor(s, s.D)
	case inst.XOR_E:
		execXor(s, s.E)
	case inst.XOR_H:
		execXor(s, s.H)
	case inst.XOR_L:
		execXor(s, s.L)
	case inst.XOR_A:
		execXor(s, s.A)
	case inst.XOR_N:
		execXor(s, uint8(imm))

	// === 8-bit logic: OR ===
	case inst.OR_B:
		execOr(s, s.B)
	case inst.OR_C:
		execOr(s, s.C)
	case inst.OR_D:
		execOr(s, s.D)
	case inst.OR_E:
		execOr(s, s.E)
	case inst.OR_H:
		execOr(s, s.H)
	case inst.OR_L:
		execOr(s, s.L)
	case inst.OR_A:
		execOr(s, s.A)
	case inst.OR_N:
		execOr(s, uint8(imm))

	// === 8-bit logic: CP ===
	case inst.CP_B:
		execCp(s, s.B)
	case inst.CP_C:
		execCp(s, s.C)
	case inst.CP_D:
		execCp(s, s.D)
	case inst.CP_E:
		execCp(s, s.E)
	case inst.CP_H:
		execCp(s, s.H)
	case inst.CP_L:
		execCp(s, s.L)
	case inst.CP_A:
		execCp(s, s.A)
	case inst.CP_N:
		execCp(s, uint8(imm))

	// === INC/DEC registers ===
	case inst.INC_A:
		execInc(s, &s.A)
	case inst.INC_B:
		execInc(s, &s.B)
	case inst.INC_C:
		execInc(s, &s.C)
	case inst.INC_D:
		execInc(s, &s.D)
	case inst.INC_E:
		execInc(s, &s.E)
	case inst.INC_H:
		execInc(s, &s.H)
	case inst.INC_L:
		execInc(s, &s.L)
	case inst.DEC_A:
		execDec(s, &s.A)
	case inst.DEC_B:
		execDec(s, &s.B)
	case inst.DEC_C:
		execDec(s, &s.C)
	case inst.DEC_D:
		execDec(s, &s.D)
	case inst.DEC_E:
		execDec(s, &s.E)
	case inst.DEC_H:
		execDec(s, &s.H)
	case inst.DEC_L:
		execDec(s, &s.L)

	// === Rotate/shift accumulator ===
	case inst.RLCA:
		s.A = (s.A << 1) | (s.A >> 7)
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & (FlagC | Flag3 | Flag5))
	case inst.RRCA:
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & FlagC)
		s.A = (s.A >> 1) | (s.A << 7)
		s.F |= s.A & (Flag3 | Flag5)
	case inst.RLA:
		old := s.A
		s.A = (s.A << 1) | (s.F & FlagC)
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & (Flag3 | Flag5)) | (old >> 7)
	case inst.RRA:
		old := s.A
		s.A = (s.A >> 1) | (s.F << 7)
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & (Flag3 | Flag5)) | (old & FlagC)

	// === Special A operations ===
	case inst.DAA:
		execDaa(s)
	case inst.CPL:
		s.A ^= 0xFF
		s.F = (s.F & (FlagC | FlagP | FlagZ | FlagS)) | (s.A & (Flag3 | Flag5)) | FlagN | FlagH
	case inst.SCF:
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & (Flag3 | Flag5)) | FlagC
	case inst.CCF:
		oldC := s.F & FlagC
		s.F = (s.F & (FlagP | FlagZ | FlagS)) | (s.A & (Flag3 | Flag5))
		if oldC != 0 {
			s.F |= FlagH
		} else {
			s.F |= FlagC
		}
	case inst.NEG:
		old := s.A
		s.A = 0
		execSub(s, old)

	// === CB prefix: rotate/shift A ===
	case inst.RLC_A:
		s.A = (s.A << 1) | (s.A >> 7)
		s.F = (s.A & FlagC) | Sz53pTable[s.A]
	case inst.RRC_A:
		s.F = s.A & FlagC
		s.A = (s.A >> 1) | (s.A << 7)
		s.F |= Sz53pTable[s.A]
	case inst.RL_A:
		old := s.A
		s.A = (s.A << 1) | (s.F & FlagC)
		s.F = (old >> 7) | Sz53pTable[s.A]
	case inst.RR_A:
		old := s.A
		s.A = (s.A >> 1) | (s.F << 7)
		s.F = (old & FlagC) | Sz53pTable[s.A]
	case inst.SLA_A:
		s.F = s.A >> 7
		s.A <<= 1
		s.F |= Sz53pTable[s.A]
	case inst.SRA_A:
		s.F = s.A & FlagC
		s.A = (s.A & 0x80) | (s.A >> 1)
		s.F |= Sz53pTable[s.A]
	case inst.SRL_A:
		s.F = s.A & FlagC
		s.A >>= 1
		s.F |= Sz53pTable[s.A]
	case inst.SLL_A:
		s.F = s.A >> 7
		s.A = (s.A << 1) | 0x01
		s.F |= Sz53pTable[s.A]

	// === CB prefix: rotate/shift B-L ===
	case inst.RLC_B:
		s.B = execRlc(s, s.B)
	case inst.RLC_C:
		s.C = execRlc(s, s.C)
	case inst.RLC_D:
		s.D = execRlc(s, s.D)
	case inst.RLC_E:
		s.E = execRlc(s, s.E)
	case inst.RLC_H:
		s.H = execRlc(s, s.H)
	case inst.RLC_L:
		s.L = execRlc(s, s.L)
	case inst.RRC_B:
		s.B = execRrc(s, s.B)
	case inst.RRC_C:
		s.C = execRrc(s, s.C)
	case inst.RRC_D:
		s.D = execRrc(s, s.D)
	case inst.RRC_E:
		s.E = execRrc(s, s.E)
	case inst.RRC_H:
		s.H = execRrc(s, s.H)
	case inst.RRC_L:
		s.L = execRrc(s, s.L)
	case inst.RL_B:
		s.B = execRl(s, s.B)
	case inst.RL_C:
		s.C = execRl(s, s.C)
	case inst.RL_D:
		s.D = execRl(s, s.D)
	case inst.RL_E:
		s.E = execRl(s, s.E)
	case inst.RL_H:
		s.H = execRl(s, s.H)
	case inst.RL_L:
		s.L = execRl(s, s.L)
	case inst.RR_B:
		s.B = execRr(s, s.B)
	case inst.RR_C:
		s.C = execRr(s, s.C)
	case inst.RR_D:
		s.D = execRr(s, s.D)
	case inst.RR_E:
		s.E = execRr(s, s.E)
	case inst.RR_H:
		s.H = execRr(s, s.H)
	case inst.RR_L:
		s.L = execRr(s, s.L)
	case inst.SLA_B:
		s.B = execSla(s, s.B)
	case inst.SLA_C:
		s.C = execSla(s, s.C)
	case inst.SLA_D:
		s.D = execSla(s, s.D)
	case inst.SLA_E:
		s.E = execSla(s, s.E)
	case inst.SLA_H:
		s.H = execSla(s, s.H)
	case inst.SLA_L:
		s.L = execSla(s, s.L)
	case inst.SRA_B:
		s.B = execSra(s, s.B)
	case inst.SRA_C:
		s.C = execSra(s, s.C)
	case inst.SRA_D:
		s.D = execSra(s, s.D)
	case inst.SRA_E:
		s.E = execSra(s, s.E)
	case inst.SRA_H:
		s.H = execSra(s, s.H)
	case inst.SRA_L:
		s.L = execSra(s, s.L)
	case inst.SRL_B:
		s.B = execSrl(s, s.B)
	case inst.SRL_C:
		s.C = execSrl(s, s.C)
	case inst.SRL_D:
		s.D = execSrl(s, s.D)
	case inst.SRL_E:
		s.E = execSrl(s, s.E)
	case inst.SRL_H:
		s.H = execSrl(s, s.H)
	case inst.SRL_L:
		s.L = execSrl(s, s.L)

	// === CB prefix: SLL B-L (undocumented) ===
	case inst.SLL_B:
		s.B = execSll(s, s.B)
	case inst.SLL_C:
		s.C = execSll(s, s.C)
	case inst.SLL_D:
		s.D = execSll(s, s.D)
	case inst.SLL_E:
		s.E = execSll(s, s.E)
	case inst.SLL_H:
		s.H = execSll(s, s.H)
	case inst.SLL_L:
		s.L = execSll(s, s.L)

	// === Wave 1: BIT n, r — test bit, set flags only ===
	case inst.BIT_0_A:
		execBit(s, s.A, 0)
	case inst.BIT_0_B:
		execBit(s, s.B, 0)
	case inst.BIT_0_C:
		execBit(s, s.C, 0)
	case inst.BIT_0_D:
		execBit(s, s.D, 0)
	case inst.BIT_0_E:
		execBit(s, s.E, 0)
	case inst.BIT_0_H:
		execBit(s, s.H, 0)
	case inst.BIT_0_L:
		execBit(s, s.L, 0)
	case inst.BIT_1_A:
		execBit(s, s.A, 1)
	case inst.BIT_1_B:
		execBit(s, s.B, 1)
	case inst.BIT_1_C:
		execBit(s, s.C, 1)
	case inst.BIT_1_D:
		execBit(s, s.D, 1)
	case inst.BIT_1_E:
		execBit(s, s.E, 1)
	case inst.BIT_1_H:
		execBit(s, s.H, 1)
	case inst.BIT_1_L:
		execBit(s, s.L, 1)
	case inst.BIT_2_A:
		execBit(s, s.A, 2)
	case inst.BIT_2_B:
		execBit(s, s.B, 2)
	case inst.BIT_2_C:
		execBit(s, s.C, 2)
	case inst.BIT_2_D:
		execBit(s, s.D, 2)
	case inst.BIT_2_E:
		execBit(s, s.E, 2)
	case inst.BIT_2_H:
		execBit(s, s.H, 2)
	case inst.BIT_2_L:
		execBit(s, s.L, 2)
	case inst.BIT_3_A:
		execBit(s, s.A, 3)
	case inst.BIT_3_B:
		execBit(s, s.B, 3)
	case inst.BIT_3_C:
		execBit(s, s.C, 3)
	case inst.BIT_3_D:
		execBit(s, s.D, 3)
	case inst.BIT_3_E:
		execBit(s, s.E, 3)
	case inst.BIT_3_H:
		execBit(s, s.H, 3)
	case inst.BIT_3_L:
		execBit(s, s.L, 3)
	case inst.BIT_4_A:
		execBit(s, s.A, 4)
	case inst.BIT_4_B:
		execBit(s, s.B, 4)
	case inst.BIT_4_C:
		execBit(s, s.C, 4)
	case inst.BIT_4_D:
		execBit(s, s.D, 4)
	case inst.BIT_4_E:
		execBit(s, s.E, 4)
	case inst.BIT_4_H:
		execBit(s, s.H, 4)
	case inst.BIT_4_L:
		execBit(s, s.L, 4)
	case inst.BIT_5_A:
		execBit(s, s.A, 5)
	case inst.BIT_5_B:
		execBit(s, s.B, 5)
	case inst.BIT_5_C:
		execBit(s, s.C, 5)
	case inst.BIT_5_D:
		execBit(s, s.D, 5)
	case inst.BIT_5_E:
		execBit(s, s.E, 5)
	case inst.BIT_5_H:
		execBit(s, s.H, 5)
	case inst.BIT_5_L:
		execBit(s, s.L, 5)
	case inst.BIT_6_A:
		execBit(s, s.A, 6)
	case inst.BIT_6_B:
		execBit(s, s.B, 6)
	case inst.BIT_6_C:
		execBit(s, s.C, 6)
	case inst.BIT_6_D:
		execBit(s, s.D, 6)
	case inst.BIT_6_E:
		execBit(s, s.E, 6)
	case inst.BIT_6_H:
		execBit(s, s.H, 6)
	case inst.BIT_6_L:
		execBit(s, s.L, 6)
	case inst.BIT_7_A:
		execBit(s, s.A, 7)
	case inst.BIT_7_B:
		execBit(s, s.B, 7)
	case inst.BIT_7_C:
		execBit(s, s.C, 7)
	case inst.BIT_7_D:
		execBit(s, s.D, 7)
	case inst.BIT_7_E:
		execBit(s, s.E, 7)
	case inst.BIT_7_H:
		execBit(s, s.H, 7)
	case inst.BIT_7_L:
		execBit(s, s.L, 7)

	// === Wave 1: RES n, r — clear bit, no flag changes ===
	case inst.RES_0_A:
		s.A &^= 1 << 0
	case inst.RES_0_B:
		s.B &^= 1 << 0
	case inst.RES_0_C:
		s.C &^= 1 << 0
	case inst.RES_0_D:
		s.D &^= 1 << 0
	case inst.RES_0_E:
		s.E &^= 1 << 0
	case inst.RES_0_H:
		s.H &^= 1 << 0
	case inst.RES_0_L:
		s.L &^= 1 << 0
	case inst.RES_1_A:
		s.A &^= 1 << 1
	case inst.RES_1_B:
		s.B &^= 1 << 1
	case inst.RES_1_C:
		s.C &^= 1 << 1
	case inst.RES_1_D:
		s.D &^= 1 << 1
	case inst.RES_1_E:
		s.E &^= 1 << 1
	case inst.RES_1_H:
		s.H &^= 1 << 1
	case inst.RES_1_L:
		s.L &^= 1 << 1
	case inst.RES_2_A:
		s.A &^= 1 << 2
	case inst.RES_2_B:
		s.B &^= 1 << 2
	case inst.RES_2_C:
		s.C &^= 1 << 2
	case inst.RES_2_D:
		s.D &^= 1 << 2
	case inst.RES_2_E:
		s.E &^= 1 << 2
	case inst.RES_2_H:
		s.H &^= 1 << 2
	case inst.RES_2_L:
		s.L &^= 1 << 2
	case inst.RES_3_A:
		s.A &^= 1 << 3
	case inst.RES_3_B:
		s.B &^= 1 << 3
	case inst.RES_3_C:
		s.C &^= 1 << 3
	case inst.RES_3_D:
		s.D &^= 1 << 3
	case inst.RES_3_E:
		s.E &^= 1 << 3
	case inst.RES_3_H:
		s.H &^= 1 << 3
	case inst.RES_3_L:
		s.L &^= 1 << 3
	case inst.RES_4_A:
		s.A &^= 1 << 4
	case inst.RES_4_B:
		s.B &^= 1 << 4
	case inst.RES_4_C:
		s.C &^= 1 << 4
	case inst.RES_4_D:
		s.D &^= 1 << 4
	case inst.RES_4_E:
		s.E &^= 1 << 4
	case inst.RES_4_H:
		s.H &^= 1 << 4
	case inst.RES_4_L:
		s.L &^= 1 << 4
	case inst.RES_5_A:
		s.A &^= 1 << 5
	case inst.RES_5_B:
		s.B &^= 1 << 5
	case inst.RES_5_C:
		s.C &^= 1 << 5
	case inst.RES_5_D:
		s.D &^= 1 << 5
	case inst.RES_5_E:
		s.E &^= 1 << 5
	case inst.RES_5_H:
		s.H &^= 1 << 5
	case inst.RES_5_L:
		s.L &^= 1 << 5
	case inst.RES_6_A:
		s.A &^= 1 << 6
	case inst.RES_6_B:
		s.B &^= 1 << 6
	case inst.RES_6_C:
		s.C &^= 1 << 6
	case inst.RES_6_D:
		s.D &^= 1 << 6
	case inst.RES_6_E:
		s.E &^= 1 << 6
	case inst.RES_6_H:
		s.H &^= 1 << 6
	case inst.RES_6_L:
		s.L &^= 1 << 6
	case inst.RES_7_A:
		s.A &^= 1 << 7
	case inst.RES_7_B:
		s.B &^= 1 << 7
	case inst.RES_7_C:
		s.C &^= 1 << 7
	case inst.RES_7_D:
		s.D &^= 1 << 7
	case inst.RES_7_E:
		s.E &^= 1 << 7
	case inst.RES_7_H:
		s.H &^= 1 << 7
	case inst.RES_7_L:
		s.L &^= 1 << 7

	// === Wave 1: SET n, r — set bit, no flag changes ===
	case inst.SET_0_A:
		s.A |= 1 << 0
	case inst.SET_0_B:
		s.B |= 1 << 0
	case inst.SET_0_C:
		s.C |= 1 << 0
	case inst.SET_0_D:
		s.D |= 1 << 0
	case inst.SET_0_E:
		s.E |= 1 << 0
	case inst.SET_0_H:
		s.H |= 1 << 0
	case inst.SET_0_L:
		s.L |= 1 << 0
	case inst.SET_1_A:
		s.A |= 1 << 1
	case inst.SET_1_B:
		s.B |= 1 << 1
	case inst.SET_1_C:
		s.C |= 1 << 1
	case inst.SET_1_D:
		s.D |= 1 << 1
	case inst.SET_1_E:
		s.E |= 1 << 1
	case inst.SET_1_H:
		s.H |= 1 << 1
	case inst.SET_1_L:
		s.L |= 1 << 1
	case inst.SET_2_A:
		s.A |= 1 << 2
	case inst.SET_2_B:
		s.B |= 1 << 2
	case inst.SET_2_C:
		s.C |= 1 << 2
	case inst.SET_2_D:
		s.D |= 1 << 2
	case inst.SET_2_E:
		s.E |= 1 << 2
	case inst.SET_2_H:
		s.H |= 1 << 2
	case inst.SET_2_L:
		s.L |= 1 << 2
	case inst.SET_3_A:
		s.A |= 1 << 3
	case inst.SET_3_B:
		s.B |= 1 << 3
	case inst.SET_3_C:
		s.C |= 1 << 3
	case inst.SET_3_D:
		s.D |= 1 << 3
	case inst.SET_3_E:
		s.E |= 1 << 3
	case inst.SET_3_H:
		s.H |= 1 << 3
	case inst.SET_3_L:
		s.L |= 1 << 3
	case inst.SET_4_A:
		s.A |= 1 << 4
	case inst.SET_4_B:
		s.B |= 1 << 4
	case inst.SET_4_C:
		s.C |= 1 << 4
	case inst.SET_4_D:
		s.D |= 1 << 4
	case inst.SET_4_E:
		s.E |= 1 << 4
	case inst.SET_4_H:
		s.H |= 1 << 4
	case inst.SET_4_L:
		s.L |= 1 << 4
	case inst.SET_5_A:
		s.A |= 1 << 5
	case inst.SET_5_B:
		s.B |= 1 << 5
	case inst.SET_5_C:
		s.C |= 1 << 5
	case inst.SET_5_D:
		s.D |= 1 << 5
	case inst.SET_5_E:
		s.E |= 1 << 5
	case inst.SET_5_H:
		s.H |= 1 << 5
	case inst.SET_5_L:
		s.L |= 1 << 5
	case inst.SET_6_A:
		s.A |= 1 << 6
	case inst.SET_6_B:
		s.B |= 1 << 6
	case inst.SET_6_C:
		s.C |= 1 << 6
	case inst.SET_6_D:
		s.D |= 1 << 6
	case inst.SET_6_E:
		s.E |= 1 << 6
	case inst.SET_6_H:
		s.H |= 1 << 6
	case inst.SET_6_L:
		s.L |= 1 << 6
	case inst.SET_7_A:
		s.A |= 1 << 7
	case inst.SET_7_B:
		s.B |= 1 << 7
	case inst.SET_7_C:
		s.C |= 1 << 7
	case inst.SET_7_D:
		s.D |= 1 << 7
	case inst.SET_7_E:
		s.E |= 1 << 7
	case inst.SET_7_H:
		s.H |= 1 << 7
	case inst.SET_7_L:
		s.L |= 1 << 7

	// === Wave 2: 16-bit register pair ops ===
	// Note: 16-bit INC/DEC do NOT affect flags (unlike 8-bit!)
	case inst.INC_BC:
		bc := (uint16(s.B)<<8 | uint16(s.C)) + 1
		s.B, s.C = uint8(bc>>8), uint8(bc)
	case inst.INC_DE:
		de := (uint16(s.D)<<8 | uint16(s.E)) + 1
		s.D, s.E = uint8(de>>8), uint8(de)
	case inst.INC_HL:
		hl := (uint16(s.H)<<8 | uint16(s.L)) + 1
		s.H, s.L = uint8(hl>>8), uint8(hl)
	case inst.INC_SP:
		s.SP++
	case inst.DEC_BC:
		bc := (uint16(s.B)<<8 | uint16(s.C)) - 1
		s.B, s.C = uint8(bc>>8), uint8(bc)
	case inst.DEC_DE:
		de := (uint16(s.D)<<8 | uint16(s.E)) - 1
		s.D, s.E = uint8(de>>8), uint8(de)
	case inst.DEC_HL:
		hl := (uint16(s.H)<<8 | uint16(s.L)) - 1
		s.H, s.L = uint8(hl>>8), uint8(hl)
	case inst.DEC_SP:
		s.SP--
	case inst.ADD_HL_BC:
		execAddHL(s, uint16(s.B)<<8|uint16(s.C))
	case inst.ADD_HL_DE:
		execAddHL(s, uint16(s.D)<<8|uint16(s.E))
	case inst.ADD_HL_HL:
		execAddHL(s, uint16(s.H)<<8|uint16(s.L))
	case inst.ADD_HL_SP:
		execAddHL(s, s.SP)
	case inst.EX_DE_HL:
		s.D, s.H = s.H, s.D
		s.E, s.L = s.L, s.E
	case inst.LD_SP_HL:
		s.SP = uint16(s.H)<<8 | uint16(s.L)

	// === Wave 4: 16-bit immediate loads + ED arithmetic ===
	case inst.LD_BC_NN:
		s.B, s.C = uint8(imm>>8), uint8(imm)
	case inst.LD_DE_NN:
		s.D, s.E = uint8(imm>>8), uint8(imm)
	case inst.LD_HL_NN:
		s.H, s.L = uint8(imm>>8), uint8(imm)
	case inst.LD_SP_NN:
		s.SP = imm

	// === ED-prefix 16-bit arithmetic ===
	case inst.ADC_HL_BC:
		execAdcHL(s, uint16(s.B)<<8|uint16(s.C))
	case inst.ADC_HL_DE:
		execAdcHL(s, uint16(s.D)<<8|uint16(s.E))
	case inst.ADC_HL_HL:
		execAdcHL(s, uint16(s.H)<<8|uint16(s.L))
	case inst.ADC_HL_SP:
		execAdcHL(s, s.SP)
	case inst.SBC_HL_BC:
		execSbcHL(s, uint16(s.B)<<8|uint16(s.C))
	case inst.SBC_HL_DE:
		execSbcHL(s, uint16(s.D)<<8|uint16(s.E))
	case inst.SBC_HL_HL:
		execSbcHL(s, uint16(s.H)<<8|uint16(s.L))
	case inst.SBC_HL_SP:
		execSbcHL(s, s.SP)

	// === NOP ===
	case inst.NOP:
		// do nothing

	// === Wave 5: Memory ops — (HL)/(BC)/(DE) indirect ===
	// All use s.M as the virtual memory byte.

	// LD r, (HL): r = M
	case inst.LD_A_HLI:
		s.A = s.M
	case inst.LD_B_HLI:
		s.B = s.M
	case inst.LD_C_HLI:
		s.C = s.M
	case inst.LD_D_HLI:
		s.D = s.M
	case inst.LD_E_HLI:
		s.E = s.M
	case inst.LD_H_HLI:
		s.H = s.M
	case inst.LD_L_HLI:
		s.L = s.M

	// LD (HL), r: M = r
	case inst.LD_HLI_A:
		s.M = s.A
	case inst.LD_HLI_B:
		s.M = s.B
	case inst.LD_HLI_C:
		s.M = s.C
	case inst.LD_HLI_D:
		s.M = s.D
	case inst.LD_HLI_E:
		s.M = s.E
	case inst.LD_HLI_H:
		s.M = s.H
	case inst.LD_HLI_L:
		s.M = s.L

	// LD (HL), n: M = imm
	case inst.LD_HLI_N:
		s.M = uint8(imm)

	// LD A, (BC)/(DE): A = M (same address assumption)
	case inst.LD_A_BCI:
		s.A = s.M
	case inst.LD_A_DEI:
		s.A = s.M

	// LD (BC), A / LD (DE), A: M = A
	case inst.LD_BCI_A:
		s.M = s.A
	case inst.LD_DEI_A:
		s.M = s.A

	// ALU A, (HL)
	case inst.ADD_A_HLI:
		execAdd(s, s.M)
	case inst.ADC_A_HLI:
		execAdc(s, s.M)
	case inst.SUB_HLI:
		execSub(s, s.M)
	case inst.SBC_A_HLI:
		execSbc(s, s.M)
	case inst.AND_HLI:
		execAnd(s, s.M)
	case inst.XOR_HLI:
		execXor(s, s.M)
	case inst.OR_HLI:
		execOr(s, s.M)
	case inst.CP_HLI:
		execCp(s, s.M)

	// INC/DEC (HL)
	case inst.INC_HLI:
		execInc(s, &s.M)
	case inst.DEC_HLI:
		execDec(s, &s.M)

	// CB-prefix rotate/shift (HL)
	case inst.RLC_HLI:
		s.M = execRlc(s, s.M)
	case inst.RRC_HLI:
		s.M = execRrc(s, s.M)
	case inst.RL_HLI:
		s.M = execRl(s, s.M)
	case inst.RR_HLI:
		s.M = execRr(s, s.M)
	case inst.SLA_HLI:
		s.M = execSla(s, s.M)
	case inst.SRA_HLI:
		s.M = execSra(s, s.M)
	case inst.SRL_HLI:
		s.M = execSrl(s, s.M)
	case inst.SLL_HLI:
		s.M = execSll(s, s.M)

	// BIT n, (HL)
	case inst.BIT_0_HLI:
		execBit(s, s.M, 0)
	case inst.BIT_1_HLI:
		execBit(s, s.M, 1)
	case inst.BIT_2_HLI:
		execBit(s, s.M, 2)
	case inst.BIT_3_HLI:
		execBit(s, s.M, 3)
	case inst.BIT_4_HLI:
		execBit(s, s.M, 4)
	case inst.BIT_5_HLI:
		execBit(s, s.M, 5)
	case inst.BIT_6_HLI:
		execBit(s, s.M, 6)
	case inst.BIT_7_HLI:
		execBit(s, s.M, 7)

	// RES n, (HL)
	case inst.RES_0_HLI:
		s.M &^= 1 << 0
	case inst.RES_1_HLI:
		s.M &^= 1 << 1
	case inst.RES_2_HLI:
		s.M &^= 1 << 2
	case inst.RES_3_HLI:
		s.M &^= 1 << 3
	case inst.RES_4_HLI:
		s.M &^= 1 << 4
	case inst.RES_5_HLI:
		s.M &^= 1 << 5
	case inst.RES_6_HLI:
		s.M &^= 1 << 6
	case inst.RES_7_HLI:
		s.M &^= 1 << 7

	// SET n, (HL)
	case inst.SET_0_HLI:
		s.M |= 1 << 0
	case inst.SET_1_HLI:
		s.M |= 1 << 1
	case inst.SET_2_HLI:
		s.M |= 1 << 2
	case inst.SET_3_HLI:
		s.M |= 1 << 3
	case inst.SET_4_HLI:
		s.M |= 1 << 4
	case inst.SET_5_HLI:
		s.M |= 1 << 5
	case inst.SET_6_HLI:
		s.M |= 1 << 6
	case inst.SET_7_HLI:
		s.M |= 1 << 7

	default:
		panic("unhandled opcode in Exec")
	}
	return inst.TStates(op)
}

// --- ALU helpers, ported from remogatto/z80 ---

func execAdd(s *State, value uint8) {
	addtemp := uint16(s.A) + uint16(value)
	lookup := ((s.A & 0x88) >> 3) | ((value & 0x88) >> 2) | uint8((addtemp&0x88)>>1)
	s.A = uint8(addtemp)
	s.F = bsel(addtemp&0x100 != 0, FlagC, 0) |
		HalfcarryAddTable[lookup&0x07] |
		OverflowAddTable[lookup>>4] |
		Sz53Table[s.A]
}

func execAdc(s *State, value uint8) {
	adctemp := uint16(s.A) + uint16(value) + uint16(s.F&FlagC)
	lookup := uint8(((uint16(s.A) & 0x88) >> 3) | ((uint16(value) & 0x88) >> 2) | ((adctemp & 0x88) >> 1))
	s.A = uint8(adctemp)
	s.F = bsel(adctemp&0x100 != 0, FlagC, 0) |
		HalfcarryAddTable[lookup&0x07] |
		OverflowAddTable[lookup>>4] |
		Sz53Table[s.A]
}

func execSub(s *State, value uint8) {
	subtemp := uint16(s.A) - uint16(value)
	lookup := ((s.A & 0x88) >> 3) | ((value & 0x88) >> 2) | uint8((subtemp&0x88)>>1)
	s.A = uint8(subtemp)
	s.F = bsel(subtemp&0x100 != 0, FlagC, 0) | FlagN |
		HalfcarrySubTable[lookup&0x07] |
		OverflowSubTable[lookup>>4] |
		Sz53Table[s.A]
}

func execSbc(s *State, value uint8) {
	sbctemp := uint16(s.A) - uint16(value) - uint16(s.F&FlagC)
	lookup := ((s.A & 0x88) >> 3) | ((value & 0x88) >> 2) | uint8((sbctemp&0x88)>>1)
	s.A = uint8(sbctemp)
	s.F = bsel(sbctemp&0x100 != 0, FlagC, 0) | FlagN |
		HalfcarrySubTable[lookup&0x07] |
		OverflowSubTable[lookup>>4] |
		Sz53Table[s.A]
}

func execAnd(s *State, value uint8) {
	s.A &= value
	s.F = FlagH | Sz53pTable[s.A]
}

func execOr(s *State, value uint8) {
	s.A |= value
	s.F = Sz53pTable[s.A]
}

func execXor(s *State, value uint8) {
	s.A ^= value
	s.F = Sz53pTable[s.A]
}

func execCp(s *State, value uint8) {
	cptemp := uint16(s.A) - uint16(value)
	lookup := ((s.A & 0x88) >> 3) | ((value & 0x88) >> 2) | uint8((cptemp&0x88)>>1)
	s.F = bsel(cptemp&0x100 != 0, FlagC, bsel(cptemp != 0, 0, FlagZ)) |
		FlagN |
		HalfcarrySubTable[lookup&0x07] |
		OverflowSubTable[lookup>>4] |
		(value & (Flag3 | Flag5)) |
		uint8(cptemp&uint16(FlagS))
}

func execInc(s *State, reg *uint8) {
	*reg++
	s.F = (s.F & FlagC) |
		bsel(*reg == 0x80, FlagV, 0) |
		bsel(*reg&0x0F != 0, 0, FlagH) |
		Sz53Table[*reg]
}

func execDec(s *State, reg *uint8) {
	s.F = (s.F & FlagC) | bsel(*reg&0x0F != 0, 0, FlagH) | FlagN
	*reg--
	s.F |= bsel(*reg == 0x7F, FlagV, 0) | Sz53Table[*reg]
}

func execDaa(s *State) {
	var add, carry uint8
	carry = s.F & FlagC
	if (s.F&FlagH) != 0 || (s.A&0x0F) > 9 {
		add = 6
	}
	if carry != 0 || s.A > 0x99 {
		add |= 0x60
	}
	if s.A > 0x99 {
		carry = FlagC
	}
	if (s.F & FlagN) != 0 {
		execSub(s, add)
	} else {
		execAdd(s, add)
	}
	s.F = (s.F & ^(FlagC | FlagP)) | carry | ParityTable[s.A]
}

// CB-prefix rotate/shift helpers (return the new value)
func execRlc(s *State, v uint8) uint8 {
	v = (v << 1) | (v >> 7)
	s.F = (v & FlagC) | Sz53pTable[v]
	return v
}

func execRrc(s *State, v uint8) uint8 {
	s.F = v & FlagC
	v = (v >> 1) | (v << 7)
	s.F |= Sz53pTable[v]
	return v
}

func execRl(s *State, v uint8) uint8 {
	old := v
	v = (v << 1) | (s.F & FlagC)
	s.F = (old >> 7) | Sz53pTable[v]
	return v
}

func execRr(s *State, v uint8) uint8 {
	old := v
	v = (v >> 1) | (s.F << 7)
	s.F = (old & FlagC) | Sz53pTable[v]
	return v
}

func execSla(s *State, v uint8) uint8 {
	s.F = v >> 7
	v <<= 1
	s.F |= Sz53pTable[v]
	return v
}

func execSra(s *State, v uint8) uint8 {
	s.F = v & FlagC
	v = (v & 0x80) | (v >> 1)
	s.F |= Sz53pTable[v]
	return v
}

func execSrl(s *State, v uint8) uint8 {
	s.F = v & FlagC
	v >>= 1
	s.F |= Sz53pTable[v]
	return v
}

// execAddHL implements ADD HL, rr: 16-bit add, sets H (bit 11 carry), N=0, C (bit 15 carry).
// Preserves S, Z, P/V flags.
func execAddHL(s *State, value uint16) {
	hl := uint16(s.H)<<8 | uint16(s.L)
	result := uint32(hl) + uint32(value)
	// Half-carry from bit 11
	hc := (hl & 0x0FFF) + (value & 0x0FFF)
	s.F = (s.F & (FlagS | FlagZ | FlagP)) | // preserve S, Z, P/V
		bsel(hc&0x1000 != 0, FlagH, 0) | // half-carry from bit 11
		bsel(result&0x10000 != 0, FlagC, 0) | // carry from bit 15
		(uint8(result>>8) & (Flag3 | Flag5)) // undocumented bits from high byte
	s.H = uint8(result >> 8)
	s.L = uint8(result)
}

// execAdcHL implements ADC HL, rr: 16-bit add with carry.
// Full flag computation: S, Z, H (bit 11), P/V (overflow), N=0, C.
// From remogatto/z80: uses lookup tables for half-carry and overflow.
func execAdcHL(s *State, value uint16) {
	hl := uint16(s.H)<<8 | uint16(s.L)
	carry := uint(s.F & FlagC)
	result := uint(hl) + uint(value) + carry
	// Lookup: bits 11 and 15 of hl, value, result → 3-bit index for half-carry, 3-bit for overflow
	lookup := byte(((uint(hl) & 0x8800) >> 11) | ((uint(value) & 0x8800) >> 10) | ((result & 0x8800) >> 9))
	s.H = uint8(result >> 8)
	s.L = uint8(result)
	s.F = bsel(result&0x10000 != 0, FlagC, 0) |
		OverflowAddTable[lookup>>4] |
		(s.H & (Flag3 | Flag5 | FlagS)) |
		HalfcarryAddTable[lookup&0x07] |
		bsel(s.H|s.L != 0, 0, FlagZ)
}

// execSbcHL implements SBC HL, rr: 16-bit subtract with carry.
// Full flag computation: S, Z, H (bit 11), P/V (overflow), N=1, C.
// From remogatto/z80: uses lookup tables for half-carry and overflow.
func execSbcHL(s *State, value uint16) {
	hl := uint16(s.H)<<8 | uint16(s.L)
	carry := uint(s.F & FlagC)
	result := uint(hl) - uint(value) - carry
	lookup := byte(((uint(hl) & 0x8800) >> 11) | ((uint(value) & 0x8800) >> 10) | (((result) & 0x8800) >> 9))
	s.H = uint8(result >> 8)
	s.L = uint8(result)
	s.F = bsel(result&0x10000 != 0, FlagC, 0) |
		FlagN |
		OverflowSubTable[lookup>>4] |
		(s.H & (Flag3 | Flag5 | FlagS)) |
		HalfcarrySubTable[lookup&0x07] |
		bsel(s.H|s.L != 0, 0, FlagZ)
}

// execBit implements BIT n, r: test bit n of register, set flags accordingly.
// From remogatto/z80: F = (F & C) | H | (r & (flag3|flag5)); if bit is zero → F |= P|Z; if n==7 && bit set → F |= S.
func execBit(s *State, r uint8, bit uint8) {
	s.F = (s.F & FlagC) | FlagH | (r & (Flag3 | Flag5))
	if r&(1<<bit) == 0 {
		s.F |= FlagP | FlagZ
	}
	if bit == 7 && r&0x80 != 0 {
		s.F |= FlagS
	}
}

// execSll implements the undocumented SLL: shift left, set bit 0 to 1.
func execSll(s *State, v uint8) uint8 {
	s.F = v >> 7
	v = (v << 1) | 0x01
	s.F |= Sz53pTable[v]
	return v
}

// bsel returns a if cond is true, else b. Branchless flag selection.
func bsel(cond bool, a, b uint8) uint8 {
	if cond {
		return a
	}
	return b
}

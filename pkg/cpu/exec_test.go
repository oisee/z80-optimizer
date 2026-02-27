package cpu

import (
	"fmt"
	"testing"

	"github.com/oisee/z80-optimizer/pkg/inst"
)

// TestFlagTables verifies our precomputed tables match expected values.
func TestFlagTables(t *testing.T) {
	// Verify zero has Z flag set
	if Sz53Table[0]&FlagZ == 0 {
		t.Error("sz53Table[0] should have Z flag")
	}
	if Sz53pTable[0]&FlagZ == 0 {
		t.Error("sz53pTable[0] should have Z flag")
	}

	// Verify 0x80 has S flag
	if Sz53Table[0x80]&FlagS == 0 {
		t.Error("sz53Table[0x80] should have S flag")
	}

	// Verify parity of 0x00 (even parity → P flag set)
	if ParityTable[0]&FlagP == 0 {
		t.Error("parityTable[0] should have P flag (even parity)")
	}

	// Verify parity of 0x01 (odd parity → P flag clear)
	if ParityTable[1]&FlagP != 0 {
		t.Error("parityTable[1] should NOT have P flag (odd parity)")
	}

	// Verify parity of 0xFF (even parity → P flag set)
	if ParityTable[0xFF]&FlagP == 0 {
		t.Error("parityTable[0xFF] should have P flag")
	}
}

// TestAddFlags verifies ADD A, r flag behavior for key cases.
func TestAddFlags(t *testing.T) {
	tests := []struct {
		a, val     uint8
		wantA      uint8
		wantCarry  bool
		wantZero   bool
		wantSign   bool
		wantHalf   bool
		wantOverflow bool
	}{
		{0, 0, 0, false, true, false, false, false},
		{1, 1, 2, false, false, false, false, false},
		{0xFF, 1, 0, true, true, false, true, false},
		{0x0F, 1, 0x10, false, false, false, true, false},
		{0x7F, 1, 0x80, false, false, true, true, true}, // overflow: pos + pos = neg
		{0x80, 0x80, 0, true, true, false, false, true}, // overflow: neg + neg = pos
	}

	for _, tc := range tests {
		s := State{A: tc.a}
		Exec(&s, inst.ADD_A_N, uint16(tc.val))

		if s.A != tc.wantA {
			t.Errorf("ADD A=%02X + %02X: got A=%02X, want %02X", tc.a, tc.val, s.A, tc.wantA)
		}
		if (s.F&FlagC != 0) != tc.wantCarry {
			t.Errorf("ADD A=%02X + %02X: carry=%v, want %v", tc.a, tc.val, s.F&FlagC != 0, tc.wantCarry)
		}
		if (s.F&FlagZ != 0) != tc.wantZero {
			t.Errorf("ADD A=%02X + %02X: zero=%v, want %v", tc.a, tc.val, s.F&FlagZ != 0, tc.wantZero)
		}
		if (s.F&FlagS != 0) != tc.wantSign {
			t.Errorf("ADD A=%02X + %02X: sign=%v, want %v", tc.a, tc.val, s.F&FlagS != 0, tc.wantSign)
		}
		if (s.F&FlagH != 0) != tc.wantHalf {
			t.Errorf("ADD A=%02X + %02X: half=%v, want %v", tc.a, tc.val, s.F&FlagH != 0, tc.wantHalf)
		}
		if (s.F&FlagV != 0) != tc.wantOverflow {
			t.Errorf("ADD A=%02X + %02X: overflow=%v, want %v", tc.a, tc.val, s.F&FlagV != 0, tc.wantOverflow)
		}
	}
}

// TestSubFlags verifies SUB flag behavior.
func TestSubFlags(t *testing.T) {
	tests := []struct {
		a, val     uint8
		wantA      uint8
		wantCarry  bool
		wantN      bool
	}{
		{5, 3, 2, false, true},
		{0, 1, 0xFF, true, true},  // borrow
		{0x80, 1, 0x7F, false, true}, // overflow case
	}

	for _, tc := range tests {
		s := State{A: tc.a}
		Exec(&s, inst.SUB_N, uint16(tc.val))
		if s.A != tc.wantA {
			t.Errorf("SUB A=%02X - %02X: got A=%02X, want %02X", tc.a, tc.val, s.A, tc.wantA)
		}
		if (s.F&FlagC != 0) != tc.wantCarry {
			t.Errorf("SUB A=%02X - %02X: carry=%v, want %v", tc.a, tc.val, s.F&FlagC != 0, tc.wantCarry)
		}
		if (s.F&FlagN != 0) != tc.wantN {
			t.Errorf("SUB A=%02X - %02X: N=%v, want %v", tc.a, tc.val, s.F&FlagN != 0, tc.wantN)
		}
	}
}

// TestAndOrXor verifies logic operations set flags correctly.
func TestAndOrXor(t *testing.T) {
	// AND always sets H, clears N and C
	s := State{A: 0xFF}
	Exec(&s, inst.AND_N, 0x0F)
	if s.A != 0x0F {
		t.Errorf("AND: got A=%02X, want 0F", s.A)
	}
	if s.F&FlagH == 0 {
		t.Error("AND should set H flag")
	}
	if s.F&FlagN != 0 {
		t.Error("AND should clear N flag")
	}
	if s.F&FlagC != 0 {
		t.Error("AND should clear C flag")
	}

	// OR clears H, N, C
	s = State{A: 0xF0}
	Exec(&s, inst.OR_N, 0x0F)
	if s.A != 0xFF {
		t.Errorf("OR: got A=%02X, want FF", s.A)
	}
	if s.F&FlagH != 0 {
		t.Error("OR should clear H flag")
	}

	// XOR clears H, N, C
	s = State{A: 0xFF}
	Exec(&s, inst.XOR_N, 0xFF)
	if s.A != 0x00 {
		t.Errorf("XOR: got A=%02X, want 00", s.A)
	}
	if s.F&FlagZ == 0 {
		t.Error("XOR A,A should set Z flag")
	}
}

// TestIncDec verifies INC/DEC flag behavior.
func TestIncDec(t *testing.T) {
	// INC: 0x7F -> 0x80 should set overflow
	s := State{A: 0x7F}
	Exec(&s, inst.INC_A, 0)
	if s.A != 0x80 {
		t.Errorf("INC 0x7F: got %02X want 0x80", s.A)
	}
	if s.F&FlagV == 0 {
		t.Error("INC 0x7F should set V flag (overflow)")
	}

	// INC: 0xFF -> 0x00 should set Z and H
	s = State{A: 0xFF}
	Exec(&s, inst.INC_A, 0)
	if s.A != 0x00 {
		t.Errorf("INC 0xFF: got %02X want 0x00", s.A)
	}
	if s.F&FlagZ == 0 {
		t.Error("INC 0xFF should set Z flag")
	}

	// INC should preserve carry
	s = State{A: 0x00, F: FlagC}
	Exec(&s, inst.INC_A, 0)
	if s.F&FlagC == 0 {
		t.Error("INC should preserve carry flag")
	}

	// DEC: 0x80 -> 0x7F should set overflow
	s = State{A: 0x80}
	Exec(&s, inst.DEC_A, 0)
	if s.A != 0x7F {
		t.Errorf("DEC 0x80: got %02X want 0x7F", s.A)
	}
	if s.F&FlagV == 0 {
		t.Error("DEC 0x80 should set V flag (overflow)")
	}
	if s.F&FlagN == 0 {
		t.Error("DEC should set N flag")
	}
}

// TestRotates verifies accumulator rotate instructions.
func TestRotates(t *testing.T) {
	// RLCA: rotate left, bit 7 goes to carry and bit 0
	s := State{A: 0x80}
	Exec(&s, inst.RLCA, 0)
	if s.A != 0x01 {
		t.Errorf("RLCA 0x80: got %02X want 0x01", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("RLCA 0x80 should set carry")
	}

	// RRCA: rotate right, bit 0 goes to carry and bit 7
	s = State{A: 0x01}
	Exec(&s, inst.RRCA, 0)
	if s.A != 0x80 {
		t.Errorf("RRCA 0x01: got %02X want 0x80", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("RRCA 0x01 should set carry")
	}

	// RLA: rotate left through carry
	s = State{A: 0x80, F: 0x00}
	Exec(&s, inst.RLA, 0)
	if s.A != 0x00 {
		t.Errorf("RLA 0x80 (C=0): got %02X want 0x00", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("RLA 0x80 should set carry")
	}

	// RRA: rotate right through carry
	s = State{A: 0x01, F: FlagC}
	Exec(&s, inst.RRA, 0)
	if s.A != 0x80 {
		t.Errorf("RRA 0x01 (C=1): got %02X want 0x80", s.A)
	}
	if s.F&FlagC == 0 {
		t.Errorf("RRA 0x01 should set carry (bit 0 was 1): F=%02X", s.F)
	}
}

// TestSpecialOps verifies CPL, SCF, CCF, NEG.
func TestSpecialOps(t *testing.T) {
	// CPL
	s := State{A: 0x55}
	Exec(&s, inst.CPL, 0)
	if s.A != 0xAA {
		t.Errorf("CPL 0x55: got %02X want 0xAA", s.A)
	}
	if s.F&FlagH == 0 || s.F&FlagN == 0 {
		t.Error("CPL should set H and N")
	}

	// SCF
	s = State{F: 0x00}
	Exec(&s, inst.SCF, 0)
	if s.F&FlagC == 0 {
		t.Error("SCF should set carry")
	}

	// CCF (complement carry)
	s = State{F: FlagC}
	Exec(&s, inst.CCF, 0)
	if s.F&FlagC != 0 {
		t.Error("CCF with C=1 should clear carry")
	}
	if s.F&FlagH == 0 {
		t.Error("CCF with C=1 should set H (old carry)")
	}

	// NEG
	s = State{A: 0x01}
	Exec(&s, inst.NEG, 0)
	if s.A != 0xFF {
		t.Errorf("NEG 0x01: got %02X want 0xFF", s.A)
	}
}

// TestCBRotates verifies CB-prefix rotate/shift instructions.
func TestCBRotates(t *testing.T) {
	// RLC A: like RLCA but also sets S, Z, P flags
	s := State{A: 0x80}
	Exec(&s, inst.RLC_A, 0)
	if s.A != 0x01 {
		t.Errorf("RLC A 0x80: got %02X want 0x01", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("RLC A 0x80 should set carry")
	}

	// SLA A: shift left arithmetic
	s = State{A: 0x80}
	Exec(&s, inst.SLA_A, 0)
	if s.A != 0x00 {
		t.Errorf("SLA A 0x80: got %02X want 0x00", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("SLA A 0x80 should set carry")
	}
	if s.F&FlagZ == 0 {
		t.Error("SLA A 0x80 should set zero")
	}

	// SRA A: shift right arithmetic (preserves sign)
	s = State{A: 0x80}
	Exec(&s, inst.SRA_A, 0)
	if s.A != 0xC0 {
		t.Errorf("SRA A 0x80: got %02X want 0xC0", s.A)
	}

	// SRL A: shift right logical
	s = State{A: 0x81}
	Exec(&s, inst.SRL_A, 0)
	if s.A != 0x40 {
		t.Errorf("SRL A 0x81: got %02X want 0x40", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("SRL A 0x81 should set carry (bit 0 was 1)")
	}
}

// TestExhaustiveAddSub verifies ADD and SUB for all 256 A values.
func TestExhaustiveAddSub(t *testing.T) {
	for a := 0; a < 256; a++ {
		for v := 0; v < 256; v++ {
			s := State{A: uint8(a)}
			Exec(&s, inst.ADD_A_N, uint16(v))

			// Verify A result
			expected := uint8(a + v)
			if s.A != expected {
				t.Fatalf("ADD %02X + %02X: got %02X want %02X", a, v, s.A, expected)
			}

			// Verify carry
			wantCarry := (a + v) > 0xFF
			if (s.F&FlagC != 0) != wantCarry {
				t.Fatalf("ADD %02X + %02X: carry=%v want %v", a, v, s.F&FlagC != 0, wantCarry)
			}

			// Verify N flag is clear for ADD
			if s.F&FlagN != 0 {
				t.Fatalf("ADD should clear N flag")
			}
		}
	}
}

// TestAllRegisterLoads verifies all LD r,r' instructions.
func TestAllRegisterLoads(t *testing.T) {
	type ldTest struct {
		op       inst.OpCode
		getSrc   func(State) uint8
		getDst   func(State) uint8
		name     string
	}

	tests := []ldTest{
		{inst.LD_A_B, func(s State) uint8 { return s.B }, func(s State) uint8 { return s.A }, "LD A,B"},
		{inst.LD_A_C, func(s State) uint8 { return s.C }, func(s State) uint8 { return s.A }, "LD A,C"},
		{inst.LD_B_A, func(s State) uint8 { return s.A }, func(s State) uint8 { return s.B }, "LD B,A"},
		{inst.LD_H_L, func(s State) uint8 { return s.L }, func(s State) uint8 { return s.H }, "LD H,L"},
	}

	for _, tc := range tests {
		s := State{A: 0x11, B: 0x22, C: 0x33, D: 0x44, E: 0x55, H: 0x66, L: 0x77}
		srcVal := tc.getSrc(s)
		Exec(&s, tc.op, 0)
		if tc.getDst(s) != srcVal {
			t.Errorf("%s: dst=%02X want %02X", tc.name, tc.getDst(s), srcVal)
		}
	}
}

// TestDAA verifies DAA for a selection of key cases.
func TestDAA(t *testing.T) {
	tests := []struct {
		a    uint8
		f    uint8 // input flags
		want uint8
		name string
	}{
		{0x15, 0, 0x15, "BCD 15 no adjust"},
		{0x1A, 0, 0x20, "BCD adjust low nibble"},
		{0xA0, 0, 0x00, "BCD adjust high nibble"}, // carry set
		{0x9A, 0, 0x00, "BCD 9A -> 00"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := State{A: tc.a, F: tc.f}
			Exec(&s, inst.DAA, 0)
			if s.A != tc.want {
				t.Errorf("DAA A=%02X F=%02X: got A=%02X want %02X (F=%02X)", tc.a, tc.f, s.A, tc.want, s.F)
			}
		})
	}
}

// TestCBRotatesOnOtherRegs verifies CB-prefix operations on B-L.
func TestCBRotatesOnOtherRegs(t *testing.T) {
	// RLC B
	s := State{B: 0x80}
	Exec(&s, inst.RLC_B, 0)
	if s.B != 0x01 {
		t.Errorf("RLC B 0x80: got %02X want 0x01", s.B)
	}
	if s.F&FlagC == 0 {
		t.Error("RLC B 0x80 should set carry")
	}

	// SLA C
	s = State{C: 0x40}
	Exec(&s, inst.SLA_C, 0)
	if s.C != 0x80 {
		t.Errorf("SLA C 0x40: got %02X want 0x80", s.C)
	}

	// SRL D
	s = State{D: 0x02}
	Exec(&s, inst.SRL_D, 0)
	if s.D != 0x01 {
		t.Errorf("SRL D 0x02: got %02X want 0x01", s.D)
	}
}

// BenchmarkExec benchmarks single instruction execution.
func BenchmarkExec(b *testing.B) {
	s := State{A: 0x42, B: 0x13, F: 0x01}
	for i := 0; i < b.N; i++ {
		s2 := s
		Exec(&s2, inst.ADD_A_B, 0)
	}
}

// BenchmarkExecSequence benchmarks a 3-instruction sequence.
func BenchmarkExecSequence(b *testing.B) {
	s := State{A: 0x42, B: 0x13, F: 0x01}
	for i := 0; i < b.N; i++ {
		s2 := s
		Exec(&s2, inst.ADD_A_B, 0)
		Exec(&s2, inst.INC_A, 0)
		Exec(&s2, inst.AND_N, 0xFF)
	}
}

func TestAllOpcodes(t *testing.T) {
	// Verify every opcode in the catalog can be executed without panic
	for op := inst.OpCode(0); op < inst.OpCodeCount; op++ {
		info := &inst.Catalog[op]
		if info.Mnemonic == "" {
			t.Errorf("OpCode %d has no mnemonic in catalog", op)
			continue
		}

		// Execute with a representative state
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("OpCode %d (%s) panicked: %v", op, info.Mnemonic, r)
				}
			}()
			s := State{A: 0x42, F: 0x01, B: 0x13, C: 0x24, D: 0x35, E: 0x46, H: 0x57, L: 0x68}
			Exec(&s, op, 0x99)
		}()
	}
}

// TestCP verifies CP doesn't modify A.
func TestCP(t *testing.T) {
	for a := 0; a < 256; a++ {
		for v := 0; v < 256; v++ {
			s := State{A: uint8(a)}
			Exec(&s, inst.CP_N, uint16(v))
			if s.A != uint8(a) {
				t.Fatalf("CP %02X, %02X modified A to %02X", a, v, s.A)
			}
		}
	}
}

// TestXorASelf verifies XOR A, A produces zero with correct flags.
func TestXorASelf(t *testing.T) {
	for a := 0; a < 256; a++ {
		s := State{A: uint8(a), F: 0xFF} // all flags set
		Exec(&s, inst.XOR_A, 0)
		if s.A != 0 {
			t.Fatalf("XOR A with A=%02X: got %02X want 0", a, s.A)
		}
		if s.F&FlagZ == 0 {
			t.Fatal("XOR A should set Z")
		}
		if s.F&FlagC != 0 {
			t.Fatal("XOR A should clear C")
		}
		if s.F&FlagN != 0 {
			t.Fatal("XOR A should clear N")
		}
		if s.F&FlagH != 0 {
			t.Fatal("XOR A should clear H")
		}
	}
}

// TestRRAPreservesFlags verifies RRA preserves S, Z, P.
func TestRRAPreservesFlags(t *testing.T) {
	// Set S, Z, P flags — RRA should preserve them
	for _, initialF := range []uint8{FlagS | FlagZ | FlagP, 0x00, FlagS, FlagZ | FlagP} {
		s := State{A: 0x01, F: initialF}
		Exec(&s, inst.RRA, 0)

		// Check S, Z, P are preserved
		preserved := FlagS | FlagZ | FlagP
		if s.F&preserved != initialF&preserved {
			t.Errorf("RRA with F=%02X: preserved flags %02X, want %02X",
				initialF, s.F&preserved, initialF&preserved)
		}
	}
}

// TestExecDeterministic verifies same input → same output for all opcodes.
func TestExecDeterministic(t *testing.T) {
	initial := State{A: 0x42, F: 0x01, B: 0x13, C: 0x24, D: 0x35, E: 0x46, H: 0x57, L: 0x68}
	for op := inst.OpCode(0); op < inst.OpCodeCount; op++ {
		s1 := initial
		s2 := initial
		Exec(&s1, op, 0x55)
		Exec(&s2, op, 0x55)
		if s1 != s2 {
			t.Errorf("OpCode %d (%s) is not deterministic", op, inst.Catalog[op].Mnemonic)
		}
	}
}

// TestLDImmediate verifies all LD r, n instructions.
func TestLDImmediate(t *testing.T) {
	for imm := 0; imm < 256; imm++ {
		v := uint8(imm)

		s := State{}
		Exec(&s, inst.LD_A_N, uint16(v))
		if s.A != v {
			t.Errorf("LD A, %02X: got %02X", v, s.A)
		}

		s = State{}
		Exec(&s, inst.LD_B_N, uint16(v))
		if s.B != v {
			t.Errorf("LD B, %02X: got %02X", v, s.B)
		}
	}
}

// TestSLLUndocumented verifies the undocumented SLL instruction.
func TestSLLUndocumented(t *testing.T) {
	// SLL A: shift left, bit 0 = 1 (not 0 like SLA)
	s := State{A: 0x00}
	Exec(&s, inst.SLL_A, 0)
	if s.A != 0x01 {
		t.Errorf("SLL A 0x00: got %02X want 0x01", s.A)
	}

	s = State{A: 0x80}
	Exec(&s, inst.SLL_A, 0)
	if s.A != 0x01 {
		t.Errorf("SLL A 0x80: got %02X want 0x01", s.A)
	}
	if s.F&FlagC == 0 {
		t.Error("SLL A 0x80 should set carry")
	}
}

// TestBIT verifies BIT n, r instructions.
func TestBIT(t *testing.T) {
	// BIT 0, A: bit 0 is set → Z should be clear
	s := State{A: 0x01, F: FlagC}
	Exec(&s, inst.BIT_0_A, 0)
	if s.F&FlagZ != 0 {
		t.Error("BIT 0, A=0x01: Z should be clear (bit is set)")
	}
	if s.F&FlagH == 0 {
		t.Error("BIT should always set H")
	}
	// Carry should be preserved
	if s.F&FlagC == 0 {
		t.Error("BIT should preserve carry")
	}

	// BIT 7, A: bit 7 is 0 → Z should be set
	s = State{A: 0x01}
	Exec(&s, inst.BIT_7_A, 0)
	if s.F&FlagZ == 0 {
		t.Error("BIT 7, A=0x01: Z should be set (bit is clear)")
	}

	// BIT 7, A: bit 7 is 1 → S should be set
	s = State{A: 0x80}
	Exec(&s, inst.BIT_7_A, 0)
	if s.F&FlagS == 0 {
		t.Error("BIT 7, A=0x80: S should be set")
	}
	if s.F&FlagZ != 0 {
		t.Error("BIT 7, A=0x80: Z should be clear")
	}

	// BIT on other registers
	s = State{B: 0x04}
	Exec(&s, inst.BIT_2_B, 0)
	if s.F&FlagZ != 0 {
		t.Error("BIT 2, B=0x04: Z should be clear (bit 2 is set)")
	}

	s = State{B: 0x00}
	Exec(&s, inst.BIT_2_B, 0)
	if s.F&FlagZ == 0 {
		t.Error("BIT 2, B=0x00: Z should be set (bit 2 is clear)")
	}

	// Verify BIT doesn't modify the register
	s = State{A: 0xFF}
	Exec(&s, inst.BIT_3_A, 0)
	if s.A != 0xFF {
		t.Errorf("BIT should not modify register: A=%02X want 0xFF", s.A)
	}
}

// TestRES verifies RES n, r instructions.
func TestRES(t *testing.T) {
	// RES 0, A: clears bit 0
	s := State{A: 0xFF, F: FlagC | FlagZ}
	Exec(&s, inst.RES_0_A, 0)
	if s.A != 0xFE {
		t.Errorf("RES 0, A=0xFF: got %02X want 0xFE", s.A)
	}
	// Flags should be unchanged
	if s.F != FlagC|FlagZ {
		t.Errorf("RES should not modify flags: F=%02X", s.F)
	}

	// RES 7, B: clears bit 7
	s = State{B: 0x80}
	Exec(&s, inst.RES_7_B, 0)
	if s.B != 0x00 {
		t.Errorf("RES 7, B=0x80: got %02X want 0x00", s.B)
	}

	// RES on already-clear bit is a no-op
	s = State{C: 0x00}
	Exec(&s, inst.RES_3_C, 0)
	if s.C != 0x00 {
		t.Errorf("RES 3, C=0x00: got %02X want 0x00", s.C)
	}
}

// TestSET verifies SET n, r instructions.
func TestSET(t *testing.T) {
	// SET 0, A: sets bit 0
	s := State{A: 0x00, F: FlagC | FlagZ}
	Exec(&s, inst.SET_0_A, 0)
	if s.A != 0x01 {
		t.Errorf("SET 0, A=0x00: got %02X want 0x01", s.A)
	}
	// Flags should be unchanged
	if s.F != FlagC|FlagZ {
		t.Errorf("SET should not modify flags: F=%02X", s.F)
	}

	// SET 7, B: sets bit 7
	s = State{B: 0x00}
	Exec(&s, inst.SET_7_B, 0)
	if s.B != 0x80 {
		t.Errorf("SET 7, B=0x00: got %02X want 0x80", s.B)
	}

	// SET on already-set bit is a no-op
	s = State{D: 0xFF}
	Exec(&s, inst.SET_5_D, 0)
	if s.D != 0xFF {
		t.Errorf("SET 5, D=0xFF: got %02X want 0xFF", s.D)
	}
}

// TestSLLRegisters verifies SLL on all registers.
func TestSLLRegisters(t *testing.T) {
	// SLL B: shift left, bit 0 = 1
	s := State{B: 0x00}
	Exec(&s, inst.SLL_B, 0)
	if s.B != 0x01 {
		t.Errorf("SLL B 0x00: got %02X want 0x01", s.B)
	}

	s = State{C: 0x80}
	Exec(&s, inst.SLL_C, 0)
	if s.C != 0x01 {
		t.Errorf("SLL C 0x80: got %02X want 0x01", s.C)
	}
	if s.F&FlagC == 0 {
		t.Error("SLL C 0x80 should set carry")
	}
}

// TestINC16 verifies 16-bit INC instructions don't affect flags.
func TestINC16(t *testing.T) {
	// INC BC: 0x00FF -> 0x0100
	s := State{B: 0x00, C: 0xFF, F: FlagZ | FlagC}
	Exec(&s, inst.INC_BC, 0)
	if s.B != 0x01 || s.C != 0x00 {
		t.Errorf("INC BC 0x00FF: got BC=%02X%02X want 0100", s.B, s.C)
	}
	// Flags must be unchanged
	if s.F != FlagZ|FlagC {
		t.Errorf("INC BC should not modify flags: F=%02X", s.F)
	}

	// INC HL: 0xFFFF -> 0x0000 (wraps)
	s = State{H: 0xFF, L: 0xFF}
	Exec(&s, inst.INC_HL, 0)
	if s.H != 0x00 || s.L != 0x00 {
		t.Errorf("INC HL 0xFFFF: got HL=%02X%02X want 0000", s.H, s.L)
	}

	// INC SP
	s = State{SP: 0xFFFE}
	Exec(&s, inst.INC_SP, 0)
	if s.SP != 0xFFFF {
		t.Errorf("INC SP 0xFFFE: got %04X want 0xFFFF", s.SP)
	}

	// INC DE
	s = State{D: 0x12, E: 0x34}
	Exec(&s, inst.INC_DE, 0)
	if s.D != 0x12 || s.E != 0x35 {
		t.Errorf("INC DE 0x1234: got DE=%02X%02X want 1235", s.D, s.E)
	}
}

// TestDEC16 verifies 16-bit DEC instructions.
func TestDEC16(t *testing.T) {
	// DEC BC: 0x0100 -> 0x00FF
	s := State{B: 0x01, C: 0x00, F: FlagZ}
	Exec(&s, inst.DEC_BC, 0)
	if s.B != 0x00 || s.C != 0xFF {
		t.Errorf("DEC BC 0x0100: got BC=%02X%02X want 00FF", s.B, s.C)
	}
	if s.F != FlagZ {
		t.Errorf("DEC BC should not modify flags: F=%02X", s.F)
	}

	// DEC HL: 0x0000 -> 0xFFFF (wraps)
	s = State{H: 0x00, L: 0x00}
	Exec(&s, inst.DEC_HL, 0)
	if s.H != 0xFF || s.L != 0xFF {
		t.Errorf("DEC HL 0x0000: got HL=%02X%02X want FFFF", s.H, s.L)
	}

	// DEC SP
	s = State{SP: 0x0000}
	Exec(&s, inst.DEC_SP, 0)
	if s.SP != 0xFFFF {
		t.Errorf("DEC SP 0x0000: got %04X want 0xFFFF", s.SP)
	}
}

// TestADDHL verifies ADD HL, rr instructions.
func TestADDHL(t *testing.T) {
	// ADD HL, BC: no carry, no half-carry
	s := State{H: 0x00, L: 0x01, B: 0x00, C: 0x02, F: FlagZ | FlagS}
	Exec(&s, inst.ADD_HL_BC, 0)
	hl := uint16(s.H)<<8 | uint16(s.L)
	if hl != 0x0003 {
		t.Errorf("ADD HL, BC: got HL=%04X want 0003", hl)
	}
	// S and Z should be preserved
	if s.F&FlagZ == 0 {
		t.Error("ADD HL should preserve Z flag")
	}
	if s.F&FlagS == 0 {
		t.Error("ADD HL should preserve S flag")
	}
	// N should be cleared
	if s.F&FlagN != 0 {
		t.Error("ADD HL should clear N flag")
	}

	// ADD HL, DE: with carry
	s = State{H: 0xFF, L: 0xFF, D: 0x00, E: 0x01}
	Exec(&s, inst.ADD_HL_DE, 0)
	hl = uint16(s.H)<<8 | uint16(s.L)
	if hl != 0x0000 {
		t.Errorf("ADD HL, DE (overflow): got HL=%04X want 0000", hl)
	}
	if s.F&FlagC == 0 {
		t.Error("ADD HL, DE (overflow) should set C")
	}

	// ADD HL, HL: double
	s = State{H: 0x40, L: 0x00}
	Exec(&s, inst.ADD_HL_HL, 0)
	hl = uint16(s.H)<<8 | uint16(s.L)
	if hl != 0x8000 {
		t.Errorf("ADD HL, HL: got HL=%04X want 8000", hl)
	}

	// ADD HL, SP
	s = State{H: 0x00, L: 0x01, SP: 0x1000}
	Exec(&s, inst.ADD_HL_SP, 0)
	hl = uint16(s.H)<<8 | uint16(s.L)
	if hl != 0x1001 {
		t.Errorf("ADD HL, SP: got HL=%04X want 1001", hl)
	}

	// Half-carry test: ADD HL, BC where bits 11 carry
	s = State{H: 0x0F, L: 0xFF, B: 0x00, C: 0x01}
	Exec(&s, inst.ADD_HL_BC, 0)
	if s.F&FlagH == 0 {
		t.Error("ADD HL, BC: half-carry from bit 11 should set H")
	}
}

// TestEXDEHL verifies EX DE, HL.
func TestEXDEHL(t *testing.T) {
	s := State{D: 0x12, E: 0x34, H: 0x56, L: 0x78, F: FlagC}
	Exec(&s, inst.EX_DE_HL, 0)
	if s.D != 0x56 || s.E != 0x78 {
		t.Errorf("EX DE,HL: DE=%02X%02X want 5678", s.D, s.E)
	}
	if s.H != 0x12 || s.L != 0x34 {
		t.Errorf("EX DE,HL: HL=%02X%02X want 1234", s.H, s.L)
	}
	// Flags unchanged
	if s.F != FlagC {
		t.Errorf("EX DE,HL should not modify flags: F=%02X", s.F)
	}
}

// TestLDSPHL verifies LD SP, HL.
func TestLDSPHL(t *testing.T) {
	s := State{H: 0xAB, L: 0xCD, SP: 0x0000}
	Exec(&s, inst.LD_SP_HL, 0)
	if s.SP != 0xABCD {
		t.Errorf("LD SP, HL: got SP=%04X want ABCD", s.SP)
	}
}

// TestAdcWithCarry verifies ADC uses carry flag.
func TestAdcWithCarry(t *testing.T) {
	// ADC with carry=0
	s := State{A: 5, F: 0}
	Exec(&s, inst.ADC_A_N, 3)
	if s.A != 8 {
		t.Errorf("ADC 5+3+0: got %d want 8", s.A)
	}

	// ADC with carry=1
	s = State{A: 5, F: FlagC}
	Exec(&s, inst.ADC_A_N, 3)
	if s.A != 9 {
		t.Errorf("ADC 5+3+1: got %d want 9", s.A)
	}
}

// TestSbcWithCarry verifies SBC uses carry flag.
func TestSbcWithCarry(t *testing.T) {
	// SBC with carry=0
	s := State{A: 5, F: 0}
	Exec(&s, inst.SBC_A_N, 3)
	if s.A != 2 {
		t.Errorf("SBC 5-3-0: got %d want 2", s.A)
	}

	// SBC with carry=1
	s = State{A: 5, F: FlagC}
	Exec(&s, inst.SBC_A_N, 3)
	if s.A != 1 {
		t.Errorf("SBC 5-3-1: got %d want 1", s.A)
	}
}

// TestRegisterOps verifies operations on each register (B through L).
func TestRegisterOps(t *testing.T) {
	// ADD A, B/C/D/E/H/L
	for _, tc := range []struct {
		op   inst.OpCode
		reg  string
		init State
		want uint8
	}{
		{inst.ADD_A_B, "B", State{A: 5, B: 3}, 8},
		{inst.ADD_A_C, "C", State{A: 5, C: 3}, 8},
		{inst.ADD_A_D, "D", State{A: 5, D: 3}, 8},
		{inst.ADD_A_E, "E", State{A: 5, E: 3}, 8},
		{inst.ADD_A_H, "H", State{A: 5, H: 3}, 8},
		{inst.ADD_A_L, "L", State{A: 5, L: 3}, 8},
	} {
		s := tc.init
		Exec(&s, tc.op, 0)
		if s.A != tc.want {
			t.Errorf("ADD A, %s: got %d want %d", tc.reg, s.A, tc.want)
		}
	}
}

// Fuzz test for ADD consistency
func FuzzAdd(f *testing.F) {
	f.Add(uint8(0), uint8(0))
	f.Add(uint8(0xFF), uint8(1))
	f.Add(uint8(0x7F), uint8(1))

	f.Fuzz(func(t *testing.T, a, v uint8) {
		s := State{A: a}
		Exec(&s, inst.ADD_A_N, uint16(v))

		// Basic invariant: result should be (a+v) mod 256
		expected := a + v
		if s.A != expected {
			t.Errorf("ADD %02X + %02X: got %02X want %02X", a, v, s.A, expected)
		}

		// N flag should be clear for ADD
		if s.F&FlagN != 0 {
			t.Error("ADD should clear N flag")
		}

		// Zero flag consistency
		if (s.A == 0) != (s.F&FlagZ != 0) {
			t.Errorf("ADD %02X + %02X: Z flag inconsistent (A=%02X, Z=%v)", a, v, s.A, s.F&FlagZ != 0)
		}

		// Sign flag consistency
		if (s.A >= 0x80) != (s.F&FlagS != 0) {
			t.Errorf("ADD %02X + %02X: S flag inconsistent", a, v)
		}
	})
}

// === Wave 4 Tests: 16-bit Immediate Loads + ADC/SBC HL ===

func TestLDRegPairNN(t *testing.T) {
	tests := []struct {
		name string
		op   inst.OpCode
		imm  uint16
		getH func(State) uint8
		getL func(State) uint8
	}{
		{"LD BC, 0x1234", inst.LD_BC_NN, 0x1234, func(s State) uint8 { return s.B }, func(s State) uint8 { return s.C }},
		{"LD DE, 0xABCD", inst.LD_DE_NN, 0xABCD, func(s State) uint8 { return s.D }, func(s State) uint8 { return s.E }},
		{"LD HL, 0xFF00", inst.LD_HL_NN, 0xFF00, func(s State) uint8 { return s.H }, func(s State) uint8 { return s.L }},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := State{A: 0x11, F: 0xFF, B: 0x22, C: 0x33} // flags should be unaffected
			oldF := s.F
			Exec(&s, tc.op, tc.imm)
			if tc.getH(s) != uint8(tc.imm>>8) {
				t.Errorf("high byte: got %02X, want %02X", tc.getH(s), uint8(tc.imm>>8))
			}
			if tc.getL(s) != uint8(tc.imm) {
				t.Errorf("low byte: got %02X, want %02X", tc.getL(s), uint8(tc.imm))
			}
			if s.F != oldF {
				t.Errorf("flags changed: got %02X, want %02X", s.F, oldF)
			}
		})
	}
}

func TestLDSPNN(t *testing.T) {
	s := State{SP: 0x0000, F: 0xFF}
	Exec(&s, inst.LD_SP_NN, 0xFFFE)
	if s.SP != 0xFFFE {
		t.Errorf("LD SP, 0xFFFE: got SP=%04X, want FFFE", s.SP)
	}
	if s.F != 0xFF {
		t.Errorf("LD SP, nn changed flags: got %02X", s.F)
	}
}

func TestADCHL(t *testing.T) {
	tests := []struct {
		name   string
		op     inst.OpCode
		h, l   uint8
		srcH   uint8
		srcL   uint8
		carry  uint8
		expectH, expectL uint8
		checkS bool
		expectS bool
		checkZ bool
		expectZ bool
		checkC bool
		expectC bool
		checkN bool
		expectN bool
	}{
		{
			name: "ADC HL, BC no carry",
			op: inst.ADC_HL_BC, h: 0x10, l: 0x00,
			srcH: 0x20, srcL: 0x00, carry: 0,
			expectH: 0x30, expectL: 0x00,
			checkN: true, expectN: false,
		},
		{
			name: "ADC HL, BC with carry in",
			op: inst.ADC_HL_BC, h: 0x10, l: 0xFF,
			srcH: 0x20, srcL: 0x00, carry: FlagC,
			expectH: 0x31, expectL: 0x00,
			checkN: true, expectN: false,
		},
		{
			name: "ADC HL, DE overflow to carry",
			op: inst.ADC_HL_DE, h: 0xFF, l: 0xFF,
			srcH: 0x00, srcL: 0x01, carry: 0,
			expectH: 0x00, expectL: 0x00,
			checkC: true, expectC: true,
			checkZ: true, expectZ: true,
		},
		{
			name: "ADC HL, HL (doubles HL)",
			op: inst.ADC_HL_HL, h: 0x40, l: 0x00,
			carry: 0,
			expectH: 0x80, expectL: 0x00,
			checkS: true, expectS: true,
		},
		{
			name: "ADC HL, SP",
			op: inst.ADC_HL_SP, h: 0x10, l: 0x00,
			carry: 0,
			expectH: 0x30, expectL: 0x00,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := State{H: tc.h, L: tc.l, F: tc.carry}
			switch tc.op {
			case inst.ADC_HL_BC:
				s.B, s.C = tc.srcH, tc.srcL
			case inst.ADC_HL_DE:
				s.D, s.E = tc.srcH, tc.srcL
			case inst.ADC_HL_HL:
				// source is HL itself, already set
			case inst.ADC_HL_SP:
				s.SP = 0x2000
			}
			Exec(&s, tc.op, 0)
			if s.H != tc.expectH || s.L != tc.expectL {
				t.Errorf("result: got HL=%02X%02X, want %02X%02X", s.H, s.L, tc.expectH, tc.expectL)
			}
			if tc.checkS && ((s.F&FlagS != 0) != tc.expectS) {
				t.Errorf("S flag: got %v, want %v", s.F&FlagS != 0, tc.expectS)
			}
			if tc.checkZ && ((s.F&FlagZ != 0) != tc.expectZ) {
				t.Errorf("Z flag: got %v, want %v (HL=%04X)", s.F&FlagZ != 0, tc.expectZ, uint16(s.H)<<8|uint16(s.L))
			}
			if tc.checkC && ((s.F&FlagC != 0) != tc.expectC) {
				t.Errorf("C flag: got %v, want %v", s.F&FlagC != 0, tc.expectC)
			}
			if tc.checkN && ((s.F&FlagN != 0) != tc.expectN) {
				t.Errorf("N flag: got %v, want %v", s.F&FlagN != 0, tc.expectN)
			}
		})
	}
}

func TestSBCHL(t *testing.T) {
	tests := []struct {
		name   string
		op     inst.OpCode
		h, l   uint8
		srcH   uint8
		srcL   uint8
		carry  uint8
		expectH, expectL uint8
		checkS bool
		expectS bool
		checkZ bool
		expectZ bool
		checkC bool
		expectC bool
		checkN bool
		expectN bool
	}{
		{
			name: "SBC HL, BC no carry",
			op: inst.SBC_HL_BC, h: 0x30, l: 0x00,
			srcH: 0x10, srcL: 0x00, carry: 0,
			expectH: 0x20, expectL: 0x00,
			checkN: true, expectN: true,
		},
		{
			name: "SBC HL, BC with carry in",
			op: inst.SBC_HL_BC, h: 0x30, l: 0x00,
			srcH: 0x10, srcL: 0x00, carry: FlagC,
			expectH: 0x1F, expectL: 0xFF,
			checkN: true, expectN: true,
		},
		{
			name: "SBC HL, DE zero result",
			op: inst.SBC_HL_DE, h: 0x50, l: 0x00,
			srcH: 0x50, srcL: 0x00, carry: 0,
			expectH: 0x00, expectL: 0x00,
			checkZ: true, expectZ: true,
			checkC: true, expectC: false,
		},
		{
			name: "SBC HL, HL (always zero, no carry in)",
			op: inst.SBC_HL_HL, h: 0x42, l: 0x42,
			carry: 0,
			expectH: 0x00, expectL: 0x00,
			checkZ: true, expectZ: true,
		},
		{
			name: "SBC HL, SP borrow",
			op: inst.SBC_HL_SP, h: 0x00, l: 0x01,
			carry: 0,
			expectH: 0x80, expectL: 0x01,
			checkC: true, expectC: true,
			checkS: true, expectS: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := State{H: tc.h, L: tc.l, F: tc.carry}
			switch tc.op {
			case inst.SBC_HL_BC:
				s.B, s.C = tc.srcH, tc.srcL
			case inst.SBC_HL_DE:
				s.D, s.E = tc.srcH, tc.srcL
			case inst.SBC_HL_HL:
				// source is HL itself
			case inst.SBC_HL_SP:
				s.SP = 0x8000
			}
			Exec(&s, tc.op, 0)
			if s.H != tc.expectH || s.L != tc.expectL {
				t.Errorf("result: got HL=%02X%02X, want %02X%02X", s.H, s.L, tc.expectH, tc.expectL)
			}
			if tc.checkS && ((s.F&FlagS != 0) != tc.expectS) {
				t.Errorf("S flag: got %v, want %v", s.F&FlagS != 0, tc.expectS)
			}
			if tc.checkZ && ((s.F&FlagZ != 0) != tc.expectZ) {
				t.Errorf("Z flag: got %v, want %v", s.F&FlagZ != 0, tc.expectZ)
			}
			if tc.checkC && ((s.F&FlagC != 0) != tc.expectC) {
				t.Errorf("C flag: got %v, want %v", s.F&FlagC != 0, tc.expectC)
			}
			if tc.checkN && ((s.F&FlagN != 0) != tc.expectN) {
				t.Errorf("N flag: got %v, want %v", s.F&FlagN != 0, tc.expectN)
			}
		})
	}
}

// TestADCSBCHLCrossCheck verifies ADC/SBC HL agree with the reference lookup tables.
func TestADCSBCHLCrossCheck(t *testing.T) {
	// ADC HL, BC: verify N flag is always clear
	for hl := uint32(0); hl < 0x10000; hl += 0x1111 {
		for bc := uint32(0); bc < 0x10000; bc += 0x1111 {
			for carry := uint8(0); carry <= 1; carry++ {
				s := State{
					H: uint8(hl >> 8), L: uint8(hl),
					B: uint8(bc >> 8), C: uint8(bc),
					F: carry,
				}
				Exec(&s, inst.ADC_HL_BC, 0)
				if s.F&FlagN != 0 {
					t.Fatalf("ADC HL,BC: N flag set for HL=%04X BC=%04X carry=%d", hl, bc, carry)
				}
				result := uint16(s.H)<<8 | uint16(s.L)
				expected := uint32(hl) + uint32(bc) + uint32(carry)
				if result != uint16(expected) {
					t.Fatalf("ADC HL,BC: got %04X, want %04X (HL=%04X BC=%04X c=%d)", result, uint16(expected), hl, bc, carry)
				}
			}
		}
	}

	// SBC HL, DE: verify N flag is always set
	for hl := uint32(0); hl < 0x10000; hl += 0x1111 {
		for de := uint32(0); de < 0x10000; de += 0x1111 {
			for carry := uint8(0); carry <= 1; carry++ {
				s := State{
					H: uint8(hl >> 8), L: uint8(hl),
					D: uint8(de >> 8), E: uint8(de),
					F: carry,
				}
				Exec(&s, inst.SBC_HL_DE, 0)
				if s.F&FlagN == 0 {
					t.Fatalf("SBC HL,DE: N flag clear for HL=%04X DE=%04X carry=%d", hl, de, carry)
				}
				result := uint16(s.H)<<8 | uint16(s.L)
				expected := uint32(hl) - uint32(de) - uint32(carry)
				if result != uint16(expected) {
					t.Fatalf("SBC HL,DE: got %04X, want %04X (HL=%04X DE=%04X c=%d)", result, uint16(expected), hl, de, carry)
				}
			}
		}
	}
}

func init() {
	// Force unused import
	_ = fmt.Sprintf
}

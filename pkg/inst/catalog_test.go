package inst

import (
	"testing"
)

// TestCatalogCompleteness verifies every OpCode has a catalog entry.
func TestCatalogCompleteness(t *testing.T) {
	for op := OpCode(0); op < OpCodeCount; op++ {
		info := &Catalog[op]
		if info.Mnemonic == "" {
			t.Errorf("OpCode %d has no mnemonic", op)
		}
		if len(info.Bytes) == 0 {
			t.Errorf("OpCode %d (%s) has no encoding bytes", op, info.Mnemonic)
		}
		if info.TStates == 0 {
			t.Errorf("OpCode %d (%s) has 0 T-states", op, info.Mnemonic)
		}
	}
}

// TestEncodingMatchesMinzTS verifies our encodings match minz-ts/pkg/z80asm/opcodes.go.
func TestEncodingMatchesMinzTS(t *testing.T) {
	// Reference values from minz-ts/minzc/pkg/z80asm/opcodes.go
	expected := map[OpCode][]byte{
		// Loads
		LD_B_B: {0x40}, LD_B_C: {0x41}, LD_B_D: {0x42}, LD_B_E: {0x43},
		LD_B_H: {0x44}, LD_B_L: {0x45}, LD_B_A: {0x47},
		LD_C_B: {0x48}, LD_C_C: {0x49}, LD_C_D: {0x4A}, LD_C_E: {0x4B},
		LD_C_H: {0x4C}, LD_C_L: {0x4D}, LD_C_A: {0x4F},
		LD_A_B: {0x78}, LD_A_C: {0x79}, LD_A_D: {0x7A}, LD_A_E: {0x7B},
		LD_A_H: {0x7C}, LD_A_L: {0x7D}, LD_A_A: {0x7F},

		// Immediate loads
		LD_A_N: {0x3E}, LD_B_N: {0x06}, LD_C_N: {0x0E},
		LD_D_N: {0x16}, LD_E_N: {0x1E}, LD_H_N: {0x26}, LD_L_N: {0x2E},

		// Arithmetic
		ADD_A_B: {0x80}, ADD_A_C: {0x81}, ADD_A_D: {0x82}, ADD_A_E: {0x83},
		ADD_A_H: {0x84}, ADD_A_L: {0x85}, ADD_A_A: {0x87},
		ADC_A_B: {0x88}, ADC_A_C: {0x89}, ADC_A_A: {0x8F},
		SUB_B: {0x90}, SUB_C: {0x91}, SUB_A: {0x97},
		SBC_A_B: {0x98}, SBC_A_A: {0x9F},

		// Logic
		AND_B: {0xA0}, AND_A: {0xA7},
		XOR_B: {0xA8}, XOR_A: {0xAF},
		OR_B: {0xB0}, OR_A: {0xB7},
		CP_B: {0xB8}, CP_A: {0xBF},

		// Immediate arithmetic/logic
		ADD_A_N: {0xC6}, ADC_A_N: {0xCE}, SUB_N: {0xD6}, SBC_A_N: {0xDE},
		AND_N: {0xE6}, XOR_N: {0xEE}, OR_N: {0xF6}, CP_N: {0xFE},

		// INC/DEC
		INC_A: {0x3C}, INC_B: {0x04}, INC_C: {0x0C},
		INC_D: {0x14}, INC_E: {0x1C}, INC_H: {0x24}, INC_L: {0x2C},
		DEC_A: {0x3D}, DEC_B: {0x05}, DEC_C: {0x0D},

		// Special
		RLCA: {0x07}, RRCA: {0x0F}, RLA: {0x17}, RRA: {0x1F},
		DAA: {0x27}, CPL: {0x2F}, SCF: {0x37}, CCF: {0x3F},
		NEG: {0xED, 0x44},
		NOP: {0x00},

		// CB prefix
		RLC_A: {0xCB, 0x07}, RLC_B: {0xCB, 0x00},
		RRC_A: {0xCB, 0x0F},
		RL_A: {0xCB, 0x17},
		RR_A: {0xCB, 0x1F},
		SLA_A: {0xCB, 0x27},
		SRA_A: {0xCB, 0x2F},
		SRL_A: {0xCB, 0x3F},
	}

	for op, wantBytes := range expected {
		info := &Catalog[op]
		if len(info.Bytes) != len(wantBytes) {
			t.Errorf("%s: encoding length %d, want %d", info.Mnemonic, len(info.Bytes), len(wantBytes))
			continue
		}
		for i := range wantBytes {
			if info.Bytes[i] != wantBytes[i] {
				t.Errorf("%s: byte[%d] = 0x%02X, want 0x%02X", info.Mnemonic, i, info.Bytes[i], wantBytes[i])
			}
		}
	}
}

// TestByteSize verifies byte size calculations.
func TestByteSize(t *testing.T) {
	// Non-immediate: 1 byte
	if ByteSize(ADD_A_B) != 1 {
		t.Errorf("ADD A,B size: got %d want 1", ByteSize(ADD_A_B))
	}
	// Immediate: opcode + immediate = 2 bytes
	if ByteSize(ADD_A_N) != 2 {
		t.Errorf("ADD A,n size: got %d want 2", ByteSize(ADD_A_N))
	}
	// CB prefix: 2 bytes
	if ByteSize(RLC_A) != 2 {
		t.Errorf("RLC A size: got %d want 2", ByteSize(RLC_A))
	}
	// NEG: 2 bytes (ED prefix)
	if ByteSize(NEG) != 2 {
		t.Errorf("NEG size: got %d want 2", ByteSize(NEG))
	}
}

// TestTStates verifies timing.
func TestTStates(t *testing.T) {
	if TStates(ADD_A_B) != 4 {
		t.Errorf("ADD A,B: got %d T-states, want 4", TStates(ADD_A_B))
	}
	if TStates(ADD_A_N) != 7 {
		t.Errorf("ADD A,n: got %d T-states, want 7", TStates(ADD_A_N))
	}
	if TStates(RLC_A) != 8 {
		t.Errorf("RLC A: got %d T-states, want 8", TStates(RLC_A))
	}
	if TStates(NEG) != 8 {
		t.Errorf("NEG: got %d T-states, want 8", TStates(NEG))
	}
	if TStates(NOP) != 4 {
		t.Errorf("NOP: got %d T-states, want 4", TStates(NOP))
	}
}

// TestDisassemble verifies mnemonic generation.
func TestDisassemble(t *testing.T) {
	tests := []struct {
		instr Instruction
		want  string
	}{
		{Instruction{ADD_A_B, 0}, "ADD A, B"},
		{Instruction{LD_A_N, 0x00}, "LD A, 00h"},
		{Instruction{LD_A_N, 0xFF}, "LD A, 0FFh"},
		{Instruction{XOR_A, 0}, "XOR A"},
		{Instruction{NOP, 0}, "NOP"},
	}

	for _, tc := range tests {
		got := Disassemble(tc.instr)
		if got != tc.want {
			t.Errorf("Disassemble(%v): got %q want %q", tc.instr, got, tc.want)
		}
	}
}

// TestHasImmediate verifies immediate detection.
func TestHasImmediate(t *testing.T) {
	immOps := []OpCode{LD_A_N, LD_B_N, ADD_A_N, ADC_A_N, SUB_N, SBC_A_N, AND_N, XOR_N, OR_N, CP_N}
	for _, op := range immOps {
		if !HasImmediate(op) {
			t.Errorf("%s should be immediate", Catalog[op].Mnemonic)
		}
	}

	nonImmOps := []OpCode{ADD_A_B, XOR_A, INC_A, NOP, NEG, RLCA, RLC_A}
	for _, op := range nonImmOps {
		if HasImmediate(op) {
			t.Errorf("%s should NOT be immediate", Catalog[op].Mnemonic)
		}
	}
}

// TestAllOpsCount verifies the total number of opcodes.
func TestAllOpsCount(t *testing.T) {
	all := AllOps()
	if len(all) != int(OpCodeCount) {
		t.Errorf("AllOps() returned %d, want %d", len(all), OpCodeCount)
	}
}

// TestSeqByteSize verifies sequence byte size calculation.
func TestSeqByteSize(t *testing.T) {
	seq := []Instruction{
		{ADD_A_B, 0},  // 1 byte
		{LD_A_N, 0x42}, // 2 bytes
	}
	if SeqByteSize(seq) != 3 {
		t.Errorf("SeqByteSize: got %d want 3", SeqByteSize(seq))
	}
}

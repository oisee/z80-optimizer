package inst

// Info holds static metadata for an instruction opcode.
type Info struct {
	Mnemonic string  // Assembly mnemonic (e.g., "ADD A, B")
	Bytes    []uint8 // Raw encoding (without immediate), e.g., {0x80} or {0xCB, 0x07}
	TStates  int     // Clock cycles
}

// Catalog maps each OpCode to its Info.
var Catalog [OpCodeCount]Info

// AllOps returns all valid OpCode values (for enumeration).
func AllOps() []OpCode {
	ops := make([]OpCode, 0, OpCodeCount)
	for i := OpCode(0); i < OpCodeCount; i++ {
		ops = append(ops, i)
	}
	return ops
}

// NonImmediateOps returns all OpCodes that don't take an immediate.
func NonImmediateOps() []OpCode {
	ops := make([]OpCode, 0)
	for i := OpCode(0); i < OpCodeCount; i++ {
		if !HasImmediate(i) {
			ops = append(ops, i)
		}
	}
	return ops
}

// ImmediateOps returns all OpCodes that take an 8-bit immediate byte.
// Does NOT include 16-bit immediate ops (use Imm16Ops for those).
func ImmediateOps() []OpCode {
	ops := make([]OpCode, 0)
	for i := OpCode(0); i < OpCodeCount; i++ {
		if HasImmediate(i) && !HasImm16(i) {
			ops = append(ops, i)
		}
	}
	return ops
}

// Imm16Ops returns all OpCodes that take a 16-bit immediate.
func Imm16Ops() []OpCode {
	ops := make([]OpCode, 0)
	for i := OpCode(0); i < OpCodeCount; i++ {
		if HasImm16(i) {
			ops = append(ops, i)
		}
	}
	return ops
}

// TStates returns the T-state cost of an instruction.
func TStates(op OpCode) int {
	return Catalog[op].TStates
}

// ByteSize returns the total byte size of an instruction (encoding + immediate).
func ByteSize(op OpCode) int {
	n := len(Catalog[op].Bytes)
	if HasImm16(op) {
		n += 2
	} else if HasImmediate(op) {
		n++
	}
	return n
}

// Disassemble returns assembly text for an instruction.
func Disassemble(instr Instruction) string {
	info := &Catalog[instr.Op]
	if HasImm16(instr.Op) {
		return disasmImm16(info.Mnemonic, instr.Imm)
	}
	if HasImmediate(instr.Op) {
		return disasmImm8(info.Mnemonic, uint8(instr.Imm))
	}
	return info.Mnemonic
}

func disasmImm8(mnemonic string, imm uint8) string {
	// Replace "n" placeholder with hex value
	buf := make([]byte, 0, len(mnemonic)+4)
	for i := 0; i < len(mnemonic); i++ {
		if mnemonic[i] == 'n' && (i == len(mnemonic)-1 || mnemonic[i+1] == ' ' || mnemonic[i+1] == ',') {
			buf = appendHex8(buf, imm)
		} else {
			buf = append(buf, mnemonic[i])
		}
	}
	return string(buf)
}

func disasmImm16(mnemonic string, imm uint16) string {
	// Replace "nn" placeholder with 16-bit hex value
	buf := make([]byte, 0, len(mnemonic)+6)
	for i := 0; i < len(mnemonic); i++ {
		if i+1 < len(mnemonic) && mnemonic[i] == 'n' && mnemonic[i+1] == 'n' {
			buf = appendHex16(buf, imm)
			i++ // skip second 'n'
		} else {
			buf = append(buf, mnemonic[i])
		}
	}
	return string(buf)
}

func appendHex8(buf []byte, v uint8) []byte {
	const hex = "0123456789ABCDEF"
	if v >= 0xA0 {
		buf = append(buf, '0')
	}
	buf = append(buf, hex[v>>4], hex[v&0x0F], 'h')
	return buf
}

func appendHex16(buf []byte, v uint16) []byte {
	const hex = "0123456789ABCDEF"
	if v>>12 >= 0xA {
		buf = append(buf, '0')
	}
	buf = append(buf, hex[v>>12], hex[(v>>8)&0x0F], hex[(v>>4)&0x0F], hex[v&0x0F], 'h')
	return buf
}

// SeqByteSize returns total byte size for a sequence of instructions.
func SeqByteSize(seq []Instruction) int {
	n := 0
	for i := range seq {
		n += ByteSize(seq[i].Op)
	}
	return n
}

// SeqTStates returns total T-states for a sequence of instructions.
func SeqTStates(seq []Instruction) int {
	t := 0
	for i := range seq {
		t += TStates(seq[i].Op)
	}
	return t
}

func init() {
	// === V1: 8-bit register operations (206 opcodes) ===

	// Register-to-register loads: 4 T-states, 1 byte
	regLoads := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{LD_B_B, "LD B, B", 0x40}, {LD_B_C, "LD B, C", 0x41},
		{LD_B_D, "LD B, D", 0x42}, {LD_B_E, "LD B, E", 0x43},
		{LD_B_H, "LD B, H", 0x44}, {LD_B_L, "LD B, L", 0x45},
		{LD_B_A, "LD B, A", 0x47},
		{LD_C_B, "LD C, B", 0x48}, {LD_C_C, "LD C, C", 0x49},
		{LD_C_D, "LD C, D", 0x4A}, {LD_C_E, "LD C, E", 0x4B},
		{LD_C_H, "LD C, H", 0x4C}, {LD_C_L, "LD C, L", 0x4D},
		{LD_C_A, "LD C, A", 0x4F},
		{LD_D_B, "LD D, B", 0x50}, {LD_D_C, "LD D, C", 0x51},
		{LD_D_D, "LD D, D", 0x52}, {LD_D_E, "LD D, E", 0x53},
		{LD_D_H, "LD D, H", 0x54}, {LD_D_L, "LD D, L", 0x55},
		{LD_D_A, "LD D, A", 0x57},
		{LD_E_B, "LD E, B", 0x58}, {LD_E_C, "LD E, C", 0x59},
		{LD_E_D, "LD E, D", 0x5A}, {LD_E_E, "LD E, E", 0x5B},
		{LD_E_H, "LD E, H", 0x5C}, {LD_E_L, "LD E, L", 0x5D},
		{LD_E_A, "LD E, A", 0x5F},
		{LD_H_B, "LD H, B", 0x60}, {LD_H_C, "LD H, C", 0x61},
		{LD_H_D, "LD H, D", 0x62}, {LD_H_E, "LD H, E", 0x63},
		{LD_H_H, "LD H, H", 0x64}, {LD_H_L, "LD H, L", 0x65},
		{LD_H_A, "LD H, A", 0x67},
		{LD_L_B, "LD L, B", 0x68}, {LD_L_C, "LD L, C", 0x69},
		{LD_L_D, "LD L, D", 0x6A}, {LD_L_E, "LD L, E", 0x6B},
		{LD_L_H, "LD L, H", 0x6C}, {LD_L_L, "LD L, L", 0x6D},
		{LD_L_A, "LD L, A", 0x6F},
		{LD_A_B, "LD A, B", 0x78}, {LD_A_C, "LD A, C", 0x79},
		{LD_A_D, "LD A, D", 0x7A}, {LD_A_E, "LD A, E", 0x7B},
		{LD_A_H, "LD A, H", 0x7C}, {LD_A_L, "LD A, L", 0x7D},
		{LD_A_A, "LD A, A", 0x7F},
	}
	for _, rl := range regLoads {
		Catalog[rl.op] = Info{Mnemonic: rl.mnemonic, Bytes: []uint8{rl.enc}, TStates: 4}
	}

	// Immediate loads: 7 T-states, 2 bytes (opcode + immediate)
	immLoads := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{LD_A_N, "LD A, n", 0x3E},
		{LD_B_N, "LD B, n", 0x06},
		{LD_C_N, "LD C, n", 0x0E},
		{LD_D_N, "LD D, n", 0x16},
		{LD_E_N, "LD E, n", 0x1E},
		{LD_H_N, "LD H, n", 0x26},
		{LD_L_N, "LD L, n", 0x2E},
	}
	for _, il := range immLoads {
		Catalog[il.op] = Info{Mnemonic: il.mnemonic, Bytes: []uint8{il.enc}, TStates: 7}
	}

	// 8-bit arithmetic/logic with registers: 4 T-states, 1 byte
	aluReg := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{ADD_A_B, "ADD A, B", 0x80}, {ADD_A_C, "ADD A, C", 0x81},
		{ADD_A_D, "ADD A, D", 0x82}, {ADD_A_E, "ADD A, E", 0x83},
		{ADD_A_H, "ADD A, H", 0x84}, {ADD_A_L, "ADD A, L", 0x85},
		{ADD_A_A, "ADD A, A", 0x87},
		{ADC_A_B, "ADC A, B", 0x88}, {ADC_A_C, "ADC A, C", 0x89},
		{ADC_A_D, "ADC A, D", 0x8A}, {ADC_A_E, "ADC A, E", 0x8B},
		{ADC_A_H, "ADC A, H", 0x8C}, {ADC_A_L, "ADC A, L", 0x8D},
		{ADC_A_A, "ADC A, A", 0x8F},
		{SUB_B, "SUB B", 0x90}, {SUB_C, "SUB C", 0x91},
		{SUB_D, "SUB D", 0x92}, {SUB_E, "SUB E", 0x93},
		{SUB_H, "SUB H", 0x94}, {SUB_L, "SUB L", 0x95},
		{SUB_A, "SUB A", 0x97},
		{SBC_A_B, "SBC A, B", 0x98}, {SBC_A_C, "SBC A, C", 0x99},
		{SBC_A_D, "SBC A, D", 0x9A}, {SBC_A_E, "SBC A, E", 0x9B},
		{SBC_A_H, "SBC A, H", 0x9C}, {SBC_A_L, "SBC A, L", 0x9D},
		{SBC_A_A, "SBC A, A", 0x9F},
		{AND_B, "AND B", 0xA0}, {AND_C, "AND C", 0xA1},
		{AND_D, "AND D", 0xA2}, {AND_E, "AND E", 0xA3},
		{AND_H, "AND H", 0xA4}, {AND_L, "AND L", 0xA5},
		{AND_A, "AND A", 0xA7},
		{XOR_B, "XOR B", 0xA8}, {XOR_C, "XOR C", 0xA9},
		{XOR_D, "XOR D", 0xAA}, {XOR_E, "XOR E", 0xAB},
		{XOR_H, "XOR H", 0xAC}, {XOR_L, "XOR L", 0xAD},
		{XOR_A, "XOR A", 0xAF},
		{OR_B, "OR B", 0xB0}, {OR_C, "OR C", 0xB1},
		{OR_D, "OR D", 0xB2}, {OR_E, "OR E", 0xB3},
		{OR_H, "OR H", 0xB4}, {OR_L, "OR L", 0xB5},
		{OR_A, "OR A", 0xB7},
		{CP_B, "CP B", 0xB8}, {CP_C, "CP C", 0xB9},
		{CP_D, "CP D", 0xBA}, {CP_E, "CP E", 0xBB},
		{CP_H, "CP H", 0xBC}, {CP_L, "CP L", 0xBD},
		{CP_A, "CP A", 0xBF},
	}
	for _, ar := range aluReg {
		Catalog[ar.op] = Info{Mnemonic: ar.mnemonic, Bytes: []uint8{ar.enc}, TStates: 4}
	}

	// 8-bit arithmetic/logic immediate: 7 T-states, 2 bytes
	aluImm := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{ADD_A_N, "ADD A, n", 0xC6},
		{ADC_A_N, "ADC A, n", 0xCE},
		{SUB_N, "SUB n", 0xD6},
		{SBC_A_N, "SBC A, n", 0xDE},
		{AND_N, "AND n", 0xE6},
		{XOR_N, "XOR n", 0xEE},
		{OR_N, "OR n", 0xF6},
		{CP_N, "CP n", 0xFE},
	}
	for _, ai := range aluImm {
		Catalog[ai.op] = Info{Mnemonic: ai.mnemonic, Bytes: []uint8{ai.enc}, TStates: 7}
	}

	// INC/DEC registers: 4 T-states, 1 byte
	incDec := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{INC_A, "INC A", 0x3C}, {INC_B, "INC B", 0x04}, {INC_C, "INC C", 0x0C},
		{INC_D, "INC D", 0x14}, {INC_E, "INC E", 0x1C}, {INC_H, "INC H", 0x24},
		{INC_L, "INC L", 0x2C},
		{DEC_A, "DEC A", 0x3D}, {DEC_B, "DEC B", 0x05}, {DEC_C, "DEC C", 0x0D},
		{DEC_D, "DEC D", 0x15}, {DEC_E, "DEC E", 0x1D}, {DEC_H, "DEC H", 0x25},
		{DEC_L, "DEC L", 0x2D},
	}
	for _, id := range incDec {
		Catalog[id.op] = Info{Mnemonic: id.mnemonic, Bytes: []uint8{id.enc}, TStates: 4}
	}

	// Accumulator rotates: 4 T-states, 1 byte
	Catalog[RLCA] = Info{"RLCA", []uint8{0x07}, 4}
	Catalog[RRCA] = Info{"RRCA", []uint8{0x0F}, 4}
	Catalog[RLA] = Info{"RLA", []uint8{0x17}, 4}
	Catalog[RRA] = Info{"RRA", []uint8{0x1F}, 4}

	// Special instructions
	Catalog[DAA] = Info{"DAA", []uint8{0x27}, 4}
	Catalog[CPL] = Info{"CPL", []uint8{0x2F}, 4}
	Catalog[SCF] = Info{"SCF", []uint8{0x37}, 4}
	Catalog[CCF] = Info{"CCF", []uint8{0x3F}, 4}
	Catalog[NEG] = Info{"NEG", []uint8{0xED, 0x44}, 8}
	Catalog[NOP] = Info{"NOP", []uint8{0x00}, 4}

	// CB-prefix rotate/shift: 8 T-states, 2 bytes (CB + opcode)
	cbOps := []struct {
		op       OpCode
		mnemonic string
		enc      uint8
	}{
		{RLC_B, "RLC B", 0x00}, {RLC_C, "RLC C", 0x01}, {RLC_D, "RLC D", 0x02},
		{RLC_E, "RLC E", 0x03}, {RLC_H, "RLC H", 0x04}, {RLC_L, "RLC L", 0x05},
		{RLC_A, "RLC A", 0x07},
		{RRC_B, "RRC B", 0x08}, {RRC_C, "RRC C", 0x09}, {RRC_D, "RRC D", 0x0A},
		{RRC_E, "RRC E", 0x0B}, {RRC_H, "RRC H", 0x0C}, {RRC_L, "RRC L", 0x0D},
		{RRC_A, "RRC A", 0x0F},
		{RL_B, "RL B", 0x10}, {RL_C, "RL C", 0x11}, {RL_D, "RL D", 0x12},
		{RL_E, "RL E", 0x13}, {RL_H, "RL H", 0x14}, {RL_L, "RL L", 0x15},
		{RL_A, "RL A", 0x17},
		{RR_B, "RR B", 0x18}, {RR_C, "RR C", 0x19}, {RR_D, "RR D", 0x1A},
		{RR_E, "RR E", 0x1B}, {RR_H, "RR H", 0x1C}, {RR_L, "RR L", 0x1D},
		{RR_A, "RR A", 0x1F},
		{SLA_B, "SLA B", 0x20}, {SLA_C, "SLA C", 0x21}, {SLA_D, "SLA D", 0x22},
		{SLA_E, "SLA E", 0x23}, {SLA_H, "SLA H", 0x24}, {SLA_L, "SLA L", 0x25},
		{SLA_A, "SLA A", 0x27},
		{SRA_B, "SRA B", 0x28}, {SRA_C, "SRA C", 0x29}, {SRA_D, "SRA D", 0x2A},
		{SRA_E, "SRA E", 0x2B}, {SRA_H, "SRA H", 0x2C}, {SRA_L, "SRA L", 0x2D},
		{SRA_A, "SRA A", 0x2F},
		{SRL_B, "SRL B", 0x38}, {SRL_C, "SRL C", 0x39}, {SRL_D, "SRL D", 0x3A},
		{SRL_E, "SRL E", 0x3B}, {SRL_H, "SRL H", 0x3C}, {SRL_L, "SRL L", 0x3D},
		{SRL_A, "SRL A", 0x3F},
		{SLL_A, "SLL A", 0x37}, // Undocumented
		// SLL B-L (undocumented): CB 30-35
		{SLL_B, "SLL B", 0x30}, {SLL_C, "SLL C", 0x31}, {SLL_D, "SLL D", 0x32},
		{SLL_E, "SLL E", 0x33}, {SLL_H, "SLL H", 0x34}, {SLL_L, "SLL L", 0x35},
	}
	for _, cb := range cbOps {
		Catalog[cb.op] = Info{Mnemonic: cb.mnemonic, Bytes: []uint8{0xCB, cb.enc}, TStates: 8}
	}

	// === Wave 1: BIT / RES / SET on registers (174 opcodes) ===

	// BIT n, r: CB 40-7F (register-only, excluding (HL))
	// Encoding: CB [01 bbb rrr] where bbb=bit number, rrr=register (B=000,C=001,D=010,E=011,H=100,L=101,A=111)
	regEnc := [7]uint8{7, 0, 1, 2, 3, 4, 5} // A,B,C,D,E,H,L encoding
	regNames := [7]string{"A", "B", "C", "D", "E", "H", "L"}
	bitOps := [7][8]OpCode{
		{BIT_0_A, BIT_1_A, BIT_2_A, BIT_3_A, BIT_4_A, BIT_5_A, BIT_6_A, BIT_7_A},
		{BIT_0_B, BIT_1_B, BIT_2_B, BIT_3_B, BIT_4_B, BIT_5_B, BIT_6_B, BIT_7_B},
		{BIT_0_C, BIT_1_C, BIT_2_C, BIT_3_C, BIT_4_C, BIT_5_C, BIT_6_C, BIT_7_C},
		{BIT_0_D, BIT_1_D, BIT_2_D, BIT_3_D, BIT_4_D, BIT_5_D, BIT_6_D, BIT_7_D},
		{BIT_0_E, BIT_1_E, BIT_2_E, BIT_3_E, BIT_4_E, BIT_5_E, BIT_6_E, BIT_7_E},
		{BIT_0_H, BIT_1_H, BIT_2_H, BIT_3_H, BIT_4_H, BIT_5_H, BIT_6_H, BIT_7_H},
		{BIT_0_L, BIT_1_L, BIT_2_L, BIT_3_L, BIT_4_L, BIT_5_L, BIT_6_L, BIT_7_L},
	}
	for ri := 0; ri < 7; ri++ {
		for bit := 0; bit < 8; bit++ {
			enc := 0x40 | uint8(bit<<3) | regEnc[ri]
			Catalog[bitOps[ri][bit]] = Info{
				Mnemonic: "BIT " + string('0'+byte(bit)) + ", " + regNames[ri],
				Bytes:    []uint8{0xCB, enc},
				TStates:  8,
			}
		}
	}

	// RES n, r: CB 80-BF
	resOps := [7][8]OpCode{
		{RES_0_A, RES_1_A, RES_2_A, RES_3_A, RES_4_A, RES_5_A, RES_6_A, RES_7_A},
		{RES_0_B, RES_1_B, RES_2_B, RES_3_B, RES_4_B, RES_5_B, RES_6_B, RES_7_B},
		{RES_0_C, RES_1_C, RES_2_C, RES_3_C, RES_4_C, RES_5_C, RES_6_C, RES_7_C},
		{RES_0_D, RES_1_D, RES_2_D, RES_3_D, RES_4_D, RES_5_D, RES_6_D, RES_7_D},
		{RES_0_E, RES_1_E, RES_2_E, RES_3_E, RES_4_E, RES_5_E, RES_6_E, RES_7_E},
		{RES_0_H, RES_1_H, RES_2_H, RES_3_H, RES_4_H, RES_5_H, RES_6_H, RES_7_H},
		{RES_0_L, RES_1_L, RES_2_L, RES_3_L, RES_4_L, RES_5_L, RES_6_L, RES_7_L},
	}
	for ri := 0; ri < 7; ri++ {
		for bit := 0; bit < 8; bit++ {
			enc := 0x80 | uint8(bit<<3) | regEnc[ri]
			Catalog[resOps[ri][bit]] = Info{
				Mnemonic: "RES " + string('0'+byte(bit)) + ", " + regNames[ri],
				Bytes:    []uint8{0xCB, enc},
				TStates:  8,
			}
		}
	}

	// SET n, r: CB C0-FF
	setOps := [7][8]OpCode{
		{SET_0_A, SET_1_A, SET_2_A, SET_3_A, SET_4_A, SET_5_A, SET_6_A, SET_7_A},
		{SET_0_B, SET_1_B, SET_2_B, SET_3_B, SET_4_B, SET_5_B, SET_6_B, SET_7_B},
		{SET_0_C, SET_1_C, SET_2_C, SET_3_C, SET_4_C, SET_5_C, SET_6_C, SET_7_C},
		{SET_0_D, SET_1_D, SET_2_D, SET_3_D, SET_4_D, SET_5_D, SET_6_D, SET_7_D},
		{SET_0_E, SET_1_E, SET_2_E, SET_3_E, SET_4_E, SET_5_E, SET_6_E, SET_7_E},
		{SET_0_H, SET_1_H, SET_2_H, SET_3_H, SET_4_H, SET_5_H, SET_6_H, SET_7_H},
		{SET_0_L, SET_1_L, SET_2_L, SET_3_L, SET_4_L, SET_5_L, SET_6_L, SET_7_L},
	}
	for ri := 0; ri < 7; ri++ {
		for bit := 0; bit < 8; bit++ {
			enc := 0xC0 | uint8(bit<<3) | regEnc[ri]
			Catalog[setOps[ri][bit]] = Info{
				Mnemonic: "SET " + string('0'+byte(bit)) + ", " + regNames[ri],
				Bytes:    []uint8{0xCB, enc},
				TStates:  8,
			}
		}
	}

	// === Wave 2: 16-bit register pair ops (14 opcodes) ===
	Catalog[INC_BC] = Info{"INC BC", []uint8{0x03}, 6}
	Catalog[INC_DE] = Info{"INC DE", []uint8{0x13}, 6}
	Catalog[INC_HL] = Info{"INC HL", []uint8{0x23}, 6}
	Catalog[INC_SP] = Info{"INC SP", []uint8{0x33}, 6}
	Catalog[DEC_BC] = Info{"DEC BC", []uint8{0x0B}, 6}
	Catalog[DEC_DE] = Info{"DEC DE", []uint8{0x1B}, 6}
	Catalog[DEC_HL] = Info{"DEC HL", []uint8{0x2B}, 6}
	Catalog[DEC_SP] = Info{"DEC SP", []uint8{0x3B}, 6}
	Catalog[ADD_HL_BC] = Info{"ADD HL, BC", []uint8{0x09}, 11}
	Catalog[ADD_HL_DE] = Info{"ADD HL, DE", []uint8{0x19}, 11}
	Catalog[ADD_HL_HL] = Info{"ADD HL, HL", []uint8{0x29}, 11}
	Catalog[ADD_HL_SP] = Info{"ADD HL, SP", []uint8{0x39}, 11}
	Catalog[EX_DE_HL] = Info{"EX DE, HL", []uint8{0xEB}, 4}
	Catalog[LD_SP_HL] = Info{"LD SP, HL", []uint8{0xF9}, 6}

	// === Wave 4: 16-bit immediate loads + ED arithmetic (12 opcodes) ===

	// 16-bit immediate loads: 10 T-states, 3 bytes (opcode + 2-byte immediate)
	Catalog[LD_BC_NN] = Info{"LD BC, nn", []uint8{0x01}, 10}
	Catalog[LD_DE_NN] = Info{"LD DE, nn", []uint8{0x11}, 10}
	Catalog[LD_HL_NN] = Info{"LD HL, nn", []uint8{0x21}, 10}
	Catalog[LD_SP_NN] = Info{"LD SP, nn", []uint8{0x31}, 10}

	// ED-prefix 16-bit arithmetic: 15 T-states, 2 bytes
	Catalog[ADC_HL_BC] = Info{"ADC HL, BC", []uint8{0xED, 0x4A}, 15}
	Catalog[ADC_HL_DE] = Info{"ADC HL, DE", []uint8{0xED, 0x5A}, 15}
	Catalog[ADC_HL_HL] = Info{"ADC HL, HL", []uint8{0xED, 0x6A}, 15}
	Catalog[ADC_HL_SP] = Info{"ADC HL, SP", []uint8{0xED, 0x7A}, 15}
	Catalog[SBC_HL_BC] = Info{"SBC HL, BC", []uint8{0xED, 0x42}, 15}
	Catalog[SBC_HL_DE] = Info{"SBC HL, DE", []uint8{0xED, 0x52}, 15}
	Catalog[SBC_HL_HL] = Info{"SBC HL, HL", []uint8{0xED, 0x62}, 15}
	Catalog[SBC_HL_SP] = Info{"SBC HL, SP", []uint8{0xED, 0x72}, 15}
}

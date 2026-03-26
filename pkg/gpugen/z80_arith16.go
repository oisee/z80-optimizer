package gpugen

// Z80Arith16 defines the 33-op pool for 16-bit arithmetic idiom search.
// Includes 16-bit level ops + per-byte H/L operations + full ALU.
var Z80Arith16 = ISA{
	Name:       "z80_arith16",
	InputReg:   "l",           // input loaded into L
	InputRegs:  []string{"a", "l"},  // both A and L get input (matches CUDA kernel)
	OutputReg:  "a",           // placeholder, OutputExpr used instead
	OutputExpr: "((UINT16)h << 8) | l",
	OutputType: U16,
	QuickCheck: []uint8{0, 1, 127, 255},
	Locals: []Var{
		{Name: "hl", Type: U16},
		{Name: "bc", Type: U16},
		{Name: "de", Type: U16},
		{Name: "r16", Type: U16},
		{Name: "th", Type: U8},
		{Name: "tl", Type: U8},
		{Name: "hbit", Type: U8},
		{Name: "cc", Type: U8},
	},
	State: []Reg{
		{Name: "a", Type: U8},
		{Name: "b", Type: U8},
		{Name: "c", Type: U8},
		{Name: "d", Type: U8},
		{Name: "e", Type: U8},
		{Name: "h", Type: U8},
		{Name: "l", Type: U8},
		{Name: "carry", Type: Bool},
	},
	Ops: []Op{
		// 16-bit level ops (0-8)
		{Name: "ADD HL,HL", Cost: 11, Body: `hl = ((UINT16)h << 8) | l; r16 = hl + hl; h = (UINT8)(r16 >> 8); l = (UINT8)r16;`},
		{Name: "ADD HL,BC", Cost: 11, Body: `hl = ((UINT16)h << 8) | l; bc = ((UINT16)b << 8) | c; r16 = hl + bc; h = (UINT8)(r16 >> 8); l = (UINT8)r16;`},
		{Name: "LD C,A", Cost: 4, Body: `c = a;`},
		{Name: "SWAP_HL", Cost: 11, Body: `h = l; l = 0;`},
		{Name: "SUB HL,BC", Cost: 15, Body: `hl = ((UINT16)h << 8) | l; bc = ((UINT16)b << 8) | c; hl = hl - bc; h = (UINT8)(hl >> 8); l = (UINT8)hl;`},
		{Name: "EX DE,HL", Cost: 4, Body: `th = h; tl = l; h = d; l = e; d = th; e = tl;`},
		{Name: "ADD HL,DE", Cost: 11, Body: `hl = ((UINT16)h << 8) | l; de = ((UINT16)d << 8) | e; r16 = hl + de; h = (UINT8)(r16 >> 8); l = (UINT8)r16;`},
		{Name: "SUB HL,DE", Cost: 15, Body: `hl = ((UINT16)h << 8) | l; de = ((UINT16)d << 8) | e; hl = hl - de; h = (UINT8)(hl >> 8); l = (UINT8)hl;`},
		{Name: "SHR_HL", Cost: 16, Body: `hbit = h & 1; h = h >> 1; l = (l >> 1) | (hbit << 7);`},
		// Per-byte ops (9-20)
		{Name: "XOR A", Cost: 4, Body: `a = 0; carry = CFALSE;`},
		{Name: "SUB L", Cost: 4, Body: `carry = (a < l) ? CTRUE : CFALSE; a = a - l;`},
		{Name: "SUB H", Cost: 4, Body: `carry = (a < h) ? CTRUE : CFALSE; a = a - h;`},
		{Name: "ADD A,L", Cost: 4, Body: `r = (UINT16)a + l; carry = r > 0xFF ? CTRUE : CFALSE; a = (UINT8)r;`},
		{Name: "ADD A,H", Cost: 4, Body: `r = (UINT16)a + h; carry = r > 0xFF ? CTRUE : CFALSE; a = (UINT8)r;`},
		{Name: "SBC A,A", Cost: 4, Body: `cc = carry ? 1 : 0; carry = cc > 0 ? CTRUE : CFALSE; a = (cc != 0) ? 0xFF : 0x00;`},
		{Name: "LD L,A", Cost: 4, Body: `l = a;`},
		{Name: "LD H,A", Cost: 4, Body: `h = a;`},
		{Name: "LD A,L", Cost: 4, Body: `a = l;`},
		{Name: "LD A,H", Cost: 4, Body: `a = h;`},
		{Name: "OR L", Cost: 4, Body: `a = a | l; carry = CFALSE;`},
		{Name: "NEG", Cost: 8, Body: `carry = (a != 0) ? CTRUE : CFALSE; a = (UINT8)(0 - a);`},
		// Full ALU per-byte (21-32)
		{Name: "ADC A,L", Cost: 4, Body: `cc = carry ? 1 : 0; r = (UINT16)a + l + cc; carry = r > 0xFF ? CTRUE : CFALSE; a = (UINT8)r;`},
		{Name: "ADC A,H", Cost: 4, Body: `cc = carry ? 1 : 0; r = (UINT16)a + h + cc; carry = r > 0xFF ? CTRUE : CFALSE; a = (UINT8)r;`},
		{Name: "SBC A,L", Cost: 4, Body: `cc = carry ? 1 : 0; carry = ((INT16)a - (INT16)l - cc) < 0 ? CTRUE : CFALSE; a = a - l - (UINT8)cc;`},
		{Name: "SBC A,H", Cost: 4, Body: `cc = carry ? 1 : 0; carry = ((INT16)a - (INT16)h - cc) < 0 ? CTRUE : CFALSE; a = a - h - (UINT8)cc;`},
		{Name: "INC L", Cost: 4, Body: `l = l + 1;`},
		{Name: "INC H", Cost: 4, Body: `h = h + 1;`},
		{Name: "DEC L", Cost: 4, Body: `l = l - 1;`},
		{Name: "DEC H", Cost: 4, Body: `h = h - 1;`},
		{Name: "AND L", Cost: 4, Body: `a = a & l; carry = CFALSE;`},
		{Name: "XOR L", Cost: 4, Body: `a = a ^ l; carry = CFALSE;`},
		{Name: "XOR H", Cost: 4, Body: `a = a ^ h; carry = CFALSE;`},
		{Name: "OR H", Cost: 4, Body: `a = a | h; carry = CFALSE;`},
	},
}

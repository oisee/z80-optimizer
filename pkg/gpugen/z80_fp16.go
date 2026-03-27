package gpugen

// Z80FP16 defines the op pool for Z80-FP16 format: H=[exp] L=[sign+mant7].
// Brute-force search for normalize, compare, convert operations.
var Z80FP16 = ISA{
	Name:       "z80_fp16",
	InputReg:   "l",
	OutputReg:  "l",
	OutputExpr: "((UINT16)h << 8) | l",
	OutputType: U16,
	QuickCheck: []uint8{0, 1, 64, 127, 128, 255},
	Locals: []Var{
		{Name: "tmp", Type: U8},
	},
	State: []Reg{
		{Name: "h", Type: U8}, // exponent
		{Name: "l", Type: U8}, // sign(7) + mantissa(6:0)
		{Name: "carry", Type: Bool},
	},
	Ops: []Op{
		// Exponent ops
		{Name: "INC H", Cost: 4, Body: `h = h + 1;`},
		{Name: "DEC H", Cost: 4, Body: `h = h - 1;`},
		// Mantissa shift (via combined HL shift)
		{Name: "ADD HL,HL", Cost: 11, Body: `tmp = h; h = (h << 1) | (l >> 7); l = l << 1; carry = (tmp >> 7) ? CTRUE : CFALSE;`},
		{Name: "SHR HL", Cost: 16, Body: `tmp = l; l = (l >> 1) | ((h & 1) << 7); h = h >> 1; carry = (tmp & 1) ? CTRUE : CFALSE;`},
		// Sign/mantissa ops
		{Name: "RES 7,L", Cost: 8, Body: `l = l & 0x7F;`},           // abs
		{Name: "SET 7,L", Cost: 8, Body: `l = l | 0x80;`},           // force negative
		{Name: "BIT 7,L", Cost: 8, Body: `carry = (l & 0x80) ? CTRUE : CFALSE;`}, // test sign
		{Name: "XOR L,80", Cost: 8, Body: `l = l ^ 0x80;`},          // flip sign
		// Register moves
		{Name: "LD A,H", Cost: 4, Body: `/* a = h — implicit in output */`},
		{Name: "LD A,L", Cost: 4, Body: `/* a = l */`},
		{Name: "LD H,0", Cost: 7, Body: `h = 0;`},
		// Carry ops
		{Name: "SCF", Cost: 4, Body: `carry = CTRUE;`},
		{Name: "CCF", Cost: 4, Body: `carry = !carry;`},
		// Bit test
		{Name: "AND L,7F", Cost: 7, Body: `l = l & 0x7F; carry = CFALSE;`}, // mask mantissa
		{Name: "OR H", Cost: 4, Body: `l = l | h; carry = CFALSE;`},  // for zero test
	},
}

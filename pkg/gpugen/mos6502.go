package gpugen

// MOS6502Mul defines the op pool for 6502 constant multiply search.
// 6502 has no multiply — shift-add chains through A, X, Y + carry.
var MOS6502Mul = ISA{
	Name:       "mos6502_mulopt",
	InputReg:   "a",
	OutputReg:  "a",
	QuickCheck: []uint8{1, 2, 127, 255},
	State: []Reg{
		{Name: "a", Type: U8},
		{Name: "x", Type: U8},
		{Name: "y", Type: U8},
		{Name: "carry", Type: Bool},
		{Name: "zp0", Type: U8}, // zero-page scratch
	},
	Ops: []Op{
		{Name: "ASL A", Cost: 2, Body: `carry = (a & 0x80) != 0; a = a << 1;`},
		{Name: "LSR A", Cost: 2, Body: `carry = (a & 1) != 0; a = a >> 1;`},
		{Name: "ROL A", Cost: 2, Body: `bit = carry ? 1 : 0; carry = (a & 0x80) != 0; a = (a << 1) | bit;`},
		{Name: "ROR A", Cost: 2, Body: `bit = carry ? 0x80 : 0; carry = (a & 1) != 0; a = (a >> 1) | bit;`},
		{Name: "TAX", Cost: 2, Body: `x = a;`},
		{Name: "TXA", Cost: 2, Body: `a = x;`},
		{Name: "TAY", Cost: 2, Body: `y = a;`},
		{Name: "TYA", Cost: 2, Body: `a = y;`},
		{Name: "CLC", Cost: 2, Body: `carry = 0;`},
		{Name: "SEC", Cost: 2, Body: `carry = 1;`},
		{Name: "STA zp", Cost: 3, Body: `zp0 = a;`},
		{Name: "ADC zp", Cost: 3, Body: `c = carry ? 1 : 0; r = (UINT16)a + zp0 + c; carry = r > 0xFF; a = (UINT8)r;`},
		{Name: "SBC zp", Cost: 3, Body: `c = carry ? 0 : 1; r = (UINT16)a - zp0 - c; carry = r <= 0xFF; a = (UINT8)r;`},
		{Name: "EOR #$FF", Cost: 2, Body: `a = a ^ 0xFF;`},
	},
}

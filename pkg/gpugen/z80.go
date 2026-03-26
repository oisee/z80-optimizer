package gpugen

// Z80Mul defines the 14-op reduced pool for Z80 constant multiply search.
// 7 never-used ops removed for 38x speedup (empirically verified).
var Z80Mul = ISA{
	Name:       "z80_mulopt",
	InputReg:   "a",
	OutputReg:  "a",
	QuickCheck: []uint8{1, 2, 127, 255},
	State: []Reg{
		{Name: "a", Type: U8},
		{Name: "b", Type: U8},
		{Name: "carry", Type: Bool},
	},
	Ops: []Op{
		{Name: "ADD A,A", Cost: 4, Body: `r = (UINT16)a + a; carry = r > 0xFF; a = (UINT8)r;`},
		{Name: "ADD A,B", Cost: 4, Body: `r = (UINT16)a + b; carry = r > 0xFF; a = (UINT8)r;`},
		{Name: "SUB B", Cost: 4, Body: `carry = (a < b); a = a - b;`},
		{Name: "LD B,A", Cost: 4, Body: `b = a;`},
		{Name: "ADC A,B", Cost: 4, Body: `c = carry ? 1 : 0; r = (UINT16)a + b + c; carry = r > 0xFF; a = (UINT8)r;`},
		{Name: "ADC A,A", Cost: 4, Body: `c = carry ? 1 : 0; r = (UINT16)a + a + c; carry = r > 0xFF; a = (UINT8)r;`},
		{Name: "SBC A,B", Cost: 4, Body: `c = carry ? 1 : 0; carry = ((INT16)a - (INT16)b - c) < 0; a = a - b - (UINT8)c;`},
		{Name: "SBC A,A", Cost: 4, Body: `c = carry ? 1 : 0; carry = c > 0; a = -(UINT8)c;`},
		{Name: "SRL A", Cost: 8, Body: `carry = (a & 1) != 0; a = a >> 1;`},
		{Name: "RLA", Cost: 4, Body: `bit = carry ? 1 : 0; carry = (a & 0x80) != 0; a = (a << 1) | bit;`},
		{Name: "RRA", Cost: 4, Body: `bit = carry ? 0x80 : 0; carry = (a & 1) != 0; a = (a >> 1) | bit;`},
		{Name: "RLCA", Cost: 4, Body: `carry = (a & 0x80) != 0; a = (a << 1) | (a >> 7);`},
		{Name: "RRCA", Cost: 4, Body: `carry = (a & 1) != 0; a = (a >> 1) | (a << 7);`},
		{Name: "NEG", Cost: 8, Body: `carry = (a != 0); a = (UINT8)(0 - a);`},
	},
}

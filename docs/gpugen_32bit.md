# 32-bit Arithmetic Search via gpugen

How to add 32-bit Z80 arithmetic (HL:H'L' = 32-bit value) to gpugen.

## What's needed in the DSL

### 1. Add U32 type

In `pkg/gpugen/isa.go`:
```go
const (
    U8   Type = iota
    U16
    U32
    Bool
)
```

In `pkg/gpugen/emit.go`, add type mapping:
```go
func (e *emitter) u32Full() string {
    switch e.backend {
    case CUDA:   return "uint32_t"
    case Vulkan: return "uint"     // already 32-bit
    default:     return "uint"
    }
}
```

Add `UINT32` to `expandTypes`:
```go
"UINT32", e.u32Full(),
```

### 2. Define ISA

Create `pkg/gpugen/z80_arith32.go`:

```go
var Z80Arith32 = ISA{
    Name:       "z80_arith32",
    InputReg:   "l",
    InputRegs:  []string{"a", "l"},  // A=L=input
    OutputReg:  "l",
    // Result = H'L':HL (32-bit): high word in shadow, low word in main
    OutputExpr: "((UINT32)hS << 24) | ((UINT32)lS << 16) | ((UINT32)h << 8) | l",
    OutputType: U32,  // needs new type
    QuickCheck: []uint8{0, 1, 2, 127, 255},
    Locals: []Var{
        {Name: "hl", Type: U32},
        {Name: "de", Type: U32},
        {Name: "bc", Type: U32},
        {Name: "r32", Type: U32},
        {Name: "r16", Type: U16},
        {Name: "cc", Type: U8},
        {Name: "th", Type: U8}, {Name: "tl", Type: U8},
        {Name: "tb", Type: U8}, {Name: "tc", Type: U8},
        {Name: "td", Type: U8}, {Name: "te", Type: U8},
    },
    State: []Reg{
        // Main registers
        {Name: "a", Type: U8},
        {Name: "b", Type: U8}, {Name: "c", Type: U8},
        {Name: "d", Type: U8}, {Name: "e", Type: U8},
        {Name: "h", Type: U8}, {Name: "l", Type: U8},
        {Name: "carry", Type: Bool},
        // Shadow registers (EXX swaps B↔B', C↔C', D↔D', E↔E', H↔H', L↔L')
        {Name: "bS", Type: U8}, {Name: "cS", Type: U8},
        {Name: "dS", Type: U8}, {Name: "eS", Type: U8},
        {Name: "hS", Type: U8}, {Name: "lS", Type: U8},
    },
    Ops: []Op{
        // 16-bit add/sub with carry propagation
        {Name: "ADD HL,HL", Cost: 11, Body: `
            hl = (UINT16)h<<8|l; r32 = hl+hl;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "ADD HL,BC", Cost: 11, Body: `
            hl = (UINT16)h<<8|l; bc = (UINT16)b<<8|c; r32 = hl+bc;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "ADD HL,DE", Cost: 11, Body: `
            hl = (UINT16)h<<8|l; de = (UINT16)d<<8|e; r32 = hl+de;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "ADC HL,HL", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; cc = carry?1:0; r32 = hl+hl+cc;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "ADC HL,BC", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; bc = (UINT16)b<<8|c; cc = carry?1:0; r32 = hl+bc+cc;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "ADC HL,DE", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; de = (UINT16)d<<8|e; cc = carry?1:0; r32 = hl+de+cc;
            carry = r32 > 0xFFFF ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "SBC HL,HL", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; cc = carry?1:0; r32 = hl-hl-cc;
            carry = (r32 > 0xFFFF) ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "SBC HL,BC", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; bc = (UINT16)b<<8|c; cc = carry?1:0; r32 = hl-bc-cc;
            carry = (r32 > 0xFFFF) ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},
        {Name: "SBC HL,DE", Cost: 15, Body: `
            hl = (UINT16)h<<8|l; de = (UINT16)d<<8|e; cc = carry?1:0; r32 = hl-de-cc;
            carry = (r32 > 0xFFFF) ? CTRUE : CFALSE;
            h = (UINT8)(r32>>8); l = (UINT8)r32;`},

        // Register bank swap
        {Name: "EXX", Cost: 4, Body: `
            th=h;tl=l;h=hS;l=lS;hS=th;lS=tl;
            tb=b;tc=c;b=bS;c=cS;bS=tb;cS=tc;
            td=d;te=e;d=dS;e=eS;dS=td;eS=te;`},

        // Per-byte ops
        {Name: "LD C,A", Cost: 4, Body: `c = a;`},
        {Name: "LD E,A", Cost: 4, Body: `e = a;`},
        {Name: "LD L,A", Cost: 4, Body: `l = a;`},
        {Name: "LD H,A", Cost: 4, Body: `h = a;`},
        {Name: "LD A,L", Cost: 4, Body: `a = l;`},
        {Name: "LD A,H", Cost: 4, Body: `a = h;`},
        {Name: "XOR A", Cost: 4, Body: `a = 0; carry = CFALSE;`},
        {Name: "NEG", Cost: 8, Body: `carry = (a != 0) ? CTRUE : CFALSE; a = (UINT8)(0 - a);`},
    },
}
```

### 3. Key design points

**Carry propagation**: ADD HL sets carry from bit 15 overflow. EXX preserves carry.
ADC HL uses carry. This is the 32-bit chain: `ADD HL,DE : EXX : ADC HL,DE`.

**Shadow state**: 6 extra U8 registers (bS,cS,dS,eS,hS,lS). Total state = 14 × U8 + 1 bool.
Vulkan masking already handles this (all U8 get `&= 0xFF`).

**Search space**: ~18 ops, len-5 = 18^5 = 1.9M (instant). len-7 = 18^7 = 612M (~seconds on GPU).
32-bit multiply ×K needs: load K into BC (or inline), then shift-add chain with EXX.

**Output**: 32-bit result from 4 registers. OutputExpr handles this:
```
((uint32_t)hS << 24) | ((uint32_t)lS << 16) | ((uint32_t)h << 8) | l
```

**QuickCheck**: For 32-bit, comparison is `(uint32_t)(input * k)` — works up to k=255,
input=255: 255×255 = 65025 fits in u16. But for larger patterns (e.g. x*x or polynomial),
output may need full 32 bits.

### 4. emit.go changes needed

- Add `U32` case to `typeStr()`, `regType()`
- Add `UINT32` to `expandTypes()`
- Vulkan masking: `case U32: e.w("reg &= 0xFFFFFFFF;\n")` (noop on uint but documents intent)
- QuickCheck comparison: `(uint32_t)(N * k)` or `& 0xFFFFFFFF` for Vulkan
- `run_seq` return type: `uint32_t` / `uint`

### 5. Practical search: 32-bit multiply

Goal: HL:H'L' = input × K (mod 2^32) for all 256 8-bit inputs.

Typical pattern:
```asm
; HL = input (already loaded)
; BC = K_low, B'C' = K_high (loaded by setup, not counted)
ADD HL,HL      ; ×2, carry set
EXX
ADC HL,HL      ; upper half ×2 + carry
EXX
; repeat shift-add chain
```

For brute-force: ~18 ops × len-7 = 612M candidates.
M2 Metal: ~100M/sec → ~6 seconds per constant.
RTX 4060 Ti: ~1B/sec → <1 second.

### 6. Generate and run

```bash
# When z80_arith32.go is ready:
go run cmd/gpugen/main.go -isa z80_arith32 -backend metal > arith32.metal
xcrun -sdk macosx metal -O2 -c arith32.metal -o arith32.air
xcrun -sdk macosx metallib arith32.air -o /tmp/mulopt.metallib

# Generate op names
go run cmd/gpugen/main.go -isa z80_arith32 -backend host-header | \
    grep '"' | sed 's/.*"\(.*\)".*/\1/' > arith32_ops.txt

# Run
./metal_mulopt --k 3 --max-len 7 --ops-file arith32_ops.txt --no-verify
```

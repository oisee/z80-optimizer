# Floating Point Type Integration for MinZ/nanz

## For frontend (nanz language) and backend (MinZ compiler) teams

---

## Type Hierarchy

| Type | Size | Z80 Location | Layout | Precision | Range |
|------|------|-------------|--------|-----------|-------|
| `f8.8` | 16-bit | HL | H=integer, L=fraction | 8.8 fixed | 0..255.996 |
| `fp8` | 8-bit | A | [EEEE.MMMM] (E4M3 or E5M2) | ~3 digits | ~2^-14..2^15 |
| `fp16` | 16-bit | HL | H=[exp8] L=[sign+mant7] | ~2 digits | ~2^-126..2^127 |
| `fp24` | 24-bit | A+HL | A=[exp8] H=[sign+mant_hi] L=[mant_lo] | ~4.5 digits | ~2^-126..2^127 |
| `fp32` | 32-bit | HL+H'L' | HL=[exp8+sign+mant_hi] H'L'=[mant_lo] via EXX | ~7 digits | ~2^-126..2^127 |
| `fp40` | 40-bit | A+HL+H'L' | A=[exp8] HL=[sign+mant_hi] H'L'=[mant_lo] via EXX | ~9 digits | ~2^-126..2^127 |

### Existing
- `f8.8` already in nanz as `~= u16` with fixed-point semantics

### New (this package)
- `fp16` — primary recommendation for Z80. 2 bytes, fast ops.
- `fp24` — when you need more precision. 3 bytes. Uses A register.
- `fp8` — minimal float, fits in single register.

### Future
- `fp32`, `fp40` — need EXX shadow register bank. Complex.

---

## Key Design: Byte-Aligned Exponent

All Z80-FP formats have the exponent as a whole byte (not bit-packed like IEEE).

**This means:**
```
x2  = INC exp_reg   (4 T-states, 1 byte)
/2  = DEC exp_reg   (4 T-states, 1 byte)
```

Compare this to IEEE 754 where x2 requires bit extraction, increment, and repacking (~30-40T).

---

## fp16: Primary Float Type

### Memory Layout
```
Byte 0 (H): EEEEEEEE  — 8-bit exponent, bias 127
Byte 1 (L): SMMMMMMM  — S=sign, M=7-bit mantissa (implicit leading 1)
```

### Register Convention
- Load: `LD HL, (addr)` — H gets exponent, L gets sign+mantissa
- Store: `LD (addr), HL`
- Passes in HL, returns in HL

### Operations and Cost

| Operation | Z80 Code | T-states | Notes |
|-----------|----------|----------|-------|
| negate | `LD A,L; XOR 0x80; LD L,A` | 15T | flip sign bit |
| abs | `RES 7,L` | 8T | clear sign bit |
| x2 | `INC H` | 4T | increment exponent |
| /2 | `DEC H` | 4T | decrement exponent |
| is_zero | `LD A,L; AND 0x7F; OR H` | 15T | Z flag set if zero |
| compare | exp first, then mant | ~45T | see pkg/fp16 |
| add/sub | align + add + normalize | ~80-120T | loop-based normalize |
| mul (const K) | add exp + mul8 table | ~30-60T | uses our 254-entry table! |
| mul (general) | add exp + shift-add mant | ~120-160T | 7x7 bit multiply |
| to_uint8 | shift mant right by (134-exp) | ~40-60T | loop-based |
| from_uint8 | find leading 1, set exp | ~40-60T | loop-based |

### Constant Multiply is Special!

For `fp16 * 3.14159`:
1. Precompute constant as fp16: exp=128, mant=0x49, sign=0
2. Exponent: `LD A,H; ADD A,1; LD H,A` (128-127=1, add offset)
3. Mantissa: use `mulopt.Emit8(0xC9)` from our table (0x49|0x80=0xC9)
4. Total: ~20-30T vs ~120T for general multiply

**The compiler should constant-fold fp16 multiplies at compile time!**

---

## fp24: Extended Precision

### Memory Layout
```
Byte 0 (A): EEEEEEEE  — 8-bit exponent, bias 127
Byte 1 (H): SMMMMMMM  — S=sign, M=mantissa bits 14:8
Byte 2 (L): MMMMMMMM  — mantissa bits 7:0
```

### Register Convention
- Uses A for exponent — **A is NOT free during fp24 operations!**
- Load: `LD A,(addr); LD HL,(addr+1)` or custom 3-byte load
- Mantissa in HL: can use ADD HL,HL for shift, ADD HL,BC for add

### Operations
- negate: `LD B,A; LD A,H; XOR 0x80; LD H,A; LD A,B` (23T)
- abs: `RES 7,H` (8T)
- x2: `INC A` (4T)
- /2: `DEC A` (4T)
- normalize: `ADD HL,HL; DEC A` loop (max 14 iterations)
- mul mantissa: 15x15 multiply — use mul16 table for constant K!

---

## Language Integration (nanz)

### Type Declarations
```nanz
var x: fp16       // 2 bytes in HL
var y: fp24       // 3 bytes in A+HL
var z: f8.8       // 2 bytes in HL (existing)
```

### Implicit Conversions
```
f8.8 → fp16:  find leading bit, set exponent  (~50T)
fp16 → f8.8:  shift mantissa by exponent      (~50T)
fp16 → fp24:  expand mantissa, move exponent   (~20T)
fp24 → fp16:  truncate mantissa, move exponent (~20T)
uint8 → fp16: find leading 1, normalize        (~50T)
fp16 → uint8: denormalize, extract integer     (~50T)
```

### Operator Mapping
```nanz
x * 3.14    // compiler: fp16_mul_const(pi_exp, pi_mant, 0) → uses mul8 table
x * y       // compiler: fp16_mul(HL, DE) → general 7x7 multiply
x + y       // compiler: fp16_add(HL, DE) → align + add + normalize
x * 2       // compiler: INC H (4T!) — detected as power-of-2
x / 2       // compiler: DEC H (4T!)
abs(x)      // compiler: RES 7,L (8T)
-x          // compiler: LD A,L; XOR 0x80; LD L,A (15T)
```

### Register Allocation Implications
- `fp16` uses HL pair — same as `f8.8` and `u16`
- `fp24` uses A+HL — A is not available for other purposes
- Compiler must track which register holds what type
- Two fp16 values: one in HL, one in DE. Swap via EX DE,HL (4T)
- Two fp24 values: very constrained — one in A+HL, other needs BC+DE? Not practical.

### Recommendation
- `fp16` as the standard float type for Z80
- `fp24` only when explicitly requested (scientific/financial)
- `f8.8` remains for graphics/game code where fixed-point is natural
- `fp8` for packed arrays where precision doesn't matter

---

## Package API (Go)

```go
import "github.com/oisee/z80-optimizer/pkg/fp16"

// Get Z80 assembly for any operation:
seq := fp16.Negate()          // → {Ops: ["LD A, L", "XOR 0x80", "LD L, A"], Cost: 15}
seq := fp16.Abs()             // → {Ops: ["RES 7, L"], Cost: 8}
seq := fp16.Double()          // → {Ops: ["INC H"], Cost: 4}
seq := fp16.Add()             // → full addition sequence
seq := fp16.Mul()             // → general multiplication
seq := fp16.IntToFP16()       // → integer conversion
seq := fp16.IEEEToZ80()       // → IEEE half → Z80-FP16
seq := fp16.FP24Normalize()   // → 24-bit normalize

// Each Seq has:
//   .Name   string     operation name
//   .Ops    []string   Z80 assembly instructions
//   .Cost   int        T-state cost (0 = variable/loop-based)
//   .Bytes  int        code size
//   .Note   string     explanation
```

---

## Cross-Format Conversion Matrix

| From \ To | fp8 | fp16 | fp24 | f8.8 | uint8 | IEEE half | Bfloat16 |
|-----------|-----|------|------|------|-------|-----------|----------|
| fp8 | - | expand | expand | denorm | trunc | repack | repack |
| fp16 | trunc | - | expand | denorm | denorm | repack | repack |
| fp24 | trunc | trunc | - | denorm | denorm | trunc+repack | trunc+repack |
| f8.8 | norm | norm | norm | - | trunc H | norm+repack | norm+repack |
| uint8 | norm | norm | norm | `LD H,A; LD L,0` | - | norm+repack | norm+repack |
| IEEE half | repack | repack | repack+expand | repack+denorm | repack+denorm | - | repack |
| Bfloat16 | repack | repack | repack+expand | repack+denorm | repack+denorm | repack | - |

All Z80-FP formats share the same exponent bias (127) and byte-aligned exponent,
so conversion between them is just mantissa truncation/expansion + register shuffle.

---

*Package: `pkg/fp16/` in z80-optimizer repo*
*See also: `docs/adr/adr-fp-format.md`, `docs/sprint_fp.md`*

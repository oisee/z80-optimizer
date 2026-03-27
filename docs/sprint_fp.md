# Sprint FP: Z80-Optimal Floating Point Arithmetic

## Goal
Complete floating-point library for Z80 with provably optimal operations,
cross-format conversion, all via GPU brute-force across 4 backends.

---

## Phase 1: Format Operations (per-format, brute-forceable)

### For each format: find optimal add, sub, mul, compare, convert

#### Z80-FP16: H=[EEEEEEEE] L=[SMMMMMMM] (16-bit, Tier 2)

| Operation | Abstract | Estimated Z80 | Brute-force? |
|-----------|----------|---------------|-------------|
| negate | flip L.7 | SET/RES 7,L (8T) | trivial |
| abs | clear L.7 | RES 7,L (8T) | trivial |
| ×2 | exp++ | INC H (4T) | trivial, verify edge cases |
| ÷2 | exp-- | DEC H (4T) | trivial |
| compare | cmp exp, then mant | CP H then... (~12T) | brute-force |
| is_zero | exp==0 AND mant==0 | LD A,H / OR L (~8T) | brute-force |
| add | align exp, add mant, norm | ~30-50 insts | decompose |
| sub | align exp, sub mant, norm | ~30-50 insts | decompose |
| mul | add exp, mul mant | mul7×mul7 + combine | use mul8 table! |
| div | sub exp, div mant | div7 + combine | use div8 table! |
| normalize | shift mant, adjust exp | loop or unrolled | brute-force |
| to_int | extract integer part | shift by (bias-exp) | brute-force |
| from_int | find leading 1, set exp | normalize | brute-force |

**Key: mul mantissa = 7-bit × 7-bit. Our mul8 table covers this!**

#### s1.E8.M15: A=[EEEEEEEE] H=[SMMMMMMM] L=[MMMMMMMM] (24-bit, Tier 3)

| Operation | Abstract | Estimated Z80 | Notes |
|-----------|----------|---------------|-------|
| negate | flip H.7 | LD B,A / LD A,H / XOR 0x80 / LD H,A / LD A,B | brute-force! |
| abs | clear H.7 | RES 7,H (8T) | trivial |
| ×2 | A++ | INC A (4T) | trivial |
| ÷2 | A-- | DEC A (4T) | trivial |
| add | align, add HL mantissas | ~40-60 insts | decompose into steps |
| mul | add exp, mul mantissa | A=A1+A2, HL=M1*M2>>15 | mul16 table! |
| normalize | ADD HL,HL + DEC A loop | ~5-8T per bit | brute-force unrolled |
| to_f8.8 | convert to fixed | shift mant by exp | brute-force |
| from_f8.8 | convert from fixed | find leading 1 | brute-force |

**Key: mul mantissa = 15-bit × 15-bit. Decompose: hi×hi + hi×lo + lo×hi.**

---

## Phase 2: Cross-Format Conversion (brute-forceable!)

All conversions between our formats. Each is an A→A or A,HL→A,HL function.

| From | To | State | Method |
|------|----|-------|--------|
| Z80-FP16 (HL) | f8.8 (HL) | HL→HL | shift mant by (bias-exp) |
| f8.8 (HL) | Z80-FP16 (HL) | HL→HL | find leading 1, set exp |
| Z80-FP16 (HL) | s1.E8.M15 (A+HL) | HL→A,HL | extract exp to A, expand mant |
| s1.E8.M15 | Z80-FP16 | A,HL→HL | pack exp to H, truncate mant |
| IEEE FP16 (HL) | Z80-FP16 (HL) | HL→HL | repack bit fields |
| Bfloat16 (HL) | Z80-FP16 (HL) | HL→HL | repack bit fields |
| Z80-FP16 | IEEE FP16 | HL→HL | repack bit fields |
| Z80-FP16 | Bfloat16 | HL→HL | repack bit fields |
| FP8 (A) | Z80-FP16 (HL) | A→HL | expand to 16-bit |
| Z80-FP16 (HL) | FP8 (A) | HL→A | truncate to 8-bit |
| integer (A) | Z80-FP16 (HL) | A→HL | normalize |
| Z80-FP16 (HL) | integer (A) | HL→A | denormalize |

**Each conversion = target function for GPU brute-force!**
State is small (2-3 bytes). Pool: ~20 ops. Depth 6-10.
ALL conversions solvable by exhaustive search.

---

## Phase 3: Decomposed Float Addition

Float add = 3 sub-steps, each brute-forceable:

### Step 1: Exponent alignment
```
Input:  two floats (A1,M1) and (A2,M2)
Output: aligned mantissas with same exponent
Method: shift smaller mantissa right by |A1-A2| positions
```
**Brute-force target:** given exp_diff in A, shift HL right by A positions.
This is the "variable shift" problem — not a fixed sequence!
Solution: unrolled shift table (like our RLCA/SRL sleds) or loop.

### Step 2: Mantissa addition/subtraction
```
Input:  two aligned mantissas
Output: sum/difference
Method: ADD HL,BC or SBC HL,BC
```
Already solved! Our 16-bit idioms cover this.

### Step 3: Normalization
```
Input:  unnormalized result (mantissa + exponent)
Output: normalized (leading 1 in mantissa, adjusted exp)
Method: shift mantissa left, decrement exp per shift
```
**Brute-force target:** for each bit-width of result, optimal normalize sequence.
At most 15 shifts for 15-bit mantissa. Unrolled = up to 15 × (ADD HL,HL + DEC A).

---

## Phase 4: Float Multiplication via Our Tables!

Float mul is EASY because we already have multiply tables!

```
(A1, M1) × (A2, M2) = (A1+A2-bias, M1×M2 >> mantissa_bits)
```

### Z80-FP16 mul:
```asm
; Input: HL = float1, DE = float2
; Exponent: H + D - bias
LD A, H        ; exp1
ADD A, D       ; exp1 + exp2
SUB bias       ; adjust bias
; Mantissa: L.6:0 × E.6:0 → 14-bit product → take high 7 bits
; This is a 7×7 multiply! Our mul8 table covers A×K for any K.
; But here BOTH operands are variable...
; Need: general 7×7 multiply (not constant-K)
; Options:
;   1. LUT: 128×128 = 16KB table (too big?)
;   2. Shift-add loop: ~100T
;   3. Quarter-square: A×B = ((A+B)²-(A-B)²)/4 with 256-byte table
```

### s1.E8.M15 mul:
```
; Exponent: A1 + A2 - bias → ADD A,D / SUB bias
; Mantissa: 15×15 → 30-bit → take high 15 bits
; This is a full 16-bit multiply! Use our mul16 approach:
;   ADD HL,HL chains with BC as second operand
; But VARIABLE × VARIABLE, not constant K...
; Need: general 16×16 multiply (our MULU112 from Dark's article!)
; Cost: ~200T for general mul, vs 20-40T for constant mul from table
```

**For constant float mul (e.g. ×3.14159): use our table directly!**
Convert constant to s1.E8.M15, then:
- exp: ADD A, const_exp - bias
- mantissa: mul16_K(mantissa) where K = const_mantissa

---

## Phase 5: DSL + Multi-Backend Brute-Force

### New ISA definitions for gpugen:

```go
var Z80FP16 = ISA{
    Name: "z80_fp16",
    State: []Reg{
        {Name: "h", Type: U8},  // exponent
        {Name: "l", Type: U8},  // sign + mantissa
    },
    Ops: []Op{
        {Name: "INC H", Cost: 4, Body: `h = h + 1;`},
        {Name: "DEC H", Cost: 4, Body: `h = h - 1;`},
        {Name: "ADD HL,HL", Cost: 11, Body: `...`},
        {Name: "SRL H / RR L", Cost: 16, Body: `...`},
        {Name: "RES 7,L", Cost: 8, Body: `l = l & 0x7F;`},
        // ... etc
    },
}

var Z80FP24 = ISA{
    Name: "z80_fp24",
    State: []Reg{
        {Name: "a", Type: U8},  // exponent
        {Name: "h", Type: U8},  // sign + mantissa hi
        {Name: "l", Type: U8},  // mantissa lo
    },
    // ... 12 abstract ops from our design
}
```

### Generate and deploy:
```bash
gpugen -isa z80_fp16 -backend cuda   > fp16_search.cu    # for main GPUs
gpugen -isa z80_fp16 -backend vulkan > fp16_search.comp   # for i3 RX 580
gpugen -isa z80_fp16 -backend metal  > fp16_search.metal  # for M2
gpugen -isa z80_fp16 -backend opencl > fp16_search.cl     # universal
```

### Search targets:
```
fp16_normalize: HL → HL (normalize mantissa, adjust exp)
fp16_to_int:    HL → A (extract integer part)
fp16_from_int:  A → HL (convert integer to float)
fp16_to_f88:    HL → HL (convert to f8.8 fixed)
fp16_from_ieee: HL → HL (convert IEEE half → Z80-FP16)
fp16_to_ieee:   HL → HL (convert Z80-FP16 → IEEE half)
fp16_to_bfloat: HL → HL (convert Z80-FP16 → Bfloat16)
```

---

## Schedule

| Day | Task | Machine | Target |
|-----|------|---------|--------|
| 1 | Define z80_fp16 + z80_fp24 ISA in gpugen | CPU | DSL |
| 1 | Generate kernels for all 4 backends | CPU | CUDA/Metal/Vulkan/OpenCL |
| 1 | Brute-force FP16 single-float ops | all GPUs | neg, abs, compare, is_zero |
| 2 | Brute-force normalize + to/from_int | all GPUs | key conversion ops |
| 2 | Brute-force IEEE/Bfloat16 converters | all GPUs | cross-format |
| 3 | Decompose float add into sub-steps | design | alignment + add + norm |
| 3 | Brute-force each add sub-step | all GPUs | complete float add |
| 4 | Float mul via constant table | design | connect to mul16 table |
| 4 | Package as pkg/fp16/ Go library | code | compiler integration |

## Success Criteria
- [ ] FP16 normalize: optimal instruction sequence found
- [ ] FP16 ↔ integer conversion: optimal sequences
- [ ] FP16 ↔ IEEE/Bfloat16: optimal converters
- [ ] Float add decomposed and each step brute-forced
- [ ] Float constant-mul via existing mul16 table
- [ ] All results cross-verified on 4 GPU backends
- [ ] pkg/fp16/ Go package for MinZ compiler

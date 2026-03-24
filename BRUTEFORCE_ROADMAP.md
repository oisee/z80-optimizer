# Z80 Bruteforce Optimization Roadmap

**The idea is simple:** instead of hand-writing clever Z80 code, let the computer try EVERY possible instruction sequence and pick the shortest one that works.

A GPU with 5000 cores can test billions of sequences per second. A human can't. The GPU finds tricks no human would think of.

---

## Already Done

### Peephole Rules (602,008 entries)
- **What:** For every pair of Z80 instructions, find a shorter replacement
- **Search space:** 4,215² = 17.8M pairs (length-2)
- **Results:** 602K proven optimizations in `results-len2.json`
- **Example:** `SLA A / RR A` → `OR A` (saves 3 bytes)

### Register Allocation Table (61 entries)
- **What:** For each function shape, find the optimal register assignment
- **Search space:** up to 15^12 = 129B per function
- **Results:** 61 provably optimal assignments in `regalloc_table.json`
- **Example:** `add(a,b)→a+b` → `{v0→A, v1→B, v2→A}` cost=4T

---

## In Progress

### Optimal Constant Multiplication (×2..×255)

Z80 has no MUL instruction. Currently: shift-and-add loop (~80 T-states).
For constant K, there's always a shorter sequence of ADD/SUB/shift.

**4-tier incremental search:**

| Tier | Instructions | Pool | Search/constant |
|------|-------------|------|-----------------|
| 1 | ADD A,A, ADD A,B, SUB B, LD B,A, SLA A, SRL A | 6 | 6⁸ = 1.6M |
| 2 | + ADC A,B, ADC A,A, SBC A,B, OR A | 10 | 10⁸ = 100M |
| 3 | + EX AF,AF' (dual accumulator) | 11 | 11⁸ = 214M |
| 4 | + RLA, RRA, RLCA, RRCA, SRA A, RLC A, RRC A | 18 | 18⁶ = 34M (CPU) |

**Why 4 tiers?** To measure what each instruction class contributes. If Tier 2 (carry chain) doesn't improve over Tier 1 for any constant, we know ADC/SBC aren't useful for multiplication. If Tier 3 finds shorter sequences, EX AF,AF' as dual-accumulator is proven valuable.

**Key insight — carry chains:**
```z80
SLA A         ; A = 2x, CF = bit7 (overflow from shift!)
ADC A, B      ; A = 2x + x + CF  ← carry bit carries information
```
Rotations feed carry to ADC/SBC. The bruteforcer finds these patterns automatically.

**Output:** `mul_table.json` — 254 entries, one per constant K=2..255:
```json
{"k": 3, "sequence": ["LD B,A", "ADD A,A", "ADD A,B"], "cost": 12, "tier": 1}
{"k": 23, "sequence": ["LD B,A", "SLA A", "SLA A", "ADD A,B", "SLA A", "SUB B"], "cost": 28, "tier": 1}
```

**Integration:** VIR compiler's ISLE combiner uses the table: `mul(x, 23)` → emit the precomputed 6-instruction sequence instead of calling `__mul8` (~80T).

---

## Planned

### Optimal Constant Division (÷2..÷255)

Even more expensive than multiply (~120T for 8-bit div). For constant K, reciprocal multiplication tricks exist:

```z80
; div by 3: multiply by 86 (≈256/3), take high byte
; A = x/3 ≈ (x * 86) >> 8
```

**Search:** find shortest sequence where output = input / K for all inputs 0..255.
Separate tables for quotient and remainder.

### ZX Spectrum Screen Address

The holy grail of Spectrum programming. Every game computes this thousands of times per frame:

```
Y (0-191) → HL = video RAM address

Formula: 0x4000 + ((Y & 0xC0) << 5) + ((Y & 0x38) << 2) + ((Y & 0x07) << 8)
```

The Y coordinate is split into 3 bit-fields that get shuffled into the address. Current hand-optimized implementations: 10-15 instructions.

**Search:** input A = Y (0..191), output HL = correct address. Try all sequences up to length 12.

### Sign Extend (i8 → i16)

```z80
; Current: 4 instructions
LD L, A       ; L = value
RLCA          ; bit7 → CF
SBC A, A      ; A = (CF ? 0xFF : 0x00)
LD H, A       ; H = sign extension
```

Can bruteforce beat 4 instructions? Maybe with different register choices.

### ABS(A) — Absolute Value

```z80
; Current best: 4 instructions
OR A          ; test sign
JP P, .done   ; if positive, skip
NEG           ; A = -A
.done:
```

Is there a branchless version? Bruteforce would find it.

### Bit Reverse

Reverse all 8 bits of A. Used in FFT, CRC, memory-mapped I/O.

```
Input:  A = 0b10110001
Output: A = 0b10001101
```

No known fast algorithm on Z80. Bruteforce search space: try rotate/shift/mask combos.

### Population Count (Hamming Weight)

Count set bits in A (0..8). Used in chess engines (bitboards), compression, error detection.

```
Input:  A = 0b10110001
Output: A = 4 (four bits set)
```

Classic bit-manipulation problem. x86 has POPCNT; Z80 has nothing. What's the shortest sequence?

### BCD Conversion

Binary → packed BCD. Z80 has DAA (Decimal Adjust Accumulator) but it's tricky to use correctly.

```
Input:  A = 42 (binary)
Output: A = 0x42 (BCD: high nibble=4, low nibble=2)
```

### Approximate sin(A)

NOT a lookup table — actual computation in Z80 instructions.

Input: A = 0..255 (maps to 0..2π). Output: A = 0..255 (maps to -1..+1 unsigned).

**Search:** find the shortest sequence that produces output within ±2 of the true value for all 256 inputs. This is an "approximate" bruteforce — we allow small errors.

### CRC-8 Without Lookup Table

Compute CRC-8 of a single byte without a 256-byte lookup table. Used in Dallas/Maxim 1-Wire protocol, SMBUS.

### Pixel Plot on ZX Spectrum

Given (X, Y) → set the correct bit in video RAM. Combines screen address calculation with bit-position logic.

```
Input:  B = Y (0-191), C = X (0-255)
Output: set bit at (X,Y) in video RAM at 0x4000-0x57FF
```

---

## How It Works

### The Bruteforce Pattern

Every problem follows the same template:

1. **Define the function:** input → expected output (for all valid inputs)
2. **Define the instruction pool:** which Z80 instructions to try
3. **Search:** try all sequences of length 1, then 2, then 3, ...
4. **Verify:** for each sequence, test against ALL possible inputs
5. **Report:** first correct sequence found is the shortest (optimal)

### Verification Strategy

- **QuickCheck (8 vectors):** rejects 99.99% of candidates instantly
- **MidCheck (32 vectors):** catches most remaining false positives
- **ExhaustiveCheck (all inputs):** proves correctness for every possible input

### GPU Acceleration

For large search spaces (>100M), the GPU evaluates millions of candidates in parallel:

```
Thread 0:     try sequence [ADD A,A, LD B,A, ADD A,B, ...]
Thread 1:     try sequence [ADD A,A, LD B,A, ADD A,A, ...]
Thread 2:     try sequence [ADD A,A, LD B,A, SUB B, ...]
...
Thread 999999: try sequence [SRL A, RRA, RLCA, ...]
```

Each thread tests one sequence against the QuickCheck vectors. Survivors get full verification.

---

## File Structure

```
z80-optimizer/
├── results-len2.json         # 602K peephole rules (done)
├── regalloc_table.json       # 61 register assignments (done)
├── mul_table_tier1.json      # constant multiply (in progress)
├── mul_table_tier2.json      # + ADC/SBC
├── mul_table_tier3.json      # + EX AF,AF'
├── mul_table_tier4.json      # + rotations
├── div_table.json            # constant division (planned)
├── screen_addr_table.json    # ZX screen address (planned)
└── special_functions.json    # abs, sign_ext, bitrev, popcnt (planned)
```

---

*"Let the GPU try everything. Ship the winners."*

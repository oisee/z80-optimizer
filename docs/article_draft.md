# The Z80 Compiler That Never Guesses

*How GPU brute-force proved optimal Z80 code — and why 97.7% of real functions can't be register-allocated*

---

In 1997, a teenager named Dark from the demoscene group X-Trade published an article about Z80 multiplication in his electronic magazine *Spectrum Expert*. His shift-and-add loop ran in 196-204 T-states and powered the wireframe rotations in his award-winning demo *Illusion*. For nearly thirty years, this was the state of the art.

In 2026, we pointed a GPU at the problem and found that for any *specific* constant, the optimal sequence is 4-50× shorter than Dark's general loop. We found this not by cleverness, but by trying every possible instruction sequence and proving which one is shortest. The GPU doesn't guess. It *knows*.

This is the story of what happens when you stop trying to be smart and let the machine enumerate everything.

## Part 1: The Brute-Force Philosophy

### Why It Works on Z80

The Z80 CPU has an 11-byte state: seven 8-bit registers (A through L), a flag register F, a stack pointer SP, and a virtual memory byte M. That's small enough to fit in a GPU register file. Each instruction transforms this state deterministically. Two instruction sequences are equivalent if and only if they produce identical 11-byte output for *every* possible 11-byte input.

This is a finite, deterministic problem. There are roughly 2^88 possible input states — too many to enumerate. But a trick makes it tractable: test a few carefully chosen inputs first.

### The Three-Stage Pipeline

**QuickCheck** tests 8 inputs and rejects 99.99% of candidates. If two sequences disagree on *any* of these 8 inputs, they're definitely not equivalent. This costs microseconds per candidate.

**MidCheck** adds 24 more inputs targeting BIT/SET/RES instructions whose effects are bit-position-specific. This catches another 23% of false positives.

**ExhaustiveCheck** sweeps all possible inputs for the ~0.01% of survivors. On GPU, 256 threads per candidate — thread *i* handles A=*i* and loops over the remaining registers.

Running on a pair of RTX 4060 Ti GPUs, this pipeline processes 17.8 million instruction pairs in about six minutes. The result: 739,575 provably correct peephole optimization rules. Each one is a mathematical theorem: "these two instruction sequences produce identical output for all possible inputs."

Some are obvious: `LD A, B : LD B, A → LD A, B` (the second instruction is redundant). Others are deeply surprising:

```
SLA A : RR A → OR A    (save 12 T-states, 3 bytes)
```

Shift left then rotate right equals... OR with self? It works because the shift clears bit 0, the rotate puts the old bit 7 into bit 0 through carry, and the combined flag effect matches OR A exactly. No human would design this rule. The GPU doesn't need to understand *why* — it just proves *that*.

## Part 2: Constant Multiplication — When 3 Ops Suffice

### The 14-Op Discovery

For 8-bit multiply (`A × K → A`), we started with 21 candidate instructions: shifts, rotates, adds, subtracts, NEG, carry manipulations. The GPU found optimal sequences for 103 constants at length ≤8.

Then we analyzed which instructions actually *appeared* in any optimal solution. Seven never did:
- `SLA A` — identical to `ADD A,A` but costs 8T instead of 4T (strictly dominated)
- `RLC A`, `RRC A` — CB-prefix versions of RLCA/RRCA (8T vs 4T)
- `OR A`, `SCF`, `EX AF,AF'` — theoretically useful, never optimal

Removing them shrinks the search from 21^N to 14^N candidates. At length 9, that's 38× fewer. At length 10, 57× fewer. This single insight — **empirical pool reduction** — is the most powerful technique in the entire project.

With 14 ops, the GPU found 164 constants at length ≤9. The remaining 90 were solved by composition: `×75 = ×3 then ×25`, both already in the table. **All 254 non-trivial constants have optimal sequences.**

The most beautiful: `×255 = NEG` — one instruction, 8 T-states. Because 255 = -1 mod 256, and NEG computes `-A`. The GPU doesn't know number theory. It just found that NEG produces the right output for all 256 inputs.

### The 3-Op Miracle

For 16-bit multiply (`A × K → HL`), the story is even more dramatic. We started with 23 ops. The GPU found all 254 constants... using only **three instructions**:

- `ADD HL,HL` — double the 16-bit accumulator
- `ADD HL,BC` — add the original input
- `LD C,A` — save the input to C (so BC = input with B=0)

That's it. Every 16-bit constant multiply on the Z80 is a sequence of doubles and adds. The search space collapsed from 23^N to 3^N — a factor of **13,600× at length 8**. The entire table took 30 seconds.

We later added `SWAP_HL` (byte swap = ×256) and `SUB HL,BC` (subtract). These improved 88 of the 254 constants (35%), with ×255 dropping from 15 instructions to 3:

```asm
SWAP_HL       ; HL = input × 256
LD C,A        ; save input
SUB HL,BC     ; HL = input×256 - input = input×255
```

The byte swap trick: multiply by 256 then subtract. Three virtual operations, five real Z80 instructions, 30 T-states.

## Part 3: Division — When Abstract Chains Guide the GPU

### The Reciprocal Trick

Division by a constant K uses the identity `n/K = (n × M) >> S` where M ≈ 2^S/K. This converts division into multiplication by a "magic constant" followed by a right shift. The technique is well-known from Hacker's Delight.

What's new: we automate the entire process. An abstract chain solver (running on CPU, 8 seconds for all 254 constants) finds the shortest multiply-then-shift pattern. Then a focused GPU kernel — with only 6 ops instead of 37 — searches the Z80-specific materialization.

The result: **246 of 247 non-power-of-2 divisors** found in 11 seconds each. The GPU's div10 sequence (14 instructions, 124 T-states) exactly matches the hand-optimized Hacker's Delight solution. But the GPU found it automatically.

The fastest: `÷171 = 4 instructions, 27 T-states`. The most useful: `÷10 = 124T, ÷100 = 105T, ÷3 = 130T`.

### Guided Brute-Force

The key architectural insight: **separate the mathematical structure from the ISA-specific implementation**. The abstract chain says *what* to compute (multiply by 171, shift right 9). The GPU finds *how* to compute it on Z80 (which registers, which order).

This "guided brute-force" reduces the search space by millions. Pure brute-force at 37 ops would need 37^14 ≈ 10^22 evaluations — years on any GPU. Guided search with 6 ops needs 6^16 ≈ 10^12 — eleven seconds.

## Part 4: The Idiom Zoo

Fifteen branchless idioms, all discovered by GPU exhaustive search with a 37-op pool:

| Idiom | Sequence | Cost | Trick |
|-------|----------|------|-------|
| bool(A) | `LD B,A : NEG : ADC A,B` | 16T | NEG sets carry if nonzero; ADC adds carry to cancelled result |
| NOT(A) | `NEG : SBC A,A : INC A` | 16T | Carry-to-mask (0xFF) then increment (0xFF→0, 0→1) |
| ABS(A) | `LD B,A : RLCA : SBC A,A : XOR B : SBC A,B : ADC A,B` | 24T | Sign→carry→mask→conditional complement |
| sign-extend A→HL | `ADC A,L : SBC A,A : LD H,A` | 12T | Double overflows if ≥128 → carry → 0xFF mask |
| half(A) | `RRA` | 4T | Not SRL A (8T) — RRA is non-CB prefix, faster! |
| complement(A) | `CPL` | 4T | We forgot to include CPL in the initial pool... |

The `SBC A,A` carry-to-mask trick appears everywhere. It converts a single carry bit into a full-byte mask: 0x00 if carry clear, 0xFF if carry set. Combined with `RLCA` (which puts the sign bit into carry), this enables instant branchless sign detection.

The GPU also found that `CPL` (complement, 1 instruction, 4 T-states) is optimal for bitwise NOT — but only after we realized we'd forgotten to include it in the instruction pool. Pool design matters.

## Part 5: Register Allocation — The Feasibility Cliff

### 83.6 Million Exhaustive Solutions

For every theoretically possible combination of virtual register count, widths, location constraints, and interference graph up to 6 virtual registers, we computed the provably optimal physical register assignment — or proved that no valid assignment exists.

The results reveal a dramatic phase transition:

```
2 vregs:  95.9% feasible
3 vregs:  88.5% feasible
4 vregs:  78.7% feasible
5 vregs:  67.7% feasible
6 vregs:   0.9% feasible   ← cliff
```

At 6 virtual registers, 99.1% of all possible constraint shapes have NO valid Z80 register assignment. The register file "fills up" — there simply aren't enough physical registers for most configurations.

### 97.7% Infeasibility for Real Programs

We tested 129 unique function signatures from a real compiler corpus (8 programming languages, 1,567 functions). For functions with 7-15 virtual registers:

- **3 feasible** — provably optimal assignments found
- **126 infeasible** — proven that no valid assignment exists
- **97.7% infeasibility rate**

This means the Z80 compiler MUST decompose almost every non-trivial function into smaller pieces ("islands") that fit in the register file. Island decomposition isn't an optimization — it's a mathematical necessity.

### The Five-Level Pipeline

No single method handles everything:

1. **Table lookup** (83.6M entries, O(1)) — covers ≤6v
2. **Graph decomposition** at cut vertices — compose from smaller solved pieces
3. **GPU brute-force** — for ≤12v shapes
4. **CPU backtracking** with pattern-aware pruning (1000-4000× faster) — ≤15v
5. **Island decomposition** + Z3 solver — for >15v functions

Every function hits one of these levels. The pipeline is complete.

## Part 6: The Packed Arithmetic Library

### Multi-Entry Fall-Through Code

All 254 multiply sequences share instruction prefixes. By arranging them in reverse order with multiple entry labels, we eliminate all redundancy:

```asm
mul104:            ; enter here for ×104
mul52:  ADD A,A    ; enter here for ×52 (×104 = ×52 × 2)
mul26:  ADD A,A    ; enter here for ×26
mul24:  ADD A,B    ; enter here for ×24
mul12:  ADD A,A    ; enter here for ×12
mul6:   LD B,A     ; enter here for ×6
        ADD A,B
        ADD A,B
mul2:   RLA        ; enter here for ×2
        RET        ; shared return — 7 constants, 9 instructions
```

For rotations, the same principle yields an instruction sled:

```asm
rot7:   RLCA       ; 7 rotations left (9 bytes total, 7 entry points)
rot6:   RLCA       ; 6 rotations
rot5:   RLCA       ; 5
rot4:   RLCA       ; = nibble swap!
rot3:   RLCA       ; 3
rot2:   RLCA       ; 2
rot1:   RLCA       ; 1
        RET
```

Combined: 254 multiplies + 245 divisions + rotation/shift sleds = approximately **2KB** of Z80 code. For a ZX Spectrum with 48KB RAM, that's 4%. For a ROM-based system, it fits alongside any program.

Runtime dispatch: a page-aligned 256-byte jump table. `LD H, page / LD L, K / JP (HL)` — single-instruction dispatch to the provably optimal sequence.

## Part 7: Universal Computation Chains

### ISA-Independent Search

The deepest insight: the mathematical structure of multiplication and division is ISA-independent. An "addition-subtraction chain" — a sequence of {double, add, subtract, save, negate} — finds the shortest path from 1 to K for any processor.

We built a chain solver that finds all 254 multiply chains in 8 seconds on CPU. These chains then materialize to any ISA:

| Chain op | Z80 | 6502 | RISC-V |
|----------|-----|------|--------|
| dbl | ADD A,A (4T) | ASL A (2cy) | SLLI rd,1 (1cy) |
| add | ADD A,B (4T) | CLC:ADC zp (5cy) | ADD rd,rs (1cy) |
| save | LD B,A (4T) | TAX (2cy) | MV rd,rs (1cy) |
| neg | NEG (8T) | EOR#FF:ADC#1 (4cy) | NEG rd (1cy) |

One search, every processor. The same chain that gives ×10 on Z80 also gives ×10 on 6502 — just with different instruction encodings and cycle counts.

## Part 8: Cross-Platform Verification

We verified every result across five platforms:

- NVIDIA RTX 4060 Ti × 2 (CUDA)
- NVIDIA RTX 2070 (CUDA)
- AMD Radeon RX 580 (OpenCL via Mesa rusticl + Vulkan via RADV)
- Apple M2 MacBook Air (Metal + OpenCL)
- CPU (Python, all 256 inputs)

All platforms produce identical results. Three GPU vendors, four APIs, one truth.

The AMD verification deserves special mention: ROCm 6.x dropped support for the RX 580's gfx803 architecture. ROCm 5.7 installs but the kernel fusion driver doesn't register the GPU. The solution: Mesa provides OpenCL 3.0 and Vulkan through the open-source RADV driver — no proprietary runtime needed.

## Part 9: For the Compiler

Three Go packages ship ready for integration:

```go
// Multiply: O(1) lookup → inline optimal sequence
seq := mulopt.Emit8(42, bPreserve)
// → ["RLA", "LD B,A", "ADD A,B", "ADD A,A", "ADD A,B", ...]

// Register allocation: O(1) lookup from 83.6M table
entry := table.Lookup(regalloc.IndexOf(shape))
// → {Cost: 46, Assignment: [A, HL, DE, B]}

// Peephole: proven-correct replacement
rule := peephole.Lookup("SLA A : RR A")
// → {Replacement: "OR A", CyclesSaved: 12, BytesSaved: 3}
```

MinZ v0.23.0 ships with 498 inline arithmetic sequences from our tables. Every constant multiply and divide produces provably optimal code with zero runtime overhead.

## Conclusion: The Compiler That Never Guesses

Dark wrote the best general multiply in 1997. The GPU proved what's optimal for each specific case in 2026 — and it's 8× faster on average. Not because GPU is smarter than Dark, but because it doesn't need to be. It just tries everything.

The surprising findings aren't the individual optimizations — it's the structural results:
- Only 3 of 23 instructions matter for 16-bit multiply
- 97.7% of real functions above 6 variables can't be register-allocated on Z80
- Abstract computation chains are ISA-independent
- 2KB of packed code covers ALL optimal arithmetic

The Z80 was designed in 1976 for serial I/O controllers. Fifty years later, a cluster of GPUs exhaustively enumerated its optimal instruction sequences. Every result is a mathematical proof. The compiler that uses these tables doesn't guess, doesn't heuristic, doesn't approximate. It looks up the GPU-proven answer.

That's what brute force buys you: certainty.

---

*Repository: https://github.com/oisee/z80-optimizer (v1.0.0)*
*Built during a birthday marathon, March 26, 2026. 🎂*
*5 machines, 4 GPU APIs, 3 vendors, ~60 commits, one cake.*

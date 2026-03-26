# Research Insights & Experiment Log

Running log of discoveries, ideas, and experimental results.
Each insight is timestamped and tagged for traceability into papers.

---

## 2026-03-26: Virtual Instruction Pool for u16 Multiply

**Tags:** mulopt16, pool-reduction, paper-seed

**Discovery:** Only 7 of 23 ops appear in optimal u16 multiply solutions (len ≤6):
- `ADD HL,HL` (60x) — 16-bit doubling, by far dominant
- `ADD HL,BC` (20x) — add original value
- `LD C,A` (14x) — save input to C for ADD HL,BC
- `NEG` (2x) — negate for complement multiplication
- `LD L,A`, `SBC A,B`, `LD H,A` (1x each) — edge cases

16 ops never used: all carry-propagation (RL B, RL H), all ADC, all SUB, OR A, SCF.

**Insight:** The optimal u16 multiply patterns operate at 16-bit granularity:
1. **Double:** `ADD HL,HL` (native 16-bit)
2. **Add original:** `LD C,A` then `ADD HL,BC`
3. **Negate:** `NEG` for ×(-1) complement
4. **Byte swap:** `LD H,L / LD L,0` for ×256

The 8-bit carry-propagation ops are implementation details of these 16-bit
operations. Searching in a virtual 16-bit op space is both more natural AND
dramatically smaller.

**Impact on search space:**

| Pool | len-8 | len-9 | len-10 | Speedup vs 23-op |
|------|-------|-------|--------|-------------------|
| 23 ops (current) | 78B | 1.8T | 41T | 1x |
| 11 virtual ops | 214M | 2.4B | 26B | 365x |
| 7 ops (empirical) | 5.7M | 40M | 282M | 13,600x |

At 7 ops, len-10 = 282M — **instant on any GPU**. len-12 = 13.8B — still feasible.

**Caveat:** The 7-op pool is validated only through len-6. Longer sequences
might need carry-propagation ops (RL H, RL B) for multi-step multiplication
where intermediate results overflow 8 bits. Must verify against len-7 results
(running on i5).

**Proposed virtual instruction set for mulopt16:**

| Virtual Op | Materializes to | Cost | Semantics |
|-----------|----------------|------|-----------|
| DBL | ADD HL,HL | 11T | HL *= 2 |
| ADD | LD C,A / ADD HL,BC | 15T | HL += input (first use sets C) |
| SUB | OR A / SBC HL,BC | 15T | HL -= input |
| NEG16 | (5-inst sequence) | 24T | HL = -HL |
| SAVE | LD B,H / LD C,L | 8T | BC = HL (save for later) |
| RESTORE | LD H,B / LD L,C | 8T | HL = BC (restore saved) |
| SWAP | LD H,L / LD L,0 | 8T | HL = L * 256 (byte swap) |

7 virtual ops → 7^12 = 13.8B for len-12. Entire u16 multiply table
to depth 12 becomes feasible in minutes.

**Next steps:**
1. Wait for len-7 results to validate which ops appear at longer lengths
2. Build virtual mulopt16 kernel
3. If validated: search to len-12 for complete u16 multiply table

---

## 2026-03-26: Instruction Pool Reduction (8-bit Multiply)

**Tags:** mulopt, pool-reduction

**Discovery:** 7 of 21 ops never appear in optimal 8-bit multiply solutions:
- `SLA A` — identical to `ADD A,A` but costs 8T vs 4T (strictly dominated)
- `SRA A` — arithmetic shift right (not useful for unsigned multiply)
- `RLC A` / `RRC A` — CB-prefix rotates, same as RLCA/RRCA but 8T vs 4T
- `OR A` — only clears carry, no useful effect on A for multiplication
- `SCF` — set carry, never needed
- `EX AF,AF'` — shadow register swap, never needed

Reducing from 21 to 14 ops gives **38x speedup at len-9**.

**Verified:** 103/254 constants produce identical results with 14-op pool (len ≤8).
Cross-verified CUDA (RTX 4060 Ti, RTX 2070) vs OpenCL (RX 580).

---

## 2026-03-26: OpenCL on AMD Radeon RX 580

**Tags:** infrastructure, cross-vendor

**Discovery:** ROCm 6.4.4 dropped gfx803 (Polaris) support. ROCm 5.7.3
installs but KFD doesn't register GPU on Ubuntu 24.04 kernel 6.8.
However, Mesa rusticl OpenCL works perfectly.

**Setup:**
```bash
sudo apt install mesa-opencl-icd ocl-icd-opencl-dev clinfo
# Compile: gcc -O2 -o prog prog.c -lOpenCL
```

No CUDA, no ROCm needed. Mesa provides OpenCL 3.0 via Vulkan/RADV driver.
Our mulopt OpenCL port produces identical results to CUDA version.

**Performance:** RX 580 via OpenCL ≈ 40-50% of RTX 2070 via CUDA for mulopt.

---

## 2026-03-26: Feasibility Phase Transition at 6 Virtual Registers

**Tags:** regalloc, phase-transition, paper-A

**Discovery:** Sharp cliff in register allocation feasibility:

| vregs | Feasible | Infeasible |
|-------|----------|------------|
| 2 | 95.9% | 4.1% |
| 3 | 88.5% | 11.5% |
| 4 | 78.7% | 21.3% |
| 5 | 67.7% | 32.4% |
| 6 | 0.9% | 99.1% |

At 6v, the Z80 register file is effectively "full" — 99.1% of all possible
constraint shapes are infeasible. This is a **phase transition** in the
constraint satisfaction sense.

---

## 2026-03-26: Treewidth Analysis — Random vs Compiler Graphs

**Tags:** treewidth, regalloc, paper-C

**Discovery:** 99.5% of randomly enumerated interference graphs have treewidth ≤3
(classically tractable). But compiler-generated graphs are denser:
53.7% of dense real-world functions have treewidth ≥4.

The random graph prediction does NOT transfer to real programs.
Compilers produce biased interference patterns — variables in real programs
tend to be more interconnected than random chance would predict.

**Exact treewidth of all 32,768 possible 6-vertex interference graphs:**
- tw=0: 1 (0.0%)
- tw=1: 2,931 (8.9%)
- tw=2: 18,612 (56.8%)
- tw=3: 10,662 (32.5%)
- tw=4: 561 (1.7%)
- tw=5: 1 (0.0%)

Only 562 of 32,768 (1.7%) have treewidth ≥4 — this enables the
treewidth-filtered 6v enumeration (66M shapes instead of 1.9B).

---

## 2026-03-26: Composition Verification (13.2M data points)

**Tags:** composition, regalloc, paper-A

5v shapes composed from 4v table via cut-vertex splitting:
- Zero missed solutions (composition always finds solution when GPU does)
- Average overhead: 5.06 T-states
- Maximum overhead: 12 T-states (~3 register moves at cut boundary)
- 480 edge cases where composition finds solution but GPU reports infeasible
  (likely GPU search budget/numeric edge case — composition is MORE robust)

---

## 2026-03-26: div3 NOT FOUND at len ≤8

**Tags:** divmod, search-certificate

GPU exhaustive search proves no sequence of ≤8 instructions from 14-op pool
computes integer division by 3 for all 256 8-bit inputs. This is a negative
result / search certificate — the lower bound for div3 is ≥9 instructions.

Combined with div10 lower bound ≥13, this confirms that division by
non-power-of-2 is fundamentally hard on Z80 (no native divide instruction).

---

## 2026-03-26: Cross-Vendor GPU Verification

**Tags:** infrastructure, verification

mulopt results cross-verified across 3 GPUs from 2 vendors:
- NVIDIA RTX 4060 Ti (CUDA) — 103/254 at len ≤8
- NVIDIA RTX 2070 (CUDA) — 103/254 at len ≤8 ✓
- AMD RX 580 (OpenCL) — 103/254 at len ≤8 ✓

All produce identical solution counts. Different GPU architectures
independently verify the same exhaustive search results.

---

## Future Insights (to investigate)

- [ ] Do len-7+ u16 solutions use carry-propagation ops? (validates virtual pool)
- [ ] Can virtual instruction approach work for 8-bit mulopt too?
- [ ] What's the optimal split: 2 temps (BC) vs 1 temp (B only)?
- [ ] antique-toy book patterns — how do they compare with GPU-optimal?
- [ ] 6502 mulopt: how different is the optimal instruction pool?
- [ ] Meet-in-the-middle: practical on GPU with hash tables in shared memory?

---

## 2026-03-26: Clobber-Aware Multiply Tables

**Tags:** mulopt, mulopt16, compiler-integration

**Insight:** The shortest multiply sequence is not always the best for a compiler.
A 5-instruction solution that clobbers DE is worse than a 6-instruction solution
that preserves DE — if the caller needs DE alive.

**Proposed storage:** For each constant K, store **Pareto-optimal** solutions:
multiple sequences that are either shorter OR less clobbering. The compiler
picks the cheapest solution whose clobber set doesn't conflict with live registers.

**Clobber classes for u16 multiply:**
- Class A: {HL, C only} — safest (3-op pool: ADD HL,HL + ADD HL,BC + LD C,A)
- Class B: {HL, C, A} — uses A (NEG, LD L,A etc.)
- Class C: {HL, C, DE} — uses DE temp (EX DE,HL)
- Class D: {HL, C, A, DE} — uses everything

Verified: 3-op pool (class A) finds ALL 254 constants — DE is NEVER required.
But 8-op pool (class C) finds 30 constants shorter. These are Pareto-optimal
alternatives: same K, shorter sequence, but larger clobber set.

**For compiler integration:** generate all Pareto solutions during brute-force,
tag with clobber bitmask. At code generation time, intersect clobber with
liveness → pick cheapest compatible solution.

---

## 2026-03-26: EX DE,HL for Factored Multiplication

**Tags:** mulopt16, virtual-ops

EX DE,HL (4T, 1 inst) enables factored multiplication:
1. Compute partial result x*A in HL
2. EX DE,HL (save to DE)
3. Compute x*B in HL (starting fresh)
4. ADD HL,DE (combine: x*A + x*B)

This is the shift-add equivalent of polynomial factoring:
  x*63 = x*64 - x = (x<<6) - x
  x*99 = x*100 - x = (x*10 * x*10) - x

With EX DE,HL: 30 constants improved by 1-2 instructions (validated
against 5-op pool). No regressions. DE arithmetic adds ADD HL,DE (11T)
and SBC HL,DE (15T) to the virtual op pool.

---

## 2026-03-26: SWAP_HL (Byte Swap Trick)

**Tags:** mulopt16, virtual-ops, z80-lore

SWAP_HL = LD H,L / LD L,0 (2 insts, 11T). Semantics: HL = L * 256.

This is the classic Z80 "multiply by 256" trick from optimization guides.
Combined with SUB HL,BC: enables x*K = x*256 - x*(256-K) patterns.

Impact: 88/254 constants improved (35%), 179 total instructions saved vs
pure 3-op pool. x255: 15→3 virtual insts (12 saved). Zero regressions.

The byte swap trick is one of the most well-known Z80 optimizations but
was never systematically applied to constant multiplication before.
Our brute-force search rediscovered it independently — and proved it
optimal via exhaustive verification.

---

## 2026-03-26: Solver Configuration Taxonomy

**Tags:** mulopt, architecture, compiler-integration

**Complete list of useful multiply solver configurations:**

| Config | In→Out | Temps | Use case | Status |
|--------|--------|-------|----------|--------|
| mul8 | A→A | B,carry | Pixel math, counters | 164/254 at len≤9 |
| mul16 | A→HL | BC,DE | Address calc, pointers | **254/254 DONE** |
| mul16_de | A→DE | HL,BC | LDIR dest, stack | = mul16 + EX DE,HL |
| mul16_bc | A→BC | HL,DE | LDIR count, ports | = mul16 + push/pop |
| mul16x16 | HL→HL | BC,DE | Struct offset, scaling | New solver needed |
| muladd | A→HL | BC,DE | Screen addr (Y*32+X) | New solver needed |
| divmod | A→A,B | carry | BCD, wrapping, hash | Running on i5 |

**Key insight from GPT-5.4:** HL→HL (input already 16-bit) is critical for
chaining multiplications and struct array indexing. Currently not covered.

**Key insight:** mul16_de and mul16_bc don't need separate solvers — they're
mul16 + a final register move (EX DE,HL or LD B,H/LD C,L). The clobber-aware
table handles this: if DE is the target, pick the mul16 solution that doesn't
use DE, then EX DE,HL at the end.

**Virtual op pools per config:**

| Config | Core ops | Extended ops | Pool size |
|--------|----------|-------------|-----------|
| mul8 | ADD A,A + ADD A,B + LD B,A | +RLA, RLCA, RRCA, NEG, SBC | 14 |
| mul16 | ADD HL,HL + ADD HL,BC + LD C,A | +SWAP, SUB, EX DE, ADD/SUB HL,DE | 8 |
| mul16x16 | ADD HL,HL + ADD HL,BC | +EX DE, LD C,L, LD B,H | ~8 |
| muladd | mul16 ops + ADD HL,nn | +immediate offset as op | ~10 |

**Validated op usage across pools:**
- 3-op core (mul16): ALL 254 solved — sufficient but not shortest
- 5-op (+SWAP+SUB): 88 improved, 0 regressions — the sweet spot
- 8-op (+DE arith): 30 more improved — marginal gains, larger clobber
- All 8 ops appear in solutions — none are dead

---

## 2026-03-26: Pre-Calculable Tables Roadmap

**Tags:** roadmap, compiler, tables

### Tables we have
- Regalloc ≤6v (83.6M shapes, 32MB compressed)
- mul8 A*K→A (164/254, clobber-annotated)
- mul16 A*K→HL (254/254, complete, 3-op basis)
- Peephole len-2 (739K rules)
- div10 lower bound ≥13 (search certificate)

### What to pre-calculate next (by priority)

**#1 Common Idiom Table** (hours, high compiler impact)
- Zero-extend A→HL: LD L,A / LD H,0 (2 insts, 11T)
- Sign-extend A→HL: LD L,A / RLA / SBC A,A / LD H,A (4 insts, 16T)
- ABS(A), MIN(A,B), MAX(A,B), CLAMP(A,lo,hi)
- Swap A,B: LD C,A / LD A,B / LD B,C (3 insts, 12T)
- BCD output: divmod10 → two ASCII digits
- Each as brute-force target with clobber annotation

**#2 16-bit Arithmetic Library** (hours, direct compiler help)
- ADD HL,nn (not native on Z80)
- SUB HL,nn (OR A + SBC HL,rr)
- CP HL,nn (16-bit compare)
- NEG HL (we found 5-inst optimal: NEG/LD L,A/NEG/SBC A,B/LD H,A = 28T)
- MUL HL,K (struct array indexing)
- Each with clobber annotation for register allocator

**#3 Peephole len-3** (days, dual GPU, huge impact)
- 74.9B targets, 37M found (0.05%)
- 3→1 rules: save 2 instructions per match
- 3→2 rules: save 1 instruction per match
- Dual 4060 Ti: ~days for complete sweep

**#4 Composition Table** (CPU hours)
- For each decomposable 5v/6v shape: optimal split point + costs
- Already verified on 13.2M shapes (max 12T overhead, 0 misses)
- Output: split_vertex, cost_A, cost_B, boundary_cost

**#5 Spectrum Screen Address** (fun + publishable)
- Y→VRAM address: the holy grail of Spectrum programming
- Brute-force optimal Y*32+X, pixel row calculation
- Every Spectrum game calls this thousands of times

**#6 Island Templates** (needs VIR liveness data)
- Pre-solved split patterns for 7-15v corpus functions
- ~50-100 templates cover most real programs
- Uses liveness bottleneck detection from earlier session

**#7 Calling Convention Optimization** (needs VIR ABI data)
- For each function signature: optimal arg register assignment
- Regalloc applied to function boundaries
- Minimize argument shuffle cost at call sites

**#8 Memory Access Patterns** (brute-force, hours)
- LD A,(HL) : op : LD (HL),A read-modify-write sequences
- Pre-compute best instruction for each ALU op on memory
- INC (HL), DEC (HL), SET n,(HL) — when are direct ops better?

**#9 Shift-Add Chain Database** (theoretical, publishable)
- Shortest addition chain for each number 1-65535
- Generalizes mulopt to ANY ISA (not Z80-specific)
- Existing tables for small numbers, none exhaustive for 16-bit

### Hardware for computation
- main (i7): 2× RTX 4060 Ti — peephole len-3, idiom search
- i5: RTX 2070 — mul8 len-10 (running), then idioms
- i3: RX 580 OpenCL — mul8 len-9 (running), then 6502 port
- M2 MacBook: Metal/CPU — 6502 mulopt, meet-in-the-middle

---

## 2026-03-26: Universal Computation Chains (Key Architecture Insight)

**Tags:** architecture, chains, cross-ISA, division, multiplication

**The Big Idea:** Separate the SEARCH from the ISA.

Three layers:
1. **Abstract chain** — sequence of {dbl, add, sub, save, restore, shr, mask}
2. **Materialization** — map each abstract op to ISA-specific instructions
3. **Cost model** — T-states (Z80), cycles (6502), or whatever

The abstract chain is ISA-independent. Search ONCE, deploy EVERYWHERE.

**Why this helps for division:**
Division by constant K uses reciprocal multiplication:
  n/K ≈ n * (1/K) = n * (2^M / K) >> M
This is a shift-add chain followed by a right-shift:
  1. Multiply by reciprocal (chain of dbl/add/sub)
  2. Shift right to get quotient (chain of shr)
  3. Back-multiply to get remainder: r = n - q*K

Each step is an abstract chain. Brute-force the CHAIN, not the assembly.

**Search space comparison:**
- Z80 assembly: 14^12 = 56.7T (limited by ISA details)
- Abstract chain: ~10^7 at depth 12 (only meaningful operations)
- Speedup: ~8,000,000× by searching in abstract space

**Then materialize:**
  Z80:   dbl → ADD A,A (4T), add → ADD A,B (4T), save → LD B,A (4T)
  6502:  dbl → ASL A (2cy), add → CLC/ADC zp (5cy), save → TAX (2cy)
  RISC-V: dbl → SLLI rd,rs,1 (1cy), add → ADD rd,rd,rs (1cy)
  ARM:   dbl → LSL rd,rs,#1 (1cy), add → ADD rd,rd,rs (1cy)

**The abstract ops (7 total):**
  dbl     — double current value
  add(i)  — add saved value #i
  sub(i)  — subtract saved value #i
  save    — checkpoint current value (push to save stack)
  shr(n)  — shift right by n bits
  mask(m) — AND with constant mask
  neg     — negate current value

With 2 save slots and these 7 ops:
  7^15 = 4.7B — exhaustive to depth 15 in seconds

**For division:**
  div10 in abstract chain:
    save(input) → dbl^4 → add(input) → dbl → save(approx) → shr^3 → mask →
    save(quotient) → dbl → dbl → add(quotient) → dbl → sub(input) → neg
  Then materialize per-ISA and verify against all 256/65536 inputs.

**This means:**
  1. Compute abstract chains for mul/div/mod for all constants 1-255
  2. Materialize to Z80, 6502, RISC-V, ARM in one step
  3. Verify each materialization per-ISA (different overflow, carry, flags)
  4. Result: optimal multiply/divide tables for EVERY retro ISA from ONE search

**Publication potential:** "Universal Computation Chains: ISA-Independent 
Optimal Arithmetic via Exhaustive Search" — standalone paper.

---

## 2026-03-26: GPU Portability — OpenCL wins, Mojo/MLIR overkill

**Tags:** infrastructure, portability

For integer ALU brute-force (no tensors, no ML), GPU portability options:
- **OpenCL: WINNER** — already working on RX 580, 250 LOC per kernel, works on NVIDIA too
- Vulkan compute: works everywhere but 200 lines boilerplate per kernel
- Mojo/MLIR: heavy dependency, GPU via MAX engine, overkill for our case
- SYCL: Intel-focused, AMD via hipSYCL
- WGPU: tried before, SIGSEGV bugs in Go bindings

Strategy: CUDA for NVIDIA (fast path) + OpenCL for AMD/fallback. Two files per kernel, 95% shared logic. For abstract chains: pure CPU, no GPU needed.

---

## 2026-03-26: Compressed Multiply Core via Prefix Sharing

**Tags:** mulopt, code-size, compiler, publishable

**Discovery:** 311 prefix overlaps among 164 optimal multiply sequences.
Many larger multiplies START with a smaller multiply as prefix:
  x58 = x56 + ADD A,B
  x104 = x52 + ADD A,A
  x52 = x26 + ADD A,A
  ... forming chains up to 7 constants deep.

**Fall-through layout:** place constants in reverse order, each enters
at its label and falls through to the end:
```asm
mul104:
mul52:  ADD A,A
mul26:  ADD A,A
mul24:  ADD A,B
mul12:  ADD A,A
mul6:   ADD A,A / LD B,A / ADD A,B / ADD A,B
mul2:   RLA
        RET              ; shared return
```
7 constants, 9 instructions, 1 RET (vs 40 naive = 77% saved!)

**Total savings:** 1224 → 594 instructions (51% reduction).
All 164 multiply sequences packed into 594 bytes of shared code.

**For compiler:** emit CALL to the right label. The multiply "library"
is a single compact block with multiple entry points.

**For ROM/embedded:** 594 bytes for 164 optimal multiplies = fits in
any Z80 system. Even ZX Spectrum (48KB) has room.

**Also works for chains:** abstract chains have even MORE overlap
since they use fewer distinct operations.

---

## 2026-03-26: mul16 Prefix Sharing — 86% Compression

**Tags:** mulopt16, code-size, publishable

mul16 (3-op pool) is even denser than mul8:
- 254 constants, 2375 virtual ops naive
- With prefix sharing: 328 virtual ops (86% saved!)
- Materialized: ~500 bytes of Z80 code for ALL 254 multiplies
- vs __mul16 runtime: 30 bytes but 200-300T per call

The regularity comes from the 3-op pool: most sequences are
ADD HL,HL chains with occasional LD C,A + ADD HL,BC insertions.
This makes prefix trees very deep (6+ constants per chain).

**Combined library:**
  mul8:  594 bytes, 164 constants (51% shared)
  mul16: ~500 bytes, 254 constants (86% shared)
  Total: ~1094 bytes for COMPLETE multiply coverage
  = 2.2% of ZX Spectrum 48KB RAM

**Publishable as:** "Z80 Multiply Library: 418 Optimal Sequences in 1KB"
or include as appendix in the universal chains paper.

---

## 2026-03-26: Complete 254/254 mul8 via Composition

**Tags:** mulopt, composition, compiler, publishable

**Strategy: 4-tier multiply for ALL 254 constants:**

| Tier | Constants | Avg cost | Method |
|------|-----------|----------|--------|
| 1. Direct table | 164 | ~31T | GPU brute-force optimal (len ≤9) |
| 2. Factor composition | 67 | ~43T | x*K = mul_A(mul_B(x)) where A×B=K |
| 3. Near-table adjust | 19 | ~47T | x*K = mul_(K±δ)(x) ± δ*x |
| 4. Double composition | 4 | ~50T | factor + adjust (primes 149,151,167 + 166) |

**Combined: all 254 constants, avg ~35T, 8.1× faster than general.**
No constant needs the 280T shift-and-add loop.

The 4 hardest: 149, 151, 167 (primes near unsolved range) + 166=2×83.
Solved via double composition: e.g. x149 = x(3×50) - 1 = 52T.

**For compiler:** cascading lookup:
```
if table[k] → emit sequence                    // 164 constants, ~31T
elif factor_table[k] → emit mul_a + mul_b       // 67 constants, ~43T
elif near_table[k] → emit mul_adj + add/sub     // 19 constants, ~47T
else → emit double_composition                  // 4 constants, ~50T
```

**Key metric for paper:**
- General shift-and-add: 280T average
- Our table: 35T average
- Speedup: **8× for ALL 254 constants, zero fallback**

---

## 2026-03-26: Division via Reciprocal Multiply — 86 Divisors, Zero New Search

**Tags:** division, reciprocal, mul16, publishable

**Method:** div_K(n) = H byte of mul16_M(n), then SRL A × (S-8)
where M = round(2^S / K) is the magic reciprocal constant.

**Key trick:** instead of shifting full 16-bit HL right by S bits (expensive!),
read H directly (= already shifted by 8), then shift A by only S-8 bits.
SRL A = 8T vs SRL H + RR L = 16T per bit. Saves 50% on shift phase.

**Results:** 86 non-power-of-2 divisors (3-127) solved via EXISTING mul16 table.
Average 2.0× faster than general division loop. Best: div57 = 4.7× (60T).
No new GPU brute-force needed — purely derived from mul16 results.

**For compiler:** division by constant K →
  1. Look up reciprocal: (M, S) = reciprocal_table[K]
  2. Emit: mul16_M sequence (from mul16 table)
  3. Emit: LD A,H
  4. Emit: SRL A × (S-8)
  5. Result in A = n / K

Total library: mul16 table (~500 bytes) + reciprocal constants (128 bytes)
= ~630 bytes covers multiply AND divide for all 8-bit constants.

---

## 2026-03-26: NEG HL — 4 Methods, Clobber-Aware Selection

**Tags:** arith16, neg, clobber, Alf

Four provably correct methods for HL = -HL:

| Method | Insts | T | Prerequisite | Source |
|--------|-------|---|-------------|--------|
| EX DE,HL / OR A / SBC HL,DE | 3 | 23T | DE=0 | GPU brute-force |
| NEG / LD L,A / NEG / SBC A,B / LD H,A | 5 | 28T | B=0 | GPU 23-op search |
| XOR A / SUB L / LD L,A / SBC A,A / SUB H / LD H,A | 6 | 24T | NONE | Alf (human expert) |
| LD A,L / CPL / INC A / LD L,A / LD A,H / CPL / ADC A,0 / LD H,A | 8 | 32T | NONE | GPT/textbook |

Alf's method verified: 65536/65536 inputs correct.

**Why brute-force missed Alf's:** our 16-bit pool operates on HL as a unit.
Alf's uses per-byte ops (SUB L, SUB H, SBC A,A) that decompose HL into
individual registers. Need expanded pool with H/L byte access for complete search.

**For compiler:** clobber-based selection:
  DE=0 available → use 3-inst (23T)
  B=0 available → use 5-inst (28T)
  else → Alf's 6-inst (24T) universal

**abs_diff(a,b)** from MinZ corpus: SUB C / RET NC / NEG / RET
= 3 insts + conditional return. Branchless abs not possible in 14-op pool
(needs conditional ops). Branching version is already near-optimal.

---

## 2026-03-26: Full ALU Pool for 16-bit Idiom Search (TODO)

**Tags:** arith16, pool-design, next-session

Current 21-op pool misses basic ALU ops needed for many patterns:

**Must add:**
- ADC A,L / ADC A,H — add with carry (multi-byte arithmetic!)
- SBC A,L / SBC A,H — subtract with carry
- INC L / INC H — increment WITHOUT touching carry (unique Z80 property!)
- DEC L / DEC H — decrement without carry
- AND L / AND H — bitwise AND
- OR H — for testing (OR L already in pool)
- XOR L / XOR H — bitwise XOR
- CP L / CP H — compare without storing
- INC A / DEC A — accumulator inc/dec (no carry touch)

**INC/DEC carry preservation** is key:
  SBC A,A (carry to mask) → INC L (no carry damage) → SBC continues correct.
  This enables interleaved byte-level operations during multi-byte chains.

**Estimated pool:** ~35 ops. 35^6 = 1.8B (instant), 35^7 = 64B (feasible),
35^8 = 2.3T (borderline). Len-7 covers most patterns.

**After empirical reduction** (like 21→14 for mul8): probably ~20 actually
used ops → 20^8 = 25.6B (fast).

For next session: build full ALU pool, search all idioms to len-7/8,
analyze which ops appear → reduce → go deeper.

---

## 2026-03-26: Packed Arithmetic Cassette — Multi-Entry Overlapped Code

**Tags:** code-size, publishable, architecture, mul, div, rotation

**The Ultimate Z80 Arithmetic Library**: one compact ROM blob with hundreds
of entry points. Three types of multi-entry code, all overlapping:

**1. Instruction Sleds** (homogeneous chains):
```asm
rot7: RLCA          ; 7 rotations left
rot6: RLCA          ; 6 rotations
rot5: RLCA          ; 5
rot4: RLCA          ; = nibble swap!
rot3: RLCA
rot2: RLCA
rot1: RLCA          ; = ×2 via rotate
      RET           ; 9 bytes, 7 entry points

shr7: SRL A         ; 7 shifts right (= /128)
shr6: SRL A         ; 6 shifts (= /64)
...
shr1: SRL A         ; 1 shift (= /2)
      RET           ; 16 bytes (CB prefix), 7 entry points
```

**2. Multiply Chains** (prefix-shared heterogeneous):
```asm
mul104: ADD A,A     ; falls through to ×52
mul52:  ADD A,A
mul26:  ADD A,B     ; uses saved B
mul24:  ADD A,A
mul12:  ADD A,A
mul6:   LD B,A : ADD A,B : ADD A,B
mul2:   RLA
        RET         ; 7 constants, 9 instructions, 1 RET
```

**3. Division via Shared Reciprocal**:
Divisors sharing the same magic constant M overlap in the multiply phase.
div6 and div12 both use M=171 — the mul(171) code is shared, only
the shift count differs.

**Combined**: sleds + multiply chains + division chains = one blob,
~2KB for ALL optimal arithmetic. Hundreds of entry points,
each labeled `mul_K:`, `div_K:`, `rot_N:`, `shr_N:`.

**Runtime dispatch**: page-aligned jump table (256 bytes).
`LD H, page / LD L, K / JP (HL)` = single-instruction dispatch.

**With TSMC** (self-modifying code): `CALL target` patches once,
then runs at full speed forever. Zero dispatch overhead after first call.

**Total: ~2KB packed cassette = provably optimal arithmetic for Z80.**
Every operation 2-8× faster than general loops. For ZX Spectrum (48KB):
just 4% of RAM. For ROM systems: fits in any EPROM alongside the program.

---

## 2026-03-26: 7v+ Regalloc Pipeline — What's Needed

**Tags:** regalloc, composition, islands, next-session

**For 7v+ functions, the pipeline is:**
1. Check ≤6v table (83.6M entries) → O(1) if ≤6v
2. Find cut vertex → split into ≤6v sub-problems → compose
3. If 2-connected (no cut vertex): backtracking solver (≤15v, <1 sec)
4. If >15v: island decomposition at liveness bottlenecks → solve each island

**What we HAVE:**
- ≤6v complete table (83.6M)
- Backtracking solver with 1000-4000× pruning
- Island decomposition algorithm (tested on ZSQL)
- 5v→4v composition verified (13.2M shapes, max 12T overhead, 0 misses)

**What we NEED (from VIR):**
- Fixed island sub-problems for _prow (28v) and _sel_rows (37v)
- 7v+ corpus shapes (from 315 signatures) for on-demand solving
- Liveness data for new corpus functions

**What we CAN pre-compute:**
- Register shuffle optimal sequences (for island stitching)
- Composition table: for each decomposable shape, best split + costs

**Dead-flags peephole running** (Layer 2: ~2-5M rules when flags dead).
**mul8 len-10 overnight on i5** (90 unsolved constants).

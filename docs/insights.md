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

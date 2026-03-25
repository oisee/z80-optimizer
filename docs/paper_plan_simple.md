# Paper Plan in Simple Words

## Title

**"Register Allocation as a Solved Game"**

## The One-Sentence Version

We used GPUs to try every possible way to assign variables to registers on a Z80 CPU, saved the answers in a table, and discovered that real programs only need ~315 different patterns — so the compiler can just look up the answer instead of solving it every time.

## The Problem (What We're Trying to Do)

When a compiler turns your code into machine instructions, it needs to decide which variables go in which CPU registers. This is called **register allocation**. It's a hard problem (NP-hard) — the number of possibilities grows exponentially.

Traditional compilers handle this in two ways:
- **Heuristics** (SDCC, GCC): fast guesses, often good, sometimes bad. No guarantee of optimality.
- **SMT solvers** (Z3): mathematically optimal, but slow (~100ms per function).

We tried a third way: **just try all possibilities on a GPU**.

## The Surprising Discovery

We expected the table of "all solved register allocation problems" to be enormous. Instead:

1. **Only 315 unique patterns** cover 1360 real functions across 8 programming languages.
2. **Adding an entire standard library** (140 functions) only added 7 new patterns (+2.2%).
3. **Training on basic programs** (math, parsing) gives **88.2% hit rate** on completely different programs (business logic, databases, screen rendering).

The vocabulary of register allocation patterns is **tiny and universal**. Real programs reuse the same patterns over and over.

## Why This Works (The Phase Transition)

The Z80 has 15 register locations. For a function with N variables, there are 15^N possible assignments:

| Variables | Possibilities | GPU Time |
|-----------|--------------|----------|
| 4 | 50,625 | instant |
| 6 | 11 million | 1 second |
| 8 | 2.5 billion | 15 seconds |
| 10 | 576 billion | 5 minutes |

81% of real Z80 functions have ≤8 variables → solvable in seconds.

But for ARM (32 registers): 32^6 = 1 billion already for just 6 variables. The **cliff** is at ~16 register locations. Below that, brute force works. Above, it doesn't.

Z80 sits right at the sweet spot — just below the cliff.

## What We Built

Four CUDA GPU kernels:

1. **Register allocator** (`z80_regalloc.cu`): tries all register assignments, finds the cheapest one. JSON input/output, server mode for compiler integration.

2. **Multiply optimizer** (`z80_mulopt.cu`): finds shortest instruction sequence to multiply by any constant (Z80 has no MUL instruction). Found optimal sequences for 103 out of 254 constants.

3. **16-bit multiply** (`z80_mulopt16.cu`): same but for full 16-bit results. Uses ADD HL,HL (native 16-bit double) as key building block.

4. **Division/modulo** (`z80_divmod.cu`): searched for short division sequences. Proved that dividing by 10 needs at least 13 instructions — a mathematical lower bound via exhaustive search.

## The Three Key Results (for the paper)

### Result 1: Convergence
315 signatures cover everything. The table plateaus. Adding more programs barely grows it. The compiler's "dictionary" of allocation patterns is finite and small.

### Result 2: Transfer
Patterns learned from simple programs (arithmetic, parsing) work on complex programs (SQL databases, business logic). 88.2% hit rate across completely different application domains.

### Result 3: Phase Transition
There's a sharp boundary at ~16 register locations. Below: GPUs can solve everything. Above: only small functions are tractable. This applies to ALL processors:

- 6502 (3 registers): trivially solvable
- Z80 (15 locations): sweet spot — most functions solvable
- ARM Thumb (16 registers): just past the cliff
- RISC-V (32 registers): only tiny functions

## The Chess Analogy

Chess endgame tablebases: computers solved ALL positions with ≤7 pieces. Now any chess engine can look up the provably optimal move instantly. No thinking required — the game is "solved" for those positions.

We did the same for register allocation: solved ALL patterns with ≤6 variables. Now the compiler looks up the provably optimal assignment instantly. No solver required — the problem is "solved" for those patterns.

For larger functions (>6 variables): split into smaller pieces at function call boundaries (where most variables die), solve each piece from the table, connect pieces with minimal register shuffles.

## The Demo

SQL running on a ZX Spectrum (1982 home computer):
```
SELECT * FROM users
→ Alice|30, Bob|25, Charlie|35
```

Compiled by MinZ → optimized by GPU table → assembled → runs on Z80 CP/M with SQLite. Provably optimal register allocation inside a SQL database on 8-bit hardware.

## Paper Structure

1. **Introduction**: "When is exhaustive search practical for compilers?"
2. **The Table**: GPU kernel, JSON protocol, 15-location model
3. **Convergence**: 315 patterns, +2.2% from stdlib, marginal coverage curve
4. **Transfer**: Train on A, test on B → 88.2% hit rate
5. **Phase Transition**: Cliff at 16 locations, formula: max_vregs = log(budget)/log(locations)
6. **Island Decomposition**: Split large functions → table lookup per island → stitch at boundaries
7. **Instruction Synthesis**: Multiply tables, division lower bounds
8. **Generalization**: 6502, GameBoy, ARM, RISC-V — same framework, different parameters
9. **Demo**: SQL on ZX Spectrum

## Why It Matters Beyond Retro Computing

The core insight — "real programs use a tiny fraction of theoretical possibilities" — applies to any compiler optimization:

- Instruction scheduling
- SIMD vectorization patterns
- Cache-aware memory layout

If the pattern space is small enough, precompute everything offline, ship a table, compile at lookup speed. GPU is just the tool that proved the space is small.

## Status

- All GPU kernels: built, tested, committed
- Exhaustive tables: 12M+ entries (2-6 vregs)
- Corpus: 1360 functions, 315 signatures, convergence proven
- Transfer experiment: 88.2% cross-frontend
- Phase diagram: measured across 7 architecture sizes
- Demo: SQL on ZX Spectrum, screenshot ready
- Research statement: written, peer-reviewed by 3 AI researchers
- Paper draft: ready to start

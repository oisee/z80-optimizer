# GPU Brute-Force Superoptimization: A Practical Guide

*From Z80 to Universal Computation Chains*

## Part I: The Z80 Superoptimizer

### Chapter 1: Why Brute Force?
- Compilers make suboptimal choices. Can we prove what's optimal?
- The key insight: Z80's 11-byte state is GPU-friendly
- 739K proven peephole rules in minutes, not months

### Chapter 2: The Three-Stage GPU Pipeline
- QuickCheck (8 vectors, 99.99% rejection)
- MidCheck (32 vectors, BIT/SET/RES targeting)
- ExhaustiveCheck (full sweep, shared-memory early termination)
- Performance: 30× faster than CPU, 743K rules verified

### Chapter 3: Instruction Pool Reduction
- Start with 21 ops, analyze which appear in solutions
- 7 ops never used → remove → **38× speedup**
- Empirical pool reduction as a general technique
- The lesson: most instructions are useless for most computations

### Chapter 4: Constant Multiplication
- 254/254 u8 constants solved (8× faster than shift-and-add)
- 254/254 u16 constants solved in 30 seconds (3-op basis!)
- Pool reduction: 23→3 ops for u16 = **13,600× speedup**
- NEG trick: ×255 = 1 instruction (8T)
- Prefix sharing: 51% code compression, multiple entry points

### Chapter 5: Division by Constant
- The reciprocal trick: n/K = (n × M) >> S
- Abstract chains guide GPU search: 6-op focused pool
- 118/120 divisors found in 11 seconds each
- div10 = 124T matches Hacker's Delight (found automatically!)
- Guided brute-force: abstract oracle → ISA-specific materialization

### Chapter 6: The Idiom Zoo
- 15 branchless idioms found: bool, abs, sign, not, lsb, ...
- ABS in 6 insts branchless: carry-to-mask trick
- Sign-extend in 3 insts (12T): ADC overflow → SBC mask
- CPL: the instruction we forgot (complement in 1 inst vs 2)

## Part II: Register Allocation as a Solved Game

### Chapter 7: Exhaustive Register Allocation
- 83.6 million provably optimal allocations
- Feasibility phase transition: 96%(2v) → 1%(6v)
- The Z80 register file "fills up" — a mathematical cliff

### Chapter 8: Treewidth and Decomposition
- 99.5% of random graphs decompose classically
- But compiler-generated graphs are denser (53.7% tw≥4)
- The honest result: theory doesn't match practice
- Five-level pipeline: table → composition → GPU → backtrack → Z3

### Chapter 9: Cross-Function Optimization
- ZSQL: 31 functions, 5 profitable merges (210T saved)
- Island decomposition for 28v-37v functions
- Partition optimizer: bottom-up DP on call graph
- The merge decision: when CALL/RET overhead exceeds merge cost

### Chapter 10: The Backtracking Solver
- CPU fallback for GPU-intractable problems (>5T search space)
- Pattern-aware location masks: 1000-4000× pruning
- Constraint propagation + forward checking + most-constrained-first
- Why it works: sparse interference = most assignments infeasible early

## Part III: Universal Computation Chains

### Chapter 11: Abstract Chains
- ISA-independent: {dbl, add, sub, save, neg, shr}
- One search → materialize to Z80, 6502, RISC-V, ARM
- Modular arithmetic: NEG in chains vs NEG on hardware
- 254/254 multiply chains in 8 seconds on CPU

### Chapter 12: Guided Brute-Force
- Abstract chain predicts depth and structure
- GPU searches only the ISA-specific materialization space
- 6 ops instead of 37 = millions× faster
- Division: abstract says mul(M)+shr(S), GPU finds exact Z80 sequence

### Chapter 13: The Clobber Problem
- Shortest ≠ best for a compiler
- Pareto-optimal solutions: multiple sequences per constant
- B-preserving vs B-clobbering: compiler picks by liveness
- 14 B-safe multiplies, 150 B-clobbering (same lookup API)

## Part IV: Multi-Backend GPU Computing

### Chapter 14: One ISA Definition, Four Backends
- The gpugen DSL: ISA → CUDA / Metal / OpenCL / Vulkan
- 250 lines per kernel, 95% shared logic
- Cross-vendor verification: NVIDIA × AMD × Apple = identical results

### Chapter 15: The Hardware Zoo
- RTX 4060 Ti (CUDA): the workhorse
- RTX 2070 (CUDA): the validator
- Radeon RX 580 (OpenCL + Vulkan): the AMD proof
- M2 MacBook Air (Metal): the Apple proof
- ROCm broken for gfx803 — but Mesa saves the day

### Chapter 16: OpenCL via Mesa
- When ROCm fails: Mesa rusticl provides OpenCL 3.0
- No CUDA, no ROCm, no HIP — just Mesa and a GPU
- Verified: identical results to CUDA on same search
- The lesson: open drivers matter

## Part V: Results and Applications

### Chapter 17: The Complete Tables
- 83.6M regalloc entries (32MB compressed, format spec + readers)
- 254 mul8 + 254 mul16 + 118 div + 15 idioms
- 739K peephole rules
- Everything in data/ with Python/Go reader examples

### Chapter 18: Compiler Integration
- Go packages: pkg/mulopt/, pkg/regalloc/, pkg/peephole/
- INCBIN for runtime tables, inline for compile-time
- The 5-level pipeline: table → composition → GPU → backtrack → Z3
- v0.23.0: 372 arithmetic sequences shipping in production

### Chapter 19: What's Next
- 6502 constant multiplication (similar ISA, different pool)
- Meet-in-the-middle for deeper search
- 32-bit arithmetic via shadow registers (EXX)
- The self-hosting dream: Z80 allocating its own registers
- "Universal Computation Chains" as a standalone paper

## Appendices

### A: Binary Table Format (Z80T v1)
### B: Complete Multiply Table (254 entries)
### C: Complete Division Table (118 entries)
### D: Idiom Reference Card (15 branchless patterns)
### E: GPU Kernel Architecture (QuickCheck → Mid → Exhaustive)
### F: ISA DSL Reference (gpugen)
### G: Hardware Setup (CUDA + OpenCL + Vulkan + Metal)

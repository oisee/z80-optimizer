# When Is Exhaustive Search Practical for Compiler Optimization?

## Research Statement (Draft)

### The Question

Modern compilers solve NP-hard optimization subproblems (register allocation, instruction selection, scheduling) using heuristics or SMT solvers. Heuristics are fast but leave performance on the table. Solvers give optimal results but scale poorly.

We ask: **for which compiler subproblems can GPU-accelerated exhaustive search replace both heuristics and solvers?**

### The Answer (Preview)

For processors with small state spaces (≤15 register locations), exhaustive search on commodity GPUs is practical for the vast majority of real functions. The key insight is not GPU speed — it's **signature reuse**: real programs occupy a tiny fraction of the theoretical constraint space, enabling precomputed lookup tables that cover 80-90% of functions with zero runtime cost.

### Three Claims

**Claim 1: Exhaustive GPU search is practical for register allocation on small architectures.**
For the Zilog Z80 (15 register locations), we solve all possible allocation problems for 2-6 virtual registers (12M+ feasible entries) on two RTX 4060 Ti GPUs in 25 minutes. Functions with ≤6 vregs constitute 81% of a 1605-function multi-language corpus.

**Claim 2: Corpus-driven table generation achieves 7700x reduction over blind enumeration.**
Real programs exhibit massive constraint reuse: 56 unique allocation signatures cover 639 functions (11.4x reuse factor). A table built from basic programs (arithmetic, parsing) achieves 88.2% hit rate on unseen application domains (business logic, database access, UI rendering). This is a general principle applicable beyond register allocation.

**Claim 3: Exhaustive search produces certificates of hardness for instruction synthesis.**
For constant division by non-power-of-2 on 8-bit architectures, we prove a lower bound of 13 instructions via exhaustive search over all sequences up to length 12 with 21 instruction types. This closes the question of whether short division sequences exist for register-only Z80 code.

### Structure of the Full Paper

**Section 1: Tableability** — When can a compiler pass be materialized offline?
- Definition: tableability(P) = |functions served| / |table entries|
- Z80 regalloc: tableability = 11.4 (measured)
- Conditions for high tableability: finite state space, signature reuse in real programs
- Connection to chess endgame tablebases and other solved-game paradigms

**Section 2: GPU Architecture for Compiler Optimization**
- Three search modes: assignment (regalloc), synthesis (multiply), verification (divmod)
- Solver-as-a-service pattern: long-running CUDA process, JSON protocol
- Dual-GPU pipeline: 30K solves/sec throughput
- `cudaDeviceScheduleBlockingSync` for CPU-friendly operation

**Section 3: Exhaustive Register Allocation**
- 15-location model: 7 GPR + 4 pairs + 4 IX/IY + memory spill
- Per-vreg width constraints (8-bit vs 16-bit location restriction)
- Corpus-derived enumeration: extract shapes from real compiler, vary interference
- Exhaustive tables: 12M+ entries for 2-6 vregs

**Section 4: The Transfer Experiment**
- Train on frontends A-D, test on E-H (zero overlap)
- 88.2% hit rate on unseen frontends
- Per-domain analysis: ABAP 98.8%, Screen 89.6%, SQLite 45.9%
- Why it works: Z80 ALU operations are the same across domains, only composition differs
- Marginal coverage analysis: how many entries until the table is "full"?

**Section 5: Island-of-Optimality Decomposition**
- Functions with >6 vregs: split at liveness bottlenecks (call sites)
- Each island ≤K vregs → table lookup (O(1), provably optimal)
- Boundary joins: min-cost register shuffle (2-3 vregs, trivial)
- Connection to treewidth decomposition of interference graphs
- Theoretical bound: total cost ≤ sum(optimal_islands) + O(K × boundaries)

**Section 6: Instruction Synthesis and Certificates of Hardness**
- Constant multiplication: 103/254 8-bit constants solved at length ≤8
- NEG as "negative multiplication" trick (×255 = NEG, 1 instruction)
- 16-bit multiplication via carry-spill and ADD HL,HL
- Division/modulo: proved lower bound ≥13 for div-by-10
- Hacker's Delight divmod10: 27 instructions, 124T, verified 256/256

**Section 7: Generalization Beyond Z80**
- Memory hierarchy as register file: 6502 zero page (256 slots), GPU scratchpad, embedded SRAM
- Tableability across architectures: smaller state → higher tableability
- GameBoy LR35902: remove IX/IY → 7 locations, 400x smaller search space
- Modern relevance: RISC-V RV32E (16 registers), ARM Thumb (8 registers)

### Why This Matters Beyond Retro Computing

The principle "real programs occupy a tiny fraction of theoretical constraint space" applies to any compiler optimization with finite input structure:
- **Instruction scheduling** on in-order processors (finite pipeline states)
- **SIMD vectorization** patterns (finite lane configurations)
- **Memory layout optimization** for cache performance (finite cache set mappings)

If the tableability of these passes is high (as it is for register allocation), the same GPU-materialization approach could replace heuristics across modern compilers.

### Experimental Setup

- Hardware: 2× NVIDIA RTX 4060 Ti 16GB, CUDA 12.9
- Corpus: 1605 functions from 8 MinZ compiler frontends (Nanz, C89, PL/M, ABAP, SQLite, Screen, FS, Lizp)
- Baselines: Z3 SMT solver (optimal but slow), SDCC heuristic graph coloring (fast but suboptimal)
- All results: provably optimal (exhaustive search), verified against Z3 for subset

### Key Numbers

| Metric | Value |
|--------|-------|
| Exhaustive table entries (2-6 vregs) | 12.3M feasible |
| Corpus coverage from table | 81% of functions |
| Signature reuse factor | 11.4× (56 entries → 639 functions) |
| Transfer hit rate (unseen frontends) | 88.2% |
| GPU throughput | 30K solves/sec (dual GPU) |
| Multiply table coverage | 103/254 constants |
| Division lower bound | ≥13 instructions (proven) |
| Divmod10 best known | 27 instructions, 124T |

---

## Appendix: Alternative Research Framings (from peer review)

### Core Reframing

> Our main result is not merely that GPUs accelerate superoptimization. Rather, they make it practical to empirically probe the structure of backend search spaces on small irregular architectures. What emerges is not unstructured combinatorial explosion, but a sharply compressed vocabulary of recurring allocation signatures and arithmetic idioms. This suggests a broader paradigm: parts of code generation may be shifted from online search to offline knowledge compilation, with the compiler acting as a retrieval engine over precomputed optimality artifacts.

### 14 Paper Angles Identified

1. **Effective Entropy of Compiler Optimization Spaces** — Zipf-like distribution of constraint signatures; coverage curves showing when the table is "full"
2. **Retrieval-Based Backend Compilation** — Compiler as retrieval engine over canonicalized program constraints (bridge to DB/IR/knowledge compilation)
3. **Knowledge Compilation for Register Allocation** — Expensive offline reasoning → cheap online query answering; connection to BDD/SAT compilation
4. **Are Backend Decision Spaces Finite in Practice?** — Effective enumerability of ISA-specific optimization spaces
5. **Phase Transitions in Exhaustive Backend Optimization** — Sharp tractability boundary at ~6-8 vregs; architecture-specific "phase diagrams"
6. **Irregularity as Structure** — Z80's architectural irregularity reduces ambiguity and makes search space more compressible
7. **Amortized Exact Compilation** — One-time offline GPU compute amortized across all future compilations; ROI analysis
8. **Compiler Tablebases** — Chess endgame analogy made rigorous: canonical forms, exact values, composition rules
9. **Solving vs Discovering** — Register allocation (assignment) and instruction synthesis (discovery) are structurally different search problems
10. **Verified Micro-Algorithm Libraries** — Exhaustively synthesized, 256/256-verified instruction idioms as reusable compiler assets
11. **Corpus-Guided Compiler Construction** — Corpus as first-class input shaping optimizer search space, not just evaluation benchmark
12. **Calling Convention as Optimization Variable** — Within closed-world modules, parameter placement co-optimized with register allocation
13. **Exact Backend in Closed-World Setting** — Full call graph + fixed ISA enables amortizable, compositional offline solving
14. **Empirical Revision of Backend Folklore** — "Optimal allocation too expensive" → not for L≤15 and real program distributions

### Key Open Problems

- **Canonicalization**: formal graph canonicalization of constraint signatures (Nauty-based) for maximum reuse
- **Phase diagram**: map tractability boundaries across architectures (Z80, 6502, 8080, Thumb, AVR, MSP430)
- **Learned vs exact**: compare table lookup with ML-predicted allocation as hybrid router
- **Proof-carrying entries**: attach minimality witnesses/certificates to table entries
- **Backend knowledge base**: extend beyond regalloc to scheduling, idiom selection, constant materialization
- **Marginal coverage curve**: empirically measure when adding new table entries stops improving coverage

### The Strongest Angle

Not GPU speed, not dual GPUs, not CUDA engineering — but the empirical discovery that **real programs occupy a tiny, recurring fraction of theoretical backend decision spaces**, making offline precomputation practical. GPU is merely the instrument that made this visible.

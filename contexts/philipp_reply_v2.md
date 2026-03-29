# Reply to Philipp — Draft v2

Dear Philipp,

Thank you very much for taking the time to read the paper and for the detailed, constructive feedback. Every point is valid and actionable — I'll address them briefly and then share something that I think might interest you beyond this particular paper.

## Responses to your points

**SDCC version**: Thank you for confirming no CC changes since 4.2.0 — we'll note this explicitly to preempt the question from reviewers. We'd also love to add a mos6502 comparison using SDCC 4.5.0's new backend; we have a 6502 ISA definition in our superoptimizer already.

**"Nanz compiler" citation**: You're right, this needs a proper reference. Nanz is the language; the compiler/toolchain is called MinZ (MZC). Public repository: https://github.com/oisee/minz — we'll add a proper citation in the revision.

**"17x smaller" terminology**: You're absolutely right — sloppy language on my part. We'll change to "17:1 size ratio" or "reduced to 1/17th the original size" throughout.

**Multi-return methodology**: We used register return (DE:A or HL:A for two values) — PFCCO selects the optimal register pair per call site. SDCC uses stack-based return for multi-value results. We did not use pointer arguments. We'll state this explicitly in the revision to make the comparison fair and transparent.

**Separate compilation and function pointers**: This is the key architectural difference, and I think you framed it perfectly. MinZ is a whole-program compiler — it sees the complete call graph, which is the natural model for 8-bit targets with no dynamic linking. For functions whose address is taken (function pointers), we fall back to a fixed default convention, so PFCCO applies only to the "closed world" subset. As you noted, this is exactly the subset that SDCC could also optimize for static functions without address-taken — which suggests the technique has value even within a separate-compilation model.

**Inlining vs. PFCCO**: This is a fair and important point. Our view is that they're complementary:
- Inline tiny functions (1–3 ops): eliminates call overhead entirely
- PFCCO for medium functions (4–20 ops): saves 10–20T per call without code duplication
- On Z80, CALL+RET = 27T. A 20T function body inlined at 5 call sites adds 100 bytes but saves 135T. PFCCO achieves ~60% of the savings with zero code growth.
- Additionally, inlining increases register pressure at the call site, which on a 7-register machine can force spills that cost more than the call overhead saved.

We'll add a section discussing the inlining/PFCCO tradeoff explicitly.

---

## Beyond this paper — potential collaboration?

I want to be upfront: I'm not from academia, and I don't have experience with the publication process. But over the past months, our GPU superoptimizer work has produced a volume of results that — based on your papers and what I can see in the literature — might contain publishable material. I'd genuinely value your perspective on whether any of this is interesting, and whether you might want to collaborate.

Here's what we have:

### 1. Exhaustive Arithmetic Tables (verified, production-ready)
- **254/254 optimal multiply sequences** (u8 × K → A): 1–11 instructions, all proven optimal by exhaustive GPU search
- **254/254 optimal division sequences** (u8 ÷ K → A): 6 methods, average 79 T-states — faster than SDCC's generic `__div8` on the *average* divisor
  - Includes a GPU-discovered trick for K ≥ 128: `OR A; LD B,(256-K); ADC A,B; SBC A,A; AND 1` = 5 ops, 26T. Not found in any Z80 reference we're aware of.
- **254/254 mod8, divmod8** companion tables
- **Branchless idioms**: saturating add (4 ops, 16T!), sign, ABS, MIN/MAX — all exhaustively verified

These could slot directly into SDCC as lookup tables for constant division/multiply optimization, with zero risk (every sequence is verified against all 256 inputs).

### 2. Exhaustive Register Allocation (83.6 million shapes)
- For ≤6 virtual registers: every possible interference graph shape enumerated, optimal physical assignment found or infeasibility proven
- **Phase transition**: feasibility drops from 95.9% (2 vars) to 0.9% (6 vars) — a sharp cliff
- 37.6 million feasible shapes with 15 operation-aware enrichment metrics
- O(1) lookup by (interference_shape, operation_bag) signature — already integrated in our compiler

This might connect to your CC 2013 work on optimal register allocation for irregular architectures?

### 3. GPU Superoptimizer Infrastructure
- 739K peephole rules (length-2 complete), 37M length-3 (partial)
- ISA DSL that generates CUDA/Metal/OpenCL/Vulkan kernels from a single Z80 ISA definition
- Cross-verified on 5 platforms, 3 GPU vendors
- The ISA DSL already has 6502 and SM83 (Game Boy) definitions — extending to any small-state ISA is straightforward

### 4. Multi-Convention 32-bit Arithmetic
- Three u32 register packing conventions analyzed (DEHL, HLIX, HLH'L')
- HLH'L' (using shadow registers via EXX) is 30–40% faster than the standard DEHL convention
- SHA-256 feasibility: 58ms/block at 3.5MHz using proven u32 primitives

### Draft papers

Beyond the ABI paper you've already seen, we have several more drafts at various stages — all with empirical data from GPU search. I list them because some connect directly to your published work:

**Paper A: "Precomputed Optimal Register Allocation for Constrained Architectures via Corpus-Driven Exhaustive GPU Search"** — 83.6M shapes exhaustively enumerated for ≤6 virtual registers, 315 unique signatures from 1,645 functions (80% reuse), 88.2% cross-program transfer, phase transition at ~16 physical locations. This directly extends your CC 2013 work on optimal allocation for irregular architectures — the question is whether the "solved game" argument holds empirically.

**Paper B: "Composing Provably Optimal Register Allocations Across Function Boundaries"** — joint caller+callee allocation via GPU search, eliminating CALL/RET overhead. ~35T saved per eliminated boundary on Z80. Connects to the ABI paper (per-function conventions enable cross-boundary optimization).

**Paper C: "Compositional Register Allocation via Graph Decomposition"** — decomposing large interference graphs at small vertex separators, solving each sub-problem from the precomputed table, stitching with bounded-cost shuffles. Theoretical framework for scaling beyond the enumeration wall.

**Superoptimizer results (no paper yet)**: 739K peephole rules, the three-level validation methodology (analytical→composite→GPU exhaustive — each finds what the others miss), ISA DSL generating GPU kernels for Z80/6502/SM83 from a single source.

### What I'm imagining

To be honest: I'm not from academia, I don't have experience with the publication process, and I can't judge which of this reaches the bar for CGO, CC, or LCTES. That's exactly the kind of judgment you have and I lack.

If any of this looks interesting to you — whether as a potential collaboration, or just "this table would be useful in SDCC" — I'd love to explore it further:

- The arithmetic tables (div8, mul8, branchless idioms) could be contributed to SDCC directly — MIT-licensed, JSON format, verified against all inputs, trivial to integrate as constant-optimization passes
- The 6502 ISA definition in our superoptimizer means we could generate equivalent tables for SDCC's new mos6502 backend
- Paper A connects directly to your prior work — if the phase transition result and cross-program transfer data are interesting, perhaps we could develop it together
- Even a short note showing the carry_compare GPU discovery (division trick not found in any reference, found by exhaustive search) might have pedagogical value

No pressure at all. If none of this is at the right level, that's completely fine — the feedback on the ABI paper alone was already very valuable.

Everything is open source: https://github.com/oisee/z80-optimizer

Thank you again for the thoughtful feedback on the ABI paper. Whatever comes of the broader question, the paper will be stronger for your comments.

Best regards,
Alice

---

## NOTES (not for sending)
- [x] MinZ team answers received: Nanz=language, MinZ=compiler, repo=github.com/oisee/minz
- [x] Multi-return: register return (DE:A, HL:A), SDCC uses stack. Will clarify.
- [x] Function pointers: fallback to fixed CC. Direct calls = PFCCO, indirect = standard ABI.
- [ ] Fix "17x smaller" in paper before sending
- Tone: enthusiastic but not overwhelming. Let Philipp pick what interests him.
- Key sell: tables are FREE for SDCC (MIT, JSON, verified). Zero-risk integration.

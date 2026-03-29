# Week 1 Report — Birthday Marathon (March 26–29, 2026)

## Executive Summary

4 days, ~60 hours of human+AI collaboration. From v1.0.0 release to a comprehensive Z80 arithmetic optimization ecosystem. **1500+ verified sequences**, 4 CUDA kernels, 3 image search methods, cross-verified on 5 platforms.

## Day-by-Day

### Day 1 (March 26) — Release & Foundation
- **v1.0.0 released** to GitHub with article "The Z80 Compiler That Never Guesses"
- 500+ arithmetic sequences (254 mul + 246 div + idioms)
- 83.6M regalloc shapes, 97.7% infeasibility at 7–15v
- ISA DSL gpugen: 4 backends (CUDA/Metal/OpenCL/Vulkan), 3 ISA definitions
- Cross-verified: 5 platforms, 4 APIs, 3 GPU vendors
- MinZ v0.23.0 ships with our tables

### Day 2 (March 27) — Arithmetic Expansion
- FP16 library, BCD arithmetic (GPU-proven with H-flag)
- Multi-target approximate search: 13-op pool, 15 targets simultaneously
- **21-instruction universal pool thesis**: 2.7% of ISA generates ALL optimal arithmetic
- MinZ v0.24.0 (VIR). Frill→CUDA PoC for mul8 search
- Book: Appendices L(FP) + M(BCD) + N(LUT) + O(meta-analysis)

### Day 3 (March 28) — Proofs & Architecture
- **37.6M enriched shapes** with 15 op-aware metrics (78MB compressed)
- **O(1) regalloc**: signature = (shape, operation_bag) → hash lookup
- **Z flag write-only proof** (exhaustive + induction)
- **SBC A,A trick library**: CMOV, ABS, MIN/MAX, gray_decode EXACT
- Register cost graph: 11 regs, full paths
- Smart CALL save: 17T avg (was 34T) = 50% reduction
- SDCC analysis: IX/IY = 6–11%, EXX < 1%

### Day 4 (March 28–29) — Regalloc Deep Dive
- Corpus bias discovery: VIR=79% ≤6v, FatFS=ALL >10v
- Five-level pipeline: cut-vertex → O(1) → EXX → GPU → Z3
- Backtracking solver for 7–15v (alternative to Z3)
- Partition optimizer: ≤18v exhaustive (<2 min), ≤20v (~30 min)
- Phase diagram: feasibility cliff 95.9% (2v) → 0.9% (6v)

### Day 5 (March 29) — Image Search & u32
- **3 CUDA image generators**: dual-layer (cat 4.9%), segmented LFSR (Che 15%), Introspec BB port
- **u32 arithmetic library**: 13 operations, SHL32/SHR32 proven optimal
- Z80 intro attempt (che_intro.asm, 351 bytes)
- SHA-256 feasibility: ~58ms/block @3.5MHz
- divmod8 strategy decided: analytical multiply-and-shift

### Day 6 (March 29 continued) — Division Breakthrough
- **div8 254/254 complete**: v1→v2→v3, avg 154T→135T→**79T** (−49%)
- **carry_compare trick** (GPU-discovered): 5 ops, 26T for all K≥128
- **PRESHIFT trick**: (A>>P)×M>>S, dominant for K<128
- **sat_add8**: 4 ops, 16T — branchless saturating add masterpiece
- sign8, sat_sub8, abs16, neg16, min16, max16
- SHA-256 round decomposition with realistic estimate
- **4× cross-verified**: z80-optimizer, MinZ, MinZ-VIR, MinZ-ABAP
- MinZ-VIR integrated div8 into IntrinsicTable (commit 8cfba219)

## By The Numbers

| Metric | Start (Day 1) | End (Day 6) | Delta |
|--------|---------------|-------------|-------|
| Arithmetic sequences | 501 | **1500+** | +200% |
| div8 coverage | 0/254 | **254/254** | complete |
| div8 avg T-states | N/A (runtime) | **79T** | vs 80-200T SDCC |
| mul8 coverage | 254/254 | 254/254 | maintained |
| mul16 coverage | 254/254 | 254/254 | maintained |
| Regalloc shapes | 83.6M | 83.6M | enriched +15 metrics |
| Branchless idioms | 15 | **25+** | +67% |
| CUDA kernels | 2 | **6** | +200% |
| Image search methods | 0 | **3** | new capability |
| u32 operations | 0 | **13** | new capability |
| Cross-verified systems | 3 | **5+** | 4 independent verifiers |
| Data files shipped | 3 | **12** | +300% |
| Peephole rules | 739K | 739K + 37M partial | ongoing |

## Key Innovations

1. **carry_compare** — GPU-discovered division trick, not in any reference
2. **PRESHIFT division** — exploit K's factorization for shorter sequences
3. **3-level validation** — analytical → composite → GPU exhaustive
4. **sat_add8 = OR C** — 4-instruction branchless saturating add
5. **Segmented hierarchical LFSR** — guaranteed-convergence image search
6. **Subtractive carving** — AND NOT layers for face features in image search
7. **ADC HL,rr** — Z80's hidden gem for 32-bit arithmetic

## Cross-Team Impact

| Team | Integration |
|------|------------|
| MinZ (compiler) | mul8/16, div8 v3, sign/sat, u32 ops, SHA-256 primitives |
| MinZ-VIR (backend) | div8 IntrinsicTable, tryConstDiv/Mod, enriched table reader |
| MinZ-ABAP (frontend) | MIR2 crosscheck, Z3 constant folding |
| antique-toy (book) | Appendix K data, branchless idioms, 3-level methodology |

## Files Shipped This Week

### New Data (12 files)
- `data/div8_optimal.json` — 254/254 div8 sequences (v3, avg 79T)
- `data/mod8_optimal.json` — 254/254 mod8 sequences
- `data/divmod8_optimal.json` — 254/254 divmod8 sequences
- `data/sign_sat_ops.json` — sign8 (43T), sat_add8 (16T), sat_sub8 (20T)
- `data/arith16_new.json` — abs16, neg16, min16, max16, sign16
- `data/u32_ops.json` — 13 u32 operations (DEHL convention)
- `data/sha256_round.json` — SHA-256 decomposition (58ms/block)
- `data/mulopt8_clobber.json` — 254 mul8 with clobber masks
- `data/mulopt16_complete.json` — 254 mul16 sequences
- `data/z80_register_graph.json` — 11-register cost model
- `data/enriched_*.enr.zst` — 37.6M enriched regalloc shapes (78MB)
- `data/bcd_idioms.json` — BCD arithmetic

### New CUDA Kernels (4)
- `cuda/prng_hybrid_gpu.cu` — dual-layer evolutionary image generator
- `cuda/prng_segmented_search.cu` — hierarchical segmented LFSR
- `cuda/bb_search.cu` — Introspec BB algorithm port
- `cuda/prng_layered_search.cu` — layered LFSR search

### Documentation
- `TODO.md` — comprehensive roadmap (277 lines, priority matrix)
- `contexts/day[3-6]_wisdom.md` — daily knowledge dumps
- `contexts/week1_report.md` — this file
- `media/prng_images/README.md` — 29-experiment gallery

# TODO — Z80 Superoptimizer Roadmap

> Last updated: 2026-03-29 (Day 6 birthday marathon)

Legend: `[x]` done, `[-]` in progress, `[ ]` planned.
Effort: S = hours, M = day, L = days, XL = week+.

---

## 1. Arithmetic Library (for MinZ compiler)

### 1.1 Multiply — COMPLETE
- [x] **mul8**: 254/254 constants, A×K→A — `data/mulopt8_clobber.json` (S)
- [x] **mul16**: 254/254 constants, A×K→HL — `data/mulopt16_complete.json` (M)
- [ ] **mul16c**: HL×K→HL (16-bit × constant, full 16-bit) — needs new CUDA kernel (M)
  - Approach: decompose as HL×K = L×K + H×K×256, use mul16 building blocks
  - Or: new CUDA search with HL input, reduced op pool

### 1.2 Division / Modulo — COMPLETE (u8)
- [x] **div8**: 254/254 constants, A÷K→A — `data/div8_optimal.json` v3 (M)
  - **6 methods**: shift(5), mul_shift(30), preshift_mul(36), mul_add256_shift(41), double_mul_shift(15), carry_compare(127)
  - T-states: 8–188 (avg **79T**). v1→v2→v3: 154→135→79T (−49%). All exhaustively verified.
  - **carry_compare** (GPU-discovered): `OR A; LD B,(256-K); ADC A,B; SBC A,A; AND 1` = 26T for K≥128
  - **PRESHIFT**: `(A>>P)×M>>S` — exploit K's power-of-2 factor
  - Cross-verified: z80-optimizer + MinZ + MinZ-VIR + MinZ-ABAP (4 systems)
- [x] **mod8**: 254/254 — `data/mod8_optimal.json` (S)
- [x] **divmod8**: 254/254 — `data/divmod8_optimal.json` (S)
- [ ] **div16n**: HL÷K→A (narrowing division, quotient ≤ 255) (M)
  - Approach: analytical multiply-and-shift on 16-bit, or lookup table
- [ ] **mod16n**: HL%K→A (narrowing modulo) (M)

### 1.3 Branchless Primitives — COMPLETE (u8)
- [x] **ABS(A)**: 6 ops, 24T — `data/branchless_lib.json` (done day 3)
- [x] **MIN/MAX(A,B)**: 8 ops, 32T — SBC A,A + bitwise select
- [x] **CMOV CY?B:C**: 6 ops, 24T
- [x] **sign8**: 9 ops, 43T — `data/sign_sat_ops.json`
- [x] **sat_add8**: 4 ops, 16T — the `OR C` masterpiece
- [x] **sat_sub8**: 5 ops, 20T
- [ ] **sat_add8_signed**: 7 ops (branch), needs branchless variant? (S)
- [ ] **clamp8(A, lo, hi)**: A clamped to [lo,hi] range (S)
  - Compose: max(lo, min(hi, A))? Or dedicated sequence.
- [ ] **bit_reverse8**: reverse bits of A (S)
  - GPU search or manual via nibble-swap + lookup
- [ ] **popcount8**: count set bits of A → A (S)
  - Classic: A = A - ((A>>1) & 0x55); etc. Adapt to Z80.
- [ ] **approx sin8 / cos8**: polynomial, allow ±2 error (M)

### 1.4 16-bit Arithmetic — PARTIAL
- [x] **abs16**: 11 ops, 44T — `data/arith16_new.json`
- [x] **neg16**: 6 ops, 27T
- [x] **min16/max16**: 5 ops, 41–46T (branch)
- [x] **sign16**: 7 ops, 20–34T
- [x] **cmp16_zero**: 2 ops, 8T
- [ ] **sat_add16 / sat_sub16**: unsigned, 16-bit (S)
- [ ] **mul16×16→32**: HL×DE→DEHL, needed for mul16c (L)
  - Schoolbook: 4 partial products, 3 additions. ~200T estimate.

### 1.5 32-bit Arithmetic — COMPLETE
- [x] **u32 library**: 13 operations — `data/u32_ops.json`
  - SHL32(34T), SHR32(32T), SAR32(32T), ADD32(54T), SUB32(58T),
    NEG32(57T), CMP32_ZERO(16T), ZEXT/SEXT, XOR32(100T), AND32(100T), ROTR32(32–40T)
- [x] **SHA-256 round decomposition** — `data/sha256_round.json`
  - ~2570T/round, 64 rounds + message schedule = ~202K T = **58ms/block @3.5MHz**
- [ ] **OR32, NOT32**: trivial but not yet in table (S)
- [ ] **CMP32_unsigned**: compare DEHL vs IX:IY (S)
- [ ] **SHA-256 Z80 implementation**: full working .asm (XL)
  - Have decomposition; need actual register allocation + memory layout
  - 8 working variables × 4 bytes = 32 bytes in RAM
  - Message schedule: 64 words × 4 bytes = 256 bytes

---

## 2. Peephole Superoptimizer

### 2.1 Rules — PARTIAL
- [x] **len-2→len-1**: 739K rules, complete — `data/rules_l2.json` (done)
- [-] **len-3→len-1**: 37M rules found (~0.05% of 74.9B search space) (XL)
  - GPU search running intermittently. Full sweep estimated weeks on 2× RTX 4060 Ti.
  - Strategy: prioritize high-value target op ranges.
- [ ] **len-3→len-2**: not yet started. Produces size-reducing rewrites. (XL)
- [ ] **len-4→len-2**: STOKE only. Stochastic, not exhaustive. (L)
- [ ] **Dead-flags rules**: rules valid when specific flags are dead post-sequence (M)
  - Infrastructure exists (`pkg/search/verifier.go`). Need dedicated search run.

### 2.2 Reordering Optimizer
- [ ] **Dependency DAG**: build RAW/WAW/WAR graph from opReads/opWrites (M)
  - Primitives exist in `pkg/search/pruner.go`.
- [ ] **Pattern matcher with reorder**: scan basic block, match rules across independent instructions (M)
- [ ] **Multi-pass fixpoint**: repeat until no more rules apply (S)
- [ ] **Integration with MinZ**: basic-block input/output format, JSON API (M)

### 2.3 Multi-Target Search
- [ ] **Approximate search**: 13-op pool, 15 targets simultaneously (M)
  - Day 2 prototype worked. Needs cleanup for production use.
- [ ] **Focused search cores**: AF-only, HL+M-only for deeper search (L)

---

## 3. Register Allocation

### 3.1 Tables — COMPLETE
- [x] **83.6M shapes** (≤6v): enumerated, enriched, compressed (78MB)
- [x] **37.6M feasible**: each with optimal assignment + 15 metrics
- [x] **O(1) lookup**: signature = (interference_shape, operation_bag) → hash
- [x] **Enriched tables**: 43% lack A, 21% lack HL, smart CALL save 17T avg

### 3.2 Five-Level Pipeline — MOSTLY COMPLETE
- [x] Level 1: Cut vertex decomposition (free split, 87%)
- [x] Level 2: Enriched table O(1) lookup (37.6M entries, 79%)
- [x] Level 3: EXX 2-coloring (7–12v, 70% bipartite)
- [x] Level 4: GPU partition optimizer (≤18v exhaustive, <2 min)
- [x] Level 5: Z3 fallback (>18v, <0.5%)
- [-] **Backtracking solver** (`cmd/regalloc-enum/`): alternative to Z3 for 7–15v (M)
  - Basic implementation done. Needs benchmarking vs Z3.

### 3.3 Next Steps
- [ ] **7-variable table**: ~500M shapes estimated, 4–12h compute (L)
  - Feasibility: ~5% at 7v (phase cliff). Worth it for O(1) coverage.
- [ ] **MinZ integration testing**: run pipeline on full 820-fn corpus (M)
  - 246 unique signatures tested. Need end-to-end with codegen.
- [ ] **Spill cost model**: when allocation fails, estimate spill cost (M)
  - Currently: binary feasible/infeasible. Need: quantify spill penalty.

---

## 4. Image Search / Demoscene (ZX Spectrum)

### 4.1 CUDA Generators — DONE (3 methods)
- [x] **Dual-layer evolutionary** — `cuda/prng_hybrid_gpu.cu`
  - 5 layers (3 additive OR + 2 subtractive AND NOT), island model
  - Best: cat 4.9%, skull 14.7%, Einstein 15.1%
  - 557K img/s dual-layer on RTX 4060 Ti
- [x] **Segmented hierarchical LFSR** — `cuda/prng_segmented_search.cu`
  - 6 levels, brute-force 65536 seeds per segment
  - Best: Che 15.0% (1194 bytes), 31.2% (170 bytes fits 256b intro)
- [x] **BB Introspec port** — `cuda/bb_search.cu`
  - 24-bit Galois LFSR, 66 layers, 3 weighted masks
  - 0.9s per s0 value, full sweep in ~4 minutes

### 4.2 Z80 Intros
- [-] **che_intro.asm** — 351 bytes, builds with mza, LFSR mismatch produces noise (M)
  - Root cause: CUDA search uses per-layer init, Z80 code uses fixed init
  - Fix: re-run CUDA search matching exact Z80 LFSR parameters
  - Or: rewrite Z80 code to match CUDA LFSR conventions
- [ ] **256-byte intro**: Che or skull, minimal LFSR + segmented seeds (M)
  - 85 seeds × 2 bytes = 170 bytes data + ~80 bytes code = fits!
- [ ] **512-byte intro**: Che with 6 levels, LFSR playback (M)
- [ ] **1K intro**: dual-layer with evolved genome playback (L)

### 4.3 Improvements
- [ ] **Multi-pass BB refinement**: re-optimize early layers after first pass (M)
  - Introspec's secret sauce. Each re-pass reduces error monotonically.
- [ ] **Better segmented search**: levels 6–7, 32-bit seeds, multiple polynomials (M)
  - Goal: <10% error from 256 bytes
- [ ] **Face-region masks**: weighted fitness for eyes/mouth like Introspec (S)
- [ ] **New targets**: Marilyn Monroe, Darth Vader, Einstein tongue (S)
- [ ] **BB + segmented hybrid**: BB for base, segmented for detail (L)

---

## 5. GPU Infrastructure

### 5.1 CUDA Kernels — WORKING
- [x] **z80_search_v2.cu**: 3-stage pipeline (QC→Mid→Exhaustive), dual-GPU
- [x] **z80_regalloc.cu**: GPU allocator + CPU backtracking fallback
- [x] **z80_mulopt_fast.cu**: 14-op constant multiply (38× faster)
- [x] **z80_divmod_fast.cu**: 14-op division/modulo search
- [x] **z80_mulopt16.cu**: 16-bit multiply search
- [x] **z80_common.h**: shared executor, flag tables, test vectors

### 5.2 Multi-Platform DSL — WORKING
- [x] **pkg/gpugen/**: ISA DSL → CUDA, Metal, OpenCL, Vulkan
- [x] **3 ISA definitions**: Z80 (394 ops), 6502, SM83 (Game Boy)
- [x] **Cross-verified**: 5 platforms, 4 APIs, 3 vendors, zero discrepancies

### 5.3 Next Steps
- [ ] **Unified search orchestrator**: coordinate multi-GPU multi-kernel searches (L)
  - Currently: manual per-kernel scripts. Need: single coordinator.
- [ ] **Remote GPU dispatch**: ssh+scp to i5/i3, launch, collect results (M)
  - Scripts exist (`cuda/run_mulopt_remote.sh`). Need: proper pipeline.
- [ ] **Vulkan compute on AMD**: RX 580 via gpugen Vulkan backend (M)
  - ROCm broken for gfx803. Vulkan works. Need: Vulkan search driver.

---

## 6. Compiler Integration (MinZ)

### 6.1 Go Packages — SHIPPED
- [x] `pkg/mulopt/`: Emit8(k), Emit16(k), Lookup8/16
- [x] `pkg/regalloc/`: LoadBinary(path), IndexOf(shape), Lookup(idx)
- [x] `pkg/peephole/`: Lookup(source) top500, LoadRules(path) full 739K
- [x] `pkg/gpugen/`: ISA DSL for multi-platform code generation

### 6.2 Pending Integration
- [ ] **div8 inline expansion**: MinZ codegen wiring for JP __div8 → inline (S)
  - Pipeline issue: peephole transforms JP before div8 check. Fix: expansion before optimization.
- [ ] **Clobber-aware selection**: pick mul/div variant by register pressure (M)
  - Current: always use cheapest. Better: pick by what's live.
- [ ] **Pareto front export**: (tstates, bytes, clobber_mask) triples per operation (S)
  - MinZ can then pick optimal variant per call site.
- [ ] **Batch evaluation API**: JSON-in, JSON-out for testing entire corpus (M)

---

## 7. Research & Documentation

### 7.1 Papers
- [-] **Register allocation paper** — `docs/paper_seed_superopt.md` (L)
  - 8 sections drafted. Needs: image search chapter, Introspec analysis.
- [-] **Book outline** — `docs/book_outline.md`, 19 chapters (XL)
  - Appendices K(minz), L(FP), M(BCD), N(LUT), O(meta) in progress.
- [ ] **Phase diagram publication**: feasibility cliff 95.9%→0.9% (S)
  - Data complete. Need: LaTeX figure + 2-page writeup.
- [ ] **21-instruction universal pool paper**: 2.7% of ISA generates ALL arithmetic (S)
- [ ] **Dual-layer image search writeup**: subtractive carving insight (M)

### 7.2 Documentation
- [x] **CLAUDE.md**: up to date with all packages and kernels
- [x] **README.md**: v1.0.0 release notes, architecture, results
- [x] **Gallery**: `media/prng_images/README.md` (29 experiments, 3 methods)
- [ ] **Update docs/NEXT.md**: replace WebGPU section with CUDA v2 reality (S)
- [ ] **API documentation**: godoc for pkg/mulopt, pkg/regalloc, pkg/peephole (M)

---

## 8. Cleanup & Tech Debt

- [ ] **Update README.md results section**: div8 254/254, sign/sat ops, u32, SHA-256 (S)
- [ ] **Archive dead code**: WebGPU shader (abandoned), CUDA v1 search (superseded) (S)
- [ ] **Consolidate data files**: merge branchless_lib + sign_sat_ops + arith16_new into single arith8.json / arith16.json (S)
- [ ] **Test coverage**: pkg/mulopt, pkg/regalloc need integration tests (M)
- [ ] **CI pipeline**: GitHub Actions for Go build + test (S)
- [ ] **Release v1.1.0**: div8, mod8, divmod8, sign/sat, u32 ops, SHA-256, arith16 (S)

---

## Priority Matrix

| Priority | Item | Effort | Blocker for |
|----------|------|--------|-------------|
| **P0** | ~~div8/mod8/divmod8~~ | ~~M~~ | ~~MinZ codegen~~ DONE |
| **P0** | ~~sign8/sat_add8/sat_sub8~~ | ~~S~~ | ~~MinZ stdlib~~ DONE |
| **P0** | ~~abs16/neg16/min16/max16~~ | ~~S~~ | ~~MinZ u16 ops~~ DONE |
| **P0** | ~~SHA-256 decomposition~~ | ~~S~~ | ~~MinZ feasibility~~ DONE |
| **P0** | ~~div8 v3 + carry_compare~~ | ~~M~~ | ~~MinZ-VIR integrated~~ DONE |
| **P1** | div8 inline expansion fix | S | MinZ codegen pipeline |
| **P1** | Z80 intro LFSR alignment | M | Working demo |
| **P1** | mul16c (HL×K→HL) | M | MinZ u16 multiply |
| **P2** | Multi-pass BB refinement | M | Better image quality |
| **P2** | div16n/mod16n | M | MinZ u16 division |
| **P2** | len-3 peephole search | XL | Compiler quality |
| **P3** | Reordering optimizer | L | Real-world rule application |
| **P3** | 7-variable regalloc table | L | O(1) coverage expansion |
| **P3** | 256-byte ZX Spectrum intro | M | Demoscene deliverable |
| **P4** | SHA-256 full Z80 .asm | XL | Proof-of-concept |
| **P4** | approx sin/cos, popcount | M | MinZ math library |
| **P4** | Segmented search <10% | M | Research paper |
| **P5** | Paper / book completion | XL | Publication |
| **P5** | Vulkan compute on AMD | M | Multi-vendor story |

---

## Completed Milestones

| Date | Milestone |
|------|-----------|
| 2026-03-26 | v1.0.0 release, 500+ arithmetic sequences, article published |
| 2026-03-26 | 83.6M regalloc shapes, 97.7% infeasibility proof |
| 2026-03-26 | ISA DSL gpugen: 4 backends, 3 ISAs, cross-verified |
| 2026-03-27 | FP16 library, BCD arithmetic, multi-target search |
| 2026-03-27 | 21-instruction universal pool thesis |
| 2026-03-28 | 37.6M enriched shapes, O(1) regalloc lookup |
| 2026-03-28 | Branchless library: ABS/MIN/MAX/CMOV, gray_decode EXACT |
| 2026-03-28 | Z flag write-only proof, SBC A,A trick library |
| 2026-03-29 | Image search: 3 methods, cat 4.9%, Che 15%, Introspec port |
| 2026-03-29 | u32 library: 13 ops, SHL32/SHR32 proven optimal |
| 2026-03-29 | Segmented hierarchical LFSR (user's breakthrough idea) |
| 2026-03-29 | div8 254/254, mod8 254/254, divmod8 254/254 |
| 2026-03-29 | sign8, sat_add8 (16T!), sat_sub8 — all exhaustively verified |
| 2026-03-29 | abs16, neg16, min16, max16, sign16, cmp16_zero |
| 2026-03-29 | SHA-256 round decomposition: 58ms/block realistic estimate |
| 2026-03-29 | div8 v3: carry_compare trick, avg 79T (−49%), GPU-discovered |
| 2026-03-29 | 4× cross-verified: z80-optimizer + MinZ + MinZ-VIR + MinZ-ABAP |
| 2026-03-29 | MinZ-VIR integrated div8 IntrinsicTable (commit 8cfba219) |

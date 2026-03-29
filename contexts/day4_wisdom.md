# Day 4 Wisdom — March 29, 2026

## Key Discoveries

### Corpus Bias (CRITICAL)
- VIR corpus (820 funcs): 79% ≤6v, median 3v — **biased toward leaf functions**
- FatFS (ff.c, 7249 lines SDCC): 16 functions, ALL >10v, est 10-35 vregs
  - f_write: 764 instr, 30 PUSH, ~33 vregs
  - f_mkdir: 259 instr, 39 PUSH, ~35 vregs
  - LD=55% (vs VIR 34%), STACK=10% (vs VIR 7%)
- Our enriched tables (≤6v) cover **0% of FatFS**!
- Greedy partition covers v10-v32 in <30ms — solves this
- Need: FatFS + z88dk stdlib + Fuzix in corpus for unbiased stats

### IX/IY Halves = Production Safe
- IXH, IXL, IYH, IYL work on ALL known Z80 silicon
- 11 registers for allocation (not 7!)
- H↔IXH impossible direct — EX DE,HL trick (16T)
- ALU works: ADD A,IXL = 8T (DD prefix)
- IX/IY not swapped by EXX = bridge between banks
- Updated in CLAUDE.md as proven fact

### Image Search: Honest Assessment
- Single pRNG seed (10 bytes) → 2^80 possibilities → textures, NOT objects
- VGG perceptual loss: 25.5→25.3 (minimal improvement over random)
- MobileNet classification: chain_mail=7% (textures), cat=1.7% (near random)
- **Fundamental limit**: 10-byte state cannot encode 1536-byte image

### Hybrid Generator (New Approach)
- 57 searchable bytes = 2^456 space (vs 2^80 for seed-only)
- Layers: pRNG noise → tile masks → circle/gradient → threshold → symmetry
- Still fits in 256-byte Z80 intro!
- TODO: CUDA kernel (Python per-pixel too slow), or ISA DSL version

### Paper v2.2 Published
- Multi-platform GPU section (ISA DSL → 4 backends)
- Clear data breakdown table (83.6M → 37.6M → 78MB)
- A4 PDF + A5 PDF + EPUB (all with rendered Mermaid diagrams)
- v1.3.0 release on GitHub

## Files Created/Modified

### New
- `cuda/prng_dither_search.py` — VGG perceptual loss + Floyd-Steinberg dither
- `cuda/prng_hybrid_search.py` — 57-byte genome: basis + mask + symmetry + pRNG
- `media/prng_images/targets/` — 6 target images (skull, heart, star, smiley, tree, fish)
- `media/prng_images/dithered/` — 20 VGG-searched seeds + comparisons
- `media/prng_images/*/` — 7 targets × top-10 = 160 images total
- `media/prng_images/README.md` — full gallery with leaderboard
- `data/partition_greedy_v10_v32.json` — greedy partition results
- `docs/regalloc_paper_a5.pdf` — A5 format
- `contexts/day4_wisdom.md` — this file

### Modified
- `CLAUDE.md` — IX/IY halves, enriched tables, branchless library, 5-level pipeline
- `docs/regalloc_paper.md` — v2.2: data breakdown, multi-platform, corpus validation
- `docs/regalloc_paper.epub` — now with rendered Mermaid diagrams (700KB)
- `media/prng_images/README.md` — comprehensive gallery

### Commits: 8003e9f → 9f6a16b (day 4)

## Overnight Results Collected
- Partition v10-18: all DONE
- Partition v19: 272T [8+6+5] DONE
- Partition v20: 368T [9+3+3+5] DONE (both i7 GPU1 and i5)
- Corpus 172 functions: ALL partitioned, avg 90T
- Cat search (MobileNet): chain_mail=4%, bluetick=3.5%
- Multi-target: 7 classes, 160 images
- Dithered VGG: loss 25.5→25.3

## What We Told Colleagues
- MinZ (ju6yy047): corpus bias alert, FatFS analysis, IX/IY safe
- VIR (cok1cgsq): partition results 172/172, v1.3.0 released
- Book (fjimbuwe): overnight results summary, gallery, sidebars delivered
- WASM (gyfiwji1): OpStats format, sequence verifier format

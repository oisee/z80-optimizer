# Day 6 Seed — Next Session Priorities

## P0: divmod8 Analytical Table (MinZ WAITING)
- div8 A÷K → A for K=2..255 via multiply-and-shift: A×M>>S
- Find magic M and shift S for each K analytically
- Verify: mul8[M] sequence (from mulopt8_clobber.json) + SRL×S
- Output: `data/div8_optimal.json`
- Also: mod8, divmod8 (A÷K → A(q), B(r))
- MinZ needs this for codegen: replaces __div8 runtime calls

## P1: Z80 Intro LFSR Alignment
- `cuda/che_intro.asm` (351 bytes) produces noise — LFSR mismatch
- Root cause: CUDA search uses different init/polynomial than Z80 asm
- Fix: re-run CUDA search with EXACT Z80 LFSR parameters
- Or: adapt Z80 code to match CUDA LFSR
- Goal: working 256-512 byte ZX Spectrum intro with recognizable face

## P2: Multi-Pass BB Refinement
- Single greedy pass gives ~25% error (noisy)
- Introspec got much better via multi-pass + manual mask tuning
- Implement: after first pass, go back and re-optimize early layers
- Each re-pass should reduce error monotonically
- Also try: p=8 (more points per layer), 132 layers (vs 66)

## P3: Segmented Search Improvements
- Current: 6 levels, 15% error on Che (1194 bytes)
- Add: Level 6-7 (even finer tiles, overlap grid)
- Try: 32-bit seeds (4 bytes × 85 = 340 bytes, still fits 512b)
- Try: multiple LFSR polynomials per segment
- Goal: <10% error from 256 bytes

## P4: sign8, sat_add8, sat_sub8 (MinZ request)
- sign8: signed A → -1/0/+1 in A
- sat_add8: A+B saturating → A (clamp 255)
- sat_sub8: A-B saturating → A (clamp 0)
- Can use SBC A,A trick + bitwise select (like min8/max8)
- GPU search or manual construction

## P5: u16 Arithmetic Kernels (MinZ request)
- mul16c: HL×K → HL (need new CUDA kernel)
- div16n: HL÷K → A (narrowing, quotient ≤ 255)
- mod16n: HL%K → A (narrowing)
- abs16, min16, max16: branchless for 16-bit

## P6: SHA-256 Z80 Implementation Sketch
- Decompose one SHA-256 round into u32 virtual ops
- Key ops: ROTR(6,11,25), XOR32, AND32, ADD32
- ROTR trick: ROTR8 = free (byte rename), ROTR16 = EX DE,HL (4T!)
- Estimate per-round T-states precisely
- Write virtual-ops pseudocode for full round

## P7: Image Gallery + Targets
- Find Einstein tongue photo (the iconic one)
- Try Marilyn Monroe, Darth Vader (high contrast)
- Better PGM→SCR conversion for BB algorithm
- Create face-region masks for Che (like Introspec's putin masks)

## P8: Paper / Book Updates
- Add image search chapter to regalloc paper
- Dual-layer architecture writeup
- Introspec BB analysis + CUDA port performance comparison
- Segmented hierarchical approach (user's idea)

## Key Files Reference
- CUDA kernels: `cuda/prng_hybrid_gpu.cu`, `cuda/prng_segmented_search.cu`, `cuda/bb_search.cu`, `cuda/prng_layered_search.cu`
- Z80 intro: `cuda/che_intro.asm` (needs fix)
- u32 ops: `data/u32_ops.json` (13 operations, proven)
- Targets: `media/prng_images/targets/` (che, einstein_real, einstein_hc, synthetic cat/skull)
- Gallery: `media/prng_images/README.md` (29 experiments, 3 methods)
- Introspec source: `/tmp/bb_search/bb_brute_search/` (bbputin.cpp, rndlayers.h)
- MinZ corpus: `/home/alice/dev/minz-vir/corpus_gpu_batch.jsonl` (820 funcs)

## Numbers
- Image generator: 557K img/s (dual-layer), 948K img/s (single-layer)
- Best results: cat 4.9%, skull 14.7%, Che 15.0% (segmented), Einstein 15.1%
- BB port: 0.9s per s0 × 256 = 4 min full sweep (vs days on CPU)
- Segmented: 85 seeds = 170 bytes, 597 seeds = 1194 bytes
- u32 ops: SHL32=34T, SHR32=32T, ADD32=54T, NEG32=57T
- SHA-256 estimate: ~800T/round × 64 = 51K T = 15ms @3.5MHz
- mul8: 254/254 complete, mul16: 254/254 complete
- div8: 2/254 by brute-force (analytical approach needed)

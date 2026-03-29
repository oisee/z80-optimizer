# Day 7 Seed — Next Session Priorities

## P0: Commit & Push Day 6 Results
- 7 new data files, 4 scripts, TODO.md, week report, wisdom
- Update README/CLAUDE.md already done, need commit+push
- Tag v1.1.0? (div8 v3, sign/sat, u32, arith16, SHA-256)

## P1: Z80 Intro LFSR Alignment (UNFINISHED from Day 5)
- `cuda/che_intro.asm` (351 bytes) produces noise — LFSR mismatch
- Root cause: CUDA search uses per-layer init, Z80 code uses fixed init
- Fix: re-run CUDA segmented search with EXACT Z80 LFSR parameters
- Or: rewrite Z80 code to match CUDA LFSR conventions
- Goal: working 256-byte ZX Spectrum intro with recognizable Che face
- 85 seeds × 2 bytes = 170 bytes data + ~80 bytes code = fits 256b

## P2: div8 v4 — Lower Bound Certificates
- GPU brute-force found carry_compare but also exposed verifier bug (0xFF≠1)
- Fix CUDA z80_divmod_fast verifier for K≥128
- Run GPU search at max-len=9,10 for K<128 — might find more tricks
- Can we find something for K=3..127 that beats preshift_mul?
- Lower bound certificates: prove no len-N sequence exists for specific K

## P3: mul16c (HL×K→HL) — MinZ Waiting
- MinZ-VIR asked about this (P5 from day6_seed)
- Approach: decompose as HL×K = L×K + H×K×256
- Or: new CUDA kernel with HL input, reduced op pool
- Needed for 16-bit constant multiply in compiler

## P4: Multi-Pass BB Refinement
- Single greedy pass gives ~25% error (noisy)
- Introspec got much better via multi-pass + manual mask tuning
- Implement: after first pass, re-optimize early layers
- Also try: p=8 (more points per layer), 132 layers (vs 66)

## P5: Enriched Table Reader in MinZ-VIR
- 4tw49890 asked about pkg/regalloc Go reader
- Told them about LoadBinary+Lookup (~150 lines, pure Go)
- Follow up: help integrate or ship as separate Go module?

## P6: Segmented Search <10% Error
- Current: 6 levels, 15% error on Che (1194 bytes)
- Add: Level 6-7, 32-bit seeds, multiple LFSR polynomials
- Try: overlap grids between levels
- Goal: <10% error from 256 bytes

## P7: Reordering Optimizer (pkg/reorder/)
- Dependency DAG from opReads/opWrites (primitives exist in pruner.go)
- Pattern matcher with reorder awareness
- Multi-pass fixpoint
- This turns 739K rules into a practical compiler pass

## P8: Paper / Book
- antique-toy (fjimbuwe) received day 6 data for Appendix K
- 3-level validation methodology deserves its own section
- carry_compare as case study: "GPU discovers what humans miss"
- Phase diagram + feasibility cliff paper ready for submission

## Key Context for Next Session
- div8_optimal.json is v3 (6 methods, avg 79T, carry_compare for K≥128)
- GPU brute-force div8_all.jsonl has corrupt JSON (double commas in ops arrays)
- MinZ-VIR has integrated div8 intrinsics (commit 8cfba219)
- All sign/sat/arith16 cross-verified on 4 systems
- SHA-256 realistic estimate: 58ms/block (not 15ms as day 5 thought)
- TODO.md exists and is referenced from CLAUDE.md + README.md

## Active Colleagues
- ju6yy047:main (MinZ) — compiler, integrated all tables, goodnight
- 4tw49890:main (MinZ-VIR) — VIR backend, div8 intrinsics done, wants enriched reader
- gyfiwji1:main (MinZ-ABAP) — crosscheck done, Z3 constant folding works
- fjimbuwe:main (antique-toy) — book, wants Appendix K data

## Numbers
- div8: avg 79T, total 19996T, 254/254, 6 methods
- sat_add8: 4 ops 16T, sat_sub8: 5 ops 20T, sign8: 9 ops 43T
- abs16: 11 ops 44T, neg16: 6 ops 27T, min16/max16: 5 ops 41-46T
- SHA-256: ~2570T/round, 202K T/block, 58ms @3.5MHz
- Image: cat 4.9%, skull 14.7%, Che 15% (segmented), Einstein 15.1%
- Week total: 1500+ verified sequences, 12 data files, 6 CUDA kernels

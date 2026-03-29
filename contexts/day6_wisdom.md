# Day 6 Wisdom — March 29, 2026 (continued)

## Key Discoveries

### carry_compare Division Trick (GPU-DISCOVERED)
- For K≥128: `OR A; LD B,(256-K); ADC A,B; SBC A,A; AND 1` = **5 ops, 26T**
- Why it works: when K≥128, quotient ∈ {0,1}. ADC overflows iff A≥K.
- **Not found in any Z80 reference** — discovered by GPU brute-force exhaustive search
- Cross-verified on 4 independent systems: z80-optimizer, MinZ, MinZ-VIR, MinZ-ABAP
- OR A is critical: clears stale CY from previous instructions

### Three Levels of Validation (METHODOLOGY)
1. **Analytical** (Hacker's Delight formulas) → v1 baseline, avg 154T
2. **Composite search** (preshift + double_mul) → v2, avg 135T (−12%)
3. **GPU exhaustive** (no assumptions) → v3, avg 79T (−49%)
- Each level found optimizations the others missed
- GPU found carry_compare — a trick no human thought to look for
- Composite search found preshift — exploiting K's factorization structure
- Analytical provided the mathematical foundation

### PRESHIFT Division Trick
- `(A >> P) × M >> S` — shift right first, then multiply by smaller magic
- Works when K has a power-of-2 factor: divK = div(2^P) + div(K/2^P)
- div86 = `SRL A; LD H,0; LD L,A; mul16[6]; LD A,H` = 60T (was 170T)
- 80 entries improved, dominant method for K < 128

### DOUBLE_MUL Division Trick
- `A × M1 × M2 >> S` — two sequential mul16 when M factorizes nicely
- div43 = `A×3×255>>15` = 127T (was 162T)
- 26 entries improved

### div8 v3 Complete Results
- 254/254 divisors, 6 methods, avg **79T** (was 154T)
- Total: 19,996 T-states (was 39,158T) = **−49%**
- Methods: shift(5), mul_shift(30), preshift_mul(36), mul_add256_shift(41), double_mul_shift(15), carry_compare(127)
- Faster than SDCC generic __div8 (80-200T) on average

### Branchless Primitives
- sat_add8: `ADD A,B; LD C,A; SBC A,A; OR C` = **4 ops, 16T** — "masterpiece" per MinZ
- sat_sub8: `SUB B; LD C,A; SBC A,A; CPL; AND C` = 5 ops, 20T
- sign8: 9 ops, 43T (RLCA→neg_mask, NEG→nz_mask, AND 1, OR neg_mask)

### 16-bit Arithmetic
- abs16: 11 ops, 44T — branchless via (HL^mask)-mask
- neg16: 6 ops, 27T — LD A,0 preserves carry
- min16/max16: 5 ops, 41-46T — SBC+ADD+conditional EX DE,HL
- sign16: 7 ops, 20-34T — early exit for zero
- cmp16_zero: 2 ops, 8T

### SHA-256 Realistic Estimate
- ~2570T/round (not 800T as previously estimated)
- Full block: ~202K T = 58ms @3.5MHz = 17 blocks/sec
- ROTR8 = free (byte rename), ROTR16 = EX DE,HL (4T)
- XOR32/AND32 = 100T each — the real bottleneck

## Files Created/Modified
- `data/div8_optimal.json` — v3: 254/254, 6 methods, avg 79T
- `data/mod8_optimal.json` — 254/254
- `data/divmod8_optimal.json` — 254/254
- `data/sign_sat_ops.json` — sign8, sat_add8, sat_sub8
- `data/arith16_new.json` — abs16, neg16, min16, max16, sign16, cmp16_zero
- `data/sha256_round.json` — SHA-256 round decomposition
- `TODO.md` — comprehensive roadmap (277 lines, 8 sections, priority matrix)
- `scripts/gen_div8_table.py` — analytical div8 generator (v1)
- `scripts/gen_mod8_table.py` — mod8/divmod8 generator
- `scripts/composite_div_search.py` — composite search (found preshift + double_mul)
- `scripts/update_div8_with_composite.py` — v2/v3 updater
- Updated: `CLAUDE.md`, `README.md` (TODO ref, div8 results, new data files)

## Cross-Session Collaboration
- MinZ (ju6yy047): integrated all 7 JSON files, confirmed carry_compare 32768/32768
- MinZ-VIR (4tw49890): integrated div8 into VIR IntrinsicTable (commit 8cfba219), tryConstDiv+tryConstMod
- MinZ-ABAP (gyfiwji1): cross-verified sat_add8/sat_sub8/div8 via MIR2 VM + Z80 emu + LLVM lli
- antique-toy (fjimbuwe): received data for Appendix K update

## Key Decisions
- Analytical multiply-and-shift > GPU brute-force for division (brute-force can't reach len≥9)
- BUT: GPU brute-force found carry_compare that analytics missed
- Three-level validation methodology: analytical → composite → GPU exhaustive
- PRESHIFT dominant method for K<128 (80 entries)
- carry_compare dominant for K≥128 (127 entries, all 26T)

## Numbers
- div8: avg 154T → 135T → **79T** (v1 → v2 → v3)
- div8 total: 39158T → 34316T → **19996T** (−49%)
- sat_add8: 4 ops, 16T (branchless, exhaustive verified 65536)
- sign8: 9 ops, 43T
- abs16: 11 ops, 44T
- SHA-256: 58ms/block @3.5MHz (revised from 15ms)
- Cross-verified: 4 independent systems

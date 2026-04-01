# Z80 Optimizer — Data Tables

Master index of all data tables. Each entry covers: what it is, how it was produced,
key invariants, and how to use it. ADR links point to the architectural decision records
that explain *why* this approach was chosen.

---

## Quick reference by use case

| I want to... | Use |
|---|---|
| Look up optimal mul8 sequence | `mulopt8_clobber.json` |
| Look up optimal div8/mod8/divmod8 | `div8_optimal.json`, `mod8_optimal.json`, `divmod8_optimal.json` |
| Register-allocate 2-5 variables | `enriched_5v.enr.zst` |
| Check if 6v allocation is feasible | `enriched_6v_dense.enr` (tw≥4), else compose from 5v |
| Apply peephole rules | `peephole_top500.json` (fast), `peephole_len2_complete.json` (full) |
| u32 DEHL operations | `u32_ops.json` |
| 16-bit arithmetic (abs, neg, min/max) | `arith16_new.json` |
| Saturating / sign ops | `sign_sat_ops.json` |
| SHA-256 round cost on Z80 | `sha256_round.json` |
| Register move costs | `z80_register_graph.json` |

---

## 1. Arithmetic tables

### `mulopt8_clobber.json`
**254 optimal u8 constant-multiply sequences**: A × K → A, K ∈ [2..255].

- **How produced**: GPU exhaustive search (14-op pool, cuda/z80_mulopt_fast.cu), 2× RTX 4060 Ti, ~38× faster than naive pool. All 254 verified exhaustively (256 input values each).
- **Key invariants**: All sequences preserve DE. All preserve A as output (not clobbered). Clobber set per entry is minimal and explicit.
- **Format**: Array of `{k, ops[], tstates, clobbers[], de_safe}`.
- **Compiler use**: Look up K → emit ops sequence. Check clobbers against live set before call. If conflicting, save/restore (see `z80_register_graph.json` for cheapest save channel).
- **ADR**: [docs/adr/005-arithmetic-tables.md](../docs/adr/005-arithmetic-tables.md)
- **pkg**: `pkg/mulopt/` — `Emit8(k, bSafe)`, `Lookup8(k)`

### `mulopt16_complete.json` / `mulopt16_pareto.json`
**254 optimal u8×K→HL (16-bit result) sequences**.

- **How produced**: GPU search (cuda/z80_mulopt16.cu). `pareto.json` keeps only Pareto-optimal (cost, clobber) pairs.
- **Format**: Same schema as mulopt8. `mulopt16_pareto.json` has `pareto_front: true` flag.
- **Compiler use**: `pkg/mulopt/` — `Emit16(k, preamble)`. Used when result needs to fit in HL for subsequent 16-bit ops.

### `div8_optimal.json`
**254 optimal u8 constant-divide sequences**: A ÷ K → A = floor(A/K), K ∈ [2..255].

- **How produced**: 6 strategies searched by GPU + CPU (cuda/z80_divmod_fast.cu):
  - `shift` (K = power of 2)
  - `mul_shift` (A × M >> S, exact for all K)
  - `preshift_mul` ((A>>P) × M >> S, saves one shift for small K)
  - `carry_compare` (K ≥ 128: `OR A; LD B,(256-K); ADC A,B; SBC A,A; AND 1` — GPU-discovered, 26T)
  - `double_mul_shift`, `mul_add256_shift`
- **Key findings**: avg 79T (−49% vs v1). div86=60T, div172=49T. div3 = A×171>>9 (exact).
- **Verified**: All 254 × all 256 inputs = 65,024 exhaustive checks.
- **ADR**: [docs/adr/005-arithmetic-tables.md](../docs/adr/005-arithmetic-tables.md)
- **pkg**: `pkg/mulopt/` includes div lookup via `Emit8` convention

### `mod8_optimal.json` / `divmod8_optimal.json`
Same approach as div8. `divmod8` returns both quotient and remainder in one sequence.

### `u32_ops.json`
**Optimal u32 operations using DEHL convention** (D=MSB, L=LSB).

- **Includes**: shl32, shr32, add32, sub32, and32, or32, xor32, not32, neg32, load_nn32, store_nn32, rotr32, rotl32. Also HLH'L' EXX-zone variants (hlhl_add32, hlhl_shl32).
- **SHL32/SHR32 proven optimal** (exhaustive GPU verify).
- **DEHL vs HLH'L' decision guide** included in file.
- **ADR**: [docs/adr/005-arithmetic-tables.md](../docs/adr/005-arithmetic-tables.md)

### `arith16_new.json`
**16-bit arithmetic**: abs16 (44T), neg16 (27T), min16/max16 (41-46T), sign16, cmp16_zero. All exhaustively verified.

### `sign_sat_ops.json`
**Branchless u8 ops**: sign8 (43T), sat_add8 (16T — optimal!), sat_sub8 (20T). GPU-verified.

### `sha256_round.json`
**SHA-256 round decomposition** into Z80 u32 ops. 58ms/block @3.5MHz (realistic DEHL model).

### `bcd_idioms.json`
**BCD arithmetic idioms** found by 16-op CPU brute-force. Includes bcd_x10 via DAA.

### `arith16_idioms.json`
Legacy 16-bit idioms (predecessor to arith16_new). Kept for reference.

### `chains_mul8_mod256.json` / `chains_div8.json`
Intermediate chain data from mul8/div8 search. Not for direct compiler use — use the `_optimal` variants.

---

## 2. Peephole tables

### `peephole_len2_complete.json`
**602,008 proven len-2 → len-1 peephole rules** (complete, all Z80 instruction pairs).

- **How produced**: GPU exhaustive search over all 4,215² = ~17.8M pairs, 3-stage pipeline (QuickCheck → MidCheck → ExhaustiveCheck). Full-state equivalence (flags + memory).
- **Size**: 107MB uncompressed. Do not load at startup — use indexed subset.
- **Compiler use**: Full rule set for offline analysis. Runtime: use `peephole_top500.json`.
- **ADR**: [docs/adr/001-dead-flags-optimization-tier.md](../docs/adr/001-dead-flags-optimization-tier.md)
- **pkg**: `pkg/peephole/` — `LoadRules(path)`, `Lookup(source)`

### `peephole_top500.json`
Top 500 peephole rules by cycles_saved. Fits in RAM (~50KB). Use this at compiler runtime.

### `peephole_4T_plus.json`
All rules saving ≥4T. Subset of the complete table, curated for high-impact substitutions.

---

## 3. Register allocation tables

See [ENRICHED_TABLES.md](ENRICHED_TABLES.md) for full format spec and enrichment metrics.

### `exhaustive_4v.bin.zst` / `exhaustive_5v.bin.zst`
**Z80T v1 binary** — raw optimal assignments, no enrichment.

| File | Shapes | Feasible | Compressed |
|------|--------|----------|------------|
| exhaustive_4v.bin.zst | 156,506 | 123,453 (78.9%) | 64KB |
| exhaustive_5v.bin.zst | 17,366,874 | 11,762,983 (67.7%) | 8.5MB |

- **How produced**: `regalloc-enum | cuda/z80_regalloc --server`, 2× RTX 4060 Ti, 40s (4v) / 20min (5v).
- **Format**: `Z80T v1` — see format spec below.
- **Superseded by enriched variants** — prefer `enriched_*.enr.zst` for new code.

### `enriched_4v.enr` / `enriched_5v.enr.zst` / `enriched_6v_dense.enr`
**ENRT v1 binary** — enriched with 15 operation-aware cost metrics per feasible assignment.

| File | Shapes | Feasible | Size |
|------|--------|----------|------|
| enriched_4v.enr | 156,506 | 123,453 | 4MB |
| enriched_5v.enr.zst | 17,366,874 | 11,762,983 | 78MB compressed |
| enriched_6v_dense.enr | 66,118,738 | 25,772,093 | ~900MB |

- **6v dense** covers treewidth≥4 shapes only (1.7% of 6v). Remaining 98.3% compose from 5v at query time.
- **15 metrics**: u8_alu_avg, u8_best_alu, u16_add_natural, u8_mul_avg, mul8_conflicts, call_save_cost, temp_regs_avail, exx_alu_cost, djnz_conflict, ...
- **ADR**: [docs/adr/007-regalloc-enrichment.md](../docs/adr/007-regalloc-enrichment.md)
- **pkg**: `pkg/regalloc/` — `LoadBinary(path)`, `IndexOf(shape)`, `Lookup(idx)`
- **pkg/enr**: `pkg/enr/` — typed reader/writer for .enr format

### IX-expanded tables (in progress, April 2026)

| File | Status | Shapes | Notes |
|------|--------|--------|-------|
| `ix_expanded_5v.jsonl` | GPU0 running | ~60.9M target | 6 locSets8 incl IXH/IXL/IYH/IYL |
| `ix_expanded_6v_dense.jsonl` | GPU1 running | ~7.65M target | nv=6 tw≥4 IX-expanded |
| `ix_expanded_5v.bin` | pending build-ix-table | — | Z80T v2 format |

- **Why**: IXH/IXL/IYH/IYL are universal bridges (survive EXX and EX AF,AF'). Adding them as Tier 1 registers enables HLH'L' u32 and zero-cost EXX zone crossings.
- **ADR**: [docs/adr/008-18reg-exx-model.md](../docs/adr/008-18reg-exx-model.md)

---

## 4. Register cost model

### `z80_register_graph.json`
**Complete Z80 register connectivity model**: move costs between all 11 registers (A,B,C,D,E,H,L + IXH,IXL,IYH,IYL), ALU constraints, pair relationships.

- **Compiler use**: Determine cheapest save channel for a register around CALL. Compute move cost for any src→dst pair. See `call_save_cost` logic in enrich-regalloc.
- **Key values**: A↔B=4T, H↔IXH=16T (DD prefix hijack, needs EX DE,HL trick), B↔IXH=8T.

---

## 5. PRNG / image search tables

These tables are used by the ZX Spectrum LFSR image synthesis pipeline. Not relevant for compiler use.

**Animation seeds** (per-frame LFSR seeds for Che/Putin/skull/Einstein targets):
- `che_anim_*.json`, `che_anima2_*.json`, `che_cp*.json`, `che_wgt*.json`, `che_unw*.json`
- `yozhik_*.json` — hedgehog target variants
- `lissajous_anim.json` — Lissajous figure animation
- `badapple_partial.json` — Bad Apple frame seeds (partial run)

**Seed search results**:
- `cascade_seeds.json` / `foveal_cascade_seeds.json` — multi-layer AND-cascade seeds
- `flat_and4_seeds.json` / `flat_and7_seeds.json` — flat AND-combined seeds
- `budget_conf_a/b/c.json` — budget search configurations

**ADR**: [docs/adr/004-prng-codec-architecture.md](../docs/adr/004-prng-codec-architecture.md)
**Details**: [docs/2026-03-31-lfsr-animation-pipeline.md](../docs/2026-03-31-lfsr-animation-pipeline.md)

---

## 6. Corpus and partition analysis

### `corpus_7_15v_results.json`
Results of regalloc on 820-function production compiler corpus (7-15 virtual registers). 246 unique signatures. Validates the five-level pipeline.

### `partition_greedy_v10_v32.json`
Greedy cut-vertex partitioning results for 10-32 variable functions. Used to tune partition optimizer thresholds.

### `eq_compare_ops.json`
Equality and comparison operation library (branchless u8/u16 comparisons).

### `u32_conventions.json`
Documentation of DEHL vs HLH'L' u32 conventions with tradeoff analysis.

---

## 7. Binary table formats

### Z80T v1 (raw regalloc)
```
Header: "Z80T" (4) + version=1 (u32le)
Records:
  Infeasible: 0xFF
  Feasible:   nVregs(u8) + cost(u16le) + assignment[nVregs](u8)
```

### Z80T v2 (IX-expanded, 2026-04)
```
Header: "Z80T" (4) + version=2 (u32le) + nLocSets8(u8) + nLocSets16(u8) + maxVregs(u8) + n_entries(u64le)
Records: same as v1
```

### ENRT v1 (enriched)
```
Header: "ENRT" (4) + version=1 (u32le) + n_entries(u32le) + maxVregs(u8) + numPatterns(u8) + reserved(u16)
Records:
  Infeasible: 0xFF
  Feasible:   nVregs(u8) + cost(u16le) + assignment[nVregs](u8) + flags(u16le) + patterns[12×u16le]
```
Full spec: [ENRICHED_TABLES.md](ENRICHED_TABLES.md)

---

## 8. Compiler integration — quick start

```go
import (
    "github.com/oisee/z80-optimizer/pkg/mulopt"
    "github.com/oisee/z80-optimizer/pkg/regalloc"
    "github.com/oisee/z80-optimizer/pkg/peephole"
)

// Constant multiplication: A × 57 → A
seq := mulopt.Emit8(57, bSafe=true) // bSafe=true: avoid clobbering B

// Register allocation: look up shape from liveness analysis
table, _ := regalloc.LoadBinary("data/enriched_5v.enr.zst")
idx, _ := regalloc.IndexOf(shape, maxVregs=5)
entry := table.Lookup(idx)
// entry.Cost, entry.Assignment, entry.Flags (mul8_safe, djnz_conflict, ...)

// Peephole: apply top-500 rules at emission time
rules := peephole.Top500()
optimized := rules.Lookup(sourceSequence)
```

**Deeper integration**: [docs/regalloc_deep_dive.md](../docs/regalloc_deep_dive.md), [docs/regalloc_integration_plan.md](../docs/regalloc_integration_plan.md)

---

## 9. Provenance

| Table group | Generated | Hardware | Duration |
|-------------|-----------|----------|----------|
| peephole_len2_complete | 2026-03-26 | 2× RTX 4060 Ti | ~6h |
| mulopt8/16 | 2026-03-26 | 2× RTX 4060 Ti | ~2h |
| div8/mod8/divmod8 (v3) | 2026-03-29 | 2× RTX 4060 Ti | ~4h |
| exhaustive_4v/5v | 2026-03-27 | 2× RTX 4060 Ti | 40s / 20min |
| exhaustive_6v_dense | 2026-03-27 | 2× RTX 4060 Ti | ~6h |
| enriched_4v/5v/6v | 2026-03-28 | CPU (i7) | 40s / 20min / 6h |
| ix_expanded_5v | 2026-04-01 | RTX 4060 Ti (GPU0) | ~4h (running) |
| ix_expanded_6v_dense | 2026-04-01 | RTX 4060 Ti (GPU1) | ~4h (running) |

NAS backup: `/mnt/safe/z80-compiler/tables/`
MinZ integration: `research/paper-a/data/tables/`

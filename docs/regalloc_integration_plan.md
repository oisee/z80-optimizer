# Regalloc Integration Plan: 10-loc → 14-loc + EXX-zone + OFB Trim

**Goal**: Expand existing 37.6M-entry enriched tables to cover IX/IY halves and shadow zone,
enrich with Op Feasibility Bag (OFB), trim to Pareto-optimal per (shape, OFB).

---

## Current State

```
data/enriched_4v.enr.zst   — ≤4v, 156,506 shapes (78.9% feasible), 40s build
data/enriched_5v.enr.zst   — ≤5v, 17.4M shapes (67.7% feasible), 20min build
data/enriched_6v_dense.enr — 6v tw≥4, 66.1M shapes (38.9% feasible), 6h build
```

Schema: `(shape) → {cost, assignment[nVregs], 15 enrichment metrics}`

Loc indices: A=0 B=1 C=2 D=3 E=4 H=5 L=6 BC=7 DE=8 HL=9

LocSets (4+3): `{A}` `{C}` `{any GPR8}` `{non-A GPR8}` | `{HL}` `{DE}` `{any pair}`

---

## Phase 1 — Raw Data Collection (GPU, parallel)

### 1a. IX-expanded nv≤5 (running: data/ix_expanded_5v.jsonl)
New locSets 4=`{IXH..IYL}`, 5=`{A..E + IXH..IYL}`. ~70-100M shapes, ~2h on GPU0.
Output: one JSON result per shape `{cost, assignment, searchSpace, feasible}`.

### 1b. IX-expanded nv=6 tw≥4 (GPU1, ~1h)
Only the dense corner: 6 locSets8 × nv=6 tw≥4 ≈ 7.65M shapes.
```bash
./regalloc-enum --max-vregs 6 --only-nv 6 --min-treewidth 4 \
  | ./cuda/z80_regalloc --server --gpu-id 1 \
  > data/ix_expanded_6v_dense.jsonl
```

### 1c. EXX-zone nv≤5 (GPU1 after 1b, ~2h)
Shadow zone: same table structure, 8 locs {B'=1,C'=2,D'=3,E'=4,H'=5,L'=6, A=0, A'=7(+8T)}.
Shadow zone IS the existing locSets — just runs same GPU kernel, boundary cost computed separately.
**KEY INSIGHT: No new kernel needed. Shadow zone = existing table + boundary cost model.**
→ Skip GPU run for shadow; use Phase 4 composition instead.

**GPU1 freed**: use for 1b (nv=6 dense IX-expanded) then nv=6 tw≤3 weekend run.

---

## Phase 2 — JSONL → Binary Tables (CPU post-processing)

### Input format (from z80_regalloc --server):
```json
{"cost": 32, "assignment": [0, 2, 9, 13], "searchSpace": 50625, "feasible": 84}
{"cost": -1, "assignment": [], "searchSpace": 50625, "feasible": 0}
```

### Output format: extend existing Z80T v1 binary, new version Z80T v2
```
Header: "Z80T" + version(u32=2) + locSets_descriptor(var) + n_entries(u64)
Records (same as v1):
  Infeasible: 0xFF
  Feasible:   nVregs(u8) + cost(u16le) + assignment(nVregs bytes)
```

Enumeration order must match exactly between regalloc-enum and table index.
New table files: `data/ix_expanded_5v.bin`, `data/ix_expanded_6v_dense.bin`

**Tool to write**: `cmd/build-ix-table/main.go` — reads JSONL, emits Z80T v2 binary.

---

## Phase 3 — Post-Process Existing Tables: 10-loc → 14-loc Derivation

**No GPU needed.** For each feasible entry in existing enriched tables:

```
For each vreg i where assignment[i] ∈ {B=1, C=2, D=3, E=4}:
  For each IX/IY half loc j ∈ {IXH=10, IXL=11, IYH=12, IYL=13}:
    derived_cost = original_cost
      + 4T × (ops where vreg_i is src or dst)   ← extra per-use cost (8T vs 4T LD)
      - 0                                         ← no savings elsewhere
    If derived_cost < existing_best_for_shape_with_j_at_i:
      emit derived entry
```

Constraints:
- H(5)/L(6) → IXH/IXL: costs 16T per transfer (not 4T). Only derive if no H/L conflict.
- IXH/IXL cannot be used with ADC/SBC as src (only ADD/SUB/AND/OR/XOR/CP/INC/DEC).
  → derivation tagged with `alu_constraint: no_adc_sbc_src`.

**Tool to write**: `cmd/derive-ix/main.go` — reads enriched .enr, emits derived JSONL.

Output: `data/ix_derived_from_existing.jsonl` (~few hundred MB, CPU only, ~hours)

---

## Phase 4 — EXX-Zone Composition (no new table, boundary calculator)

**Architecture**: EXX zone IS the existing table. Shadow B'≡B (loc1), C'≡C (loc2), etc.
Two allocations composed at zone boundary:

```
total_cost = main_alloc_cost         (from existing table, vars in main bank)
           + shadow_alloc_cost       (from SAME existing table, vars in shadow bank)
           + boundary_cost           (computed by boundary calculator)

boundary_cost = 2 × 4T              (EXX entry + exit)
              + sum over vars crossing boundary:
                  0T  if var ∈ {A, IXH, IXL, IYH, IYL}   ← zone-invariant
                  8T  if var shuttled via A or IX half     ← one-byte bridge
                  16T if var is 16-bit, shuttled via IX    ← PUSH/POP cheaper
                  21T via PUSH/POP                         ← fallback
```

**Tool to write**: `pkg/regalloc/zone.go` — `ComposeCost(main Shape, shadow Shape) int`

---

## Phase 5 — Merge: Unified 14-loc Table

Combine all sources into one canonical table per (nVregs, locSets14, interference):

```
Sources (in priority order for same shape):
  1. GPU-direct result from ix_expanded_*.jsonl   ← ground truth optimal
  2. Derived from existing (Phase 3)              ← may miss some optima
  3. Existing table (10-loc shapes unchanged)     ← backward compat

Merge rule: keep lowest cost. Infeasible only if ALL sources say infeasible.
```

**Tool to write**: `cmd/merge-tables/main.go`

Output: `data/merged_14loc_5v.enr` (estimated ~200M entries, ~500MB compressed)

---

## Phase 6 — OFB Enrichment

**Op Feasibility Bag (OFB)** = bitmask of operations a function uses.
Each bit = one operation class:
```
bit 0: ALU_ADD_A_R      (ADD A,r — any GPR)
bit 1: ALU_ADD_A_IXH    (ADD A,IXH — requires src=IXH/IXL, so assignment must have that)
bit 2: ALU_ADC          (ADC A,r — excludes IXH as src!)
bit 3: HL_ARITH         (ADD HL,rr — requires pair in HL)
bit 4: DE_PTR           (LD A,(DE) — requires DE pair)
bit 5: HL_PTR           (LD A,(HL) — requires HL pair)
bit 6: MUL8             (calls mulopt8 — needs A free for result)
bit 7: DIV8             (calls div8 — needs A, specific constraints)
bit 8: CALL             (uses CALL — IXH/IXL survive, GPR may need save)
bit 9: EXX_ZONE         (contains EXX — triggers zone composition)
... up to 32 bits
```

For each (assignment, OFB) pair:
- Check assignment satisfies all OFB constraints
- If `ALU_ADD_A_IXH` set: at least one vreg must be at IXH/IXL
- If `ALU_ADC` set: no vreg at IXH/IXL can be ADC src
- If `EXX_ZONE` set: apply zone composition cost

**Tool to write**: `cmd/enrich-ofb/main.go` — reads merged table, annotates with OFB feasibility masks.
Extends existing 15-metric enrichment schema.

Output: `data/enriched_14loc_5v.enr` with OFB feasibility bitmask per entry.

---

## Phase 7 — Pareto Trim

For each unique (shape_signature, OFB):
- Keep only assignments that are Pareto-optimal across:
  - `total_cost` (primary)
  - `clobber_mask` (secondary — fewer clobbers = better for call boundaries)
  - `uses_ix` (tertiary — prefer not using IX if GPR suffice, leaves IX free for bridges)

```
Domination: A dominates B if A.cost ≤ B.cost AND A.clobber ⊆ B.clobber AND A.uses_ix ≤ B.uses_ix
```

Expected reduction: ~60-80% entries trimmed (many suboptimal assignments removed).
Final table size estimate: ~50-100M entries (from ~200M pre-trim).

**Tool to write**: `cmd/trim-pareto/main.go`

Output: `data/final_14loc_5v.enr` — compiler-ready.

---

## Phase 8 — pkg/regalloc v2 API

```go
// New API (backward-compatible with v1)
type TableV2 struct {
    Main   *Table   // 10-loc (existing, backward compat)
    IX     *Table   // 14-loc (new IX-expanded)
    OFB    OFBIndex // (shape, ofb) → optimal assignment
}

func (t *TableV2) LookupWithOFB(s Shape, ofb OFBMask) *Entry
func (t *TableV2) LookupZone(main, shadow Shape) (cost int)
```

---

## Build Order & Dependencies

```
Phase 1a (running) ──────────────────────────────────────┐
Phase 1b (GPU1, ~1h) ────────────────────────────────┐   │
                                                     │   │
Phase 3 (CPU, derive) ── depends on existing .enr   │   │
                                                     ▼   ▼
                              Phase 2 (JSONL→binary)─────┤
                                                         │
                                    Phase 5 (merge) ◄────┘
                                         │
                              Phase 6 (OFB enrich)
                                         │
                               Phase 7 (Pareto trim)
                                         │
                               Phase 8 (API v2)
```

---

## Estimated Timeline

| Phase | Work | Time |
|-------|------|------|
| 1a | GPU run (running) | ~done in 30min |
| 1b | GPU1 run (nv=6 dense IX) | ~1h |
| 2 | Build binary from JSONL | 1 day code + 30min run |
| 3 | Derive IX from existing | 2 days code + few hours CPU |
| 5 | Merge tables | 1 day code + 1h run |
| 6 | OFB enrichment | 2-3 days code + 2h run |
| 7 | Pareto trim | 1 day code + 30min run |
| 8 | API v2 | 1 day code |

**Total**: ~2 weeks to full enriched 14-loc compiler-ready tables.

---

## Immediate Next Steps (tonight)

1. ✅ Phase 1a running (ix_expanded_5v.jsonl, GPU0)
2. 🚀 Launch Phase 1b: nv=6 dense IX on GPU1 (command below)
3. 📝 Write `cmd/build-ix-table/main.go` (Phase 2 tool)
4. 📝 Write `pkg/regalloc/zone.go` (Phase 4 boundary calculator)

```bash
# Phase 1b — launch now on GPU1
./regalloc-enum --max-vregs 6 --only-nv 6 --min-treewidth 4 \
  | CUDA_VISIBLE_DEVICES=1 ./cuda/z80_regalloc --server \
  > data/ix_expanded_6v_dense.jsonl 2>data/ix_expanded_6v_dense.log &
```

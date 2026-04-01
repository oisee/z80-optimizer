# ADR-007: Register Allocation — Enriched Tables with Operation-Aware Metrics

**Date:** 2026-03-28
**Status:** Accepted
**Tables**: `enriched_4v.enr`, `enriched_5v.enr.zst`, `enriched_6v_dense.enr`
**Supersedes**: raw `exhaustive_*.bin.zst` tables for compiler use

---

## Context

Raw regalloc tables answer: "is this assignment feasible and what does it cost?" But a compiler needs to choose *which* feasible assignment to use. The cost in the raw table reflects the regalloc solver's internal model (minimize move costs), but doesn't account for:

- Which operations the function actually uses (u8 ALU, u16 ADD, mul8, CALL, DJNZ...)
- Whether mul8 clobber zones conflict with live variables
- How much CALL overhead is incurred (save/restore cost)
- Whether EXX could be used for a 2-bank split

A compiler that always picks the "cheapest assignment" in raw-cost terms may pick one that conflicts with a critical idiom, adding 30T of save/restore.

### Alternatives considered

| Approach | Problem |
|----------|---------|
| Query solver at compile time | Too slow for production (Z3/GPU per function) |
| Raw cost only | Misses idiom compatibility, adds save/restore overhead |
| **Enriched table: precompute 15 metrics per assignment** | **O(1) lookup at compiler runtime** |
| Multiple specialized tables per op class | Combinatorial explosion |

---

## Decision

Add 15 operation-aware cost metrics to each feasible assignment. The enriched table stores (assignment, metrics) for all 37.6M feasible shapes.

### Metrics (ENRT v1, 12 stored + 3 flags)

| Metric | What it measures | Compiler use |
|--------|-----------------|--------------|
| `u8_alu_avg` | avg cost of binary ALU between any var pair | u8-heavy functions |
| `u8_best_alu` / `u8_worst_alu` | best/worst case ALU pair | range estimate |
| `u16_add_natural` | cost of ADD HL,rr (11T if HL free, else 19T+) | u16 arithmetic |
| `u8_mul_avg` | cost of mul8 call given live set | multiply-heavy |
| `mul8_conflicts` | vars in mul8 clobber zone {C,H,L} | safe to call mul8? |
| `mul16_conflicts` | vars in mul16 clobber zone {A,C,H,L} | |
| `call_save_cost` | total save/restore T-states around CALL | CALL overhead |
| `call_free_saves` | saves that use free registers (8T) vs PUSH (21T) | |
| `temp_regs_avail` | registers free for temporaries | scratch space |
| `u16_slots_free` | remaining 16-bit pair slots | additional u16 vars |
| `u8_regs_free` | free 8-bit registers | |
| `exx_alu_cost` | cost of ALU via shadow bank | EXX-split functions |

**Flags** (uint16 bitfield):
- `no_accumulator` (bit 0): A not in assignment → u8 ALU needs 2 extra LDs
- `no_hl_pair` (bit 1): HL not in assignment → u16 ADD needs extra moves
- `mul8_safe` (bit 2): no clobber conflicts → can call mul8 without save/restore
- `djnz_conflict` (bit 3): B occupied → DJNZ loop needs save/restore
- `u16_capable` (bit 4): ≥2 pair slots available

### Key findings from enrichment

- **43% of feasible shapes lack A** → u8 ALU via these is slower than it looks
- **21% lack HL** → u16 ADD suboptimal
- **Smart CALL save: 17T avg vs 34T naive** — using free regs and IX halves as save channels instead of always PUSH/POP
- **Feasibility cliff**: 95.9% (2v) → 0.9% (6v) — phase transition at 6 variables

---

## Compiler integration

```go
table, _ := regalloc.LoadBinary("data/enriched_5v.enr.zst")

// From liveness analysis:
shape := regalloc.Shape{
    NVregs: 3,
    Widths: []int{8, 8, 16},
    LocSetIndex: []int{2, 3, 0},   // any GPR8, non-A GPR8, must-be-HL
    Interference: 0b011,            // v0 interferes with v1
}
idx, _ := regalloc.IndexOf(shape, maxVregs=5)
entry := table.Lookup(idx)

if entry.Infeasible() {
    // Decompose via cut-vertex, or escalate to GPU
}

// Pick by operation mix:
if funcHasMul8 && entry.Flags & regalloc.FlagMul8Safe == 0 {
    // This assignment will conflict with mul8 — try next candidate or add save/restore
}
cost := int(entry.Cost) + int(entry.Patterns[regalloc.PatCallSaveCost]) * numCalls
```

### Coverage (validated on 820-function corpus)

- ≤5v shapes: O(1) enriched table lookup, 79% of corpus
- 6v tw≥4: enriched_6v_dense.enr
- 6v tw≤3: compose from 5v via cut-vertex (98.3% of 6v, max 12T overhead)
- 7v+: GPU partition optimizer or Z3 (<0.5% of functions)

---

## Planned extension: OFB + 14-loc + Pareto trim (2026-04)

See `docs/regalloc_integration_plan.md` for Phase 5-8:
- **14-loc tables**: IXH/IXL/IYH/IYL as Tier 1 registers
- **OFB (Op Feasibility Bag)**: 32-bit bitmask per assignment — which op classes it supports
- **Pareto trim**: keep only assignments Pareto-optimal by (cost, clobber_mask, uses_ix)

**ADR**: [008-18reg-exx-model.md](008-18reg-exx-model.md)

---

## References

- Tool: `cmd/enrich-regalloc/`
- Package: `pkg/regalloc/`, `pkg/enr/`
- Format: [ENRICHED_TABLES.md](../../data/ENRICHED_TABLES.md)
- Deep dive: [docs/regalloc_deep_dive.md](../regalloc_deep_dive.md)

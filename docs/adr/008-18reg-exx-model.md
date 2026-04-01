# ADR-008: 18-Register Z80 Model — IX Halves as Tier 1, EXX Zone Architecture

**Date:** 2026-04-01
**Status:** Accepted
**Tables**: `ix_expanded_5v.bin` (in progress), EXX boundary: `pkg/regalloc/zone.go`
**Plan**: `docs/regalloc_integration_plan.md`

---

## Context

The existing regalloc tables cover 10 locations: A, B, C, D, E, H, L (7 GPRs) + BC, DE, HL (3 pairs). IXH/IXL/IYH/IYL were not included because:
1. They have 8T access cost (vs 4T for GPRs)
2. ADC/SBC cannot use IX halves as src (DD+ED prefix conflict)
3. H/L ↔ IX transfer costs 16T (DD prefix hijacks H/L encoding)

However, excluding them misses a critical architectural property: **IXH/IXL/IYH/IYL are universal register-file bridges**.

---

## Key architectural discovery (2026-04-01)

### Universal bridges

| Instruction | A | B,C,D,E,H,L | IXH,IXL,IYH,IYL |
|-------------|---|-------------|-----------------|
| EXX | unchanged | swap ↔ shadow | **unchanged** |
| EX AF,AF' | swap | unchanged | **unchanged** |
| CALL/RET | clobbered (return) | caller-saves | **unchanged** |

IXH/IXL/IYH/IYL survive **all** context switches. A variable assigned to IXH is the same physical register in the main bank, shadow bank, and across CALL boundaries.

This enables **zero-cost EXX zone crossings**: a variable that spans both the main and shadow EXX zones incurs 0T overhead if assigned to an IX half.

### HLH'L' u32 convention (enabled by this insight)

Previous: DEHL u32 (D=MSB, L=LSB). DE is consumed, unavailable for pointer arithmetic.

New: HLH'L' u32 — HL=low16 in main bank, H'L'=high16 in shadow (EXX) bank.
```z80
; 32-bit add via EXX:
ADD HL, BC    ; low16 add
EXX           ; switch to shadow
ADC HL, BC    ; high16 add with carry (EXX preserves CY!)
EXX           ; back to main
; Total: 34T — same as DEHL. But DE is now FREE for pointer arithmetic.
```
Critical for Che decoder (screen pointer in DE) and other memory-intensive algorithms.

---

## Decision

### Register tier model

| Tier | Registers | Access cost | Constraints |
|------|-----------|-------------|-------------|
| 0 | A, B, C, D, E, H, L | 0T (4T LD) | Full ALU; EXX swaps B-L |
| 1 | IXH, IXL, IYH, IYL | +4T (8T LD) | No ADC/SBC src; H/L↔IX=16T; universal bridge |
| 2 | B', C', D', E', H', L' | 4T (EXX) | All 6 swap simultaneously; same locs as Tier 0 |
| 2 | A' | 4T (EX AF,AF') | Individual swap |

### EXX zone table architecture

Shadow zone (B'..L') uses the **same loc indices** as main zone (B=1, C=2, ... HL=9). The EXX zone IS the existing table. No new GPU run needed for shadow zone itself.

Boundary cost formula (`pkg/regalloc/zone.go`):
```
BoundaryCost = 4T (EXX instruction)
  + 0T per var in {A, IXH, IXL, IYH, IYL}   ← zone-invariant: free crossing
  + 8T per var in {B,C,D,E,H,L} that must cross  ← A-shuttle: LD A,r; EXX; LD r,A
  OR 16T via IX bridge if A is occupied
  OR 21T via PUSH/POP for 16-bit pairs
```

### Spill tier hierarchy (complete)

| Tier | Channel | Cost | Constraints |
|------|---------|------|-------------|
| L0 | Primary GPR (A-L) | 0T | — |
| L1 | IX half (IXH/IXL/IYH/IYL) | 8T/access | No ADC/SBC src |
| L1.5 | R register (1-bit bool) | 26T round-trip | 1 bit (bit 7); IM0/1 only |
| L2a | I register (8-bit) | 18T round-trip | IM0/1 only; flags on read |
| L2b | TSMC tunnel (self-mod) | 20T 8-bit / 44T 16-bit | No recursion; DI/EI required |
| L3 | PUSH/POP (SP never swapped) | 21T/half | Interrupt-safe; universal |
| L4a | Memory via A | 13T | One byte; address in HL |
| L4b | Memory via HL | 16T | Two bytes |
| L4c | Memory via pairs | 20T | |

**R-register boolean spill** (0 named registers consumed):
```z80
; Save: materialize 0xFF/0x00 from CY, then LD R,A (9T stores bit7)
SBC A, A        ; 4T: CY → 0xFF/0x00
LD R, A         ; 9T: bit7 of A → R.bit7
; ... A and all named regs are free ...
; Restore: recover bit7 from R
LD A, R         ; 9T: bit7 intact (bits 0-6 = garbage from instruction counter)
ADD A, A        ; 4T: bit7 → CY
SBC A, A        ; 4T: CY → 0xFF/0x00
; Total round-trip: 13T + 17T = 30T (4T save-from-CY + 9T store + 9T load + 4T shift + 4T materialize)
```
Useful in EXX zone boundaries where A is the only bridge and needs to be free.

---

## Implementation plan

See `docs/regalloc_integration_plan.md` for full 8-phase plan.

Phase 1 (running 2026-04-01):
- GPU0: `ix_expanded_5v.jsonl` — 60.9M shapes, 6 locSets8
- GPU1: `ix_expanded_6v_dense.jsonl` — 7.65M shapes

Tools built:
- `cmd/build-ix-table/` — JSONL → Z80T v2 binary
- `cmd/derive-ix/` — derive IX alternatives from existing enriched tables
- `cmd/merge-tables/` — merge sources, keep best cost per shape
- `pkg/regalloc/zone.go` — EXX boundary cost calculator

---

## References

- Boundary cost: `pkg/regalloc/zone.go`
- Integration plan: `docs/regalloc_integration_plan.md`
- 18-reg memory: `/home/alice/.claude/projects/.../memory/project_18reg_model.md`
- Session: `contexts/week1_report.md` (Day 6)
- minz CLAUDE.md — TSMC tunnel definition (self-modifying code via LD immediate patch)

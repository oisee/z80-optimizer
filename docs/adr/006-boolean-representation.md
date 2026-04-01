# ADR-006: Boolean Representation — 0xFF/0x00 Over 0x01/0x00

**Date:** 2026-03-28 (formalized), 2026-04-01 (broadcast project-wide)
**Status:** Accepted — project-wide standard
**Affects**: All codegen, arithmetic tables, branchless library, carry_compare pattern

---

## Context

Z80 boolean results must be represented as bytes. Two conventions exist:

| Convention | True | False | Native ops |
|------------|------|-------|------------|
| Classical  | 0x01 | 0x00  | AND/OR/XOR require shift; CY = carry flag |
| **Mask**   | **0xFF** | **0x00** | **AND/OR/XOR/NOT free; SBC A,A materializes in 1 op** |

The classical convention is used by most languages (C, Python). The mask convention is what the Z80 flag instructions naturally produce.

---

## Decision

**Use 0xFF/0x00 as the canonical boolean representation throughout this project.**

### Why

1. **SBC A,A is the perfect materializer** (4T, 1 instruction):
   - If CY=1: A = 0xFF (true mask)
   - If CY=0: A = 0x00 (false mask)
   - No branch needed. CY is the only viable branchless boolean flag (Z→CY impossible).

2. **Logic ops are free**:
   - NOT b = `CPL` (4T, 1 instruction) — just flip bits
   - b AND c = `AND C` (4T) — no setup
   - b OR c = `OR C` (4T)
   - b XOR c = `XOR C` (4T)

3. **Arithmetic from booleans is free**:
   - b as u8 addend: `ADD A, B` where B=0xFF adds 255 ≡ -1 (mod 256)
   - `AND 1` converts to classical 0x01/0x00 when needed (interface boundary)

4. **Branchless library uses masks natively**:
   - CMOV: `SBC A,A; AND (B XOR C); XOR C` — 3 ops, 12T
   - ABS: `LD B,A; RLCA; SBC A,A; XOR B; SUB A` — 5 ops, 20T
   - MIN/MAX: built on SBC A,A + bitwise select

5. **GPU-discovered carry_compare** relies on this: `SBC A,A; AND 1` = convert CY to 0/1.

### Conversion at boundaries

When interfacing with code that expects classical 0x01/0x00:
```z80
AND 1   ; 4T — mask → classical (cheap)
```
When converting from classical to mask:
```z80
DEC A   ; 0x01→0x00, 0x00→0xFF via underflow (if input is guaranteed 0/1)
; OR:
NEG     ; 0x01→0xFF, 0x00→0x00 ... wrong for 0x01 case
; Better:
ADD A,A  ; shift bit 0 to carry
SBC A,A  ; CY → mask
```

---

## Proof: Z flag is write-only

Proven exhaustively (2026-03-28): no Z80 instruction reads the Z flag as INPUT. All ALU instructions only WRITE Z. This means:
- Z→CY conversion is impossible branchlessly in 1 instruction
- CY is the only viable single-instruction boolean → mask conversion
- Therefore SBC A,A is the canonical boolean materializer

---

## Key invariant

All sequences in the arithmetic tables that return boolean results (div8 carry_compare, sat_add8, etc.) return 0xFF (true) or 0x00 (false), not 0x01/0x00.

---

## References

- Branchless library: `data/arith16_new.json`, `data/sign_sat_ops.json`
- carry_compare pattern: `data/div8_optimal.json` (K≥128 entries)
- Z flag proof: `docs/z80_opref.md` § Boolean representation
- Session: `contexts/week1_report.md` (Day 3, session 2026-03-28)

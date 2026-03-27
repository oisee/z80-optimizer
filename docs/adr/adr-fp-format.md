# ADR: Z80-Optimal Floating Point Format

## Status: Accepted

## Context
Z80 has no FPU. IEEE 754 formats have bit fields crossing byte boundaries,
making extraction expensive (shift+mask chains). We need a format where
common operations (×2, compare, negate) map to single Z80 instructions.

## Decision
Three-tier format family, all with **byte-aligned exponent**:

| Tier | Format | Layout | ×2 cost |
|------|--------|--------|---------|
| 1 (fast) | Z80-FP16 | H=[exp] L=[sign+mant7] | INC H (4T) |
| 2 (precise) | s1.E8.M15 | A=[exp] H=[sign+mant_hi] L=[mant_lo] | INC A (4T) |
| 3 (compat) | s1.E8.M31 | A=[exp] HL=[sign+mant_hi] H'L'=[mant_lo] | INC A (4T) |

All share 8-bit exponent (same range as Bfloat16/float32), bias=127.

## Consequences
- ×2/÷2 = INC/DEC on exponent byte (4T vs 32-40T for IEEE)
- Mantissa operations reuse existing mul16/div8 tables
- NOT IEEE-compatible in memory layout — need explicit conversion
- Conversion to/from IEEE/Bfloat16 = brute-forceable (small target function)

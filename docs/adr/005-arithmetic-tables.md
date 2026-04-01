# ADR-005: Arithmetic Tables — Exhaustive GPU Search + Multiply-by-Reciprocal

**Date:** 2026-03-26 to 2026-03-29
**Status:** Accepted
**Tables**: `mulopt8_clobber.json`, `mulopt16_complete.json`, `div8_optimal.json`, `mod8_optimal.json`, `divmod8_optimal.json`, `u32_ops.json`

---

## Context

Z80 has no hardware multiply or divide. Compilers typically emit library calls (slow, ~200T) or fixed idioms. We need *provably optimal* sequences for all constants K ∈ [2..255].

### Alternatives considered

| Approach | Result | Why rejected |
|----------|--------|--------------|
| Full ISA brute-force (455 opcodes) | Intractable (455^14 ≈ 10^37) | Too large even for GPU |
| Hand-coded lookup table | Suboptimal, incomplete | Human error, coverage gaps |
| **21-instruction universal pool** | **254/254 complete** | **Adopted** |
| Barrett reduction (software) | ~200T average | Correct but slow |

---

## Decision

### Multiply (mul8)

**Method**: Exhaustive search over a curated 14-opcode pool:
```
ADD A,A  ADD A,B  ADD A,r  LD B,A  LD A,B
NEG  CPL  SUB B  SUB A  XOR A
INC A  DEC A  RLA  RLCA
```

This 14-op pool was discovered by meta-analysis: it generates ALL 254 optimal sequences. Adding more ops finds no improvements. The pool is 2.7% of the full Z80 ISA.

**Key discoveries**:
- ×255 = `NEG` (1 instruction, 8T) — multiply by subtraction from 0
- ×3 = `ADD A,A; ADD A,B` (2 instructions, 8T)
- All 254 sequences preserve A as output and are DE-safe
- Average: ~20T (vs ~200T for library call)

**Verification**: Every sequence tested against all 256 inputs exhaustively.

### Division (div8, v3)

**Method**: 6 strategies, GPU + CPU search per strategy, keep minimum cost:

```
Strategy          Condition        Example           T-states
─────────────────────────────────────────────────────────────
shift             K = 2^P          div2: SRL A       8T
mul_shift         any K            K=3: ×171>>9      141T
preshift_mul      K < 128          K=86: (A>>1)×61   60T
carry_compare     K ≥ 128          K=200: 5 ops      26T  ← GPU-discovered!
double_mul_shift  any K            alternative path   varies
mul_add256_shift  any K            alternative path   varies
```

**GPU-discovered carry_compare** (session 2026-03-29):
```z80
OR A              ; clear carry
LD B, (256-K)     ; B = 256-K
ADC A, B          ; A + (256-K); carry set iff A ≥ K
SBC A, A          ; 0x00 if A < K, 0xFF if A ≥ K
AND 1             ; 0 or 1 = floor(A/K) for K ≥ 128
```
This pattern emerges from the boolean 0xFF/0x00 representation (see ADR-006).

**Result**: avg 79T (−49% vs v1 naive approach).

**Verification**: All 254 × all 256 inputs = 65,024 exhaustive checks.

### u32 operations (DEHL convention)

All u32 ops use DEHL: D=MSB, E, H, L=LSB. SHL32/SHR32 proven optimal by GPU exhaustive search. Full list in `data/u32_ops.json`.

---

## Key invariant

All arithmetic tables satisfy: for every K and every input value in domain, the emitted sequence produces the correct result. "Verified: true" in JSON means this was checked.

---

## Compiler integration

```go
// pkg/mulopt/
seq := mulopt.Emit8(k, bSafe)  // bSafe=true keeps B register free
seq := mulopt.Emit16(k, preamble) // result in HL

// Runtime lookup — no search, just table indexing
entry := mulopt.Lookup8(k)  // returns {ops, tstates, clobbers}
```

Check `entry.Clobbers` against live set. If conflict: save/restore (see `z80_register_graph.json` for cheapest channel). Average save cost: 8T (via free register) to 21T (PUSH/POP).

---

## References

- GPU kernel: `cuda/z80_mulopt_fast.cu`, `cuda/z80_divmod_fast.cu`
- Package: `pkg/mulopt/`
- Session log: `contexts/week1_report.md` (Day 1-5)

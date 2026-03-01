# ADR 001: Dead-Flags Optimization Tier

## Status

Accepted

## Context

The Z80 superoptimizer requires **full state equivalence**: target and candidate must produce identical output for ALL possible inputs, including all 8 flag bits. This means `LD A, 0` cannot be replaced by `XOR A` (saves 1 byte, 3 T-states) because `XOR A` clobbers flags while `LD A, 0` preserves them.

In practice, most occurrences of `LD A, 0` in real Z80 code are followed by instructions that overwrite flags (e.g., `CP`, `AND`, `OR`, `ADD`, `SUB`). The flags set by `XOR A` are "dead" — never read before being overwritten. This is the single highest-impact optimization class for real Z80 code.

## Decision

We add a second tier of optimization rules tagged with which flag bits must be "dead" (not read before being overwritten) for the rule to be valid.

### Design

1. **`FlagMask` type** (`uint8`): a bitmask where set bits indicate flags that are considered dead.
   - `DeadNone` (0x00): full equivalence (existing behavior)
   - `DeadUndoc` (0x28): undocumented flag bits 3 and 5 — almost always safe to ignore
   - `DeadAll` (0xFF): all flags dead — registers-only equivalence

2. **Masked verification**: `QuickCheckMasked` and `ExhaustiveCheckMasked` compare states using `(a.F & ^deadFlags) == (b.F & ^deadFlags)` instead of `a.F == b.F`.

3. **`FlagDiff` function**: determines exactly which flag bits differ between two sequences, producing the minimal `DeadFlags` annotation for a rule.

4. **`Rule.DeadFlags` field**: tags each rule with its required dead flags. Zero means unconditional (backward compatible).

5. **Search integration**: both brute-force enumeration and STOKE stochastic search support a `--dead-flags` parameter. The brute-force worker tries full equivalence first, then falls back to masked equivalence.

### Consumer responsibility

The superoptimizer discovers and tags rules. The **consumer** (peephole optimizer, compiler pass) is responsible for liveness analysis to determine which flags are dead at each program point before applying a dead-flags rule.

## Consequences

### Positive

- Unlocks `LD A, 0 -> XOR A` and hundreds of similar flag-clobbering optimizations
- Backward compatible: existing rules with `DeadFlags=0` behave identically
- Minimal annotation: `FlagDiff` computes the precise set of flags that must be dead
- Works with both brute-force and STOKE search engines

### Negative

- Rules with `DeadFlags != 0` require liveness analysis to apply safely — the consumer must implement this
- Search time increases when dead-flags mode is enabled (two passes: full then masked)
- JSON output grows slightly with the additional fields

### Neutral

- The undocumented flags (bits 3, 5) are almost never read by real Z80 programs; `DeadUndoc` is a safe default for most use cases

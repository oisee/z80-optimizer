# z80-optimizer

A brute-force superoptimizer for the Zilog Z80 processor, inspired by [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194).

Given a sequence of Z80 instructions, it exhaustively searches for a shorter equivalent sequence that produces **identical register and flag state** for all possible inputs.

## Why?

Compilers and hand-written Z80 code often contain suboptimal patterns that are hard to spot manually. A superoptimizer finds replacements that are *provably correct* by testing every possible input state — no heuristics, no pattern databases, just brute force.

Examples it finds automatically:

| Original | Replacement | Savings |
|---|---|---|
| `AND 0FFh` | `AND A` | -1 byte, -3 cycles |
| `OR 00h` | `OR A` | -1 byte, -3 cycles |
| `XOR 00h` | `OR A` | -1 byte, -3 cycles |
| `SUB A : LD A, 0` | `SUB A` | -2 bytes, -7 cycles |
| `AND A : AND A` | `AND A` | -1 byte, -4 cycles |

Note: it correctly rejects `LD A, 0 -> XOR A` because `XOR A` modifies flags while `LD A, 0` does not. Full state equivalence means no false positives.

## How it works

1. **Enumerate** all target instruction sequences up to length N
2. **Fingerprint** each sequence by running it on 8 test vectors (fast reject for 99.99% of non-matches)
3. **Exhaustive verify** candidates that pass fingerprinting, sweeping all input registers (up to 256^3 states)
4. **Prune** redundant candidates (NOPs, self-loads, dead writes, canonical ordering)

The executor runs at **~2.7ns per instruction** on Apple M2 with zero allocations.

## Instruction coverage

406 opcodes across 4 implementation waves:

| Wave | Opcodes | What |
|---|---|---|
| V1 | 206 | 8-bit loads, ALU, rotates/shifts, specials |
| Wave 1 | +174 | BIT/RES/SET n,r (CB prefix) |
| Wave 2 | +14 | 16-bit pair ops (INC/DEC rr, ADD HL, EX DE,HL) |
| Wave 4 | +12 | LD rr,nn, ADC/SBC HL,rr (ED prefix) |

Total search space: **266,359 instructions per position**.

## Usage

```bash
# Build
go build -o z80opt ./cmd/z80opt

# Find optimal replacement for a specific sequence
z80opt target "AND 0xFF"
z80opt target "SUB A : LD A, 0"

# Enumerate all length-1 and length-2 optimizations
z80opt enumerate --max-target 2 --output rules.json

# Verify previously found rules
z80opt verify rules.json
```

## Project structure

```
cmd/z80opt/          CLI (enumerate, target, verify, export)
pkg/cpu/             Z80 state + executor (2.7ns/op, 0 alloc)
pkg/inst/            Instruction catalog (406 opcodes, encoding, timing)
pkg/search/          Verifier, enumerator, pruner, fingerprint map, workers
pkg/result/          Rule storage, checkpoint, JSON output
```

## Correctness

Flag behavior is ported from [remogatto/z80](https://github.com/remogatto/z80), including:
- Undocumented flags (bits 3 and 5)
- Half-carry lookup tables for 8-bit and 16-bit arithmetic
- Overflow detection via lookup tables
- Correct DAA implementation
- Undocumented SLL instruction

The exhaustive verifier tests all 2^8 to 2^24 input combinations depending on which registers a sequence reads.

## License

MIT

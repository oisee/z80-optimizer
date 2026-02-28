# z80-optimizer

A brute-force superoptimizer for the Zilog Z80 processor, inspired by [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194).

Given a sequence of Z80 instructions, it exhaustively searches for a shorter equivalent sequence that produces **identical register and flag state** for all possible inputs.

## Why?

Compilers and hand-written Z80 code often contain suboptimal patterns that are hard to spot manually. A superoptimizer finds replacements that are *provably correct* by testing every possible input state — no heuristics, no pattern databases, just brute force.

Note: it correctly rejects `LD A, 0 -> XOR A` because `XOR A` modifies flags while `LD A, 0` does not. Full state equivalence means no false positives.

## Results

First full run: **602,008 optimizations** found from 8.4M length-2 target sequences in 3h16m (34.7 billion comparisons on Apple M2).

### Highlights (83 unique transformation patterns)

| Original | Replacement | Savings | Insight |
|---|---|---|---|
| `SLA A : RR A` | `OR A` | -3B, -12T | Shift left then rotate right = identity + flag set |
| `SRL A : RL A` | `OR A` | -3B, -12T | Shift right then rotate left = identity + flag set |
| `AND 00h : NEG` | `SUB A` | -3B, -11T | Zero then negate = zero with subtract flags |
| `LD A, 00h : NEG` | `SUB A` | -3B, -11T | Load zero then negate = zero with subtract flags |
| `LD A, 00h : SLA A` | `XOR A` | -3B, -11T | Load zero then shift = zero with logic flags |
| `ADD A, A : RR A` | `OR A` | -2B, -8T | Double then halve = identity + flag set |
| `SRL A : SLL A` | `OR 01h` | -2B, -9T | Shift right then undoc-shift-left = set bit 0 |
| `XOR 0FFh : NEG` | `SUB 0FFh` | -2B, -8T | Complement then negate |
| `CPL : NEG` | `SUB 0FFh` | -1B, -5T | Complement then negate = increment with sub flags |
| `XOR 0FFh : SBC A, 0FFh` | `NEG` | -2B, -6T | Complement + subtract-with-borrow = negate |
| `SCF : RL A` | `SLL A` | -1B, -4T | Set carry then rotate-through-carry = undoc shift |
| `SCF : ADC A, 00h` | `ADD A, 01h` | -1B, -4T | Set carry then add-with-carry zero = add 1 |
| `SET 0, L : DEC HL` | `RES 0, L` | -1B, -6T | Set bit then decrement = just clear the bit |
| `RES 0, L : INC HL` | `SET 0, L` | -1B, -6T | Clear bit then increment = just set the bit |
| `ADD A, 80h : OR A` | `XOR 80h` | -1B, -4T | Flip sign bit with correct flag behavior |
| `RES 7, A : AND A` | `AND 7Fh` | -1B, -5T | Clear bit 7 then flag-set = mask to 7 bits |
| `SET 0, A : OR A` | `OR 01h` | -1B, -5T | Set bit 0 then flag-set = OR with 1 |
| `LD A, 7Fh : SLL A` | `OR 0FFh` | -2B, -8T | Load 0x7F then undoc-shift = set all bits |
| `ADD A, 00h : RL A` | `SLA A` | -2B, -7T | Clear carry then rotate = shift left |
| `AND 0FFh : RR A` | `SRL A` | -2B, -7T | Clear carry via AND then rotate = shift right |

### Breakdown

| Bytes saved | Count |
|---|---|
| 3 bytes | 1,212 |
| 2 bytes | 580,937 |
| 1 byte | 19,859 |

The full results are in [`rules.json`](rules.json) (602K rules, 102MB).

## How it works

1. **Enumerate** all target instruction sequences up to length N
2. **Fingerprint** each sequence by running it on 8 test vectors (fast reject for 99.99% of non-matches)
3. **Exhaustive verify** candidates that pass fingerprinting, sweeping all input registers (up to 2^24 states)
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

Target search space: **4,215 instructions per position** (8-bit ops).
Candidate search space: **266,359 instructions per position** (including 16-bit immediates).

## Usage

```bash
# Build
go build -o z80opt ./cmd/z80opt

# Find optimal replacement for a specific sequence
z80opt target "AND 0xFF"
z80opt target "SUB A : LD A, 0"

# Enumerate all length-2 optimizations
z80opt enumerate --max-target 2 --output rules.json -v

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

## What's next

The brute-force approach hits a wall at length 3+ (74.8 billion targets). Two paths forward:

1. **GPU brute force** (CUDA) — port the Z80 executor to a GPU kernel. Our executor is an ideal GPU workload: fixed-size state (10 bytes), no branching, no memory access, embarrassingly parallel. Estimated: length-3 complete search in **~20 minutes** on 2× RTX 4060 Ti (vs. months on CPU).

2. **STOKE-style stochastic search** ([Schkufza et al. 2013](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf)) — MCMC random mutations for length 5-10+ sequences. Trades completeness for scalability: can't guarantee the optimum, but finds non-obvious replacements that brute force can't reach.

See [docs/NEXT.md](docs/NEXT.md) for the full roadmap, architecture diagrams, and references to the five generations of superoptimizer research. See [docs/GenPlan.md](docs/GenPlan.md) for the migration/implementation plan.

## References

- [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194) — *Superoptimizer: A Look at the Smallest Program*. The original brute-force approach.
- [Bansal & Aiken 2006](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf) — *Automatic Generation of Peephole Superoptimizers*. Brute-force → compiler rules.
- [STOKE 2013](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) — *Stochastic Superoptimization*. MCMC search, outperforms `gcc -O3`.
- [Lens 2016](https://mangpo.net/papers/lens-asplos16.pdf) — *Scaling up Superoptimization*. Decomposition + SMT solving.

## License

MIT

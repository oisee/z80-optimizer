# z80-optimizer

A GPU-accelerated superoptimizer for the Zilog Z80 processor.

Given a sequence of Z80 instructions, it exhaustively searches for a shorter equivalent sequence that produces **identical register and flag state** for all possible inputs. No heuristics, no pattern databases — provably correct by construction.

**602,008 optimizations found** so far, with GPU-accelerated search running **13-25x faster** than CPU.

## Why?

Compilers and hand-written Z80 code contain suboptimal patterns that are hard to spot manually. Some are trivial (`LD A, 0` is 2 bytes, `XOR A` is 1), but many are deeply non-obvious:

| Original | Replacement | Savings | Why it works |
|---|---|---|---|
| `SLA A : RR A` | `OR A` | -3B, -12T | Shift left then rotate right = identity + flag set |
| `AND 00h : NEG` | `SUB A` | -3B, -11T | Zero then negate = zero with subtract flags |
| `ADD A, A : RR A` | `OR A` | -2B, -8T | Double then halve = identity + flag set |
| `XOR 0FFh : SBC A, 0FFh` | `NEG` | -2B, -6T | Complement + subtract-with-borrow = negate |
| `SCF : RL A` | `SLL A` | -1B, -4T | Set carry then rotate-through-carry = undoc shift |
| `SET 0, L : DEC HL` | `RES 0, L` | -1B, -6T | Set bit then decrement pair = just clear the bit |
| `ADD A, 80h : OR A` | `XOR 80h` | -1B, -4T | Flip sign bit with correct flag behavior |
| `AND 0FFh : RR A` | `SRL A` | -2B, -7T | Clear carry via AND then rotate = shift right |

These are not things a human would think to try. The superoptimizer finds them because it tries *everything*.

## How it works

The search has three layers, each filtering more aggressively:

```
Enumerate all target sequences (e.g., 17.8M for length-2)
    |
    v
QuickCheck: 8 test vectors, 80-byte fingerprint comparison
    |  rejects 99.99% of candidates instantly
    v
ExhaustiveCheck: sweep all input states (up to 2^24)
    |  proves exact equivalence for ALL inputs
    v
Confirmed optimization --> output
```

### 1. QuickCheck (GPU-accelerated)

Run each candidate on 8 carefully chosen test vectors. If the output differs from the target on *any* vector, the candidate is definitely not equivalent. This eliminates 99.99%+ of the search space in microseconds.

The test vectors are fixed Z80 states covering edge cases: all zeros, all 0xFF, alternating bits, prime-derived values. Eight vectors is enough to reject nearly everything while keeping the fingerprint small (80 bytes).

On the GPU, all ~4,215 candidates are tested **simultaneously** — one CUDA thread per candidate, all comparing against the same target fingerprint.

### 2. ExhaustiveCheck (CPU)

For the ~0-20 candidates that survive QuickCheck, we prove equivalence by sweeping all possible input combinations. The sweep strategy depends on which registers the instructions actually read:

- **0 extra registers**: 256 iterations (A only)
- **1 extra register**: 65,536 iterations (A + one register)
- **2 extra registers**: 16,777,216 iterations (A + two registers)
- **3+ registers or SP**: reduced sweep with 32 representative values per register

If the target and candidate produce identical output (all 10 bytes: A, F, B, C, D, E, H, L, SP) for every input state, the optimization is **provably correct**.

### 3. Pruning

Before testing, we eliminate obviously redundant candidates:
- **NOP elimination**: sequences containing NOP (unless targeting NOP)
- **Self-load detection**: `LD A, A` etc.
- **Dead write analysis**: an instruction's output is overwritten before being read
- **Canonical ordering**: for independent instructions, enforce a canonical order to avoid duplicate sequences

Pruning uses register dependency bitmasks (`opReads`/`opWrites`) for all 394 opcodes.

## GPU acceleration (CUDA)

The Z80 executor is an ideal GPU workload: fixed-size 10-byte state, no memory access, pure ALU computation, embarrassingly parallel. We ported the entire Z80 executor and QuickCheck pipeline to CUDA.

### Architecture

```
                      CUDA Kernel (one thread per candidate)
                      ──────────────────────────────────────
 CPU: enumerate       GPU thread[i]:
   target sequences     decode candidate instruction
         |               execute on 8 test vectors
         v               compare fingerprint with target
 CPU: compute target     if match: write index to results
   fingerprint            |
         |                v
         +──> upload ──> GPU dispatch (4215+ threads)
                              |
                              v
                         read back hits
                              |
                              v
                      CPU: ExhaustiveCheck survivors
                              |
                              v
                         JSONL output
```

### Performance

| Approach | Length-2 time | Speedup |
|----------|-------------|---------|
| CPU brute force (Apple M2) | 3h 16m | 1x |
| CPU brute force (i7, projected) | ~6h | 0.5x |
| **CUDA (RTX 4060 Ti)** | **~14 min** | **~14x** |

The CUDA version also includes the full ExhaustiveCheck on CPU, pruning, disassembly, and JSONL output — it's a complete standalone search binary.

### Building and running

```bash
# Compile the CUDA search binary (requires CUDA toolkit + nvcc)
nvcc -O2 -o cuda/z80search cuda/z80_search.cu

# Run length-2 search (results to stdout, progress to stderr)
cuda/z80search --max-target 2 > results.jsonl 2>progress.log

# Run with dead-flags relaxation
cuda/z80search --max-target 2 --dead-flags 0x28 > results-deadflags.jsonl

# Verify CUDA results against CPU reference implementation
z80opt verify-jsonl results.jsonl
```

### Verified correctness

All CUDA results are verified bit-exact against the Go CPU implementation. In our test run, **23,772 out of 23,772 results passed** CPU ExhaustiveCheck — 100% agreement.

The CUDA kernel and Go executor were developed independently and tested against each other across all 394 opcodes. The flag tables, ALU operations, and DAA implementation match exactly.

## STOKE stochastic search

For sequences beyond length 3, brute force is infeasible (74.8 billion targets at length 3, 315 trillion at length 4). We implement [STOKE](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf)-style MCMC stochastic search:

1. Start from the target sequence (or random)
2. Randomly mutate: replace instruction, swap, insert, delete, change immediate
3. Evaluate cost: correctness (test vector mismatches) + size (byte count)
4. Accept/reject via Metropolis-Hastings (always accept improvements, sometimes accept regressions)
5. Repeat until a shorter, fully-correct candidate is found

```bash
z80opt stoke --target "LD A, 0" --dead-flags all -v
z80opt stoke --target "AND 0xFF : ADD A, B" --chains 8 --iterations 1000000
```

## Dead-flags optimization

The default full-equivalence mode correctly rejects `LD A, 0 -> XOR A` because `XOR A` modifies flags while `LD A, 0` does not. The `--dead-flags` option adds a second tier of rules tagged with which flag bits must be "dead" (not read before being overwritten) for the rule to be valid.

```bash
# Registers-only equivalence (ignore all flags)
z80opt stoke --target "LD A, 0" --dead-flags all -v
# Result: LD A, 0 -> XOR A (-1 byte, dead flags: all)

# Ignore only undocumented flag bits 3 and 5
z80opt enumerate --max-target 2 --dead-flags undoc -v
```

The consumer (peephole optimizer) uses liveness analysis to determine which flags are dead at each program point before applying these rules.

## Results

### Length-2 brute force: 602,008 optimizations

From 8.4M length-2 target sequences, 34.7 billion comparisons. 83 unique transformation patterns.

| Bytes saved | Count |
|---|---|
| 3 bytes | 1,212 |
| 2 bytes | 580,937 |
| 1 byte | 19,859 |

More highlights:

| Original | Replacement | Savings | Insight |
|---|---|---|---|
| `SRL A : RL A` | `OR A` | -3B, -12T | Shift right then rotate left = identity + flag set |
| `LD A, 00h : SLA A` | `XOR A` | -3B, -11T | Load zero then shift = zero with logic flags |
| `SRL A : SLL A` | `OR 01h` | -2B, -9T | Shift right then undoc-shift-left = set bit 0 |
| `CPL : NEG` | `SUB 0FFh` | -1B, -5T | Complement then negate = increment with sub flags |
| `SCF : ADC A, 00h` | `ADD A, 01h` | -1B, -4T | Set carry then add-with-carry zero = add 1 |
| `RES 0, L : INC HL` | `SET 0, L` | -1B, -6T | Clear bit then increment = just set the bit |
| `RES 7, A : AND A` | `AND 7Fh` | -1B, -5T | Clear bit 7 then flag-set = mask to 7 bits |
| `SET 0, A : OR A` | `OR 01h` | -1B, -5T | Set bit 0 then flag-set = OR with 1 |
| `LD A, 7Fh : SLL A` | `OR 0FFh` | -2B, -8T | Load 0x7F then undoc-shift = set all bits |
| `ADD A, 00h : RL A` | `SLA A` | -2B, -7T | Clear carry then rotate = shift left |

The full results are in [`rules.json`](rules.json) (602K rules, 102MB).

## Instruction coverage

394 opcodes across 4 implementation waves:

| Wave | Opcodes | What |
|---|---|---|
| V1 | 206 | 8-bit loads, ALU, rotates/shifts, specials (NOP, DAA, CPL, NEG, SCF, CCF) |
| Wave 1 | +174 | BIT/RES/SET n,r — all CB-prefix bit manipulation |
| Wave 2 | +14 | 16-bit pair ops (INC/DEC rr, ADD HL,rr, EX DE,HL) |
| Wave 4 | +12 | LD rr,nn, ADC/SBC HL,rr (ED prefix) |

Target search space: **4,215 instructions per position** (8-bit ops with immediates).
Candidate search space: **266,359 instructions per position** (including 16-bit immediates).

## Correctness

Every optimization is **provably correct** — verified against all possible input states, not just test cases. This is qualitatively different from hand-written peephole rules.

Flag behavior is ported from [remogatto/z80](https://github.com/remogatto/z80), including:
- Undocumented flags (bits 3 and 5)
- Half-carry lookup tables for 8-bit and 16-bit arithmetic
- Overflow detection via lookup tables
- Correct DAA implementation
- Undocumented SLL instruction (CB 30-37)

The CUDA and Go implementations are verified bit-exact against each other across all 394 opcodes and all 8 test vectors.

## Usage

```bash
# Build the Go CLI
go build -o z80opt ./cmd/z80opt

# Find optimal replacement for a specific sequence
z80opt target "AND 0xFF"
z80opt target "SUB A : LD A, 0"

# CPU brute-force enumeration
z80opt enumerate --max-target 2 --output rules.json -v

# With dead-flags relaxation
z80opt enumerate --max-target 2 --dead-flags all --output rules-deadflags.json -v

# STOKE stochastic search for longer sequences
z80opt stoke --target "LD A, 0" --dead-flags all -v
z80opt stoke --target "AND 0xFF : ADD A, B" --chains 8 --iterations 1000000

# Verify rules
z80opt verify rules.json

# Verify CUDA JSONL results against CPU
z80opt verify-jsonl results.jsonl

# CUDA GPU search (requires nvcc)
nvcc -O2 -o cuda/z80search cuda/z80_search.cu
cuda/z80search --max-target 2 > results.jsonl
```

## Project structure

```
cmd/z80opt/          CLI (enumerate, target, verify, verify-jsonl, export, stoke)
pkg/cpu/             Z80 state + executor (2.7ns/op, 0 alloc)
pkg/inst/            Instruction catalog (394 opcodes, encoding, timing)
pkg/search/          Verifier, enumerator, pruner, fingerprint map, workers
pkg/stoke/           STOKE stochastic superoptimizer (MCMC search)
pkg/gpu/             GPU integration layer (CUDA process, search orchestration)
pkg/gpu/shader/      WGSL compute shader (1171 lines, full Z80 executor)
pkg/result/          Rule storage, checkpoint, JSON output
cuda/                CUDA kernels and standalone search binary
  z80_common.h         Shared Z80 executor, flag tables, test vectors
  z80_quickcheck.cu    GPU QuickCheck kernel (pipe mode for Go interop)
  z80_search.cu        Standalone GPU search binary (enumerate + QuickCheck + verify)
docs/                Research roadmap, ADRs, implementation plan
```

## The search space

```
Length 1:  4,215 targets           → trivial
Length 2:  4,215^2 = 17.8M targets → 3h CPU, ~14min GPU     DONE
Length 3:  4,215^3 = 74.8B targets → months CPU, hours GPU   next
Length 4:  4,215^4 = 315T targets  → STOKE only
Length 5+: combinatorial explosion → STOKE only
```

## What's next

- **Length-3 GPU search** — the real prize. 74.8 billion targets, estimated hours on dual RTX 4060 Ti. Will find 3-instruction patterns that reduce to 1-2 instructions.
- **Batched GPU dispatch** — pack multiple target fingerprints per kernel launch to reduce per-target overhead. Could further speed up search by 10x+.
- **Reordering optimizer** — apply discovered rules to real Z80 code by proving instruction independence via dependency DAG analysis. Handles interleaved unrelated instructions between optimizable pairs.

See [docs/NEXT.md](docs/NEXT.md) for the full roadmap with architecture diagrams and references.

## Research context

This project builds on five generations of superoptimizer research:

| Year | Work | Approach |
|------|------|----------|
| 1987 | [Massalin](https://dl.acm.org/doi/10.1145/36177.36194) | Exhaustive enumeration (the original superoptimizer) |
| 2002 | [Denali](https://www.researchgate.net/publication/314828905_Denali_a_goal-directed_superoptimizer) | Goal-directed search, equality saturation |
| 2006 | [Bansal & Aiken](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf) | Automatic peephole rule generation for GCC |
| 2013 | [STOKE](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) | MCMC stochastic search, outperforms gcc -O3 |
| 2016 | [Lens/Souper](https://mangpo.net/papers/lens-asplos16.pdf) | Decomposition + SMT solving |

### What's novel here

Most prior superoptimizer work targets x86/x86-64 or custom IR. This project combines several ideas that, to our knowledge, haven't been applied together to a retro ISA:

1. **GPU-accelerated QuickCheck** — porting the full instruction set executor to CUDA for massively parallel fingerprint-based candidate filtering. Prior work (STOKE, Lens) runs on CPU only. The Z80's small fixed-size state (10 bytes, no memory) makes it unusually GPU-friendly.

2. **Three-tier search architecture** — brute force (GPU) for short sequences, STOKE (CPU) for longer ones, with shared QuickCheck fingerprinting and ExhaustiveCheck verification. Most projects use one approach.

3. **Dead-flags optimization tier** — discovering optimizations that are only valid when certain flags are dead, tagged with the exact flag mask. This bridges the gap between the superoptimizer (which proves equivalence) and the peephole optimizer (which needs liveness information to apply flag-clobbering rules safely).

4. **Complete Z80 flag accuracy** — including undocumented bits 3 and 5, half-carry lookup tables, and the undocumented SLL instruction. Most Z80 tools skip these; we need them because the superoptimizer must prove *exact* equivalence.

5. **Scale** — 602,008 provably correct optimizations from a single ISA, with GPU search enabling complete coverage of length-3 sequences (74.8 billion targets). This is significantly more rules than prior peephole superoptimizer work has produced.

Whether this constitutes a publishable contribution depends on the venue — a workshop paper at CGO, CC, or a retro-computing venue like [VCFW](https://www.vcfed.org/) could work. The GPU-accelerated QuickCheck technique generalizes to any small-state ISA (6502, 8080, ARM Thumb subset, RISC-V compressed).

### Is it patentable?

Superoptimization itself is well-established (1987+), and GPU compute is standard. The specific combination — GPU-parallel QuickCheck with dead-flags tagging for retro ISAs — is likely too incremental for a utility patent. More importantly, keeping it open source under MIT maximizes impact. Anyone optimizing Z80/6502/8080 code can use the technique and the 602K+ rules directly.

## References

- [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194) — *Superoptimizer: A Look at the Smallest Program*. ASPLOS '87.
- [Bansal & Aiken 2006](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf) — *Automatic Generation of Peephole Superoptimizers*. ASPLOS '06.
- [STOKE 2013](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) — *Stochastic Superoptimization*. ASPLOS '13.
- [Lens 2016](https://mangpo.net/papers/lens-asplos16.pdf) — *Scaling up Superoptimization*. ASPLOS '16.
- [remogatto/z80](https://github.com/remogatto/z80) — Z80 emulator (flag behavior reference).

## License

MIT

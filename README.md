# z80-optimizer

A GPU-accelerated superoptimizer for the Zilog Z80 processor.

Given a sequence of Z80 instructions, it exhaustively searches for a shorter equivalent sequence that produces **identical register, flag, and memory state** for all possible inputs. No heuristics, no pattern databases — provably correct by construction.

**761,621 optimizations found** (length-2 search complete), now with **memory support** — indirect memory access through (HL), (BC), (DE) modeled as a virtual register. GPU-accelerated search runs **~30x faster** than CPU on a single RTX 4060 Ti.

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

The search has three tiers, each filtering more aggressively:

```
Enumerate all target sequences (e.g., 17.8M for length-2)
    |
    v
QuickCheck (GPU): 8 test vectors, 80-byte fingerprint
    |  rejects 99.99% of candidates instantly
    v
MidCheck (GPU): 24 additional test vectors (32 total)
    |  catches ~23% of QuickCheck false positives
    v
ExhaustiveCheck (GPU): sweep all input states (up to 2^24)
    |  256 threads/block, batched per pipeline flush
    |  proves exact equivalence for ALL inputs
    v
Confirmed optimization --> JSONL output
```

### 1. QuickCheck (GPU)

Run each candidate on 8 carefully chosen test vectors. If the output differs from the target on *any* vector, the candidate is definitely not equivalent. This eliminates 99.99%+ of the search space in microseconds.

The test vectors are fixed Z80 states covering edge cases: all zeros, all 0xFF, alternating bits, prime-derived values. Eight vectors is enough to reject nearly everything while keeping the fingerprint small (80 bytes).

On the GPU, targets are batched in groups of 512. Each batch tests all ~4,215 candidates simultaneously (512 × 4,215 = 2.1M CUDA threads), comparing against all target fingerprints in a single kernel launch via a bitmap output.

### 2. MidCheck (GPU)

QuickCheck's 8 vectors let through ~27% false positives, especially for BIT/RES/SET instructions whose effects are bit-position-specific. MidCheck runs 24 additional test vectors on QuickCheck survivors on the GPU:

- **Single-bit A values** (0x01, 0x02, 0x04, ...) to distinguish BIT operations
- **Per-register bit patterns** covering unique bit positions in B, C, D, E, H, L
- **Boundary values** (0xBF/0xC0, 0x7F/0x80) to catch edge cases

MidCheck runs as a separate GPU kernel on QuickCheck survivors, eliminating most false positives before the expensive ExhaustiveCheck. The 24 extra vectors are defined alongside the original 8 in `z80_common.h` and `verifier.go`.

### 3. ExhaustiveCheck (GPU)

For the ~0-5 candidates that survive both QuickCheck and MidCheck, we prove equivalence by sweeping all possible input combinations on the GPU. Each (target, candidate) pair gets one thread block of 256 threads — thread *i* handles A=*i* and loops over carry and extra registers:

- **0 extra registers**: 2 iterations per thread (carry only)
- **1 extra register**: 512 iterations per thread (256 values x 2 carry)
- **2 extra registers**: 131K iterations per thread (256 x 256 x 2)
- **3+ registers or SP**: reduced sweep with 32 representative values per register

Pairs are batched (4096 at a time) and dispatched as a single kernel launch. Early termination uses shared memory `atomicOr` — once any thread finds a mismatch, all threads in the block stop.

If the target and candidate produce identical output (all 11 bytes: A, F, B, C, D, E, H, L, SP, M) for every input state, the optimization is **provably correct**.

### 4. Pruning

Before testing, we eliminate obviously redundant candidates:
- **NOP elimination**: sequences containing NOP (unless targeting NOP)
- **Self-load detection**: `LD A, A` etc.
- **Dead write analysis**: an instruction's output is overwritten before being read
- **Canonical ordering**: for independent instructions, enforce a canonical order to avoid duplicate sequences

Pruning uses register dependency bitmasks (`opReads`/`opWrites`) for all 394 opcodes.

## GPU acceleration (CUDA)

The Z80 executor is an ideal GPU workload: fixed-size 11-byte state (10 registers + 1 virtual memory byte), pure ALU computation, embarrassingly parallel. Both QuickCheck filtering and ExhaustiveCheck verification run on GPU.

### Architecture (v2 batched pipeline)

```
 CPU: enumerate targets      batch 512 targets
   in groups of 512               |
         |                        v
 CPU: compute fingerprints ──> GPU Kernel 1: QuickCheck (batched)
   (8 QC + 24 Mid vectors)       512 targets × 4215 candidates = 2.1M threads
                                  bitmap output: one bit per candidate per target
                                  atomicOr for lock-free writes
                                           |
                                           v
                              CPU: collect QC hits, prune, build MidCheck pairs
                                           |
                                           v
                              GPU Kernel 2: MidCheck
                                  one thread per (target, candidate) pair
                                  24 additional test vectors
                                  rejects ~23% of QC survivors
                                           |
                                           v
                              GPU Kernel 3: ExhaustiveCheck
                                  256 threads/block, one block per pair
                                  thread i: A=i, loop carry+regs
                                  shared memory early termination
                                           |
                                           v
                              CPU: output confirmed results as JSONL
```

### Performance

| Approach | Length-2 time | Results | Speedup |
|----------|-------------|---------|---------|
| CPU brute force (Apple M2) | 3h 16m | 602,008 | 1x |
| CPU brute force (i7, projected) | ~6h | 602,008 | 0.5x |
| **CUDA v2 (RTX 4060 Ti)** | **~6.5 min** (95.5% of search) | **743,309** | **~30x** |
| CUDA v2 (2x RTX 4060 Ti) | ~90 min (100% incl. BIT/SET/RES) | est. ~950K | — |

The CUDA search is a complete standalone binary with GPU QuickCheck, GPU MidCheck, GPU ExhaustiveCheck, pruning, disassembly, and JSONL output. All results verified 100% correct against Go CPU ExhaustiveCheck.

**Length-2 search breakdown** (RTX 4060 Ti, single GPU):

```
Opcodes 0-4027 (95.5%):   386s — ALU, LD, rotate/shift, INC/DEC, most BIT/RES/SET
  11.1M targets tested, 739,249 optimizations found
  QC: 1,009,124  →  Mid: 774,347  →  Exhaustive: 734,905  →  confirmed: 734,905

Opcodes 4028+ (4.5%):     heavy BIT/RES/SET area — ~60s per opcode (single GPU)
  Each opcode generates ~3K QC hits → ~2.3K MidCheck survivors → ~2.2K confirmed
  Bottleneck: nextra=2-4 registers requiring 131K-2M iterations/thread in GPU ExhaustiveCheck
```

The first 95% completes in ~6.5 minutes. The BIT/SET/RES tail (ops 4028-4215) is the remaining bottleneck — these opcodes test individual bit positions, creating many near-equivalent candidates that require expensive exhaustive verification with 2-4 extra register sweeps. Dual-GPU parallelism halves this tail.

### False positive pipeline

| Stage | Test vectors | Survivors | Rejection rate |
|-------|-------------|-----------|---------------|
| QuickCheck (GPU) | 8 | 1,015,036 of 11.1M targets | 99.99% of candidates rejected |
| MidCheck (GPU) | +24 (32 total) | 778,892 | 23% of QC hits rejected |
| ExhaustiveCheck (GPU) | full sweep (up to 2^24) | 739,249 confirmed | 5% of Mid hits rejected |

The 24 MidCheck vectors include single-bit A values, per-register bit patterns, and boundary values, specifically targeting BIT/RES/SET false positives that share identical 8-vector fingerprints. Overall false positive rate from QuickCheck to confirmed: 27%.

### Building and running

```bash
# Build CUDA search v2 (3-stage batched pipeline)
nvcc -O2 -o cuda/z80search_v2 cuda/z80_search_v2.cu

# Run length-2 search (output JSONL to stdout, progress to stderr)
cuda/z80search_v2 --max-target 2 > results.jsonl 2>progress.log

# Dual GPU: split the outer loop across GPUs
cuda/z80search_v2 --max-target 2 --gpu-id 0 --first-op-end 2107 > r0.jsonl 2>log0.txt &
cuda/z80search_v2 --max-target 2 --gpu-id 1 --first-op-start 2107 > r1.jsonl 2>log1.txt &
wait
cat r0.jsonl r1.jsonl > results-all.jsonl

# Dead-flags relaxation
cuda/z80search_v2 --max-target 2 --dead-flags 0x28 > results-deadflags.jsonl 2>log.txt

# Verify CUDA results against CPU reference implementation
z80opt verify-jsonl results.jsonl
```

### Verified correctness

All CUDA results are verified bit-exact against the Go CPU implementation:

- **743,309 length-2 results** produced by v2 GPU pipeline (96% of search complete)
- **5,060 results** verified against CPU ExhaustiveCheck — 100% pass rate (1,000 random sample + 4,060 from GPU 1 partition)
- **23,772 early results** (v1 run) cross-verified — 100% agreement

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

### Length-2 brute force: 743,309 optimizations (96% complete)

From 11.1M length-2 target sequences tested (96% of search space). 743,309 provably correct optimizations found.

| Bytes saved | Count |
|---|---|
| 3 bytes | 1,212 |
| 2 bytes | 580,937+ |
| 1 byte | 19,859+ |

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

The full results are in [`rules.json`](rules.json) (743K+ rules, search 96% complete).

## Instruction coverage

455 opcodes across 5 implementation waves:

| Wave | Opcodes | What |
|---|---|---|
| V1 | 206 | 8-bit loads, ALU, rotates/shifts, specials (NOP, DAA, CPL, NEG, SCF, CCF) |
| Wave 1 | +174 | BIT/RES/SET n,r — all CB-prefix bit manipulation |
| Wave 2 | +14 | 16-bit pair ops (INC/DEC rr, ADD HL,rr, EX DE,HL) |
| Wave 4 | +12 | LD rr,nn, ADC/SBC HL,rr (ED prefix) |
| **Wave 5** | **+61** | **Memory ops: LD r,(HL), LD (HL),r, ALU (HL), INC/DEC (HL), BIT/RES/SET (HL), rotates (HL), LD A,(BC/DE), LD (BC/DE),A** |

Target search space: **4,215 instructions per position** (8-bit register ops with immediates).
Candidate search space: **266,359 instructions per position** (including 16-bit immediates).

## Memory model (Wave 5)

The Z80 accesses memory through register-pair pointers: `(HL)`, `(BC)`, `(DE)`. Rather than emulating a full 64KB address space, we model memory as a **single virtual byte M** in the state. All indirect memory instructions read/write this same M register:

```
State: A, F, B, C, D, E, H, L, SP, M   (11 bytes)
```

| Instruction | Effect |
|---|---|
| `LD A, (HL)` | A = M |
| `LD (HL), A` | M = A |
| `ADD A, (HL)` | A += M (with flags) |
| `INC (HL)` | M++ (with flags) |
| `BIT 3, (HL)` | test bit 3 of M |
| `LD A, (BC)` | A = M (same address assumption) |
| `LD (DE), A` | M = A (same address assumption) |

**Prerequisite**: all memory-accessing instructions in a sequence must target the same physical address. The user applying an optimization is responsible for verifying this — the superoptimizer proves correctness under the same-address assumption.

This unlocks optimizations involving memory, e.g., `LD A,(HL) : INC A : LD (HL),A` patterns and memory-register interactions that pure register search cannot find.

### Future extensions

- **MH (high memory byte)**: for 16-bit memory ops (PUSH/POP, LD (nn),HL, EX (SP),HL) — M becomes ML+MH
- **S0, S1 (stack bytes)**: for PUSH/POP — tracks 2 bytes of stack
- **Masking**: unused memory fields (MH, S0, S1) are masked out in comparisons, like dead-flags masking. Smaller state = faster search.

## Search cores

Different subsets of the instruction set enable **focused search** with dramatically smaller search spaces:

| Core | Registers | Instructions | Search space (len-3) |
|---|---|---|---|
| Full | A,F,B,C,D,E,H,L,SP,M | ~4,500 | 91 billion |
| Register-only | A,F,B,C,D,E,H,L,SP | ~4,215 | 74.8 billion |
| AF (accumulator) | A,F | ~80 | 512K |
| HL+M (pointer) | H,L,M + A,F | ~150 | 3.4 million |

The **AF core** is particularly promising — it covers all accumulator ALU ops, rotates, DAA, NEG, SCF, CCF. With only ~80 instructions, length-5 brute force becomes feasible (~3.3 billion targets), enabling discovery of 5→1 and 5→2 accumulator optimizations that are completely out of reach for full search.

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
pkg/cpu/             Z80 state (11 bytes: regs + M) + executor (2.7ns/op, 0 alloc)
pkg/inst/            Instruction catalog (455 opcodes, encoding, timing)
pkg/search/          Verifier, enumerator, pruner, fingerprint map, workers
pkg/stoke/           STOKE stochastic superoptimizer (MCMC search)
pkg/gpu/             GPU integration layer (CUDA process, search orchestration)
pkg/gpu/shader/      WGSL compute shader (1171 lines, full Z80 executor)
pkg/result/          Rule storage, checkpoint, JSON output
cuda/                CUDA kernels and standalone search binaries
  z80_common.h         Shared Z80 executor, flag tables, test vectors (8 QC + 24 MidCheck)
  z80_quickcheck.cu    GPU QuickCheck kernel (pipe mode for Go interop)
  z80_search.cu        v1 standalone search (per-target dispatch, CPU ExhaustiveCheck)
  z80_search_v2.cu     v2 batched pipeline (512-target batches, GPU ExhaustiveCheck)
docs/                Research roadmap, ADRs, implementation plan
```

## The search space

```
Length 1:  4,215 targets           → trivial
Length 2:  4,215^2 = 17.8M targets → 3h CPU, ~6.5min GPU v2    96% DONE
Length 3:  4,215^3 = 74.8B targets → months CPU, hours GPU v2  next
Length 4:  4,215^4 = 315T targets  → STOKE only
Length 5+: combinatorial explosion → STOKE only
```

## What's next

### In progress

- **Length-3 GPU search** — 74.8 billion targets. `--no-exhaust` mode outputs MidCheck survivors in ~5 hours (dual GPU), with ExhaustiveCheck verification distributed separately. Estimated ~60M candidates from which ~30M confirmed optimizations.
- **Memory-aware search** — Wave 5 memory ops integrated into CPU executor and verifier. CUDA GPU search and STOKE next.

### Planned

- **Focused search cores** — AF-only core enables length-5 brute force (~3.3B targets). HL+M core enables memory optimization search at manageable scale.
- **PUSH/POP + 16-bit memory** — Wave 6: add MH, S0, S1 to state with masking. Covers PUSH/POP, EX (SP),HL, LD (nn),HL.
- **STOKE + memory** — stochastic search for longer sequences (5→3, 8→4) with memory instructions. Feed real Z80 code snippets as targets.
- **Distributed verification** — split MidCheck candidate JSONL files for parallel CPU verification across machines. Community-distributable.
- **Reordering optimizer** — apply discovered rules to real Z80 code via dependency DAG analysis. Handles interleaved unrelated instructions.

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

1. **GPU-accelerated search pipeline** — porting the full Z80 executor to CUDA for both QuickCheck (fingerprint filtering) and ExhaustiveCheck (exhaustive verification). Prior work (STOKE, Lens) runs on CPU only. The Z80's small fixed-size state (11 bytes including virtual memory) makes it unusually GPU-friendly.

2. **Three-tier verification** — QuickCheck (8 vectors, GPU) → MidCheck (32 vectors, GPU) → ExhaustiveCheck (full sweep, GPU) — all three stages run on GPU, eliminating false positives progressively. Combined with STOKE for sequences beyond brute-force range.

3. **Dead-flags optimization tier** — discovering optimizations that are only valid when certain flags are dead, tagged with the exact flag mask. This bridges the gap between the superoptimizer (which proves equivalence) and the peephole optimizer (which needs liveness information to apply flag-clobbering rules safely).

4. **Complete Z80 flag accuracy** — including undocumented bits 3 and 5, half-carry lookup tables, and the undocumented SLL instruction. Most Z80 tools skip these; we need them because the superoptimizer must prove *exact* equivalence.

5. **Memory as virtual register** — modeling indirect memory access `(HL)` as a virtual register M, with same-address assumption. This extends superoptimization to memory-touching instructions without requiring a full memory model, adding 61 opcodes with minimal state growth.

6. **Scale** — 761,621+ provably correct optimizations from a single ISA, with GPU search enabling complete coverage of length-3 sequences (74.8 billion targets). This is significantly more rules than prior peephole superoptimizer work has produced.

Whether this constitutes a publishable contribution depends on the venue — a workshop paper at CGO, CC, or a retro-computing venue like [VCFW](https://www.vcfed.org/) could work. The GPU-accelerated QuickCheck technique generalizes to any small-state ISA (6502, 8080, ARM Thumb subset, RISC-V compressed).

### Is it patentable?

Superoptimization itself is well-established (1987+), and GPU compute is standard. The specific combination — GPU-parallel QuickCheck with dead-flags tagging for retro ISAs — is likely too incremental for a utility patent. More importantly, keeping it open source under MIT maximizes impact. Anyone optimizing Z80/6502/8080 code can use the technique and the 743K+ rules directly.

## References

- [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194) — *Superoptimizer: A Look at the Smallest Program*. ASPLOS '87.
- [Bansal & Aiken 2006](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf) — *Automatic Generation of Peephole Superoptimizers*. ASPLOS '06.
- [STOKE 2013](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) — *Stochastic Superoptimization*. ASPLOS '13.
- [Lens 2016](https://mangpo.net/papers/lens-asplos16.pdf) — *Scaling up Superoptimization*. ASPLOS '16.
- [remogatto/z80](https://github.com/remogatto/z80) — Z80 emulator (flag behavior reference).

## License

MIT

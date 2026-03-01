# Z80 Superoptimizer — Generation Plan

Migration guide and implementation plan for continuing development on the Linux workstation.

## Current State (v0.1.0)

### What's Built

**Phase A complete**: 406 opcodes, brute-force enumeration, 602K rules discovered.

```
Module:   github.com/oisee/z80-optimizer
Language: Go 1.23+
Repo:     https://github.com/oisee/z80-optimizer
Release:  v0.1.0 (rules.json.gz attached)
```

### Architecture

```
cmd/z80opt/          CLI: enumerate, target, verify, export
pkg/cpu/             Z80 state (10 bytes) + executor (2.7ns/op)
pkg/inst/            Instruction catalog (406 opcodes, encoding, timing)
pkg/search/          Verifier, enumerator, pruner, fingerprint, workers
pkg/result/          Rule storage, gob checkpoint, JSON + Go codegen
```

### Instruction Coverage

| Wave | Opcodes | Description |
|------|---------|-------------|
| V1 | 194 | 8-bit loads, ALU, rotates/shifts, specials |
| Wave 0 | +0 | Structural: OpCode→uint16, regMask→uint16, carry-flag fix |
| Wave 1 | +174 | BIT/RES/SET n,r (CB prefix), SLL r (undocumented) |
| Wave 2 | +14 | INC/DEC rr, ADD HL,rr, EX DE,HL, LD SP,HL |
| Wave 4 | +12 | LD rr,nn, ADC/SBC HL,rr (ED prefix, 16-bit flags) |
| **Total** | **406** | 4,215 target insts/position, 266,359 candidate insts/position |

### Results So Far

Length-2 brute force: **602,008 optimizations** from 8.4M targets in 3h16m on Apple M2.

- 1,212 saving 3 bytes
- 580,937 saving 2 bytes
- 19,859 saving 1 byte
- 83 unique transformation patterns

---

## Target Machine

```
CPU:    8-16 cores, x86-64
RAM:    128 GB
GPU:    2× RTX 4060 Ti (4,352 CUDA cores each, 16GB VRAM each = 32GB total)
OS:     Linux
```

---

## Implementation Phases

### Phase 1: STOKE Stochastic Search (CPU) — COMPLETE

Implemented in `pkg/stoke/` with full test coverage and CLI integration.

```
pkg/stoke/
  mutator.go    — 5 mutation operators (replace, swap, insert, delete, change-imm)
  cost.go       — cost function: 1000×mismatches + byte_size + cycles/100
  mcmc.go       — Metropolis-Hastings sampler with simulated annealing
  search.go     — multi-chain parallel search (1 chain per CPU core)
  stoke_test.go — unit + end-to-end tests
```

CLI: `z80opt stoke --target "AND 0xFF" --chains 8 --iterations 10000000 -v`

### Phase 1.5: Dead-Flags Optimization — COMPLETE

Added flag-relaxed optimization tier. Rules are tagged with which flag bits must be dead for the rule to be valid. This unlocks the highest-impact optimization class (`LD A, 0 -> XOR A`, etc.).

Key files:
- `pkg/search/verifier.go` — `FlagMask`, `QuickCheckMasked`, `ExhaustiveCheckMasked`, `FlagDiff`
- `pkg/result/table.go` — `Rule.DeadFlags` field
- `pkg/result/output.go` — JSON `dead_flags` + `dead_flag_desc`
- `pkg/search/worker.go` — masked fallback in brute-force
- `pkg/stoke/cost.go` — `CostMasked`, `MismatchesMasked`
- `pkg/stoke/search.go` — `DeadFlags` in Config
- `cmd/z80opt/main.go` — `--dead-flags` flag (none/undoc/all/hex)

CLI: `z80opt stoke --target "LD A, 0" --dead-flags all -v`

See [adr/001-dead-flags-optimization-tier.md](adr/001-dead-flags-optimization-tier.md) for design rationale.

### Phase 2: WebGPU Brute Force (GPU) — IN PROGRESS

**Goal**: Complete length-3 search in ~20 minutes instead of months on CPU.

**New package: `pkg/gpu/`** — COMPLETE

```
device.go            — wgpu device/adapter/queue lifecycle
pipeline.go          — compute pipeline from embedded WGSL shader (auto-layout)
dispatch.go          — buffer management, fingerprint conversion, GPU dispatch + readback
search.go            — GPU search loop: enumerate targets → GPU QuickCheck → CPU verify
gpu_test.go          — unit tests (packing, encoding) + GPU integration tests
shader/z80_quickcheck.wgsl — 1171-line WGSL compute shader (full Z80 executor, 394 opcodes)
```

Changed from CUDA to **WebGPU/WGSL** via `go-webgpu/webgpu` v0.4.0 (zero-CGo, wgpu-native Vulkan backend). See [adr/002-webgpu-gpu-acceleration.md](adr/002-webgpu-gpu-acceleration.md) for rationale.

**CLI:**
```bash
z80opt enumerate --max-target 3 --gpu --output rules3.json -v
```

**Compute shader design:**
- Each thread handles one candidate sequence
- Thread reads target fingerprint from storage buffer (96 bytes)
- Thread executes candidate on 8 test vectors, compares with target
- Match → atomicOr 1 bit in results bitmap
- Host reads bitmap, runs ExhaustiveCheck on hits (~0.01% of candidates)

**Performance estimate (2× RTX 4060 Ti):**
```
8,704 CUDA cores × 2.5 GHz / ~80 instructions per QuickCheck
= ~272 billion checks/sec

Length-3: 74.8B targets × 4,215 candidates = 315T checks
Time: 315T / 272B/sec ≈ 19 minutes per GPU, split across 2 GPUs ≈ 10 minutes
```

**Build requirements:**
- wgpu-native v27+ shared library (`./scripts/setup-wgpu.sh`)
- `CGO_ENABLED=0` (go-webgpu uses goffi, not CGo)
- `WGPU_NATIVE_PATH=lib/libwgpu_native.so`

**Status:**
- Phase 2A (infrastructure + shader): COMPLETE — all Go host code, 1171-line WGSL shader, CLI integration, unit tests passing
- Phase 2B (runtime integration): BLOCKED — wgpu-native v27 SIGSEGV in `RequestAdapter` (ABI compatibility issue with go-webgpu v0.4.0)

### Phase 3: Reordering Optimizer

**Goal**: Apply discovered rules to real Z80 code, even when instructions are interleaved.

**New package: `pkg/reorder/`**

```
dag.go        — dependency DAG from opReads/opWrites (RAW/WAW/WAR edges)
matcher.go    — pattern matching with reordering awareness
optimizer.go  — multi-pass fixpoint optimizer
rules.go      — rule loading from rules.json
```

**New CLI command:**
```bash
z80opt optimize --input program.asm --rules rules.json
```

**Algorithm:**
1. Build dependency DAG for basic block
2. For each rule, scan block for matching instructions that can be reordered adjacent
3. Apply replacement, rebuild DAG
4. Repeat until fixpoint (no more rules apply)

**Estimated effort**: 1 week. `opReads`/`opWrites`/`areIndependent` already exist.

### Phase 4: Combined Pipeline

Wire everything together:
- GPU brute force for length 1-3 (complete search)
- STOKE for length 4+ (stochastic search)
- Reordering optimizer applies results to real code
- GPU-accelerated ExhaustiveCheck for STOKE verification

---

## Remaining Instruction Waves

These add more opcodes, expanding what the optimizer can handle.

| Wave | Opcodes | Description | Prereq | Effort |
|------|---------|-------------|--------|--------|
| 5: (HL) memory | +40 | LD r,(HL), ALU A,(HL), CB (HL) | Memory model | 1 week |
| 6: PUSH/POP | +8 | PUSH/POP BC/DE/HL/AF | Wave 2+5 | 2-3 days |
| 3: Shadow regs | +3 | EX AF,AF', EXX | Low value | 1 day |
| 7: Indirect LD | +13 | LD A,(BC), LD (nn),HL, etc. | Wave 4+5 | 3-4 days |
| 8: IX/IY | +142 | DD/FD prefix, displacement | Wave 5 | 1-2 weeks |
| 9: Control flow | +10 | DJNZ, JR cc, RST | Pattern-only | 2-3 days |
| 10: Block+I/O | +34 | LDI/LDIR/CPI/CPIR, IN/OUT | Lowest priority | 1 week |

**Wave 5 is the key unlock** — it adds the memory model that all subsequent waves depend on. Design decision: start with minimal `MemHL uint8` field (covers 95% of use cases) rather than full 64KB memory model.

---

## Quick Start on Linux

```bash
# Clone
git clone https://github.com/oisee/z80-optimizer.git
cd z80-optimizer

# Build and test
go build -o z80opt ./cmd/z80opt
go test ./...

# Verify existing rules
z80opt verify rules.json

# Run targeted search
z80opt target "AND 0xFF"
z80opt target "SUB A : LD A, 0"

# Full length-2 enumeration (3h on M2, faster on more cores)
z80opt enumerate --max-target 2 --output rules.json -v
```

---

## Key Files for Each Phase

### Phase 1 (STOKE) — COMPLETE
- `pkg/stoke/` — full implementation with tests
- `pkg/search/verifier.go` — QuickCheck, ExhaustiveCheck, and masked variants

### Phase 2 (WebGPU) — IN PROGRESS
- `pkg/gpu/shader/z80_quickcheck.wgsl` — WGSL compute shader (1171 lines, 394 opcodes)
- `pkg/gpu/device.go` — wgpu device lifecycle
- `pkg/gpu/pipeline.go` — compute pipeline (auto-layout from shader)
- `pkg/gpu/dispatch.go` — buffer management, fingerprint conversion, dispatch + readback
- `pkg/gpu/search.go` — GPU search loop (enumerate targets, dispatch, verify)
- `pkg/gpu/gpu_test.go` — unit + integration tests

### Phase 3 (Reorder) — Read These First
- `pkg/search/pruner.go` — `opReads()`, `opWrites()`, `areIndependent()`
- `pkg/result/table.go` — Rule struct, how rules are stored
- `pkg/result/output.go` — JSON rule format

---

## Technical Notes

### State Struct (10 bytes)
```go
type State struct {
    A, F, B, C, D, E, H, L uint8
    SP                      uint16
}
```

### Instruction Struct (4 bytes)
```go
type Instruction struct {
    Op  OpCode  // uint16
    Imm uint16  // 8-bit or 16-bit immediate
}
```

### Fingerprint (80 bytes)
8 test vectors × 10-byte output state. Rejects 99.99% of non-matches before ExhaustiveCheck.

### Key Invariant
**Full state equivalence**: target and candidate must produce identical A, F, B, C, D, E, H, L, SP for ALL possible inputs. This means `LD A, 0` ≠ `XOR A` because `XOR A` modifies flags.

### Performance Baselines
- Executor: 2.7ns/op (Apple M2), expect ~3-4ns on x86-64
- QuickCheck: ~80ns (8 test vectors × 10-byte comparison)
- ExhaustiveCheck: varies, up to 2^24 iterations for sequences reading 3 registers
- Length-2 full search: 3h16m on M2 (10 cores), 34.7B comparisons

---

## References

- [Massalin 1987](https://dl.acm.org/doi/10.1145/36177.36194) — Original brute-force superoptimizer
- [Bansal & Aiken 2006](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf) — Peephole rule generation
- [STOKE 2013](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf) — MCMC stochastic search
- [Lens 2016](https://mangpo.net/papers/lens-asplos16.pdf) — Decomposition + SMT solving
- [docs/NEXT.md](NEXT.md) — Full research writeup with architecture diagrams

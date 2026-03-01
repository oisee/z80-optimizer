# Z80 Superoptimizer

Brute-force Z80 superoptimizer. Go project.

## Build & Test

```bash
go build -o z80opt ./cmd/z80opt
go test ./...
```

## Architecture

- `pkg/cpu/` — Z80 State (11 bytes: A,F,B,C,D,E,H,L + SP + M), flag tables, executor (2.7ns/op)
- `pkg/inst/` — OpCode enum (uint16, 455 opcodes), Instruction{Op uint16, Imm uint16}, catalog with encoding/disassembly/timing
- `pkg/search/` — QuickCheck (8 vectors, 88-byte fingerprint) + MidCheck (32 vectors) + ExhaustiveCheck (up to 2^24 states), enumerator, pruner (opReads/opWrites/areIndependent), worker pool
- `pkg/stoke/` — STOKE stochastic superoptimizer (MCMC search, parallel chains)
- `pkg/gpu/` — GPU integration layer (CUDA process, search orchestration)
- `pkg/result/` — Rule storage, gob checkpoint, JSON + Go codegen output
- `cmd/z80opt/` — CLI: enumerate, target, verify, verify-jsonl, export, stoke
- `cuda/` — CUDA standalone search binaries (v1, v2 batched pipeline)

## Key Invariant

Full state equivalence: target and candidate must produce identical output for ALL possible inputs, including flags and memory byte. `LD A, 0` != `XOR A` because flags differ.

## Memory Model (Wave 5)

Virtual memory byte M in State: all indirect ops (HL), (BC), (DE) share one M register. Prerequisite: all memory ops in a sequence target the same address. UsesMemory(op) checks if an opcode touches M.

## Instruction Waves

V1 (206) + Wave 1 (+174 BIT/RES/SET) + Wave 2 (+14 16-bit pair ops) + Wave 4 (+12 LD rr,nn, ADC/SBC HL) + Wave 5 (+61 memory ops) = 455 opcodes total.

## Roadmap

- [docs/NEXT.md](docs/NEXT.md) — Research roadmap: GPU brute force, STOKE stochastic search, reordering optimizer
- [docs/GenPlan.md](docs/GenPlan.md) — Implementation plan with phases, effort estimates, key files per phase

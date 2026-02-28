# Z80 Superoptimizer

Brute-force Z80 superoptimizer. Go project.

## Build & Test

```bash
go build -o z80opt ./cmd/z80opt
go test ./...
```

## Architecture

- `pkg/cpu/` — Z80 State (10 bytes: A,F,B,C,D,E,H,L + SP), flag tables, executor (2.7ns/op)
- `pkg/inst/` — OpCode enum (uint16, 406 opcodes), Instruction{Op uint16, Imm uint16}, catalog with encoding/disassembly/timing
- `pkg/search/` — QuickCheck (8 vectors, 80-byte fingerprint) + ExhaustiveCheck (up to 2^24 states), enumerator, pruner (opReads/opWrites/areIndependent), worker pool
- `pkg/result/` — Rule storage, gob checkpoint, JSON + Go codegen output
- `cmd/z80opt/` — CLI: enumerate, target, verify, export

## Key Invariant

Full state equivalence: target and candidate must produce identical output for ALL possible inputs, including flags. `LD A, 0` != `XOR A` because flags differ.

## Instruction Waves

V1 (194) + Wave 0 (fixes) + Wave 1 (+174 BIT/RES/SET) + Wave 2 (+14 16-bit pair ops) + Wave 4 (+12 LD rr,nn, ADC/SBC HL) = 406 opcodes total.

## Roadmap

- [docs/NEXT.md](docs/NEXT.md) — Research roadmap: GPU brute force, STOKE stochastic search, reordering optimizer
- [docs/GenPlan.md](docs/GenPlan.md) — Implementation plan with phases, effort estimates, key files per phase

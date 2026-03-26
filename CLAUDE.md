# Z80 Superoptimizer

Brute-force Z80 superoptimizer. Go + CUDA project.

## Build & Test

```bash
CGO_ENABLED=0 ~/go/bin/go1.24.3 build ./...
CGO_ENABLED=0 ~/go/bin/go1.24.3 test ./...
```

CUDA kernels:
```bash
nvcc -O3 -o cuda/z80search_v2 cuda/z80_search_v2.cu        # peephole search
nvcc -O3 -o cuda/z80_regalloc cuda/z80_regalloc.cu          # register allocator
nvcc -O3 -o cuda/z80_mulopt_fast cuda/z80_mulopt_fast.cu    # constant multiply (14-op)
nvcc -O3 -o cuda/z80_divmod_fast cuda/z80_divmod_fast.cu    # division/modulo (14-op)
```

## Architecture

### Go packages
- `pkg/cpu/` — Z80 State (11 bytes: A,F,B,C,D,E,H,L + SP + M), flag tables, executor (2.7ns/op)
- `pkg/inst/` — OpCode enum (uint16, 455 opcodes), Instruction{Op uint16, Imm uint16}, catalog
- `pkg/search/` — QuickCheck (8 vectors) + MidCheck (32 vectors) + ExhaustiveCheck (2^24 states)
- `pkg/stoke/` — STOKE stochastic superoptimizer (MCMC search, parallel chains)
- `pkg/gpu/` — GPU integration layer (CUDA process, search orchestration)
- `pkg/result/` — Rule storage, gob checkpoint, JSON + Go codegen output

### Go commands
- `cmd/z80opt/` — CLI: enumerate, target, verify, verify-jsonl, export, stoke
- `cmd/regalloc-enum/` — Exhaustive constraint shape enumerator (treewidth filter, dense-masks)
- `cmd/partopt/` — Call graph partition optimizer (cross-function merge decisions)
- `cmd/mulopt/` — CPU parallel multiply search

### CUDA kernels
- `cuda/z80_search_v2.cu` — Peephole superoptimizer (3-stage: QuickCheck → MidCheck → Exhaustive)
- `cuda/z80_regalloc.cu` — GPU register allocator + CPU backtracking solver fallback
- `cuda/z80_mulopt_fast.cu` — Constant multiply search (14-op reduced pool, 38x faster)
- `cuda/z80_divmod_fast.cu` — Division/modulo search (14-op, 5T limit)
- `cuda/z80_mulopt16.cu` — 16-bit multiply (u8×K=u16, result in HL)
- `cuda/z80_common.h` — Shared Z80 executor, flag tables, test vectors

### Data
- `data/` — Exhaustive register allocation tables (compressed binary + zstd)
- `data/README.md` — Binary format spec, reader examples (Python/Go), lookup instructions

### Documentation
- `docs/glossary.md` — Complete glossary of all terms and abbreviations
- `docs/paper_seed_superopt.md` — Paper/book seed: 8 sections covering all research findings
- `docs/research_statement.md` — Paper-oriented framing with phase diagram data
- `docs/paper_plan_simple.md` — Plain-language paper plan

## Key Invariant

Full state equivalence: target and candidate must produce identical output for ALL possible inputs, including flags and memory byte. `LD A, 0` != `XOR A` because flags differ.

## Results

### Peephole Superoptimizer
- 739K len-2→len-1 rules (complete)
- 37M len-3→len-1 rules (partial, ~0.05% coverage)

### Constant Multiplication
- 164/254 constants solved at length ≤9 (14-op reduced pool)
- Key finding: 7 of 21 ops never appear in optimal solutions (38x search speedup)
- NEG trick: ×255 = NEG (1 instruction, 8T)

### Division/Modulo
- div10 lower bound ≥13 instructions (GPU search certificate)
- Best known div10: 27 instructions, 124-135T (Hacker's Delight + RRA+AND)
- Gap of 14 instructions remains open

### Register Allocation Tables
- ≤4v: 156,506 shapes, 40 seconds (complete)
- ≤5v: 17,366,874 shapes, 20 minutes (complete)
- 6v dense (tw≥4): 66,118,738 shapes, ~6 hours (complete via treewidth filter)
- Feasibility cliff: 95.9% (2v) → 0.9% (6v) — phase transition
- 99.5% of random interference graphs have treewidth ≤3
- Composition verified on 13.2M shapes: max 12T overhead, 0 misses

### Five-Level Pipeline
1. Table lookup (17.4M entries) — O(1), covers 87% of corpus
2. Composition via cut vertices — O(1) per component, tw≤3
3. GPU brute-force — ≤12v, seconds
4. CPU backtracking (1000-4000x pruning) — ≤15v, <1 second
5. Island decomposition + Z3 — >15v

## Hardware

```
main:  2× RTX 4060 Ti 16GB (CUDA 12.0)
i5:    1× RTX 2070 8GB (CUDA 12.0)
i3:    1× Radeon RX 580 8GB (ROCm 6.4.4 / HIP)
```

### Remote GPU setup

```bash
# i5 (NVIDIA): copy and run
scp cuda/z80_mulopt_fast.cu cuda/run_mulopt_remote.sh i5:~/
ssh i5 'bash run_mulopt_remote.sh 9'

# i3 (AMD): needs ROCm HIP
# Install: sudo apt install -y --allow-downgrades rocminfo=1.0.0.60404-129~24.04 hip-dev rocm-hip-runtime
# Compile: hipcc -O3 -o z80_mulopt_fast z80_mulopt_fast.cu
```

## Memory Model (Wave 5)

Virtual memory byte M in State: all indirect ops (HL), (BC), (DE) share one M register. Prerequisite: all memory ops in a sequence target the same address.

## Cross-Session Communication (ddll)

```bash
ddll explore                        # list active sessions
ddll send <session>:main "message"  # send message to a session
```

Known sibling repos: minz (compiler), minz-vir (VIR backend), z80-optimizer (this)

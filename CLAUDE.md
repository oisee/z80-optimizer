# Z80 Superoptimizer

Brute-force Z80 superoptimizer. Go + CUDA project.

**Roadmap**: [TODO.md](TODO.md) — full task list with effort estimates and priority matrix.

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
- `data/enriched_*.enr.zst` — 37.6M enriched shapes with 15 op-aware metrics (78MB)
- `data/ENRICHED_TABLES.md` — Enriched format spec + usage guide
- `data/z80_register_graph.json` — Complete 11-register cost model (moves, ALU, swaps)
- `data/mulopt8_clobber.json` — 254 mul8 sequences (A×K→A) with clobber masks
- `data/mulopt16_complete.json` — 254 mul16 sequences (A×K→HL)
- `data/div8_optimal.json` — 254 div8 sequences (A÷K→A) via multiply-and-shift
- `data/mod8_optimal.json` — 254 mod8 sequences (A%K→A)
- `data/divmod8_optimal.json` — 254 divmod8 sequences
- `data/u32_ops.json` — 13 u32 operations (DEHL convention), SHL/SHR proven optimal
- `data/sign_sat_ops.json` — sign8, sat_add8 (16T!), sat_sub8
- `data/arith16_new.json` — abs16, neg16, min16, max16, sign16, cmp16_zero
- `data/sha256_round.json` — SHA-256 round decomposition (58ms/block @3.5MHz)
- `data/arith16_idioms.json` — 16-bit arithmetic idioms (legacy)
- `data/bcd_idioms.json` — BCD arithmetic (GPU-proven with H-flag)

### Documentation
- `docs/glossary.md` — Complete glossary of all terms and abbreviations
- `docs/paper_seed_superopt.md` — Paper/book seed: 8 sections covering all research findings
- `docs/research_statement.md` — Paper-oriented framing with phase diagram data
- `docs/paper_plan_simple.md` — Plain-language paper plan

## Key Invariant

Full state equivalence: target and candidate must produce identical output for ALL possible inputs, including flags and memory byte. `LD A, 0` != `XOR A` because flags differ.

## IX/IY Half Registers

IXH, IXL, IYH, IYL are considered **production-safe**. While historically called "undocumented", they work on all known Z80 silicon (Zilog, NEC, Toshiba, ST) and all Z80-compatible clones (eZ80, Z80N, R800, SM83 excluded). We use them as 4 additional 8-bit registers.

Key constraints:
- **DD/FD prefix hijacks H/L encoding** — `LD H,IXH` is impossible, `LD IXH,H` is impossible
- H↔IXH transfer requires trick: `EX DE,HL; LD IXH,D; EX DE,HL` (16T, no clobber)
- Or via A: `LD A,H; LD IXH,A` (12T, clobbers A)
- Cost: 8T per LD vs 4T for main regs (DD/FD prefix = +4T)
- Not swapped by EXX — IX/IY are **bridges between main and shadow banks**

Register count: 7 main (A,B,C,D,E,H,L) + 4 IX/IY halves = **11 registers** for allocation.

## Proven Z80 Architectural Facts

- **Z flag is write-only**: no ALU instruction reads Z as input (proven exhaustive + induction). Z→CY impossible branchless.
- **CY is the only viable branchless bool flag**: SBC A,A materializes CY→mask in 1 instruction (4T).
- **0xFF/0x00 > 0/1 for bool representation**: AND/OR/XOR/NOT are free (native Z80 logic ops).
- **Branch > branchless on Z80**: branch penalty = 5T, branchless overhead = 15-24T.
- **div3 = A×171>>9**: exact for all 256 uint8 inputs, no lookup table needed.

## Results

### Peephole Superoptimizer
- 739K len-2→len-1 rules (complete)
- 37M len-3→len-1 rules (partial, ~0.05% coverage)

### Constant Multiplication
- 254/254 constants solved (complete!)
- Key finding: 21-instruction universal pool (2.7% of ISA generates ALL optimal arithmetic)
- NEG trick: ×255 = NEG (1 instruction, 8T)
- All 254 mul8 preserve A, all DE-safe

### Division/Modulo
- div10 lower bound ≥13 instructions (GPU search certificate)
- Best known div10: 27 instructions, 124-135T (Hacker's Delight + RRA+AND)
- **div3 = A×171>>9: EXACT for all 256 inputs** (no lookup table!)

### Branchless Library (exhaustive verified)
- ABS(A) signed: 6i, 24T — `LD B,A; RLCA; SBC A,A; LD C,A; XOR B; SUB C`
- MIN/MAX(A,B) unsigned: 8i, 32T — via SBC A,A + bitwise select
- CMOV CY?B:C: 6i, 24T — `SBC A,A; LD D,A; LD A,B; XOR C; AND D; XOR C`
- gray_decode: EXACT (13 ops, found on Vulkan RX 580 in <1 second)

### Register Allocation Tables
- **83.6M total shapes enumerated** (≤6 variables)
- **37.6M feasible**, each with optimal assignment + 15 enrichment metrics
- ≤4v: 156,506 shapes (78.9% feasible), 40 seconds
- ≤5v: 17,366,874 shapes (67.7% feasible), 20 minutes
- 6v dense (tw≥4): 66,118,738 shapes (38.9% feasible), ~6 hours
- Enrichment: 43% lack A (hidden ALU infeasibility), 21% lack HL
- Smart CALL save: 17T avg (vs 34T naive) = 50% reduction
- Feasibility cliff: 95.9% (2v) → 0.9% (6v) — phase transition
- 99.5% of random interference graphs have treewidth ≤3
- O(1) lookup via signature: (interference_shape, operation_bag) → hash
- Validated on 820-function production compiler corpus (246 unique signatures)
- Compressed enriched tables: 78MB (data/enriched_*.enr.zst)

### Five-Level Pipeline
1. Cut vertex decomposition — free split, 87% of shapes
2. Enriched table O(1) lookup — 37.6M entries, 79% of corpus
3. EXX 2-coloring — dual-bank for 7-12v (70% bipartite)
4. GPU partition optimizer — ≤18v exhaustive (<2 min), ≤20v (~30 min)
5. Z3 fallback — >18v (<0.5% of functions)

### Multi-Platform GPU
- ISA DSL (`pkg/gpugen/`) → CUDA, Vulkan, Metal, OpenCL from single source
- 5 platforms, 4 APIs, 3 GPU vendors, zero discrepancies
- ISA definitions: Z80 (394 ops), 6502, SM83 (Game Boy)

## Hardware

```
main:  2× RTX 4060 Ti 16GB (CUDA 12.0) — primary search, regalloc, partition optimizer
i5:    1× RTX 2070 8GB (CUDA 12.0) — focused search, mul16, image search
i3:    1× Radeon RX 580 8GB (Vulkan/Mesa) — gray_decode EXACT found here!
M2:    Apple M2 MacBook Air (Metal) — cross-verification
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

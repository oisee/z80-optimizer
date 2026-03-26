# M2 MacBook Air — Onboarding for New Session

## Who you are
You're joining the z80-optimizer project on an M2 MacBook Air.
The main team runs on Linux with NVIDIA/AMD GPUs. Your machine
has Apple Silicon GPU accessible via Metal compute shaders.

## Project context
Read these files first:
- `CLAUDE.md` — full project overview, build instructions, results
- `docs/glossary.md` — all terms and abbreviations
- `docs/insights.md` — experiment log with timestamped discoveries
- `data/README.md` — binary table format and reader examples

## What's been done (birthday session, 2026-03-26)
- 83.6M exhaustive register allocation table (≤6v, 32MB compressed)
- 254/254 u16 constant multiply table (3-op basis, 30 sec GPU)
- 164/254 u8 constant multiply with clobber annotations
- CPU backtracking solver (1000-4000x pruning)
- OpenCL port tested on AMD RX 580
- Go packages: pkg/mulopt/ (lookup), pkg/regalloc/ (table reader)
- 4-GPU cluster: 2×RTX 4060 Ti + RTX 2070 + RX 580

## Your tasks (priority order)

### Task 1: 6502 constant multiply table
Port the mulopt concept to MOS 6502. Same brute-force approach.
The 6502 has no multiply instruction — shift-add chains are essential.

6502 multiply op pool (~12-14 ops):
```
ASL A    (shift left = ×2, 2 cycles)
LSR A    (shift right, 2 cycles)
ROL A    (rotate left through carry, 2 cycles)
ROR A    (rotate right through carry, 2 cycles)
TAX      (save A to X, 2 cycles)
TXA      (restore X to A, 2 cycles)
TAY      (save A to Y, 2 cycles)
TYA      (restore Y to A, 2 cycles)
CLC      (clear carry, 2 cycles)
SEC      (set carry, 2 cycles)
ADC_ZP   (A += mem[zp], 3 cycles — need STA first)
SBC_ZP   (A -= mem[zp], 3 cycles)
STA_ZP   (store A to zp, 3 cycles)
EOR_FF   (complement A, 2 cycles — like NEG without +1)
```

Approach:
1. Build a Metal compute shader with 6502 executor
2. Start with ~12 ops, len-8 sweep for all 254 constants
3. Analyze which ops appear → reduce pool → go deeper
4. Compare with Z80 results — how much does ISA matter?

Alternative if Metal is complex: pure CPU search (M2 has 8 fast cores).
14^9 at ~2B/sec per core × 8 cores = 16B/sec → ~1.3 sec per constant.

### Task 2: Meet-in-the-middle for mul8
The 90 unsolved mul8 constants need len-10+ (14^10 = 289B).
Meet-in-the-middle splits len-10 into two len-5 halves:
- Forward: compute all reachable states from input in 5 steps
- Backward: compute all states that reach target in 5 steps
- Match: find common states
14^5 + 14^5 = 1.07M (vs 289B brute force) — 270,000× speedup.

Challenge: state = (A, B, carry) = 131K possible states.
Forward table: 131K states × 14^5 = 70B entries → too much memory.
Practical: forward at len-4 (38K sequences) → 131K states, then
brute-force remaining 6 steps: 14^6 = 7.5M per state. Feasible.

### Task 3: Peephole len-3 search (if GPU works via Metal)
74.9B target pairs, 37M already found (0.05%).
The Z80 executor is simple (11-byte state, pure ALU).
Metal compute shader would be a great fit.

## Build on macOS

```bash
# Go (for CLI tools)
brew install go
cd z80-optimizer
go build ./...

# For Metal compute: need Xcode command line tools
xcode-select --install
# Metal shader files go in metal/ directory
# Compile: xcrun -sdk macosx metal -c shader.metal -o shader.air
# Link: xcrun -sdk macosx metallib shader.air -o shader.metallib
```

## Communication
Use ddll to communicate with other sessions:
```bash
ddll explore                          # find active sessions
ddll send <session>:main "message"    # send to a session
```

Known sessions:
- z80-optimizer (main Linux, this project)
- minz-vir (VIR backend, register allocator)
- minz (compiler frontend)

## Key numbers to remember
- Z80: 14 useful mul8 ops, 3 core mul16 ops, 15 register locations
- 6502: ~14 mul ops, 3 registers (A,X,Y) + carry + zero-page
- State for Z80 mul8: (A, B, carry) = 131K states
- State for 6502 mul: (A, X, carry) = 131K states (similar!)
- GPU search rate: ~1B candidates/sec (RTX 4060 Ti)
- CPU search rate: ~200M candidates/sec per core (M2)

## Files to look at
- `cuda/z80_mulopt_fast.cu` — reference CUDA kernel (port to Metal)
- `opencl/z80_mulopt.c` — OpenCL port (closest to Metal's API)
- `data/mulopt8_clobber.json` — 164 Z80 results (compare with 6502)
- `cmd/mulopt/main.go` — CPU Go solver (works on any platform)

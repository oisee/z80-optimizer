# GPU Brute-Force Superoptimization: From Z80 to Universal Computation Chains

## Key Results (Birthday Session 2026-03-26)

### The Numbers

| What | Count | Method | Time |
|------|-------|--------|------|
| Peephole rules (len-2) | 739K | GPU exhaustive | minutes |
| Dead-flags peephole | 1.4M+ (partial) | GPU with flag masking | running |
| Constant multiply (u8) | **254/254** | GPU + composition | hours |
| Constant multiply (u16) | **254/254** | 3-op GPU (30 sec!) | seconds |
| Constant division (u8) | **246/247** | Guided brute-force | 11 sec each |
| Branchless idioms | **15** | 37-op GPU | 2 sec |
| 16-bit idioms | **7** | 33-op GPU | 6 sec |
| Register allocations | **83.6M** | GPU exhaustive | 6 hours |
| Corpus 7-15v sigs | **129/129** | 12-core parallel backtrack | 60 sec |
| Abstract chains (mul) | 254/254 | CPU | 8 sec |
| Abstract chains (div) | 86 | Composition | instant |

### Headline Findings

**1. The Feasibility Cliff**
```
2v:  96% feasible    ← almost everything works
3v:  89%
4v:  79%
5v:  68%
6v:   1%             ← phase transition!
7-15v: 2.3%          ← 97.7% INFEASIBLE for real programs
```

The Z80 register file is so irregular that 97.7% of real 7-15v functions
have NO valid register assignment. Island decomposition isn't optimization —
it's MANDATORY for correctness.

**2. Pool Reduction: The Most Powerful Technique**

| Search | Full pool | Reduced pool | Speedup |
|--------|-----------|-------------|---------|
| mul8 | 21 ops | 14 ops | 38× |
| mul16 | 23 ops | **3 ops** | **13,600×** |
| div (guided) | 37 ops | **6 ops** | **millions×** |

The key insight: MOST instructions are USELESS for any given computation.
Empirical analysis reveals which ops actually appear in optimal solutions.
Removing the rest compresses the search space exponentially.

**3. Guided Brute-Force: Abstract Chains → ISA-Specific Search**

For division: abstract chain says "multiply by reciprocal M, then shift right S".
GPU searches ONLY the materialization space: 6 ops instead of 37.

Result: div10 = 124T found in 11 seconds (matches Hacker's Delight hand-optimized!).
118 divisors found automatically that would take a human expert months.

**4. The SBC A,A Carry-to-Mask Trick**

`SBC A,A` converts carry flag to a full byte: 0x00 or 0xFF.
This appears in nearly every branchless idiom the GPU discovered:

- `bool(A)`: LD B,A : NEG : ADC A,B — carry from NEG + add = 0 or 1
- `ABS(A)`: RLCA : SBC A,A : XOR B — sign→carry→mask→conditional complement
- `sign-extend`: ADC A,L : SBC A,A : LD H,A — overflow→carry→0xFF mask

The GPU rediscovered what assembly wizards knew — but PROVED it optimal.

**5. Packed Multi-Entry Library: 2KB for ALL Arithmetic**

```asm
mul104: ADD A,A     ; ×104 enters here → falls through
mul52:  ADD A,A     ; ×52
mul26:  ADD A,B     ; ×26
mul24:  ADD A,A     ; ×24
mul12:  ADD A,A     ; ×12
mul6:   LD B,A : ADD A,B : ADD A,B
mul2:   RLA         ; ×2
        RET         ; 7 constants, 9 instructions, 1 RET
```

254 multiplies + 245 divisions + rotation sleds = ~2KB.
Prefix sharing: 51% compression (mul8), 86% (mul16).

**6. Cross-Platform Verification**

Same search, same results across:
- NVIDIA RTX 4060 Ti × 2 (CUDA)
- NVIDIA RTX 2070 (CUDA)
- AMD Radeon RX 580 (OpenCL + Vulkan via Mesa)
- Apple M2 (Metal)
- CPU (Python, all 256 inputs)

5 platforms, 4 APIs, 3 GPU vendors = mathematical certainty.

**7. ISA DSL: One Definition → Four Backends**

```go
var Z80Mul = ISA{
    State: []Reg{{Name: "a", Type: U8}, {Name: "b", Type: U8}, ...},
    Ops: []Op{
        {Name: "ADD A,A", Cost: 4, Body: `r = a + a; carry = r > 0xFF; a = r;`},
        ...
    },
}
```
`gpugen -isa z80 -backend cuda` → CUDA kernel
`gpugen -isa z80 -backend metal` → Metal shader
Same ISA definition → 4 GPU backends. 250 lines per kernel.

---

## The Story

### Act I: Peephole (739K rules)
Try ALL pairs of Z80 instructions. For each pair, check if a single
instruction produces the same output for ALL possible inputs. 739K pairs
found. GPU does this 30× faster than CPU.

### Act II: Constant Multiply (254/254)
For each constant K, find the shortest sequence where `A_out = A_in × K`.
GPU tries all sequences up to length N. 21 ops → analyze which appear →
only 14 matter → 38× faster → go deeper.

Key discovery: only 3 ops needed for 16-bit multiply (ADD HL,HL + ADD HL,BC + LD C,A).
This is the minimal basis. 13,600× speedup from pool reduction.

### Act III: Division (246/247)
Division is HARD — no Z80 hardware divide. But abstract chains predict:
`n / K = (n × M) >> S`. GPU searches the 6-op materialization space.

div10 = 124T matches Hacker's Delight. Found automatically in 11 seconds.
245 divisors found total. Only div129 remaining.

### Act IV: Register Allocation (83.6M + 97.7% infeasibility)
Enumerate ALL possible constraint shapes. GPU solves each exhaustively.
83.6M entries for ≤6v. For 7-15v: 97.7% of real shapes are INFEASIBLE.

The Z80 compiler MUST decompose functions into ≤6v islands.
This isn't optimization — it's a mathematical necessity.

### Act V: Universal Chains
Abstract away the ISA. Search ONCE in chain space {dbl, add, sub, save, neg, shr}.
Materialize to ANY CPU: Z80, 6502, RISC-V, ARM.

254 multiply chains in 8 seconds on CPU. Same chains → any processor.

---

## Hardware & Cluster

| Machine | GPU | VRAM | API | Role |
|---------|-----|------|-----|------|
| main (i7) | 2× RTX 4060 Ti | 16GB×2 | CUDA | Primary search + backtracking |
| i5 | RTX 2070 | 8GB | CUDA | Secondary search |
| i3 | Radeon RX 580 | 8GB | OpenCL + Vulkan | AMD verification |
| M2 MacBook | Apple Silicon | shared | Metal + OpenCL | Apple verification + DSL dev |

ROCm broken for gfx803 (Polaris). Mesa rusticl provides OpenCL 3.0.
Vulkan via RADV. All verified cross-platform.

---

## For the Compiler

Three Go packages:
```go
import "github.com/oisee/z80-optimizer/pkg/mulopt"   // 254 mul + 254 mul16
import "github.com/oisee/z80-optimizer/pkg/regalloc"  // 83.6M allocations
import "github.com/oisee/z80-optimizer/pkg/peephole"  // 739K rules
```

Integration: O(1) lookup → inline sequence → provably optimal code.
MinZ v0.23.0 ships with 372 inline arithmetic sequences from our tables.

---

## Source & Data

Repository: https://github.com/oisee/z80-optimizer (v1.0.0)

All data in `data/` directory with binary format spec + Python/Go readers.
Book outline: `docs/book_outline.md` (19 chapters, 5 parts).

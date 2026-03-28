# Day 3 Wisdom Dump — March 28, 2026

## What We Proved

### Z80 Register Architecture Facts
- **Z flag is write-only.** No ALU instruction reads Z as input. Proven by exhaustive analysis of all 26 relevant instructions + induction. Z→CY impossible branchless. Only conditional branches (JR Z/NZ) read Z.
- **CY > Z for bool.** CY can materialize to register via SBC A,A (1i, 4T). Z cannot without branch.
- **0xFF/0x00 > 0/1 for bool representation.** Boolean AND/OR/XOR/NOT are FREE (native Z80 logic instructions). NEG converts between representations (1i, 8T).
- **SBC A,A preserves CY.** After SBC A,A, carry flag is unchanged. This enables chaining (mask → AND → result) without losing the condition.
- **Branch > branchless on Z80.** Branch penalty = 5T (failed conditional). Branchless overhead = 15-24T. Use branchless only for SBC A,A mask (always worth it), CMOV in hot loops, or GPU codegen.

### Register Allocation Facts
- **43% of feasible shapes lack A.** These are "hidden infeasible" for u8 ALU — need extra 8T per operation.
- **21% lack HL pair.** u16 ADD infeasible naturally — need 13T+ extra.
- **Only 9% are "ideal"** (A + HL + mul8-safe + B-free for DJNZ).
- **Phase transition at 6-7 variables.** Feasibility: 95.9%(2v) → 78.9%(4v) → 67.7%(5v) → 38.9%(6v) → ~5%(7v).
- **99.5% of real interference graphs have treewidth ≤3.** Most programs are tree-like.
- **move = 34% of all instructions** in real Z80 code. Regalloc directly eliminates these.
- **mul = 0%** in VIR corpus. Nobody multiplies on Z80 in real code.
- **CALL save: 17T smart vs 34T naive.** Free register save (8T) beats PUSH/POP (21T) — 50% reduction.

### Operation-Aware Signatures
- **(interference_shape, operation_bag) → O(1) lookup** replaces Z3 for 90% of functions.
- **operation_bag is order-independent.** ADD A,B costs 4T regardless of position in sequence. Order captured by interference graph.
- **246 unique signatures** in 820-function VIR corpus. Prediction of <500 confirmed.
- **Width matters hugely.** u16 is ~100× more constrained than u8 (only 3 pair slots: BC,DE,HL). 40% of corpus functions have u16 vars.

### Brute-Force Insights
- **Smaller pool → deeper search → better results.** gray_decode: 5-op pool found EXACT at depth 13 in <1s. 13-op pool stuck at ±3 after hours at depth 12.
- **div3 = A×171>>9 is EXACT** for all 256 uint8 inputs. No lookup table needed.
- **Exhaustive partition: ≤18v feasible** on single GPU (~2 min). ≤20v = ~30 min. VIR corpus max = 14v → all covered.

### Real Z80 Code Patterns
- **LD/MOV = 35%, ALU = 22%, Branch = 22%** across Hobbit, demos, book listings.
- **IX/IY = 6-11%** — higher than expected. Used for struct access + cross-bank bridge (not swapped by EXX).
- **EXX/EX < 1%** in explicit use — but enables dual-bank 32-bit arithmetic when used.
- **SDCC 4.5 still suboptimal:** abs_diff 7i vs optimal 4i (+75%), mul3 4i vs 3i (+33%), div10 uses library call vs our 3-instruction inline.

## Architecture Decisions

### 5-Level Regalloc Pipeline
```
Level 0: Cut vertices → free decomposition (87% of shapes)
Level 1: Enriched table O(1) lookup ≤6v (79% of corpus)
Level 2: EXX 2-coloring → dual-bank 7-12v (70% bipartite)
Level 3: GPU partition optimizer 7-18v (<2 min)
Level 4: Z3 fallback >18v (<0.5%)
Combined: 99%+ provably optimal, 91% in O(1)
```

### VIR Integration Format
```json
{
  "opBag": {"add":1, "sub":2, "cmp":1, "move":3, "call":1},
  "shape": {"nVregs":5, "edges":[[0,1],[0,2],[1,2]]},
  "enrichedSig": {"shapeHash": 123456, "opBagHash": 789012, "nVregs": 5}
}
```
- Abstract VIR ops (OpAdd etc), NOT Z80 patterns — pattern selection happens AFTER assignment
- 12 op categories: Add,Sub,Mul,Cmp,Logic,Shift,Load,Store,Call,Move,Const,Neg
- Canonical graph hash (renumbered vregs + sorted edges → SHA256)

### Register Cost Graph
- 11 registers: A,B,C,D,E,H,L,IXH,IXL,IYH,IYL
- Move: 38% at 4T, 40% at 8T, 22% at 16T
- H↔IXH impossible direct — EX DE,HL trick = 16T, no clobber
- ALU through A: natural=4T, via-A=12T, IX-operand=20T
- Cross-bank (EXX): A(0T), IX bridge(16T), stack(22T), TSMC(20T)

### Branchless Primitives (all verified exhaustive)
- CY→mask: SBC A,A (1i, 4T)
- CY?B:0: SBC A,A; AND B (2i, 8T)
- CY?B:C (CMOV): SBC A,A;LD D,A;LD A,B;XOR C;AND D;XOR C (6i, 24T)
- ABS(A): LD B,A;RLCA;SBC A,A;LD C,A;XOR B;SUB C (6i, 24T)
- MIN/MAX(A,B): 8i, 32T via SUB+CMOV
- div3: A×171>>9 (EXACT)

### RMDA Demoscene Techniques
- CALL-chain rendering: 8.5T/byte (fastest screen write, pushes return address to screen-stack)
- RL (IX+N),R: undocumented instruction, rotate+load in one op (23T for 2 operations)
- INC (HL) self-modification: 0x34↔0x35 toggle (1-instruction oscillator)
- pRNG: Patrik Rak CMWC ×253, 10-byte state, ~167T per byte, period ~2^66

## Key Files Created/Modified

### New kernels
- `cuda/z80_focused.cu` — sequential focused search with per-target minimal op pools
- `cuda/z80_regalloc_batch.cu` — GPU batch register allocator (JSON in → optimal out)
- `cuda/z80_partition_opt.cu` — GPU partition optimizer for 7-24v graphs
- `cuda/z80_prng_search.cu` — pRNG SEED search + CALL-chain simulation
- `cuda/z80_flag_idioms.c` — flag materialization exhaustive search
- `cuda/z80_branchless.c` — branchless primitives verification
- `cuda/vulkan_graydec.c` + `graydec_search.comp` — Vulkan gray_decode solver

### New data
- `data/enriched_4v.enr.zst` (168K) — 123K shapes enriched
- `data/enriched_5v.enr.zst` (22MB) — 11.7M shapes enriched
- `data/enriched_6v_dense.enr.zst` (56MB) — 25.7M shapes enriched
- `data/z80_register_graph.json` — complete register cost model
- `data/ENRICHED_TABLES.md` — binary format spec + usage guide

### New docs
- `docs/regalloc_paper.md` (1574 lines) + PDF + EPUB
- `docs/regalloc_deep_dive.md` (620 lines, Mermaid diagrams)
- `docs/build_paper.sh` — PDF build with mmdc → PNG → lualatex

### New tools
- `cmd/enrich-regalloc/` — Go tool, enriches tables with op-aware costs

### Delivered to antique-toy book
- `_in/regalloc_deep_dive.md`
- `_in/branchless_primitives.md`
- `_in/sidebar_sbc_trick.md`
- `_in/sidebar_call_chain_rmda.md`

## Running Overnight
- i7 GPU0: 19v partition exhaustive (3^19 = 275B)
- i7 GPU1: 20v partition exhaustive (3^20 = 1.1T)
- i5: 20v variant B partition exhaustive
- Corpus batch: all 172 functions 7-18v → partition results

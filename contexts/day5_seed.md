# Day 5 Seed — Next Session Priorities

## P0: Hybrid Image Generator on CUDA
- `cuda/prng_hybrid_search.py` exists but per-pixel Python = too slow
- Write CUDA kernel or ISA DSL version for GPU generation
- 57-byte genome: pRNG + masks + basis + threshold + symmetry
- Target: ~1M images/sec (vs ~10/sec in Python)
- Then run overnight: find cats, skulls, faces from noise

## P1: Go Reader for Enriched Tables
VIR waiting for `pkg/regalloc/enriched.go`:
```go
LoadEnriched(path) → *EnrichedTable
table.Lookup(shapeHash, opBagHash) → *Entry
// Entry: assignment, cost, flags, 12 pattern costs
```
Binary format: data/ENRICHED_TABLES.md
VIR corpus: /home/alice/dev/minz-vir/corpus_gpu_batch.jsonl (820 funcs)

## P2: Enrich Tables with 11 Registers (IX/IY halves)
- Current: 7 locations (A-L) + pairs
- Needed: 11 locations (+ IXH, IXL, IYH, IYL)
- Register graph already has 11 regs: data/z80_register_graph.json
- Enrichment: 7× longer (~25 min for 6v) but feasible
- ALU costs: ADD A,IXL = 8T (included in graph)

## P3: FatFS + z88dk Corpus (Fix Bias)
- MinZ will dump FatFS via VIR_DUMP_GPU_BATCH
- z88dk: 905 C files, regexp.c = 1136 lines
- Fuzix kernel if findable
- Compare: SDCC vs MinZ register allocation quality

## P4: assignmentPerPartition
Extend z80_partition_opt.cu: per-partition lookup in enriched table
Output: `{"partitions": [...], "assignments": [{...}, ...], "costs": [...]}`
VIR needs this for compose step

## P5: WASM Verifier Integration
gyfiwji1 building: OpStats + sequence verifier in WASM
Format: test(input_A) → output_A, 256 calls, ~25μs total

## P6: Peephole Bool Rules
4 rules from MinZ:
1. LD A,1; RET after CP → RET NZ
2. LD A,0; RET after CP → RET Z
3. CALL pred; CP 1; JR Z → CALL pred; JR NZ
4. CALL pred; OR A; JR NZ → CALL pred; JR NZ
Implement in pkg/peephole/

## P7: 6502 Brute-Force
gpugen ISA definition exists (6502.isa)
Generate CUDA kernel → find optimal mul8 for 6502
Only 3 regs (A, X, Y) — even more constrained than Z80

## P8: Introspec's BB Source Code
Waiting for Introspec to send his C++ brute-forcer
When arrives: port to CUDA, compare with our approach
Key insight: he used masks + regions, not just seed

## Key Files Reference
- Enriched tables: `data/enriched_{4v,5v,6v_dense}.enr.zst` (78MB)
- Register graph: `data/z80_register_graph.json`
- Batch evaluator: `cuda/z80_regalloc_batch.cu`
- Partition optimizer: `cuda/z80_partition_opt.cu`
- Image search: `cuda/prng_cat_search.py`, `prng_dither_search.py`, `prng_hybrid_search.py`
- Paper: `docs/regalloc_paper.md` (v2.2, 1600+ lines)
- Gallery: `media/prng_images/README.md`
- VIR corpus: `/home/alice/dev/minz-vir/corpus_gpu_batch.jsonl`

## Numbers
- 83.6M shapes, 37.6M feasible, 78MB enriched tables
- 820 corpus functions, 246 unique signatures
- 172/172 corpus partitioned optimally (7v+)
- Partition: v14 <1s, v18 <2min, v20 ~30min, v32 <30ms greedy
- move=34%, mul=0% in real code
- 43% no_A, 21% no_HL, 7% mul8-safe
- FatFS: 10-35v (vs corpus max 14v) — BIAS!
- ISA DSL → 4 GPU backends, 3 vendors
- 11 registers (7 main + 4 IX/IY halves)
- Paper: v2.2, A4+A5+EPUB, v1.3.0 release

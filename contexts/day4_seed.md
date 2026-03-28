# Day 4 Seed — What To Do Next

## Priority 1: Harvest Overnight Results
```bash
# Check partition optimizer results
cat /tmp/partition_19v_result.json
cat /tmp/partition_20v_result.json
ssh i5 'cat /tmp/partition_20v_b_result.json'
cat /tmp/corpus_partitions.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d)} functions partitioned')"
```

## Priority 2: Go Reader for Enriched Tables
VIR needs `pkg/regalloc/enriched.go`:
```go
type EnrichedTable struct { ... }
func LoadEnriched(path string) (*EnrichedTable, error)
func (t *EnrichedTable) Lookup(shapeHash, opBagHash uint64) *EnrichedEntry
// EnrichedEntry: assignment, cost, flags (no_accumulator, mul8_safe, etc.), 12 pattern costs
```
Binary format: see data/ENRICHED_TABLES.md

## Priority 3: assignmentPerPartition
Extend z80_partition_opt.cu to output per-partition assignment by looking up enriched table.
VIR wants: `{"partitions": [[0,1,3], [2,4,5]], "assignments": [{"0":"A","1":"C","3":"D"}, ...], "costs": [12, 8]}`

## Priority 4: VIR Corpus Full Evaluation
Run GPU batch evaluator (z80_regalloc_batch.cu) on all 820 functions from corpus.
File: /home/alice/dev/minz-vir/corpus_gpu_batch.jsonl
Compare: our optimal vs what VIR/Z3 currently produces.

## Priority 5: Peephole Bool Rules
4 rules from MinZ (ju6yy047):
1. LD A,1; RET after CP → RET NZ
2. LD A,0; RET after CP → RET Z
3. CALL pred; CP 1; JR Z → CALL pred; JR NZ
4. CALL pred; OR A; JR NZ → CALL pred; JR NZ
Implement in pkg/peephole/

## Priority 6: pRNG Visual Search
- Fix z80_prng_search.cu (was killed mid-run)
- Run CMWC + CALL-chain modes
- Find visually interesting patterns
- Share results with Maxim (RMDA)

## Priority 7: RL (IX+N),R Analysis
Undocumented instruction analysis:
- RL (IX+N),H; RL (IX+N),L; PUSH HL = 57T for 2-byte scroll+render
- vs conventional: 95T = 66% slower
- Brute-force: optimal IX offset sequences for different scroll patterns
- Add to antique-toy book as sidebar

## Priority 8: 6502 Brute-Force
gpugen ISA definition exists. Generate CUDA kernel for 6502 mul8.
6502 has only A,X,Y (3 regs!) — even more constrained than Z80.

## Context for Colleagues

### VIR (cok1cgsq)
- Committed: Tarjan cut vertices + BFS bipartite in regalloc_table.go
- Waiting: Go reader for .enr files, assignmentPerPartition format
- 91% O(1) coverage confirmed from corpus

### MinZ (ju6yy047)
- MIR2 VM bug: cross-function host poke→peek return = 0 (likely wrong dst vreg)
- Bool convention settled: CY flag, 0xFF/0x00, retFlag orthogonal
- 4 peephole rules ready for implementation

### Book (eo29c66e)
- Appendix P (regalloc), updated N (gray EXACT), K (branchless), O (flag proofs)
- Sidebars: SBC A,A trick (Ch.4), CALL-chain RMDA (Ch.3)
- Waiting: any new results, phase transition visualization

## Numbers to Remember
- 83.6M shapes, 37.6M feasible, 78MB compressed enriched tables
- 820 corpus functions, 246 unique signatures, 235 unique (shape,opBag)
- move=34%, mul=0%, call=13% in real code
- 43% no_A, 21% no_HL, 7% mul8-safe
- CALL save: 17T smart vs 34T naive
- Partition: ≤14v <1s, ≤18v <2min, ≤20v ~30min
- div3 EXACT: A×171>>9
- gray_decode EXACT: 13 ops, <1s Vulkan

## Priority 9: Image Brute-Force (Introspec + RMDA)
- z80_image_search.cu ready (exhaustive + hill climbing)
- Current metric too simple (block Hamming) → need perceptual
- Introspec: "с нормальной метрикой на VGG можно поднять качество в разы"
- Ideas: CNN face classifier as fitness, CLIP embeddings, VGG perceptual loss
- Introspec will send his C++ brute-forcer code (from BB/Big Brother demo)
- Target: 128×96 mono, pRNG SEED → image → CNN score → find best
- Also: Introspec's chunk rendering (POP HL; LDD; LD A,(HL); LD (BC),A)
  = vector quantization problem for optimal conversion table
- Book: sidebar for Ch.3 (CALL-chain) + Ch.7 (chunky pixels)
- Maxim (RMDA): Hole #17 source analyzed, CALL-chain + RL(IX+N),R

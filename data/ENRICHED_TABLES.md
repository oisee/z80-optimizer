# Enriched Register Allocation Tables

## What is this?

37.6 million Z80 register assignments, each scored with 15+ operation-aware
cost metrics. Every possible way to assign 2-6 virtual registers to Z80
physical registers, with costs for ALU, 16-bit arithmetic, multiply,
function calls, loops, and more.

**Before:** tables said "this assignment is feasible" (yes/no).
**After:** tables say "this assignment costs 4T for u8 ADD, 11T for u16 ADD,
is mul8-safe, needs 17T for CALL save, conflicts with DJNZ."

## How we computed this

### Step 1: Exhaustive enumeration (done earlier)
For each variable count (2-6), enumerate ALL possible:
- Interference graphs (which variables are live simultaneously)
- Allowed register sets per variable
- Physical register assignments

Check feasibility: can variables be assigned to registers without conflicts?

| Variables | Shapes | Feasible |
|-----------|--------|----------|
| ≤4 | 156,506 | 123,453 (78.9%) |
| ≤5 | 17,366,874 | 11,762,983 (67.7%) |
| 6 (dense) | 66,118,738 | 25,772,093 (38.9%) |
| **Total** | **83,642,118** | **37,658,529** |

### Step 2: Register cost graph
Built complete Z80 register connectivity model (`data/z80_register_graph.json`):
- Move costs: 11 registers (A,B,C,D,E,H,L,IXH,IXL,IYH,IYL)
- ALU constraints: u8 ops require A as accumulator
- 16-bit constraints: ADD HL,rr requires HL as destination
- Multiply clobber sets from 254 proven-optimal mul8 sequences
- Cross-register paths (e.g., H→IXH via EX DE,HL trick = 16T)

### Step 3: Enrichment (this computation)
For each of 37.6M feasible assignments, compute:

**u8 metrics:**
- `u8_alu_avg` — average cost of binary ALU between any variable pair
- `u8_best_alu` / `u8_worst_alu` — range of ALU costs
- `u8_mul_avg` — multiply cost via mul8 table

**u16 metrics:**
- `u16_pair_count` — how many variables sit in 16-bit pair slots (BC,DE,HL)
- `u16_add_natural` — cost of ADD HL,rr (11T if HL available, 19T+ otherwise)
- `u16_slots_free` — remaining pair slots for additional u16 variables
- `u16_add_via_u8` — fallback: u16 ADD via two u8 ops (always 24T)

**Feasibility flags:**
- `no_accumulator` — A not in assignment → u8 ALU needs extra moves
- `no_hl_pair` — HL not in assignment → u16 ADD needs extra moves
- `mul8_safe` — no live variables in mul8 clobber zone {C,F,H,L}
- `djnz_conflict` — B register occupied → DJNZ loop needs save/restore

**Idiom compatibility:**
- `mul8_conflicts` — count of variables in mul8 clobber zone
- `mul16_conflicts` — count in mul16 clobber zone {A,C,H,L}

**CALL overhead:**
- `call_save_cost` — total T-states needed to save/restore around CALL
  Uses cheapest available channel per register:
  1. Free register: LD r,r' (8T) — cheapest
  2. EX AF,AF' (8T) — A+F only
  3. IX/IY halves: LD IXH,r (16T)
  4. PUSH/POP pair (21T) — fallback
- `call_regs_to_save` — count of registers needing save
- `call_free_saves` — how many can use free-register shortcut

**Shadow bank (EXX):**
- `exx_alu_cost` — cost of one ALU op via shadow bank (12T)
- `exx_amortized_3ops` — amortized EXX cost over 3 operations

**General:**
- `inc_dec_total` — total INC/DEC cost across all variables
- `temp_regs_avail` — registers available for temporaries
- `u8_regs_free` — 8-bit registers not used by any variable

## Key findings

### Feasibility by variable count

| Metric | 4v | 5v | 6v |
|--------|-----|-----|-----|
| No A (u8 ALU infeasible) | 45% | 43% | 36% |
| No HL (u16 ADD infeasible) | 30% | 21% | 13% |
| mul8-safe | 16% | 7% | 1% |
| DJNZ conflict | 12% | 13% | 19% |
| Has A (u8 ready) | 55% | 56% | 64% |
| Has HL (u16 ready) | 70% | 78% | 87% |
| "Ideal" (A+HL+mul8-safe) | 9% | — | — |

### CALL overhead reduction
- Naive (all PUSH/POP): avg 34T per CALL
- Smart (free reg → IX → PUSH): avg 17T per CALL
- **50% reduction** just from smarter save strategy

### Width impact
- u8 ADD: 11 possible source registers, hundreds of valid assignments
- u16 ADD: only HL as accumulator, 4 source pairs → ~2 valid assignments
- u16 is ~100× more constrained than u8
- 4+ u16 variables → INFEASIBLE without IX/IY or spill

## How to use

### For compilers (VIR, SDCC, etc.)

```
1. Compute function signature:
   - interference_graph from liveness analysis
   - variable_widths (u8, u16)
   - operation_bag: {ADD:n, SUB:n, MUL:n, CALL:n, ...}

2. Lookup in enriched table:
   - Find matching shape
   - Filter by width constraints (u16 vars need pair slots)
   - Pick assignment with lowest cost for your operation_bag

3. Early infeasibility detection:
   - If shape + ops = infeasible → decompose BEFORE codegen
   - Use cut vertices from interference graph for split points
```

### For superoptimizers

Use enrichment data to guide brute-force search:
- Know which register combinations to test (skip infeasible)
- Clobber compatibility → which idioms can be applied without save/restore
- Cost bounds → prune search branches that can't beat known optimal

### For decompilers / binary analysis

Given disassembled Z80 code, reverse-engineer register allocation quality:
- Compare actual assignment against enriched optimal
- Identify suboptimal register choices (e.g., ALU through non-A register)
- Estimate potential speedup from re-allocation

### For education / research

- Phase transition analysis: feasibility cliff at 6+ variables
- Register pressure visualization per function
- Comparison: Z80 (7 regs) vs 6502 (3 regs) vs ARM (16 regs)

## Binary format (v1)

### File header (16 bytes)

| Offset | Size | Content |
|--------|------|---------|
| 0 | 4 | Magic: `ENRT` (Enriched Register Table) |
| 4 | 4 | Version: 1 (uint32 LE) |
| 8 | 4 | Number of entries (uint32 LE) |
| 12 | 4 | Max variables (uint8), num patterns (uint8), reserved (uint16) |

### Entry (fixed size per variable count)

**Infeasible marker:** single byte `0xFF`

**Feasible entry:**

| Offset | Size | Content |
|--------|------|---------|
| 0 | 1 | nVregs |
| 1 | 2 | original cost (uint16 LE) |
| 3 | nVregs | assignment (location index per vreg) |
| 3+nVregs | 2 | flags bitfield (uint16 LE) |
| 5+nVregs | N×2 | pattern costs (uint16 LE per pattern, fixed order) |

### Flags bitfield

| Bit | Meaning |
|-----|---------|
| 0 | no_accumulator (A not in assignment) |
| 1 | no_hl_pair (HL not in assignment) |
| 2 | mul8_safe (no clobber conflicts) |
| 3 | djnz_conflict (B occupied) |
| 4 | u16_capable (≥2 pair slots) |
| 5-15 | reserved |

### Pattern cost order (12 × uint16)

| Index | Pattern |
|-------|---------|
| 0 | u8_alu_avg |
| 1 | u8_best_alu |
| 2 | u8_worst_alu |
| 3 | u16_add_natural |
| 4 | u8_mul_avg |
| 5 | mul8_conflicts |
| 6 | mul16_conflicts |
| 7 | call_save_cost |
| 8 | call_free_saves |
| 9 | temp_regs_avail |
| 10 | u16_slots_free |
| 11 | u8_regs_free |

### Estimated sizes

| Table | Entries | Feasible | Est. binary size |
|-------|---------|----------|-----------------|
| 4v | 156K | 123K | ~4MB |
| 5v | 17.4M | 11.7M | ~400MB |
| 6v | 66.1M | 25.7M | ~900MB |

With zstd compression (~4:1): ~325MB total.

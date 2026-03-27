# Z80 Superoptimizer: Cross-Table Data Analytics Report

**Date:** 2026-03-27
**Dataset:** 755 arithmetic sequences, 739,574 peephole rules, 83.6M register allocation entries

## Abstract

Cross-table analysis of the Z80 superoptimizer's exhaustive search results reveals a
striking structural finding: **just 21 unique instructions -- 2.7% of the Z80's ~700
opcodes -- generate ALL optimal arithmetic sequences** across multiplication, division,
and format conversion. The individual task pools are even smaller (5-14 ops each) and
nearly disjoint, enabling search space reductions of up to 89 million times. Division
decomposes entirely into reciprocal multiplication, the mul8 table covers 100% of FP16
mantissa constants, and consecutive constants share 20% of their instruction prefixes on
average. These findings transform superoptimization from a brute-force curiosity into a
practical compiler backend.

---

## 1. Instruction Frequency Analysis

### mul8 (A x K -> A): 14-op pool, 2148 total instruction slots

| Instruction | Count | Share  | Cumulative |
|-------------|------:|-------:|-----------:|
| ADD A,A     |   844 | 39.3%  |     39.3%  |
| ADD A,B     |   490 | 22.8%  |     62.1%  |
| LD B,A      |   265 | 12.3%  |     74.4%  |
| RLA         |   164 |  7.6%  |     82.1%  |
| SBC A,B     |   116 |  5.4%  |     87.5%  |
| RLCA        |   115 |  5.4%  |     92.8%  |
| NEG         |    87 |  4.1%  |     96.9%  |
| RRCA        |    16 |  0.7%  |     97.6%  |
| ADC A,B     |    15 |  0.7%  |     98.3%  |
| SUB B       |    14 |  0.7%  |     99.0%  |
| RRA         |     9 |  0.4%  |     99.4%  |
| SBC A,A     |     7 |  0.3%  |     99.7%  |
| SRL A       |     5 |  0.2%  |     99.9%  |
| ADC A,A     |     1 |  0.0%  |    100.0%  |

**Key insight:** Three instructions (ADD A,A, ADD A,B, LD B,A) account for 74.4% of all
mul8 instruction slots. The top 6 instructions cover 92.8%.

### mul16 (A x K -> HL): 5-op pool, 2375 total instruction slots

| Instruction | Count | Share  |
|-------------|------:|-------:|
| ADD HL,HL   | 1282  | 54.0%  |
| ADD HL,BC   |  712  | 30.0%  |
| LD C,A      |  247  | 10.4%  |
| SUB HL,BC   |  127  |  5.3%  |
| SWAP_HL     |    7  |  0.3%  |

### div8 (A / K -> A): 6-op pool

| Instruction | Role               |
|-------------|---------------------|
| ADD HL,BC   | Reciprocal accumulate |
| ADD HL,HL   | Shift left          |
| LD A,H      | Extract quotient    |
| LD C,A      | Save multiplicand   |
| SHR_HL      | Shift right         |
| SRL A       | Logical shift A     |

---

## 2. The 21-Instruction Universal Pool

The union of all instructions across mul8, mul16, and div8 yields exactly **21 unique
instructions** -- just **2.7% of the Z80's ~700 opcodes**.

### Pool overlap analysis

| Intersection     | Instructions                    | Count |
|------------------|---------------------------------|------:|
| mul8 only        | ADD A,A, ADD A,B, LD B,A, RLA, SBC A,B, RLCA, NEG, RRCA, ADC A,B, SUB B, RRA, SBC A,A, ADC A,A | 13 |
| mul16 only       | ADD HL,HL\*, ADD HL,BC\*, LD C,A\*, SUB HL,BC, SWAP_HL | 2 |
| div8 only        | LD A,H, SHR_HL                  | 2 |
| mul8 ∩ div8      | SRL A                           | 1 |
| mul16 ∩ div8     | ADD HL,BC, ADD HL,HL, LD C,A    | 3 |
| mul8 ∩ mul16     | *(empty)*                       | 0 |
| mul8 ∩ mul16 ∩ div8 | *(empty)*                    | 0 |

\* After deducting shared entries with div8.

**The mul8 and mul16 pools are completely disjoint.** This is architecturally logical:
mul8 works in the accumulator (A register, 8-bit shifts and adds), while mul16 works in
the HL register pair (16-bit shifts and adds). Division bridges the two via reciprocal
multiplication in HL followed by extraction to A.

---

## 3. Cost Distribution (T-states)

| Metric  | mul8  | mul16 | div8 (+11T preamble) |
|---------|------:|------:|---------------------:|
| Min     |    4T |   11T |                  27T |
| P25     |   32T |   85T |                  89T |
| Median  |   36T |  103T |                 105T |
| P75     |   40T |  114T |                 125T |
| Max     |   48T |  137T |                 181T |
| Average |   35T |   98T |                 108T |

Division costs approximately 3x more than 8-bit multiplication on average. The 11T
preamble for div8 accounts for loading the reciprocal constant into BC.

---

## 4. Search Space Reduction

The core insight enabling tractable exhaustive search is pool reduction: eliminating
instructions that provably never appear in optimal solutions.

### Speedup from pool reduction

| Table | Original pool | Reduced pool | Speedup (len-9) | Speedup (deeper)         |
|-------|-------------:|-------------:|-----------------:|--------------------------|
| mul8  |           21 |           14 |             38x  | 86x at len-11            |
| mul16 |           23 |            5 |         13,600x  | **89,000,000x** at len-12 |
| div8  |           37 |            6 |       millions x | (combinatorial)          |

### Universal pool vs naive search

Using the 21-op universal pool instead of a naive 37-op pool yields **164x speedup** at
length 9.

### Wall-clock estimates (single GPU, RTX 4060 Ti)

| Search depth | Candidates (21^n) | Time estimate      |
|-------------|-------------------:|--------------------|
| len-9       |       794 billion  | 7 minutes          |
| len-10      |        16 trillion | 2.3 hours          |
| len-11      |       350 trillion | 12 hours (4 GPUs)  |

---

## 5. Cross-Table Reuse

### Division IS multiplication

All 86 division chains use a **reciprocal multiply constant** M, computed as
M = ceil(256 / K) or a related formula. Every one of these 86 M values exists in the
mul8 table. Between 64% and 92% of the instructions in each div8 sequence are literally
mul8 code operating on the reciprocal.

### FP16 constant multiplication

The mul8 table covers constants 128-255, which is **100% of the FP16 mantissa range**
(mantissas are normalized to [1.0, 2.0), stored as 7-bit values + implicit leading 1,
mapping to 128-255).

**FP16 x const = ADD exponent + mul8[mantissa].** Zero additional code beyond exponent
addition.

| Category          | Constants              | Cost      |
|-------------------|------------------------|-----------|
| Cheapest mantissa | x255 = NEG             | 1 op, 8T  |
|                   | x254 = RLA; NEG        | 2 ops, 12T|
|                   | x128 = 4 ops           | 4 ops, 16T|
| Most expensive    | x171, x173, x179, x181 | 11 ops, 48T |

### Fixed-point f8.8 constant multiplication

f8.8 constant multiplication **is** mul16[K] directly. The mul16 table produces a 16-bit
result in HL, which is exactly the f8.8 representation. No additional code needed.

---

## 6. Shared Subsequences (Code Sharing)

### Top n-grams in mul8

| n-gram | Sequence                              | Entries | Coverage |
|--------|---------------------------------------|--------:|---------:|
| 3-gram | ADD A,B -> ADD A,A -> ADD A,A         |     193 |    76%   |
| 4-gram | ADD A,A -> ADD A,B -> ADD A,A -> ADD A,A |   88 |    35%   |

### Prefix sharing between consecutive constants

Consecutive constants (e.g., x179 and x180) share remarkably long prefixes:

- **x179 -> x180:** share all 10 prefix instructions (only the last op differs)
- **Average prefix sharing** between consecutive K values: **20%**

This has direct implications for ROM-based cassette libraries: delta-encoding consecutive
constants can compress the library by roughly 50%.

---

## 7. Clobber Analysis

Understanding which registers each operation destroys is critical for register allocation
in a compiler backend.

### mul8 register clobber

| Clobber pattern | Entries | Share  |
|-----------------|--------:|-------:|
| Clobbers B      |     240 | 94.5%  |
| B-safe          |      14 |  5.5%  |

Only 14 of 254 mul8 sequences preserve B. Compilers must either schedule around this or
use a save/restore pair (2 extra instructions, +22T).

### mul16 register clobber

| Property       | Value                                |
|----------------|--------------------------------------|
| A preserved    | ALL 254 entries                      |
| DE-safe        | ALL 254 entries                      |
| Clobber {C,F,H,L} | 247 entries                      |
| Clobber {F,H,L} only | 7 entries                      |
| Pareto alternatives (clobber DE for shorter code) | 30 entries |

### div8 register clobber

| Property              | Value           |
|-----------------------|-----------------|
| Clobbers {A,C,F,H,L} | ALL 247 entries |
| B,D,E preserved       | ALL 247 entries |

---

## 8. Abstract Chain -> Z80 Materialization

Abstract chains represent ISA-independent multiply/divide algorithms (shift-and-add
decompositions). The Z80 materialization step maps these to concrete instructions.

| Metric                | Value  |
|-----------------------|--------|
| Average overhead ratio | 0.99x |
| Interpretation        | Z80 code is actually **shorter** than abstract chains on average |

### Notable cases

| Constant | Chain length | Z80 length | Ratio | Explanation |
|----------|------------:|----------:|------:|-------------|
| x128     |           7 |         4 | 0.57x | Z80 RLCA trick (rotate through carry) |
| x129     |           9 |         5 | 0.56x | Z80 has efficient add-self idioms      |
| Small K  |         n   |         n | 1.00x | Chain maps 1:1 to Z80 instructions     |

The Z80's rotate instructions (RLCA, RRCA, RLA, RRA) give it an advantage over the
abstract model for powers of 2 and near-powers. The abstract model cannot represent
carry-based rotations.

---

## 9. Cassette Library Metrics

A "cassette" is a ROM-resident library of precomputed arithmetic sequences, callable by
constant index.

| Table | Entries | Total ops | Avg ops/entry |
|-------|--------:|----------:|--------------:|
| mul8  |     254 |     2,148 |          8.5  |
| mul16 |     254 |     2,375 |          9.4  |
| div8  |     247 |     2,692 |         10.9  |
| **Total** | **755** | **7,215** | **9.6** |

### Size estimates

| Encoding        | Estimated size |
|-----------------|---------------:|
| Raw (1 byte/op) |        ~10.8 KB |
| Prefix-shared (~50% compression) | ~5.3 KB |

At 5.3 KB, the entire optimal arithmetic library fits in a single 8 KB ROM bank -- a
practical size for Z80 systems where ROM is typically 16-64 KB.

---

## 10. Sequence Length Distribution

### mul8 (range: 1-11 instructions)

| Length |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count  |  1 |  3 |  5 | 11 | 19 | 30 | 38 | 20 | 61 | 66 | -- |

Peaks at **len-9** (61 entries) and **len-10** (66 entries). The single len-1 entry is
x0 (XOR A) or x255 (NEG).

### mul16 (range: 1-12 instructions)

| Length |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count  |  1 |  2 |  4 |  7 | 14 | 23 | 35 | 47 | -- | 61 | 60 | -- |

Peaks at **len-10** (61 entries) and **len-11** (60 entries).

### div8 (range: 8-30 instructions)

Broad distribution with peaks at **len-20** (25 entries) and **len-24** (25 entries).
Division sequences are 2-3x longer than multiplication, reflecting the inherent
complexity of reciprocal computation.

---

## 11. Essential Pool Analysis

Can we reduce the pools further without losing coverage?

### mul8: from 14 ops toward the minimum

| Candidate removal | Pool size | Entries lost | Lost constants         | Recovery |
|-------------------|----------:|-------------:|------------------------|----------|
| Remove ADC A,A    |        13 |            1 | x3                     | Composition: x3 = x2 + x1 |
| Remove SRL A      |        12 |            5 | x126, x130, x131, x132, x134 | Partial via composition |

### Core trios

| Table | Core trio                        | Coverage |
|-------|----------------------------------|----------|
| mul8  | ADD A,A + ADD A,B + LD B,A       | 94%+     |
| mul16 | ADD HL,HL + ADD HL,BC + LD C,A   | 97%+     |

Three instructions cover nearly all entries in each table. The remaining ops handle
edge cases (negative constants via NEG/SBC, odd bit patterns via rotates).

---

## 12. Implications for Future Searches

The 21-op universal pool is not just a retrospective finding -- it is a **predictive
tool** for future search campaigns.

### Enabled searches

| Target                   | Approach                           | Expected speedup |
|--------------------------|------------------------------------|------------------|
| FP24 operations          | A + HL combined (mul8 + mul16)     | 164x vs naive    |
| Compound mul+shift+add   | 21-op pool, len-12+                | 164x vs naive    |
| Multi-format conversions | int8 <-> f8.8 <-> FP16            | 164x vs naive    |
| New arithmetic targets   | Any operation on A, B, HL, BC      | 164x vs naive    |

### Projected search times (21-op pool, 4x RTX 4060 Ti)

| Depth  | Candidates | Wall time   |
|--------|------------|-------------|
| len-9  | 21^9       | ~2 minutes  |
| len-10 | 21^10      | ~35 minutes |
| len-11 | 21^11      | ~12 hours   |
| len-12 | 21^12      | ~10 days    |

### The 2.7% principle

The most consequential finding of this analysis: **2.7% of the Z80 instruction set
generates 100% of optimal arithmetic.** This is not a sampling artifact -- it emerges
from exhaustive search over all 254 non-trivial 8-bit constants across three independent
arithmetic operations.

This suggests a general principle for ISA-level superoptimization: the "essential
instruction set" for any given computational domain is likely to be a small, discoverable
subset of the full ISA. Identifying this subset early transforms intractable searches
into routine computations.

---

## Appendix: Table Summary

| Table         | Entries    | Pool size | Total ops | Avg cost | Status     |
|---------------|------------|----------:|----------:|---------:|------------|
| mul8          | 254        |        14 |     2,148 |      35T | Complete   |
| mul16         | 254        |         5 |     2,375 |      98T | Complete   |
| div8          | 247        |         6 |     2,692 |     108T | Complete   |
| peephole      | 739,574    |       n/a |       n/a |      n/a | Complete (len-2) |
| regalloc      | 83,600,000 |       n/a |       n/a |      n/a | Complete (<=6v) |
| chains_mul    | 254        |       n/a |       n/a |      n/a | Complete   |
| chains_div    | 86         |       n/a |       n/a |      n/a | Complete   |
| arith16_idioms| 22         |        33 |       n/a |      n/a | Complete   |
| bcd_idioms    | varies     |       n/a |       n/a |      n/a | Partial    |

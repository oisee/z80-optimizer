# Z80 Exhaustive Register Allocation Tables

Provably optimal register allocation for every possible constraint shape
on the Z80 CPU, up to 6 virtual registers. 83.6 million entries, ~41MB compressed.

## What's in the tables?

For every theoretically possible combination of:
- **virtual register count** (2-6)
- **register widths** (8-bit or 16-bit per vreg)
- **allowed location sets** (e.g. "must be A", "any GPR", "any pair")
- **interference graph** (which vregs are simultaneously live)

...the table contains the **provably optimal** physical register assignment
(minimum T-states cost), or a proof that **no valid assignment exists**.

## Files

| File | Entries | Feasible | Compressed |
|------|---------|----------|------------|
| `exhaustive_4v.bin.zst` | 156,506 | 123,453 (78.9%) | 64KB |
| `exhaustive_5v.bin.zst` | 17,366,874 | 11,762,983 (67.7%) | 8.5MB |
| `exhaustive_6v_dense.bin.zst` | 66,118,738 | TBD | ~32MB |

The 6v table covers only "dense" shapes (treewidth >= 4, which is 1.7% of
all 6v shapes). The remaining 98.3% have treewidth <= 3 and are composed
at query time from the 5v table via cut-vertex decomposition (max 12T
overhead, verified on 13.2M shapes).

## How to decompress

```bash
# Requires: zstd (apt install zstd / brew install zstd)

zstd -d data/exhaustive_5v.bin.zst -o exhaustive_5v.bin

# Or decompress all:
for f in data/*.bin.zst; do zstd -d "$f"; done
```

## Binary format

Each `.bin` file has a header followed by records in deterministic
enumeration order (shape N in file = shape N from `regalloc-enum`).

### Header (8 bytes)

| Offset | Size | Content |
|--------|------|---------|
| 0 | 4 | Magic: `Z80T` (ASCII) |
| 4 | 4 | Version: `1` (uint32 little-endian) |

### Records (one per shape, variable length)

**Infeasible** (no valid assignment exists):

| Offset | Size | Content |
|--------|------|---------|
| 0 | 1 | `0xFF` marker |

**Feasible** (optimal assignment found):

| Offset | Size | Content |
|--------|------|---------|
| 0 | 1 | nVregs (uint8) |
| 1 | 2 | cost in T-states (uint16 little-endian) |
| 3 | nVregs | assignment: one byte per vreg = physical location index |

### Physical location indices

| Index | Z80 Register | Type |
|-------|--------------|------|
| 0 | A | 8-bit accumulator |
| 1 | B | 8-bit GPR |
| 2 | C | 8-bit GPR |
| 3 | D | 8-bit GPR |
| 4 | E | 8-bit GPR |
| 5 | H | 8-bit GPR |
| 6 | L | 8-bit GPR |
| 7 | BC | 16-bit pair |
| 8 | DE | 16-bit pair |
| 9 | HL | 16-bit pair |
| 10 | IXH | 8-bit IX half |
| 11 | IXL | 8-bit IX half |
| 12 | IYH | 8-bit IY half |
| 13 | IYL | 8-bit IY half |
| 14 | mem0 | memory slot |

## Reading the tables

### Python

```python
import struct

def read_table(path):
    """Yield (index, cost, assignment) for each shape."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'Z80T', "bad magic"
        version = struct.unpack('<I', f.read(4))[0]

        index = 0
        while True:
            b = f.read(1)
            if not b:
                break
            if b[0] == 0xFF:
                yield (index, -1, [])  # infeasible
            else:
                nv = b[0]
                cost = struct.unpack('<H', f.read(2))[0]
                assignment = list(f.read(nv))
                yield (index, cost, assignment)
            index += 1

# Example: print all feasible 4v shapes
LOC_NAMES = ['A','B','C','D','E','H','L','BC','DE','HL',
             'IXH','IXL','IYH','IYL','mem0']

for idx, cost, assign in read_table('exhaustive_4v.bin'):
    if cost >= 0:
        regs = [LOC_NAMES[a] for a in assign]
        print(f"Shape {idx}: {cost}T -> {regs}")
```

### Go

```go
// Read header
magic := make([]byte, 4)
io.ReadFull(r, magic) // "Z80T"
var version uint32
binary.Read(r, binary.LittleEndian, &version)

// Read records
for {
    var marker uint8
    if err := binary.Read(r, binary.LittleEndian, &marker); err != nil {
        break
    }
    if marker == 0xFF {
        // infeasible
        continue
    }
    nv := marker
    var cost uint16
    binary.Read(r, binary.LittleEndian, &cost)
    assign := make([]byte, nv)
    io.ReadFull(r, assign)
    // use cost, assign...
}
```

## Looking up a specific constraint shape

Tables are indexed by **enumeration order** from `regalloc-enum`.
The enumerator iterates in this fixed order:

```
for nVregs in 2..maxVregs:
  for widthCombo in 0..2^nVregs:        # bit K = vreg K is 16-bit
    for locSetCombo in 0..product(locSetCounts):  # per-vreg loc set
      for interferenceGraph in 0..2^(nVregs*(nVregs-1)/2):  # edge bitmask
        → shape at this index
```

To look up a shape at runtime: compute its enumeration index from
(nVregs, widths, locSets, interference) using the same nested loop
order. This is O(1) arithmetic, no search needed.

## How the tables were generated

```bash
# 1. Enumerate all possible constraint shapes
./regalloc-enum --max-vregs 5 > shapes.jsonl
# 17,366,874 shapes for ≤5v

# 2. GPU brute-force: try all 15^N assignments per shape, find optimal
cat shapes.jsonl | cuda/z80_regalloc --server > results.jsonl
# 2x NVIDIA RTX 4060 Ti 16GB, ~20 minutes for ≤5v

# 3. Pack into compact binary + compress
python3 pack_table.py results.jsonl exhaustive_5v.bin
zstd -19 exhaustive_5v.bin  # 1.3GB JSONL → 8.5MB

# 4. For 6v: only enumerate dense shapes (treewidth ≥ 4)
#    562 out of 32,768 possible interference graphs are dense
./regalloc-enum --max-vregs 6 --only-nv 6 \
    --dense-masks dense_6v_masks.txt > shapes_6v.jsonl
cat shapes_6v.jsonl | cuda/z80_regalloc --server > results_6v.jsonl
# 66M shapes, ~5.7 hours dual GPU
```

## Coverage strategy

| vregs | Method | Coverage | Table size |
|-------|--------|---------|------------|
| ≤5v | Exhaustive GPU enumeration | 100% — zero misses | 8.5MB |
| 6v (tw≤3) | Composition from ≤5v at query time | 98.3% of 6v shapes | 0 (computed) |
| 6v (tw≥4) | Direct GPU solve | 1.7% of 6v shapes | ~32MB |
| 7v+ | On-demand GPU/Z3 + caching | corpus-driven | varies |

## Key findings

### Feasibility phase transition

| vregs | Feasible | Infeasible |
|-------|----------|------------|
| 2 | 95.9% | 4.1% |
| 3 | 88.5% | 11.5% |
| 4 | 78.7% | 21.3% |
| 5 | 67.7% | 32.4% |
| 6 | 0.9% | 99.1% |

The Z80 register file "fills up" at 6 virtual registers — 99.1% of all
possible 6v constraint shapes have no valid assignment.

### Composition verification (13.2M data points)

5v shapes composed from 4v table via cut-vertex splitting:
- Zero missed solutions
- Average overhead: 5.06 T-states
- Maximum overhead: 12 T-states (~3 extra register moves)

### Treewidth

- 99.5% of random interference graphs have treewidth ≤3
- Compiler-generated graphs are denser: 53.7% of dense corpus functions have tw≥4
- All tw=4 corpus functions have ≤15v (solvable by backtracking in <1s)

## Also stored at

- **NAS**: `/mnt/safe/z80-compiler/tables/` (raw JSONL + compressed)
- **minz repo**: `research/paper-a/data/tables/`

Generated: 2026-03-26, Hardware: 2x NVIDIA RTX 4060 Ti 16GB

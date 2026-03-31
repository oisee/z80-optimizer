# Seed Stream Compressibility Analysis
**Date:** 2026-03-31
**Topic:** How well do the found LFSR-16 seed streams compress?

---

## Overview

Each frame of our LFSR-16 animation is represented as a list of seed records.
Each record is logically a 5-tuple: `(seed: u16, and_n: u3, blk: u2, ox: u4, oy: u4)` = **29 bits useful information**.

We analyze four datasets encoding the same 25-frame Che animation at different budgets.

---

## Datasets

| Dataset         | Seeds | Frames | Seeds/frame | Seeds >255 |
|----------------|-------|--------|-------------|------------|
| budget-64 (wgt) | 2,585 | 25 | 103.4 | 90% |
| budget-128 (wgt)| 4,116 | 25 | 164.6 | 94% |
| canonical (∞)   | 16,114| 25 | 644.6 | 83% |
| CP-u8           | 280   | 25 | 11.2  | 92% |

---

## Compressibility Results

### JSON (current format)

| Dataset | JSON raw | gzip | ratio | bits/seed |
|---------|----------|------|-------|-----------|
| budget-64  | 258 KB | 24 KB | **9%** | 75.5 |
| budget-128 | 424 KB | 37 KB | **8%** | 72.6 |
| canonical  | 852 KB | 115 KB | **13%** | 57.3 |
| CP-u8      | 30 KB  | 3.6 KB | **12%** | 103.6 |

JSON compresses well (8-13% of raw) because field names (`"seed"`, `"and_n"`, etc.) repeat every record. But the raw JSON is still bloated.

### Compact Binary (4 bytes/seed)

Pack as: `seed(u16) | (and_n-3)(u3) | blk_enc(u2) | pad(u3) | (ox/8)(u4) | (oy/8)(u4)`

| Dataset | Binary raw | gzip | ratio | bits/seed |
|---------|-----------|------|-------|-----------|
| budget-64   | 10.1 KB | 8.8 KB | **87%** | **27.9** |
| budget-128  | 16.1 KB | 14.1 KB | **87%** | **28.0** |
| canonical   | 62.9 KB | 51.2 KB | **82%** | **26.0** |
| CP-u8       | 1.0 KB  | 0.9 KB  | **90%** | **28.8** |

Binary barely compresses. This is the key finding.

---

## Why Seeds Don't Compress

### Seed values are nearly white noise

```
budget-64:   seed entropy = 10.99 bits  (theoretical max: 11.0 bits)
budget-128:  seed entropy = 11.76 bits  (max: 16.0)
canonical:   seed entropy = 11.46 bits
CP-u8:       seed entropy =  7.94 bits  (u8 seeds only → 8 bit max)
```

The search selects the **best** seed from a large pool — by design this looks like a pseudo-random sample. Delta coding makes it worse (seed delta entropy ≈ 11.88 bits vs 11.76 raw).

**Seeds are incompressible by construction.** This is an inherent property of brute-force search: the selected seeds look uniformly distributed in [1, 65535].

### Fields that DO compress

| Field | Entropy | Compressed to |
|-------|---------|---------------|
| `and_n` | **1.6 bits** | **1% of raw** (gzip: 73B for 4116 values) |
| `blk`   | **0.06 bits** | **<1% of raw** — 99.3% of seeds are blk=1 |
| `ox/8`  | 3.6 bits | 47-52% of raw |
| `oy/8`  | 3.3 bits | 42-49% of raw |
| `seed`  | ~12 bits | **98-100% of raw** — incompressible |

### Columnar vs interleaved

Separating fields into columns and compressing each independently:

| Approach | budget-128 | ratio |
|----------|-----------|-------|
| Interleaved 6B/seed | 24.7 KB → 13.4 KB gz | 54% |
| Columnar (all concat) | 24.7 KB → 12.3 KB gz | **49%** |

Modest 10% improvement from columnar layout. Not worth the complexity.

---

## Key Findings

### 1. The seed value IS the payload

~28 bits/seed is close to the theoretical minimum of ~12 bits (seed entropy) + ~5 bits (position) = **17 bits**. The gap (28 vs 17) comes from:
- Quantized position grid (ox/8, oy/8) wastes ~3 bits
- Padding in packing wastes ~3 bits
- 94% of seeds are u16 (not u8)

**Optimal packing**: if we restricted to blk=1 only (99.3% of records), drop blk field:
`seed(u16) + and_n(u3, packed 0-4) + ox/8(u4) + oy/8(u4)` = **27 bits → 4 bytes**.

### 2. and_n and blk are free

`and_n` and `blk` compress to essentially zero. Can be arithmetic-coded from learned phase distributions:
- 47% AND-7, 35% AND-6, 15% AND-5 (budget-128)
- blk: 99.3% are blk=1

Coding `and_n` as a 3-value Huffman: ~1.6 bits/seed instead of 3 bits. Saves ~6KB on canonical.

### 3. Position is compressible but not dramatically

ox/8 and oy/8 each compress to ~47% of raw. There's spatial clustering — seeds tend to reuse nearby positions within a frame — but it's not strong enough for dramatic gains.

### 4. CP records are extremely compact

The CP format (1 carrier + 3 payloads per frame) achieves the same visual result as 103 seeds with only ~11 seeds — **10× reduction**. At that scale, compressibility is irrelevant (a 25-frame CP animation is ~1KB binary).

### 5. JSON overhead is enormous but gzip rescues it

Raw JSON for budget-64: 258 KB → gzipped: 24 KB. The 10× compression comes entirely from repetition of field names and structured patterns, not from seed regularity.

---

## Practical Sizes for ZX Spectrum Context

| Format | 25 frames | 128 frames | bits/frame |
|--------|-----------|------------|------------|
| JSON gzipped | 24-115 KB | 120-590 KB | — |
| Binary 4B/seed (budget-64) | **10 KB** | **52 KB** | 3,307 bits |
| Binary 4B/seed (budget-128)| **16 KB** | **84 KB** | 5,267 bits |
| CP binary (4B/seed, ~12/fr)| **0.3 KB**| **1.5 KB** | **384 bits** |
| ZX Spectrum tape speed | — | — | ~1200 bits/s |

At tape speed: budget-64 (3,307 bits/frame) takes **2.7 seconds per frame** to load. CP (384 bits/frame) takes **0.32 seconds per frame** — feasible for a live loader intro!

---

## Recommendations

1. **For browser playback (current)**: JSON + gzip is fine. 24-37 KB for 25 frames loads instantly.

2. **For ZX Spectrum demo use**:
   - Use CP mode (~12 seeds/frame → ~48 bytes/frame binary)
   - Or budget-64 with u8 seeds (~103 seeds/frame × 4B = 412 bytes/frame, too slow for tape)
   - CP is the only viable path for real-time tape loading

3. **Binary format design** (if needed):
   - 2B seed (u16) + 1B packed(and_n:3, ox_hi:1, pad:4) + 1B packed(ox_lo:3, oy:4) = 4B/seed
   - No compression needed — fields are already near-entropy
   - Skip `blk` field entirely (implicit from phase schedule)

4. **Entropy coding would help ~35%**: Huffman/arithmetic on seed values using per-phase distributions could reach ~17 bits/seed (from 28). But implementation complexity outweighs the gain for browser use.

---

## Files Analyzed

- `data/che_weighted64_anim.json` — 25 frames, budget 64, heatmap-weighted
- `data/che_wgt128_anim.json` — 25 frames, budget 128, heatmap-weighted
- `data/che_anim_flat.json` — 25 frames, canonical (650 seeds/frame)
- `data/che_cp25_anim.json` — 25 frames, CP carrier-payload mode
- Catalog: `data/carrier_catalog.bin` — 9.4 MB, andN 3-8, 65535 seeds (built 236ms)

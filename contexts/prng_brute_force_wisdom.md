# pRNG Brute-Force — Accumulated Wisdom
**Compiled:** 2026-04-01
**Sessions covered:** Day 5 (foveal), Day 6 (animation pipeline), Day 7 (joint-2, morphing), Day 8 (CP encoding, catalog, video pipeline)

This file is the **seed for future PRNG/LFSR sessions**. Read this before starting any new work on image search, animation, or packing.

---

## 1. The Core Idea

**LFSR-16** (poly 0xB400, 65535 non-zero states) drives a cascade of AND-filtered bits.
For each seed, run the LFSR for 768 steps, AND every N consecutive outputs → 768-element binary buffer → place on canvas as blocks.

```
seed → LFSR stream → AND-N filter → buf[768]  (32×24 grid of blocks)
buf → render at (ox, oy) with block size blk  → XOR onto canvas
```

**Key parameters:**
- `blk` ∈ {1, 2, 4, 8}: pixel block size. blk=8 → 192 valid blocks (8×8 pixels each), blk=1 → 768 blocks (1×1)
- `andN` ∈ [3..8]: density. AND-3 → ~22% blocks active, AND-7 → ~1%. Higher andN = sparser = finer correction
- `ox, oy`: position offset, snapped to GRID=8
- `warmup`: LFSR pre-advance steps (usually 0)

**GPU kernel** (`searchKernel`): one thread per seed (65535 threads), tries all positions in list, returns best (seed, position, signed_delta). `delta = Σ(active_pixels: canvas==target ? +w : -w)`. **Negative delta = improvement.**

---

## 2. Phase Schedules

### Keyframe (first frame, from black)

Shrinking-area schedule — coarse seeds scatter wide, fine seeds converge to center:
```
blk=4  AND-3   1 seed   area=100%   (single position, covers full canvas)
blk=2  AND-4   8 seeds  area= 90%
blk=1  AND-5  16 seeds  area= 81%
blk=1  AND-6  64 seeds  area= 73%
blk=1  AND-7 167 seeds  area= 66%
```

### Delta frame (subsequent frames)

Full-area search — errors can be anywhere:
```
blk=2  AND-3   1 seed   (broad bounce, fixes big regions)
blk=1  AND-4   3 seeds
blk=1  AND-5  20 seeds
blk=1  AND-6  50 seeds
blk=1  AND-7  54 seeds  ← 43% of budget; critical for fine detail
```

**Rule of thumb:** AND-7 should get ~40-50% of total delta budget. AND-6 gets ~30-40%. AND-3/4/5 are cheap setup phases.

### Foveal cascade (still image, max quality)

For a single target image, cascade blk=4→2→1 with full 65535 seed search per step. Greedy, 1209 steps → **0.06% error on Che** (28s on RTX 4060 Ti).

---

## 3. Content Analysis — What Works

| Works great | Struggles |
|-------------|-----------|
| Dark background + bright silhouettes | Full-frame busy photo |
| Sparse: <10% lit pixels | Dense: >30% lit pixels |
| Slow motion, portraits | Fast cuts, camera shake |
| High-contrast edges (cartoon, line art) | Smooth gradients |
| Inverted (dark face, white bg) | Positive photos (usually worse) |

**Iconic results:**
- Cat (synthetic): **4.9% error**, 128B — near-perfect for simple shapes
- Ёжик в тумане: avg dt budget used = 149/256 — algorithm stops early, scene too sparse
- Bad Apple: works but needs full budget (>30% pixels/frame on average)
- Che Guevara: best subject for demos, well-studied

**Contrast boost recipe** (for any dark/silhouette video):
```
ffmpeg -i input.mp4 \
  -vf "scale=128:96,format=gray,curves=all='0/0 0.3/0 0.6/1 1/1'" \
  output_contrasted.mp4
```

---

## 4. Methods Hierarchy

### Method 1: Foveal / Quadtree (best quality/size ratio)

Hierarchical: coarse regions first (blk=4), then subdivide error zones.
- Quadtree: `prng_segmented_search --mode quadtree`
- Face-aware: `--mode face` (OpenCV Haar for eyes/nose/mouth attention)
- **Best results**: Che 15%, Marilyn 14.9%, Einstein 15.3%, Mona Lisa 15.2% @ 1194B each

### Method 2: Dual-Layer Evolutionary (best for sparse targets)

5-layer architecture: 3 additive (OR) + 2 subtractive (AND-NOT). Island model CUDA.
- **Best results**: Cat 4.9%, Skull 14.7% @ 128B each
- Kernel: `cuda/prng_hybrid_gpu.cu`

### Method 3: Budget-Constrained + Area-Shrinking (video / animation)

Phase schedule as above. Per-frame JSON output, assembled to `animation_flat`.
- Kernel: `cuda/prng_budget_search.cu`
- ~2-4s/frame on RTX 4060 Ti (independent of budget)

### Method 4: Carrier-Payload (CP) — ultra-low bitrate delta

Two-level hierarchy. See Section 6 below.

### Method 5: Introspec BB Port (demoscene-authentic)

24-bit Galois LFSR, 66 layers, 2×2 XOR plots. Faithful port of BB (Multimatograf 2014).
- 4 minutes per full s0 sweep
- Kernel: `cuda/bb_search.cu`

### Method 6: XOR Morphing Chain

Cumulative canvas: each target bruteforced ON TOP of previous canvas. Seeds correct the delta. `--canvas prev.pgm` flag for chaining.

---

## 5. Animation Pipeline

### Full encode pipeline

```bash
# 1. Build binary (once)
nvcc -O3 -o cuda/prng_budget_search cuda/prng_budget_search.cu -lm

# 2. Encode video
python3 cuda/encode_anim.py \
  --input video.mp4 \
  --out data/my_anim.json \
  --budget 256 --kf-budget 512 \
  --every 3 \                        # take every Nth source frame
  --kf-every 30 \                    # force keyframe every N anim frames
  --name "My Animation" \
  --gpu 0
```

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--budget N` | 128 | Seeds per delta frame |
| `--kf-budget N` | 2×budget | Seeds for keyframe |
| `--every N` | 5 | Source frame stride |
| `--kf-every N` | 0 (off) | Periodic keyframe reset |
| `--kf-error X` | 0 (off) | Adaptive KF when error > X% |
| `--weighted` | off | OpenCV heatmap face/edge priority |
| `--auto-bounce` | off | Auto-pick blk for delta L0 |
| `--cp` | off | Use CP mode for delta frames |
| `--cp-seeds N` | 255 | 255=CPU u8, 65535=GPU u16 |
| `--cp-andN-lo/hi` | 3/8 | Carrier andN search range |
| `--cp-catalog path` | off | Use prebuilt catalog for carrier |
| `--cp-build-catalog path` | — | Build and save catalog, then exit |

### JSON format: animation_flat

```json
{
  "type": "animation_flat",
  "n_frames": 63,
  "total_seeds": 16490,
  "frame_starts": [0, 512, 768, ...],    // expanded seed index per frame
  "frame_sizes":  [512, 256, 248, ...],  // seeds per frame (expanded)
  "frame_types":  ["kf", "dt", "dt", ...],
  "seeds": [ {seed record}, ... ]
}
```

**frame_starts must track EXPANDED indices** — CP records expand to 1+N_payloads entries in the renderer. `_cp_expanded_size(r) = 1 + len(r['ps'])` for CP records, 1 otherwise.

### Typical encode times

| Content | Budget | Time/frame | Total (100fr) |
|---------|--------|-----------|---------------|
| Dense (Bad Apple) | 256 | ~3-4s | ~5-7 min |
| Sparse (Ёжик) | 256 | ~2s | ~3 min |
| Any | 64 | ~1.5s | ~2.5 min |
| CP mode | 32 | ~1s | ~1.7 min |

---

## 6. Carrier-Payload (CP) Encoding — Full Detail

### Concept

Split the delta budget into a hierarchy:

**Carrier (blk=8):** One seed at (ox=0, oy=0) tiles the entire 128×96 canvas into 192 non-overlapping 8×8 blocks. Choose the carrier seed that activates the blocks with the most errors → a coarse "error map" in 1 seed.

**Payloads (blk=4→2→1):** Sub-seed search, scored and applied ONLY within carrier-active 8×8 zones. Focuses the remaining budget.

### Why blk=8 at (0,0) is special

With blk=8 and (ox=0, oy=0), blocks have bx∈[0..15], by∈[0..11] → exactly 192 blocks, perfectly tiling the 128×96 canvas with NO overlap. Any blk=8 seed is a perfect partition of the canvas.

### CP JSON record

```json
{"type":"cp", "cs":3960, "cx":0, "cy":0, "can":6,
 "ps":[[171,0,0,4,3],[194,32,8,2,4],[199,48,72,1,5]]}
```
- `cs`: carrier seed, `cx/cy`: carrier offset (always 0,0), `can`: carrier andN
- `ps`: payloads as `[seed, ox, oy, blk, andN]`

### AND-N density tradeoff for carrier

| andN | ~% blocks active | Good when |
|------|-----------------|-----------|
| 3 | ~22% (42/192) | >20% error density |
| 4 | ~15% (29/192) | ~15% error density |
| 5 | ~10% (19/192) | ~10% error density |
| 6 | ~7% (13/192) | ~7% error density (sweet spot for delta frames) |
| 7 | ~3% (6/192) | very sparse deltas |
| 8 | ~1.5% (3/192) | near-zero error frames |

**Rule:** AND-N ≈ -log2(error_rate). For 8% error frames, AND-6 is optimal.

### AND-3 trap

AND-3 activates 22% of blocks. At 8% frame error density (134/192 "hot" blocks), AND-3 looks great by block-count heuristic — but it flips ~6000 pixels per application, most of which are CORRECT. Actual pixel delta goes **positive** (+886). Always use AND-N ≥ 5 for typical delta frames.

### CP performance vs plain

At budget=32 (kf=128, dt=32) on Ёжик frames 50-82 (source 250-400):
| Mode | Seeds | Error |
|------|-------|-------|
| plain | 448 | ~18% avg |
| weighted | 448 | ~21% avg |
| CP (u16 GPU) | **138** | ~25% avg |

CP uses 3× fewer seeds at ~7% more error. Best for ultra-low bitrate.

---

## 7. Carrier Catalog

### What it is

Precomputed 192-bit bitmap for every (seed 1..65535, andN 3..8) pair. Enables instant CPU carrier selection without any LFSR forward computation.

### File format

```
"CPCT" magic (4B) | andN_lo (u8) | andN_hi (u8) | n_seeds (u32le=65535) |
data: [n_andN layers × 65535 entries × 24 bytes]
Each entry: uint32_t[6] = 192-bit bitmap, bit (by*16+bx) = block active
```

Total: 10 + 6×65535×24 = **9.4MB** for andN 3-8.

### Build and use

```bash
# Build (236ms on RTX 4060 Ti):
./cuda/prng_budget_search --cp-build-catalog data/carrier_catalog.bin \
                          --cp-andN-lo 3 --cp-andN-hi 8

# Use in CP encoding:
./cuda/prng_budget_search --cp --target frame.pgm --init-canvas prev.pgm \
  --cp-catalog data/carrier_catalog.bin \
  --out result.json --out-pgm result.pgm
```

### Query algorithm

1. Build `hot_px[192]`: per-block error count (0-64, actual pixel errors in each 8×8 block)
2. For each of 65535×6 catalog entries: `score = Σ(active blocks: 64 - 2×hot_px[block])`
3. Keep top-16 by score
4. Pixel-rescore top-16 with actual LFSR forward computation → pick true winner

**Why weighted (not binary) hot map**: binary "block has any error" hits AND-3 trap (134/192 hot → AND-3 looks great but is terrible). Weighted by actual error count correctly penalizes large-but-sparse carriers.

---

## 8. Seed Stream Compressibility

### Core finding: seeds are incompressible by construction

The GPU search selects the *best* seed from 65535 — this looks like a uniform random sample.

```
budget-128:  seed entropy = 11.76 bits  → gzip achieves 98% of raw (useless)
budget-64:   seed entropy = 10.99 bits
delta coding: makes it WORSE (11.88 bits)
```

### Field-by-field entropy

| Field | Entropy | gzip ratio |
|-------|---------|-----------|
| `seed` (u16) | ~12 bits | **98%** — incompressible |
| `and_n` | **1.6 bits** | **1%** — nearly free |
| `blk` | **0.06 bits** | **<1%** — 99.3% are blk=1 |
| `ox/8` | 3.6 bits | 47% |
| `oy/8` | 3.3 bits | 42% |

### Optimal binary format

```
4 bytes/seed:
  seed (u16)     — 2 bytes
  (and_n-3):3 | blk_enc:2 | pad:3  — 1 byte
  (ox/8):4 | (oy/8):4               — 1 byte
```

No compression needed. Fields are already at near-entropy. Total size at budget-64: **10KB for 25 frames**. At budget-128: **16KB for 25 frames**.

### ZX Spectrum tape math (1200 bps)

| Format | Bytes/frame | Load time |
|--------|-------------|-----------|
| budget-256 | 1000B | 6.7s — impractical |
| budget-64 | 256B | 1.7s — marginal |
| CP ~4 seeds | **16B** | **0.1s — viable!** |

**CP mode is the only path to real-time ZX Spectrum tape playback.**

---

## 9. Heatmap Weighting

`cuda/make_heatmap.py` generates per-pixel uint8 weight maps via OpenCV Haar cascade (face detection) + edge detection (Canny).

**Effect on searchKernel:** `delta += (cb==tb) ? w : -w` instead of `±1`.
High-weight pixels (face zone) contribute more to score → kernel prioritizes them.

**Results at budget=64:**
- Face zone error: 21.94% → **11.54%** (−47%)
- Overall error: slightly worse (+5.76%) — tradeoff

**Sweet spot:** budget 64-128. At budget 600+, error is near 0% everywhere anyway.

Usage: `--weighted` flag in encode_anim.py, `--weight-map file.wmap` in prng_budget_search.

---

## 10. Joint-2 Optimization

After greedy finds all seeds, re-optimize overlapping pairs jointly.

**How:** Undo seeds i and j, then brute-force all 65535² pairs for their combined region. XOR(A,B) creates patterns neither A nor B produces alone.

**Results:** Left eye: error 500→156 (−68%), right eye: 495→155 (−67%).
**Same bytes, different seeds — purely better combinations.**

CUDA kernel: `cuda/joint2_search.cu`. Grid(65535, 256) × block(256) = 4.3B threads, ~132s per pair.

**When to use:** After foveal cascade, as a polishing pass. Most value for overlapping patches (same region, different scales). Independent patches (left eye vs. right eye) can still benefit marginally.

---

## 11. XOR Morphing

Cumulative canvas: each target bruteforced ON TOP of previous. Delta seeds correct the transition.

**Reverse-pyramid V-shape:** 1→2→4→8→4→2→1 seeds per layer for "dissolve-reconverge" effect. Error profile: 18% → 46% (peak dissolve) → 18% (new face).

**Polarity rule:** Invert white-face targets (Che, Einstein, Lenin) before bruteforce. All targets should be black=subject, white=background for consistent palette.

**Pop-art coloring:** Per-target palette, polarity detection automatic. Zero extra bytes.

---

## 12. Web Player Notes

`docs/renderer.html` — full browser playback.

**CP expansion in renderer:** CP records `{type:'cp', cs, ps:[...]}` must be expanded at load time:
1. Compute `cBuf = makeBuf(cs, cw, can)` — carrier buffer
2. For each payload `[ps, pox, poy, pblk, pan]`:
   - Compute `pBuf = makeBuf(ps, pw, pan)`
   - `masked = maskPayloadBuf(pBuf, pox, poy, pblk, cBuf)` — restrict to carrier zones
   - Push as regular seed entry with `_cpBuf: masked`

**frame_starts tracking:** Must count EXPANDED seeds (CP record = 1 + N_payloads), not raw JSON records.

**Sidebar order (current):** stats → layers → playback → GIF export → dataset presets

---

## 13. Key Files

| File | Purpose |
|------|---------|
| `cuda/prng_budget_search.cu` | Main encoder: keyframe/delta/CP, catalog build/query |
| `cuda/prng_budget_search` | Built binary |
| `cuda/prng_segmented_search.cu` | Quadtree/face-aware segmented search |
| `cuda/prng_hybrid_gpu.cu` | Dual-layer evolutionary (best for sparse) |
| `cuda/bb_search.cu` | Introspec BB port (demoscene authentic) |
| `cuda/joint2_search.cu` | Joint-2 pair optimization |
| `cuda/make_heatmap.py` | OpenCV weight map generator |
| `cuda/encode_anim.py` | Full video→JSON pipeline |
| `docs/renderer.html` | Browser player |
| `data/carrier_catalog.bin` | 9.4MB prebuilt carrier catalog (andN 3-8) |
| `media/prng_images/` | All experiment results and galleries |
| `media/prng_images/README.md` | Methods overview, hall of fame |
| `media/prng_images/foveal_gallery/` | Face-aware scaling, Warhol, all 4 faces |
| `media/prng_images/morph_chain/` | 6-face cumulative morphing + animated GIF |

---

## 14. Open Problems / Next Attack Lines

### P1: CP quality improvement

CP error is ~7% worse than plain at same seed count. Better carrier selection:
- Weight hot_px by **gradient magnitude** (edges matter more than flat regions)
- Try carrier + payload with JOINT optimization (see Section 10)
- Multi-carrier: 2 blk=8 seeds before payloads

### P2: Z80 intro alignment

`cuda/che_intro.asm` (351 bytes) produces noise — LFSR mismatch between CUDA search and Z80 execution. Need to match exactly: warmup count, poly, register layout. This is the path to a working 256-byte ZX Spectrum intro.

### P3: Foveal joint-2 pass

Run joint-2 after foveal cascade: 0.06% → 0.02-0.03% target. Pick overlapping (same-region, different-scale) pairs first.

### P4: Sub-10% error from 256 bytes

Current: 15% @ 1194B (6-level segmented). Add level 7-8, overlap grids between levels, or 32-bit seeds.

### P5: Real-time ZX Spectrum playback

CP at 4 seeds/frame = 16 bytes → 0.1s/frame at tape speed. Need:
- Z80 CP decoder (apply carrier seed, then payloads masked to carrier zones)
- ~50 bytes Z80 decoder code
- Total intro: 50B code + 16B/frame × N frames

### P6: Catalog-accelerated full pipeline

Currently catalog is only used for carrier selection in CP mode. Could accelerate:
- Keyframe search: catalog lookup for blk=8 phase (tiny gain, phases are fast)
- Multi-seed joint search: for a given canvas state, find best N seeds simultaneously using catalog intersection

### P7: Better video contrast pipeline

Current: simple curves `0/0 0.3/0 0.6/1 1/1`. Could use:
- Adaptive histogram equalization (CLAHE) per-frame
- Sobel edge extraction (pure outlines, very sparse)
- Bilateral filter then threshold (cartoon look)

### P8: Animation format for ZX Spectrum

Design the on-tape binary format:
```
[header: n_frames, frame_types[]]
[per frame: n_seeds(u8) + seeds as 4B each, or CP records as 1+1+N×3B]
```
Write Z80 decoder, test in emulator.

---

## 15. Numbers Cheat Sheet

| Metric | Value |
|--------|-------|
| Canvas | 128×96 = 12,288 pixels, 1-bit |
| LFSR | 16-bit poly 0xB400, 65,535 non-zero states |
| buf[768] | 32×24 blocks; only 192 valid for blk=8 at (0,0) |
| AND-3 density | ~22% blocks (42/192 for blk=8) |
| AND-6 density | ~7% blocks (13/192) |
| AND-7 density | ~3% blocks (6/192) |
| Best error, 128B | **4.9%** (cat, dual-layer evo) |
| Best error, 1194B | **14.9-15.3%** (faces, quadtree) |
| Foveal baseline | **0.06%** @ 1209 steps, 28s |
| Catalog build | 236ms, 9.4MB, RTX 4060 Ti |
| Encode speed | ~2-4s/frame (budget-independent) |
| CP seeds/frame | ~4 (1 carrier + 3 payloads) |
| CP bytes/frame | ~16B (4B/seed binary) |
| Tape viable | CP only (0.1s/frame vs 1.7s for budget-64) |
| GPU | 2× RTX 4060 Ti 16GB (main), RTX 2070 (i5), RX 580 (i3) |

# LFSR-16 AND-Cascade Animation — Canonical Snapshot

**Date:** 2026-03-31
**Result:** 25-frame animation, avg **0.072% error**, **16,114 total seeds** (delta encoding)
**Hardware:** RTX 4060 Ti 16GB, CUDA 12.0, ~28s per frame
**Source video:** `сhe-anima-2.mp4` (5.21s, 24fps, 624×624 → every 5th frame → 128×96 binary)

---

## What It Does

This encodes a short animation of Che Guevara as a sequence of LFSR-16 AND-cascade seeds.
Each seed is a 16-bit integer. The decoder is ~20 lines of code on any hardware from 1982 onward.

### The core idea

An LFSR-16 with polynomial `0xB400` generates a pseudo-random bitstream of 65535 unique states.
Given a seed `s` and warmup `w`, we fast-forward the LFSR by `w` steps, then take 768 consecutive
bits (32×24 blocks), AND each block's `N` consecutive bits together (probability 2^-N of being 1),
and XOR that sparse mask into the canvas at position `(ox, oy)` with block size `blk`.

```
lfsr16(s) = (s >> 1) ^ (s & 1 ? 0xB400 : 0)

buf[block] = AND(bit_0, bit_1, ..., bit_{N-1})   # 1 with prob 2^-N
canvas XOR= buf at (ox, oy, blk)                  # XOR = free undo
```

The CUDA search finds the seed that most reduces the pixel error — across ALL 65,535 seeds and
ALL ~130 valid positions — in one kernel launch. 8.5 million combinations evaluated per step (~23ms).

---

## Delta Encoding

Instead of encoding each frame from scratch (independent), we **start each frame from the
canvas of the previous frame** and run the cascade to adjust toward the new target.

```
canvas_0 = blank (all zeros)
canvas_1 = foveal_cascade(target=frame_1, init=canvas_0)   # ~1049 seeds
canvas_2 = foveal_cascade(target=frame_2, init=canvas_1)   # ~542 seeds
canvas_3 = foveal_cascade(target=frame_3, init=canvas_2)   # ~632 seeds
...
canvas_25 = foveal_cascade(target=frame_25, init=canvas_24) # ~497 seeds
```

The canvas evolves continuously — it's never reset. Each frame's seeds are a *correction* to
the previous state, not a full repaint.

### Why this works

XOR is its own inverse: applying a seed buffer twice cancels it. So the cascade naturally
"undoes" what no longer belongs and "adds" what does. With a good initial canvas (the previous
frame), the cascade only needs to correct the changed regions.

Later frames benefit from temporal coherence: frames 19-22 achieved **0.02–0.03%** because
the scene barely changed and the canvas was already close.

---

## Phase Schedule (foveal cascade)

| Phase | AND-N | Block | Prob(1) | Patch size | Seeds |
|-------|-------|-------|---------|------------|-------|
| L0    | 3     | 4×4   | 12.5%   | 128×96     | 1     |
| L1    | 3     | 2×2   | 12.5%   | 64×48      | 8     |
| L2    | 4     | 1×1   | 6.25%   | 32×24      | 16    |
| L3    | 5     | 1×1   | 3.1%    | 32×24      | 128   |
| L4    | 6     | 1×1   | 1.6%    | 32×24      | 256   |
| L5    | 7     | 1×1   | 0.78%   | 32×24      | 800   |

Total budget: 1209 seeds per frame (used: 497–1049 depending on frame).

---

## Foveal Position Search — The Key Insight

At each step the CUDA kernel evaluates all 65,535 seeds × all ~130 positions.
The critical fix: rank positions by **delta** (error reduction), not by **newErr** (absolute error).

```
// WRONG: always polishes the already-good corner
if (baseErr[pos] + delta < bestNewErr) { bestNewErr = ...; }

// CORRECT: finds the most error-dense region
if (delta < bestDelta) { bestDelta = delta; bestPos = pos; }
```

This makes the algorithm "foveal": it naturally concentrates on the face (highest error density)
without any explicit face detection. Position (56,24) — the face region — was found at step 5.

---

## Results

### Per-frame error (delta encoding)

| Frame | Mode        | Seeds used | Error  |
|-------|-------------|------------|--------|
| 001   | independent | 1049       | 0.09%  |
| 002   | delta       | 542        | 0.07%  |
| 003   | delta       | 632        | 0.10%  |
| 004   | delta       | 610        | 0.07%  |
| 005   | delta       | 651        | 0.11%  |
| 006   | delta       | 638        | 0.11%  |
| 007   | delta       | 676        | 0.11%  |
| 008   | delta       | 669        | 0.11%  |
| 009   | delta       | 655        | 0.08%  |
| 010   | delta       | 501        | 0.11%  |
| 011   | delta       | 679        | 0.11%  |
| 012   | delta       | 716        | 0.06%  |
| 013   | delta       | 724        | 0.08%  |
| 014   | delta       | 727        | **0.02%** |
| 015   | delta       | 736        | 0.05%  |
| 016   | delta       | 665        | 0.10%  |
| 017   | delta       | 518        | 0.07%  |
| 018   | delta       | 674        | 0.05%  |
| 019   | delta       | 705        | **0.02%** |
| 020   | delta       | 711        | 0.03%  |
| 021   | delta       | 660        | 0.03%  |
| 022   | delta       | 424        | **0.02%** |
| 023   | delta       | 520        | 0.04%  |
| 024   | delta       | 535        | 0.09%  |
| 025   | delta       | 497        | 0.07%  |

### Delta vs independent comparison

| Metric            | Independent | Delta(L0) |
|-------------------|-------------|-----------|
| Avg error         | 0.094%      | **0.072%** |
| Total seeds       | 26,268      | **16,114** |
| Seeds saved       | —           | **−39%**  |
| Best frame        | 0.05%       | **0.02%** |
| Frames where better | —         | 14/25     |

Delta wins overall and uses 39% fewer seeds. The canvas "warms up" — later frames
in a coherent sequence need far fewer corrections.

---

## Files

```
anim_canonical/
├── README.md                    — this file
├── result_001.pgm … result_025.pgm   — encoded frames (delta cascade output)
├── targets/
│   └── frame_001.pgm … frame_025.pgm — original target frames (from video)
├── seeds/
│   └── seeds_001.json … seeds_025.json — per-frame seed records
├── che_anim_flat.json           — combined animation for web renderer (16,114 seeds)
├── anim_delta_gallery.png       — comparison gallery (orig / independent / delta)
└── prng_cascade_search.cu       — CUDA source (--init-canvas, --phase-from flags)
```

### Reproduce

```bash
# Extract frames
ffmpeg -i сhe-anima-2.mp4 \
  -vf "select='not(mod(n,5))',scale=128:96,format=gray" \
  -vsync vfr /tmp/che_anim_frames/frame_%03d.pgm

# Build CUDA
nvcc -O3 -o cuda/prng_cascade_search cuda/prng_cascade_search.cu -lm

# Run delta cascade
mkdir -p /tmp/results
# Frame 1: independent
cuda/prng_cascade_search --target frame_001.pgm --out seeds_001.json --gpu 0
cp /tmp/cuda_cascade_result.pgm result_001.pgm

# Frame N: delta from previous
cuda/prng_cascade_search \
  --target frame_002.pgm \
  --init-canvas result_001.pgm \
  --out seeds_002.json --gpu 0
cp /tmp/cuda_cascade_result.pgm result_002.pgm
# ... repeat for frames 3-25
```

### Play in browser

Open `docs/renderer.html` → select **"Che animation 🎬"** preset.
The renderer loads `data/che_anim_flat.json` and plays all 16,114 seeds as one continuous stream,
showing frame N/25 in the stats panel. Use the speed slider and frame-pause slider to control playback.

---

## Why This Matters

LFSR-16 AND-cascade is a **lossless-capable, streaming image codec** with:
- **Decoder complexity**: ~20 instructions per seed on Z80 (≈ 1983 hardware)
- **Seed size**: 16 bits + 8-bit position + 4-bit params = ~32 bits/seed
- **Animation budget**: 16,114 × 32 bits = **~64KB for 25 frames of 128×96 binary video**
- **No lookup tables, no multiplication, no division** — pure bitwise ops + LFSR shift

The delta approach makes it a viable streaming codec: the encoder needs a GPU to find seeds,
but the decoder is a shift register and a few XOR operations, runnable on any 8-bit CPU at full speed.

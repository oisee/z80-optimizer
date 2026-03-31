# LFSR-16 Animation Pipeline: From Video to ZX Spectrum Intro
**Date:** 2026-03-31

> Turn any MP4 into a playable LFSR-16 animation — and understand why the result is nearly incompressible by construction.

---

## What Is This?

A ZX Spectrum demo intro stores a few hundred bytes and generates a recognizable image on screen using a 16-bit LFSR (Linear Feedback Shift Register). The key insight: instead of storing pixels, store the *seed* that causes the LFSR to paint the right pixels.

This project extends that idea to **video**: encode each frame as a sequence of LFSR seeds, play them back in a browser, and get a recognizable animation from a tiny data stream.

---

## The Pipeline

```
MP4 / YouTube URL
      │
      ▼ ffmpeg (scale 128×96, grayscale, contrast boost)
      │
      ▼ encode_anim.py
      │  ├─ extract frames (--every N)
      │  ├─ [optional] OpenCV heatmap weights
      │  └─ CUDA brute-force per frame
      │       ├─ KEYFRAME: blk=4→2→1, shrinking area
      │       └─ DELTA:    blk=2→1→1→1, full canvas, from prev result
      │
      ▼ animation_flat JSON
      │  { type, n_frames, frame_starts[], frame_sizes[], frame_types[], seeds[] }
      │
      ▼ docs/renderer.html (browser)
         LFSR-16 replay, frame-by-frame, seed list, GIF export
```

---

## CUDA Search: How One Frame Is Encoded

Each seed covers the canvas with a pattern of **blocks** (8×8, 4×4, 2×2, or 1×1 pixels). The LFSR-16 (poly 0xB400) with AND-N consecutive bits controls which blocks are active — AND-7 gives ~1% density, AND-3 gives ~22%.

The GPU kernel (`searchKernel`) tries all 65,535 seeds simultaneously. For each seed it computes the **signed delta** against the current canvas: how many pixels would improve vs. worsen if this seed were applied. The best seed is selected and applied, then the next seed is searched on the updated canvas.

**Phase schedule (delta frame, budget=256):**
```
blk=2  AND-3   1 seed   (coarse bounce, full canvas)
blk=1  AND-4   3 seeds
blk=1  AND-5  20 seeds
blk=1  AND-6  50 seeds
blk=1  AND-7  54 seeds  ← 43% of budget here, finest detail
```

**Keyframe** (first frame or scene cut) uses a shrinking-area schedule: coarse seeds scatter everywhere, fine seeds concentrate toward the image center.

---

## Carrier-Payload (CP) Delta Encoding

Standard delta encoding searches the full canvas for each seed. CP introduces a **two-level hierarchy**:

### Level 0: Carrier (blk=8)

One seed at (0,0) with blk=8 covers the entire 128×96 canvas in 192 non-overlapping 8×8 blocks. The carrier seed is chosen to activate blocks that have the most errors — it's a coarse "error map" in one seed.

```
carrier: { type:'cp', cs:3960, cx:0, cy:0, can:6,
           ps:[[171,0,0,4,3],[194,32,8,2,4],[199,48,72,1,5]] }
```

### Level 1-3: Payloads (blk=4→2→1)

Payload seeds are scored and applied **only within carrier-active zones** — pixels in 8×8 blocks that the carrier touched. This focuses the remaining budget on areas that actually need fixing.

**Result:** 4 seeds total (1 carrier + 3 payloads) vs. 32 seeds in plain mode — **8× fewer seeds** for comparable quality on sparse content.

### CP JSON format

```json
{
  "type": "cp",
  "cs": 3960,   "cx": 0, "cy": 0, "can": 6,
  "ps": [
    [171, 0,  0,  4, 3],
    [194, 32, 8,  2, 4],
    [199, 48, 72, 1, 5]
  ]
}
```
Each payload: `[seed, ox, oy, blk, and_n]`.

---

## Carrier Catalog: O(1) Carrier Search

For each possible (seed, andN) pair, the LFSR generates the same carrier pattern every time. We can precompute all 65,535 × 6 patterns and store them as **192-bit bitmaps** (one bit per 8×8 block).

**Build once** (236ms on RTX 4060 Ti):
```bash
./cuda/prng_budget_search --cp-build-catalog data/carrier_catalog.bin \
                          --cp-andN-lo 3 --cp-andN-hi 8
# Output: 9.4MB file, format "CPCT" + andN range + 65535×6×24 bytes
```

**Query per frame** (CPU, ~2ms):
1. Build 192-bit `hot_bits` mask: which 8×8 blocks have errors?
2. Build `hot_px[192]`: error count per block (0-64)
3. Scan all (65535 × 6) catalog entries: `score = Σ(active blocks: 64 - 2×err_count)`
4. Take top-16 candidates, pixel-rescore with actual delta → best wins

**Use in encoding:**
```bash
./cuda/prng_budget_search --cp --target frame.pgm --init-canvas prev.pgm \
  --cp-catalog data/carrier_catalog.bin \
  --out result.json --out-pgm result.pgm
```

---

## Content Analysis: What Works

| Works great | Struggles |
|-------------|-----------|
| Dark background, bright silhouettes | Full-frame busy content |
| Slow motion, portraits | Fast cuts, camera shake |
| High-contrast edges | Uniform gradients |
| <10% lit pixels/frame | >30% lit pixels/frame |

### Ёжик в тумане (Hedgehog in the Fog)

The 1975 Soviet animated film is near-perfect content: dark silhouettes on misty background, slow movement, high contrast after a simple curves adjustment.

```bash
yt-dlp -f best -o /tmp/yozhik.mkv "https://www.youtube.com/watch?v=Klt8bVaycQw"
ffmpeg -i /tmp/yozhik.mkv \
  -vf "scale=128:96,format=gray,curves=all='0/0 0.3/0 0.6/1 1/1'" \
  yozhik_contrasted.mp4
python3 cuda/encode_anim.py --input yozhik_contrasted.mp4 \
  --out data/yozhik_b256.json --budget 256 --kf-budget 512 --every 5
```

Result: **104 frames, avg delta budget used = 149/256** — the algorithm stops early because the scene is so sparse. Each delta frame changes only a small fraction of the canvas.

---

## Seed Stream Compressibility

A critical question: can we compress the output further?

### Seeds are incompressible by construction

The brute-force search selects the *best* seed from 65,535 candidates. By definition, this looks like a uniform random sample — the found seeds have no predictable structure.

```
budget-128: seed entropy = 11.76 bits  (gzip achieves 98% of raw — useless)
budget-64:  seed entropy = 10.99 bits
```

Delta coding makes it worse: seed[i] - seed[i-1] has even higher entropy.

### Fields that do compress

| Field | Entropy | gzip ratio |
|-------|---------|-----------|
| `and_n` | 1.6 bits | **1%** — nearly free |
| `blk`   | 0.06 bits | **<1%** — 99.3% are blk=1 |
| `ox/8`  | 3.6 bits | 47% |
| `oy/8`  | 3.3 bits | 42% |
| `seed`  | ~12 bits | **98%** — incompressible |

### Optimal binary format

```
4 bytes per seed:
  seed      u16   (2 bytes)
  and_n-3   u3    packed  } 1 byte
  blk_enc   u2    packed  }
  ox/8      u4    packed  } 1 byte
  oy/8      u4    packed  }
```

No compression needed — the fields are already at entropy. This is the correct on-wire format for any streaming or tape-loading use case.

### ZX Spectrum tape math

At 1200 bps tape speed:

| Encoding | Seeds/frame | Bytes/frame | Load time/frame |
|----------|-------------|-------------|-----------------|
| budget-256 | 250 | 1000B | **6.7 seconds** — impractical |
| budget-64  | 64  | 256B  | **1.7 seconds** — marginal |
| CP mode    | ~4  | ~16B  | **0.1 seconds** — viable! |

**CP mode is the only path to real-time ZX Spectrum playback.** A 4-seed CP frame (1 carrier + 3 payloads) encodes in ~16 bytes — fast enough to load and display at ~10 fps from tape.

---

## Web Player

Open `docs/renderer.html`. Presets include:

- **Ёжик в тумане 🦔** — 104 frames, contrast-boosted, budget 256
- **Che Anima 2 🎬** — 63 frames at budget 256 and 64
- **CP variants** — carrier-payload comparison at low budget
- **Lissajous 〰️** — ideal LFSR content (near-zero error)

Controls: play/pause, frame scrubber, per-layer toggle, GIF export.

---

## Build & Run

```bash
# Build CUDA search binary
nvcc -O3 -o cuda/prng_budget_search cuda/prng_budget_search.cu -lm

# Build carrier catalog (once, ~236ms)
./cuda/prng_budget_search --cp-build-catalog data/carrier_catalog.bin

# Encode any video
python3 cuda/encode_anim.py \
  --input your_video.mp4 \
  --out data/my_anim.json \
  --budget 256 --kf-budget 512 \
  --every 3 \
  --name "My Animation"

# Encode with CP (ultra-low bitrate)
python3 cuda/encode_anim.py \
  --input your_video.mp4 \
  --out data/my_anim_cp.json \
  --budget 32 --kf-budget 128 \
  --cp --cp-catalog data/carrier_catalog.bin \
  --every 3
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Canvas | 128×96 pixels, 1-bit |
| LFSR | 16-bit, poly 0xB400, 65,535 states |
| Encoding speed | ~2-4s/frame on RTX 4060 Ti |
| Catalog build | 236ms (9.4MB, andN 3-8) |
| Catalog query | ~2ms/frame (CPU popcount) |
| Best error (Che, 1194B) | **15.0%** — segmented quadtree |
| Best error (Cat, 128B) | **4.9%** — dual-layer evolutionary |
| CP mode overhead vs plain | **3× fewer seeds**, ~5% more error |
| Tape-viable format | CP, ~16 bytes/frame |

---

*Inspired by [BB](https://www.pouet.net/prod.php?which=63074) (Introspec, ZX 256b, Multimatograf 2014) and [Mona](https://www.pouet.net/prod.php?which=62917) (Ilmenit, Atari 256b).*

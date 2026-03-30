# ADR-003: Foveal Descent + Bounded Morph for Animated Portraits

**Date:** 2026-03-30
**Status:** Accepted
**Context:** GPU-bruteforced image search for ZX Spectrum demoscene intros

## Decision

Two-phase rendering for animated portraits from LFSR seeds:

### Phase 1: Face-Aware Descent (first frame)

Progressive resolution increase, full screen coverage:

```
Level 0:  1 seed,   8×8 blocks  → coarse silhouette (full 256×192)
Level 1:  4+ seeds, 4×4 blocks  → face shape + grid quadrants
Level 2:  16+ seeds, 2×2 blocks → eyes, nose, mouth detail
Level 3:  64+ seeds, 1×1 pixels → fine features (pupils, lips, brows)
Level 4+: 256 seeds, 1×1 fine   → pixel-level correction
```

Each level XORs on top of all previous (cumulative). Face-aware regions get extra overlapping seeds at each level. Background gets minimal coverage.

### Phase 2: Bounded Morph (subsequent frames)

Transition between keyframes with **maximum block size = 4×4** (never touch 8×8):

```
Step 1:  1×1 seeds  → fine dissolve (noise at pixel level)
Step 2:  2×2 seeds  → medium dissolve
Step 3:  4×4 seeds  → coarsest disruption (MAX — preserves 8×8 silhouette!)
Step 4:  2×2 seeds  → reconverge to new target
Step 5:  1×1 seeds  → fine convergence
Step 6:  fine seeds → pixel correction
```

The 8×8 block structure from Phase 1 remains **stable through all animation frames**. Only 4×4 and finer detail changes between keyframes.

## Why not bounce to 8×8?

Bouncing to 8×8 destroys the silhouette — the recognizable face outline. The V-shape error profile goes to ~46% (noise) before reconverging. With max 4×4:
- Error peaks at ~35% (still recognizable face shape)
- 8×8 grid stays locked from first frame
- Transitions look like "detail shifting" not "face disappearing"

## Data Budget

```
Phase 1 (descent):   341 seeds × 2 bytes = 682 bytes
Phase 2 (per morph): 420 seeds × 2 bytes = 840 bytes

1 keyframe + 4 morphs = 682 + 4×840 = 4,042 bytes (3.9 KB)
1 keyframe + 9 morphs = 682 + 9×840 = 8,242 bytes (8.0 KB)

Z80 playback code: ~200 bytes
Total for 5-frame animation: ~4.2 KB
Total for 10-frame animation: ~8.4 KB
```

## Z80 Playback

```z80
; For each segment in sequence:
;   1. Load seed (2 bytes from data)
;   2. Init LFSR: state = seed, warm up with seg_id
;   3. For each block in region:
;      - Step LFSR
;      - If bit set: XOR solid block onto screen
;   4. Next segment

; Block XOR masks (precomputed):
;   8×8: XOR byte with 0xFF (CPL)
;   4×4: XOR with 0xF0 or 0x0F (nibble)
;   2×2: XOR with 0xC0/0x30/0x0C/0x03
;   1×1: XOR with 0x80..0x01
```

## Segment Table Compression

213 face-aware segments compress to 14 grid descriptors:
- Each grid: (base_x, base_y, tile_w, tile_h, step_x, step_y, nx, ny, blk) = 9 bytes
- Total: ~211 bytes (was 832 bytes for individual segments)

## Applications

- **256-byte intro**: 1 static face (face-aware descent, 48-170 bytes seeds)
- **512-byte intro**: 1 face + artistic pixelation
- **1K intro**: 1 animated face (descent + 1-2 morphs)
- **4K intro**: animated portrait with 5 keyframes + pop-art color
- **Morphing demo**: 7 iconic faces cycling with V-shape transitions

## References

- `cuda/prng_segmented_search.cu` — CUDA kernel with `--canvas` flag for cumulative morphing
- `media/prng_images/foveal_gallery/` — face-aware scaling experiments
- `media/prng_images/animated_portrait/morph_v3/` — bounded morph implementation
- `media/prng_images/morph_v5/` — pop-art multi-face morphing

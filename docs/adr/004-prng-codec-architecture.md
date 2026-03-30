# ADR-004: PRNG Seed Codec — Unified Static + Dynamic Architecture

**Date:** 2026-03-30
**Status:** Accepted
**Context:** GPU-bruteforced LFSR seed compression for ZX Spectrum images and video

## Core Primitive

One buffer, one function, all resolutions:

```
Buffer: ALWAYS 32×24 = 96 bytes = 768 bits
One LFSR seed (u16) → fills one 32×24 buffer

fn xor_buffer(screen, seed, cx, cy, block_size)
  1. LFSR(seed) → 768 bits
  2. For each bit: if set → XOR solid block at (cx + bx*bs, cy + by*bs)
```

Block sizes control scale:
```
block_size=8: buffer covers 256×192 = full ZX Spectrum screen
block_size=4: buffer covers 128×96, placed at (cx,cy)
block_size=2: buffer covers 64×48
block_size=1: buffer covers 32×24
```

## Static Mode (Keyframe)

Face-aware descent — multiple seeds at different scales:

```
Seed 0:   xor_buffer(seed, 0, 0, 8)     → coarse silhouette (whole screen)
Seed 1-4: xor_buffer(seed, qx, qy, 4)  → quadrant detail
Seed 5+:  xor_buffer(seed, fx, fy, 2)   → face features (eyes, nose, mouth)
Seed N+:  xor_buffer(seed, fx, fy, 1)   → fine detail (pupils, lips)
```

Each seed = 2 bytes. Segment descriptor = (seed: u16, cx: u8, cy: u8, block_size: u8) = 5 bytes.
Or packed: 4 bytes if cx/cy fit in nibbles.

Proven: 213 seeds (426B) → 26.5% error on Che Guevara face.
Periphery stays 8×8 pixelated (artistic), face gets 1×1 detail.

## Dynamic Mode (Delta Frame)

Two-layer masked codec — coarse is both DATA and MASK:

```
Layer 1 — Coarse (8×8):
  seed_coarse → 32×24 buffer → XOR onto screen at block_size=8
  Result: ~80 of 768 char cells flipped
  Side effect: activated cells = MASK for fine layer

Layer 2 — Fine (1×1):
  seed_fine → 32×24 buffer → XOR onto screen at block_size=1
  BUT: only inside char cells that coarse layer activated!
  Fine buffer bit N → pixel at (cx + bx, cy + by) ONLY IF coarse bit for
  that char cell was set
```

Why masking is critical:
- Without mask: fine seed wastes correlation budget on 80% of screen (no-change zones)
- With mask: fine seed's entire 768 bits serve active zone only
- 10× better signal/noise ratio for the same u16 search space

### Data per delta frame

```
seed_coarse (u16) + seed_fine (u16) = 4 bytes per delta frame
```

Joint-2 optimization: both seeds searched together as u32 (65536² = 4.3B candidates).
GPU: 132 seconds per frame. Coarse and fine know about each other.

### Z80 Decoder

```z80
; === One frame decode ===

; Coarse pass
ld hl, (seed_ptr)         ; load seed_coarse
call lfsr_fill_buffer     ; fill 96-byte buffer from LFSR
call xor_buffer_8x8       ; XOR 8×8 blocks onto screen
; Buffer now serves as coarse mask (which cells were activated)

; Fine pass
ld hl, (seed_ptr + 2)     ; load seed_fine
call lfsr_fill_buffer_2   ; fill SECOND 96-byte buffer
call xor_buffer_1x1_masked ; XOR 1×1 pixels ONLY inside coarse-active cells

; xor_buffer_1x1_masked:
;   for each char cell (cx, cy):
;     if coarse_buffer[cy*4 + cx/8] bit (cx%8) is SET:
;       apply fine_buffer bits to this cell's 8×8 pixels
;     else:
;       skip (cell unchanged)
```

Code size:
- lfsr_fill_buffer: ~30 bytes
- xor_buffer_8x8: ~40 bytes (CPL trick, INC H between lines)
- xor_buffer_1x1_masked: ~60 bytes (same + mask check)
- Frame loop: ~20 bytes
- **Total: ~150 bytes decoder**

T-states per frame:
- Coarse: 768 cells × ~12T (check bit + conditional XOR) = ~9K T
- Fine: ~80 active cells × 64 pixels × ~12T = ~61K T
- **Total: ~70K T = 20ms @ 3.5MHz = 50 fps**

### Segment Descriptor

```
Static mode:  (seed: u16, cx: u8, cy: u8, block_size: u8) = 5 bytes
Delta mode:   (seed_coarse: u16, seed_fine: u16) = 4 bytes
```

## Video Bitrate

```
Content             Seeds/frame  Bytes/frame  FPS  Bitrate    3 min total
Keyframe            213          426          —    —          426B
Delta (slow motion) 1+1          4            10   320 bps    5.9 KB
Delta (medium)      1+2          6            10   480 bps    8.8 KB
Delta (fast)        2+3          10           10   800 bps    14.6 KB
Full animation      213/frame    426          5    17 Kbps    46.8 KB
```

## Search Pipeline

### Keyframe (GPU, ~0.5 sec)

```
For each segment (cx, cy, block_size):
  For each seed 0..65535:
    Generate 32×24 buffer from LFSR(seed)
    XOR onto canvas at (cx, cy, block_size)
    Count error vs target
  Pick best seed → lock → next segment
```

### Delta with Joint-2 (GPU, ~132 sec)

```
For each (seed_coarse, seed_fine) in 0..65535 × 0..65535:
  Generate coarse buffer, XOR 8×8 blocks
  Record which cells activated (mask)
  Generate fine buffer, XOR 1×1 ONLY inside mask
  Count error vs target
Pick best (seed_coarse, seed_fine) pair → 4 bytes
```

### Optimization hierarchy

```
Tier 1: Joint u32 (65536²)     — 132s, best quality
Tier 2: Sampled u32 (8192²)    — 2s, 87% of tier 1 quality
Tier 3: Greedy u16 (65536+65536) — 0.001s, baseline quality
```

## Proven Results

- Static Che face4x: 213 seeds, 426B, 26.5% error (face-aware descent)
- Joint-2 left eye: 156/512 err vs greedy 184/512 = −15.2% improvement
- 25-frame Midjourney animation: 10.4KB, 26% per frame
- Nanz decoder: 256 bytes Z80 binary (lfsr_step + screen_addr, MinZ compiled)

## Open Questions

1. Joint-2 on large regions (whole quadrant + face overlay) — improvement?
2. Adaptive mask threshold (not just 50% LFSR) — tune coarse sensitivity?
3. Multiple fine passes with different masks — progressive refinement?
4. 32-bit LFSR seeds — 4 bytes per seed, 4B candidates, better patterns?
5. Temporal seed sharing — reuse coarse seed across similar delta frames?

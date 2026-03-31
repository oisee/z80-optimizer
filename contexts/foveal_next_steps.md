# Foveal Cascade — Next Attack Lines

**Baseline result:** Foveal AND-3→7 cascade, LFSR-16, greedy (seed × position), 0.06% @1209 steps, 28s on RTX 4060 Ti.

---

## Attack Line 1: Joint-2 Layer Optimization

**Idea:** After greedy foveal finds all layers (seeds locked), do a second pass:
pick pairs of overlapping layers and jointly re-optimize their two seeds.

**Why it helps:**
- Greedy is locally optimal (each seed is best given the previous canvas state)
- But two seeds that overlap can cancel each other — joint search finds the true
  2D optimum in that region
- XOR reversibility: to test pair (i, j), undo both seeds (re-apply = XOR undo),
  brute-force 65535² pairs for the combined region, re-apply best

**Implementation sketch:**
```
for each pair (i, j) where patch_i ∩ patch_j ≠ ∅:
    undo seed_i (XOR apply again)
    undo seed_j (XOR apply again)
    brute_force:
        for s1 in 1..65535:
            for s2 in 1..65535:
                delta = improvement of applying buf(s1,pos_i) then buf(s2,pos_j)
                track best (s1, s2)
    apply best (s1, s2)
```

**Search space:** 65535² = 4.3B per pair — too large for CPU, good for CUDA.
Reduce by: fix s1, find best s2 for each s1 (65535 × 130 positions = 8.5M per outer step).

**Practical plan:**
1. Load foveal_cascade_seeds.json
2. Find all pairs where |patch_i ∩ patch_j| > threshold (e.g. > 100 pixels)
3. Pick 4 random overlapping pairs
4. For each pair: undo, joint brute-force (CUDA), re-apply best
5. Compare global error before/after

**Expected gain:** small but measurable — maybe 0.06% → 0.02-0.03%.

---

## Attack Line 2: Animation Delta (Frame-to-Frame Foveal)

**Idea:** Given a sequence of frames (animation), use foveal to encode
each frame incrementally — canvas carries over between frames.

**Protocol:**
```
Frame 0:
    canvas = all zeros
    run full foveal cascade (1209 steps) → canvas_0 ≈ frame_0

Frame 1 (delta from frame_0):
    target = frame_1
    canvas = canvas_0  (already close to frame_0)
    run foveal at blk=2 or blk=1 only (skip blk=4/blk=2 broad strokes)
    → canvas_1 ≈ frame_1
    delta_seeds = seeds added in this pass

Frame N:
    same, starting from canvas_{N-1}
```

**Block size strategy for delta:**
- 8×8 (blk=8) or 4×4 (blk=4) — if frames differ a lot (scene cut)
- 2×2 (blk=2) or 1×1 (blk=1) — if frames are similar (slow motion, pan)
- Auto-detect: if global_error(canvas_prev, frame_N) > threshold → start from blk=4

**Seed encoding:**
Each frame's delta = small list of (seed, ox, oy, andN) records.
On decode: apply seeds sequentially, XOR on canvas.
To "rewind" to frame K: replay from frame 0 through frame K.

**Target animation:**
- ZX Spectrum demo-style: 128×96 binary, ~8-25 fps
- Test with a simple 2-3 frame sequence first (e.g. Che → Einstein → skull)
- Then a proper motion sequence

**Expected data budget:**
- Frame 0: ~939 seeds × 3 bytes = ~2.8 KB
- Delta frame: maybe 50-200 seeds × 3 bytes = 150-600 bytes per frame
- At 10fps: ~6 KB/sec — viable for ZX Spectrum tape/ROM

**Implementation steps:**
1. Take 2-3 target PGMs (existing: che.pgm, einstein_photo_bin.pgm, skull.pgm)
2. Run foveal on frame 0 → save canvas state as binary
3. Run foveal delta (blk=1 only, shorter budget) on frame 1 starting from canvas_0
4. Measure: how many delta seeds needed to reach 1% error on frame 1?
5. Compare to: running full foveal fresh on frame 1

---

## Status
- [x] Foveal cascade baseline: 0.06% @1209 steps, 28s CUDA
- [x] Canonical snapshot: media/prng_images/foveal_canonical/
- [ ] Joint-2 optimization (Attack Line 1)
- [ ] Animation delta encoding (Attack Line 2)

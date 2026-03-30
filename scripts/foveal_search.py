#!/usr/bin/env python3
"""
Foveal recursive image search — multiple region placement strategies.

Strategies:
  1. golden  — golden ratio spiral focus
  2. mondrian — random Mondrian-style rectangles
  3. center  — face-centered concentric
  4. random  — pure random placement
  5. spiral  — Fibonacci lattice spiral

Each strategy produces N regions at varying resolutions.
For each region, brute-force all 65536 seeds to find best match.
"""

import numpy as np
import struct
import sys
import os
from pathlib import Path

# LFSR-16: x^16 + x^14 + x^13 + x^11 + 1 (taps = 0xB400)
def lfsr16_stream(seed, n):
    """Generate n bits from 16-bit LFSR."""
    state = seed & 0xFFFF
    if state == 0:
        state = 1
    bits = []
    for _ in range(n):
        bit = state & 1
        bits.append(bit)
        feedback = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = ((state >> 1) | (feedback << 15)) & 0xFFFF
    return bits

def load_pgm(path):
    """Load PGM (P5 binary) as numpy array."""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'):
                break
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    return data

def binarize(img, threshold=128):
    """Convert grayscale to binary (0/1)."""
    return (img >= threshold).astype(np.uint8)

def save_pgm(path, img):
    """Save binary image as PGM."""
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write((img * 255).astype(np.uint8).tobytes())

# === Region placement strategies ===

PHI = (1 + np.sqrt(5)) / 2

def strategy_golden(img_w, img_h, n_levels=4, seeds_per_level=[1,2,3,4]):
    """Golden ratio spiral focus regions."""
    regions = []
    # L0: whole image
    regions.append({"level": 0, "x": 0, "y": 0, "w": img_w, "h": img_h, "block": 8})

    # Focus point: golden ratio
    cx, cy = img_w / PHI, img_h / PHI

    for lv in range(1, n_levels):
        block = [8, 4, 2, 1][lv]
        n_segs = seeds_per_level[lv]
        # Each segment covers shrinking region around golden point
        for i in range(n_segs):
            angle = i * 2 * np.pi / PHI  # golden angle
            radius = (0.3 - 0.05 * lv) * min(img_w, img_h)
            fx = cx + radius * np.cos(angle) * (0.5 + 0.2 * i)
            fy = cy + radius * np.sin(angle) * (0.5 + 0.2 * i)

            rw = int(img_w * (0.6 - 0.12 * lv))
            rh = int(img_h * (0.6 - 0.12 * lv))
            rx = max(0, min(img_w - rw, int(fx - rw/2)))
            ry = max(0, min(img_h - rh, int(fy - rh/2)))

            regions.append({"level": lv, "x": rx, "y": ry, "w": rw, "h": rh, "block": block})

    return regions

def strategy_mondrian(img_w, img_h, n_levels=4, seeds_per_level=[1,2,3,4], rng=None):
    """Mondrian-style: random rectangular subdivisions."""
    if rng is None:
        rng = np.random.RandomState(42)

    regions = []
    regions.append({"level": 0, "x": 0, "y": 0, "w": img_w, "h": img_h, "block": 8})

    for lv in range(1, n_levels):
        block = [8, 4, 2, 1][lv]
        n_segs = seeds_per_level[lv]

        for _ in range(n_segs):
            # Random rectangle, biased toward center
            cx = rng.normal(img_w * 0.5, img_w * 0.15)
            cy = rng.normal(img_h * 0.45, img_h * 0.12)  # slightly above center (faces)
            rw = int(img_w * rng.uniform(0.25, 0.65 - 0.1 * lv))
            rh = int(img_h * rng.uniform(0.25, 0.65 - 0.1 * lv))
            rx = max(0, min(img_w - rw, int(cx - rw/2)))
            ry = max(0, min(img_h - rh, int(cy - rh/2)))

            regions.append({"level": lv, "x": rx, "y": ry, "w": rw, "h": rh, "block": block})

    return regions

def strategy_center(img_w, img_h, n_levels=4, seeds_per_level=[1,2,3,4]):
    """Concentric face-centered regions."""
    regions = []
    regions.append({"level": 0, "x": 0, "y": 0, "w": img_w, "h": img_h, "block": 8})

    cx, cy = img_w // 2, int(img_h * 0.42)  # face center slightly above middle

    for lv in range(1, n_levels):
        block = [8, 4, 2, 1][lv]
        n_segs = seeds_per_level[lv]

        scale = 0.7 - 0.15 * lv
        rw = int(img_w * scale)
        rh = int(img_h * scale)

        for i in range(n_segs):
            if i == 0:
                # Center region
                rx = max(0, cx - rw//2)
                ry = max(0, cy - rh//2)
            elif i == 1:
                # Upper region (eyes)
                rx = max(0, cx - rw//2)
                ry = max(0, cy - int(rh * 0.8))
            else:
                # Offset regions
                angle = (i - 1) * np.pi / (n_segs - 1)
                ox = int(rw * 0.3 * np.cos(angle))
                oy = int(rh * 0.3 * np.sin(angle))
                rx = max(0, min(img_w - rw, cx - rw//2 + ox))
                ry = max(0, min(img_h - rh, cy - rh//2 + oy))

            regions.append({"level": lv, "x": rx, "y": ry, "w": min(rw, img_w-rx),
                          "h": min(rh, img_h-ry), "block": block})

    return regions

def strategy_random(img_w, img_h, n_levels=4, seeds_per_level=[1,2,3,4], rng=None):
    """Pure random placement."""
    if rng is None:
        rng = np.random.RandomState(123)

    regions = []
    regions.append({"level": 0, "x": 0, "y": 0, "w": img_w, "h": img_h, "block": 8})

    for lv in range(1, n_levels):
        block = [8, 4, 2, 1][lv]
        for _ in range(seeds_per_level[lv]):
            rw = int(img_w * rng.uniform(0.2, 0.6))
            rh = int(img_h * rng.uniform(0.2, 0.6))
            rx = rng.randint(0, max(1, img_w - rw))
            ry = rng.randint(0, max(1, img_h - rh))
            regions.append({"level": lv, "x": rx, "y": ry, "w": rw, "h": rh, "block": block})

    return regions

# === Brute-force search ===

def region_error(target_bin, result, region):
    """Count mismatched pixels in region."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    block = region["block"]

    errors = 0
    total = 0
    for by in range(0, h, block):
        for bx in range(0, w, block):
            px, py = x + bx, y + by
            if px < target_bin.shape[1] and py < target_bin.shape[0]:
                # Compare block average
                bx2 = min(bx + block, w)
                by2 = min(by + block, h)
                px2 = min(px + block, target_bin.shape[1])
                py2 = min(py + block, target_bin.shape[0])
                target_val = target_bin[py:py2, px:px2].mean() >= 0.5
                result_val = result[py:py2, px:px2].mean() >= 0.5 if result[py:py2, px:px2].size > 0 else 0
                if target_val != result_val:
                    errors += 1
                total += 1
    return errors, total

def apply_seed(canvas, region, seed):
    """Apply LFSR seed to a region on canvas (XOR)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    block = region["block"]

    n_blocks_x = max(1, w // block)
    n_blocks_y = max(1, h // block)
    n_blocks = n_blocks_x * n_blocks_y

    bits = lfsr16_stream(seed, n_blocks)

    i = 0
    for by_idx in range(n_blocks_y):
        for bx_idx in range(n_blocks_x):
            if i >= len(bits):
                break
            if bits[i]:
                px = x + bx_idx * block
                py = y + by_idx * block
                px2 = min(px + block, canvas.shape[1])
                py2 = min(py + block, canvas.shape[0])
                canvas[py:py2, px:px2] ^= 1
            i += 1

def search_region(target_bin, canvas, region):
    """Brute-force all 65536 seeds for a region, return best."""
    best_seed = 0
    best_errors = 999999

    for seed in range(1, 65536):
        # Apply seed
        test = canvas.copy()
        apply_seed(test, region, seed)

        # Count errors in region only
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        x2 = min(x + w, target_bin.shape[1])
        y2 = min(y + h, target_bin.shape[0])

        diff = np.sum(test[y:y2, x:x2] != target_bin[y:y2, x:x2])

        if diff < best_errors:
            best_errors = diff
            best_seed = seed

    return best_seed, best_errors

# === Main ===

def run_strategy(name, regions, target_bin, verbose=True):
    """Run foveal search with given regions."""
    canvas = np.zeros_like(target_bin)
    seeds = []
    total_pixels = target_bin.size

    for i, region in enumerate(regions):
        seed, errors = search_region(target_bin, canvas, region)
        apply_seed(canvas, region, seed)
        seeds.append(seed)

        # Global error after this region
        global_err = np.sum(canvas != target_bin)
        pct = global_err / total_pixels * 100

        if verbose:
            print(f"  R{i} (L{region['level']}, {region['block']}×{region['block']}, "
                  f"{region['w']}×{region['h']} at {region['x']},{region['y']}): "
                  f"seed={seed:5d}, global_err={pct:.1f}%")

    final_err = np.sum(canvas != target_bin) / total_pixels * 100
    return canvas, seeds, final_err

def main():
    # Find target
    targets = [
        "media/prng_images/targets/che.pgm",
        "media/prng_images/segmented_che/target.pgm",
    ]
    target_path = None
    for t in targets:
        if os.path.exists(t):
            target_path = t
            break

    if not target_path:
        print("No target found, using synthetic 128×96")
        target = np.random.RandomState(42).randint(0, 2, (96, 128)).astype(np.uint8)
    else:
        raw = load_pgm(target_path)
        # Resize to 128×96 if needed
        if raw.shape != (96, 128):
            from PIL import Image
            img = Image.fromarray(raw).resize((128, 96), Image.NEAREST)
            raw = np.array(img)
        target = binarize(raw)
        print(f"Target: {target_path} ({target.shape[1]}×{target.shape[0]})")

    img_w, img_h = target.shape[1], target.shape[0]

    # Asymmetric: 1+2+3+4 = 10 seeds
    spl = [1, 2, 3, 4]

    strategies = {
        "golden":   strategy_golden(img_w, img_h, seeds_per_level=spl),
        "mondrian": strategy_mondrian(img_w, img_h, seeds_per_level=spl),
        "center":   strategy_center(img_w, img_h, seeds_per_level=spl),
        "random":   strategy_random(img_w, img_h, seeds_per_level=spl),
    }

    # Also try multiple Mondrian random seeds
    for i in range(3):
        strategies[f"mondrian_{i+2}"] = strategy_mondrian(
            img_w, img_h, seeds_per_level=spl, rng=np.random.RandomState(100+i))

    results = {}

    for name, regions in strategies.items():
        n_seeds = len(regions)
        data_bytes = n_seeds * 4  # seed(2) + region(2)
        print(f"\n=== Strategy: {name} ({n_seeds} seeds, {data_bytes} bytes) ===")
        canvas, seeds, err = run_strategy(name, regions, target)
        results[name] = {"error": err, "seeds": seeds, "n_seeds": n_seeds, "bytes": data_bytes}
        print(f"  FINAL: {err:.1f}% error, {n_seeds} seeds, {data_bytes} bytes")

        # Save result
        outdir = f"media/prng_images/foveal_{name}"
        os.makedirs(outdir, exist_ok=True)
        save_pgm(f"{outdir}/result.pgm", canvas)
        save_pgm(f"{outdir}/target.pgm", target)

        # Save comparison (side by side)
        compare = np.zeros((img_h, img_w * 2), dtype=np.uint8)
        compare[:, :img_w] = target
        compare[:, img_w:] = canvas
        save_pgm(f"{outdir}/compare.pgm", compare)

    # Summary
    print("\n=== Summary ===\n")
    print(f"{'Strategy':15s} {'Error':>7s} {'Seeds':>6s} {'Bytes':>6s}")
    for name in sorted(results, key=lambda n: results[n]["error"]):
        r = results[name]
        print(f"{name:15s} {r['error']:6.1f}% {r['n_seeds']:6d} {r['bytes']:6d}")

if __name__ == "__main__":
    main()

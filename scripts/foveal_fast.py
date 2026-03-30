#!/usr/bin/env python3
"""Fast foveal search — numpy vectorized, all strategies."""

import numpy as np
import os, sys

def load_pgm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5'
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'): break
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)

def save_pgm(path, img):
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write((img * 255).astype(np.uint8).tobytes())

def lfsr16_block(seed, n_bits):
    """Generate n_bits from LFSR-16, return as numpy bool array."""
    state = seed & 0xFFFF
    if state == 0: state = 1
    out = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        out[i] = state & 1
        fb = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = ((state >> 1) | (fb << 15)) & 0xFFFF
    return out

def make_block_target(target_bin, x, y, w, h, block):
    """Downsample target region to block grid (majority vote per block)."""
    nbx = max(1, int(w) // int(block))
    nby = max(1, int(h) // int(block))
    n = nbx * nby
    blocks = np.zeros(n, dtype=np.uint8)
    idx = 0
    for by in range(nby):
        for bx in range(nbx):
            px, py = x + bx*block, y + by*block
            px2 = min(px+block, target_bin.shape[1])
            py2 = min(py+block, target_bin.shape[0])
            if py2 > py and px2 > px:
                blocks[idx] = 1 if target_bin[py:py2, px:px2].mean() >= 0.5 else 0
            idx += 1
    return blocks, nbx, nby

def search_best_seed(target_blocks, canvas_blocks):
    """Brute-force all 65536 seeds, find best XOR match."""
    n = len(target_blocks)
    # XOR target with current canvas to get desired flip pattern
    n = min(len(target_blocks), len(canvas_blocks))
    target_blocks = target_blocks[:n]
    canvas_blocks = canvas_blocks[:n]
    desired = target_blocks ^ canvas_blocks

    best_seed = 1
    best_match = 0

    for seed in range(1, 65536):
        bits = lfsr16_block(seed, n)
        match = np.sum(bits == desired)
        if match > best_match:
            best_match = match
            best_seed = seed
            if match == n:
                break  # perfect

    return best_seed, n - best_match  # seed, errors

def apply_seed_to_canvas(canvas, x, y, w, h, block, seed):
    """XOR LFSR pattern onto canvas at block resolution."""
    nbx = max(1, w // block)
    nby = max(1, h // block)
    bits = lfsr16_block(seed, nbx * nby)
    idx = 0
    for by in range(nby):
        for bx in range(nbx):
            if idx < len(bits) and bits[idx]:
                px, py = x + bx*block, y + by*block
                px2 = min(px+block, canvas.shape[1])
                py2 = min(py+block, canvas.shape[0])
                canvas[py:py2, px:px2] ^= 1
            idx += 1

def get_canvas_blocks(canvas, x, y, w, h, block):
    """Read current canvas state as block grid."""
    nbx = max(1, w // block)
    nby = max(1, h // block)
    blocks = np.zeros(nbx * nby, dtype=np.uint8)
    idx = 0
    for by in range(nby):
        for bx in range(nbx):
            px, py = x + bx*block, y + by*block
            px2 = min(px+block, canvas.shape[1])
            py2 = min(py+block, canvas.shape[0])
            if py2 > py and px2 > px:
                blocks[idx] = 1 if canvas[py:py2, px:px2].mean() >= 0.5 else 0
            idx += 1
    return blocks

# === Strategies ===
PHI = (1 + np.sqrt(5)) / 2

def make_regions(strategy, W, H, spl=[1,2,3,4], rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    regions = [{"lv": 0, "x": 0, "y": 0, "w": W, "h": H, "block": 8}]

    for lv in range(1, len(spl)):
        block = [8, 4, 2, 1][lv]
        for i in range(spl[lv]):
            if strategy == "golden":
                angle = i * 2 * np.pi / PHI + lv * 0.5
                r = (0.35 - 0.06*lv) * min(W, H)
                cx = W/PHI + r * np.cos(angle)
                cy = H/PHI + r * np.sin(angle)
                rw = int(W * (0.55 - 0.1*lv))
                rh = int(H * (0.55 - 0.1*lv))
            elif strategy == "mondrian":
                cx = rng.normal(W*0.5, W*0.12)
                cy = rng.normal(H*0.42, H*0.1)
                rw = int(W * rng.uniform(0.3, 0.55 - 0.07*lv))
                rh = int(H * rng.uniform(0.3, 0.55 - 0.07*lv))
            elif strategy == "center":
                cx = W/2 + (i - spl[lv]/2) * W * 0.08
                cy = H*0.42 + (lv - 2) * H * 0.05
                rw = int(W * (0.6 - 0.12*lv))
                rh = int(H * (0.6 - 0.12*lv))
            elif strategy == "random":
                cx = rng.uniform(W*0.2, W*0.8)
                cy = rng.uniform(H*0.15, H*0.75)
                rw = int(W * rng.uniform(0.2, 0.55))
                rh = int(H * rng.uniform(0.2, 0.55))
            else:
                cx, cy = W/2, H/2
                rw, rh = int(W*0.5), int(H*0.5)

            rx = max(0, min(W - max(rw, block), int(cx - rw/2)))
            ry = max(0, min(H - max(rh, block), int(cy - rh/2)))
            rw = min(rw, W - rx)
            rh = min(rh, H - ry)
            # Align to block size
            rw = (rw // block) * block
            rh = (rh // block) * block
            if rw >= block and rh >= block:
                regions.append({"lv": lv, "x": rx, "y": ry, "w": rw, "h": rh, "block": block})

    return regions

def run_search(target_bin, strategy, spl=[1,2,3,4], rng_seed=42):
    H, W = target_bin.shape
    regions = make_regions(strategy, W, H, spl, rng_seed)
    canvas = np.zeros_like(target_bin)
    seeds = []

    for i, r in enumerate(regions):
        tb = make_block_target(target_bin, r["x"], r["y"], r["w"], r["h"], r["block"])
        cb = get_canvas_blocks(canvas, r["x"], r["y"], r["w"], r["h"], r["block"])
        seed, err = search_best_seed(tb, cb)
        apply_seed_to_canvas(canvas, r["x"], r["y"], r["w"], r["h"], r["block"], seed)
        seeds.append(seed)

        global_err = np.sum(canvas != target_bin) / target_bin.size * 100
        print(f"  R{i:2d} L{r['lv']} {r['block']}×{r['block']} {r['w']:3d}×{r['h']:3d} "
              f"@({r['x']:3d},{r['y']:3d}) seed={seed:5d} err={global_err:5.1f}%")

    final_err = np.sum(canvas != target_bin) / target_bin.size * 100
    return canvas, seeds, final_err, regions

# === Main ===
def main():
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
        print("No target, generating synthetic")
        target = np.random.RandomState(42).randint(0, 2, (96, 128)).astype(np.uint8)
    else:
        raw = load_pgm(target_path)
        if raw.shape != (96, 128):
            # Simple resize via block averaging
            sy, sx = raw.shape[0]//96, raw.shape[1]//128
            if sy > 0 and sx > 0:
                raw = raw[:96*sy, :128*sx].reshape(96, sy, 128, sx).mean(axis=(1,3)).astype(np.uint8)
            else:
                raw = raw[:96, :128]
        target = (raw >= 128).astype(np.uint8)
        print(f"Target: {target_path} → {target.shape[1]}×{target.shape[0]}")

    # Asymmetric seeds: 1+2+3+4 = 10 (40 bytes)
    spl_10 = [1, 2, 3, 4]
    # Denser: 1+3+5+7 = 16 (64 bytes)
    spl_16 = [1, 3, 5, 7]

    experiments = [
        ("golden_10",    "golden",   spl_10, 42),
        ("center_10",    "center",   spl_10, 42),
        ("mondrian_10",  "mondrian", spl_10, 42),
        ("mondrian_10b", "mondrian", spl_10, 77),
        ("mondrian_10c", "mondrian", spl_10, 256),
        ("random_10",    "random",   spl_10, 42),
        ("golden_16",    "golden",   spl_16, 42),
        ("mondrian_16",  "mondrian", spl_16, 42),
    ]

    results = {}

    for name, strat, spl, rseed in experiments:
        n_seeds = sum(spl)
        data_bytes = n_seeds * 4
        print(f"\n=== {name} ({n_seeds} seeds, {data_bytes}B) ===")
        canvas, seeds, err, regions = run_search(target, strat, spl, rseed)
        results[name] = err

        outdir = f"media/prng_images/foveal_{name}"
        os.makedirs(outdir, exist_ok=True)
        save_pgm(f"{outdir}/result.pgm", canvas)
        compare = np.zeros((target.shape[0], target.shape[1]*2), dtype=np.uint8)
        compare[:, :target.shape[1]] = target
        compare[:, target.shape[1]:] = canvas
        save_pgm(f"{outdir}/compare.pgm", compare)

    print(f"\n{'='*50}")
    print(f"{'Strategy':18s} {'Error':>7s} {'Seeds':>6s}")
    for name in sorted(results, key=results.get):
        n = sum(experiments[[e[0] for e in experiments].index(name)][2])
        print(f"{name:18s} {results[name]:6.1f}% {n:6d}")

if __name__ == "__main__":
    main()

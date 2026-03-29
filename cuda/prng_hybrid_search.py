#!/usr/bin/env python3
"""
Hybrid Image Generator Search — Basis + Mask + Symmetry + pRNG

Instead of one pRNG seed, we search a rich parameter space:
  - pRNG seed (10 bytes): base noise
  - Tile masks (24 bytes): force bits in 8×8 blocks (1 bit per tile = 192 tiles / 8)
  - Basis params (16 bytes): circle(cx,cy,r,fill), gradient(angle,threshold), stripes(period,phase)
  - Threshold map (6 bytes): per-region density (6 regions: 2×3 grid)
  - Symmetry mode (1 byte): 0=none, 1=H-mirror, 2=V-mirror, 3=both, 4=radial

Total: 57 searchable bytes = 2^456 space (vs 2^80 for seed-only)
Fits in 256b intro with ~200 bytes for Z80 generator code + music

Usage: python3 prng_hybrid_search.py --target image.png [--synthetic] [--pop 500] [--gens 500]
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from prng_cat_search import CMWC

W, H = 128, 96
BW = W // 8  # 16 bytes per row
IMG_SIZE = BW * H  # 1536 bytes

# ====== Genome: searchable parameters ======
class Genome:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.seed = int(rng.randint(0, 2**63))
        self.tile_mask = rng.randint(0, 256, size=24, dtype=np.uint8)  # 192 bits
        self.circle = rng.randint(0, 256, size=4, dtype=np.uint8)  # cx, cy, r, fill_density
        self.gradient = rng.randint(0, 256, size=4, dtype=np.uint8)  # angle, threshold, strength, offset
        self.stripes = rng.randint(0, 256, size=4, dtype=np.uint8)  # period, phase, angle, thickness
        self.shapes = rng.randint(0, 256, size=4, dtype=np.uint8)  # extra shapes params
        self.threshold = rng.randint(0, 256, size=6, dtype=np.uint8)  # 2×3 region density
        self.symmetry = rng.randint(0, 5)  # 0-4

    def mutate(self, rng, strength=1):
        """Mutate a random subset of parameters."""
        child = Genome.__new__(Genome)
        child.seed = self.seed
        child.tile_mask = self.tile_mask.copy()
        child.circle = self.circle.copy()
        child.gradient = self.gradient.copy()
        child.stripes = self.stripes.copy()
        child.shapes = self.shapes.copy()
        child.threshold = self.threshold.copy()
        child.symmetry = self.symmetry

        # Pick what to mutate
        what = rng.randint(0, 8)
        if what == 0:  # seed
            child.seed ^= (1 << rng.randint(0, 64))
        elif what == 1:  # tile mask
            idx = rng.randint(0, 24)
            child.tile_mask[idx] ^= (1 << rng.randint(0, 8))
        elif what == 2:  # circle
            idx = rng.randint(0, 4)
            child.circle[idx] = (child.circle[idx] + rng.randint(-20, 21)) & 0xFF
        elif what == 3:  # gradient
            idx = rng.randint(0, 4)
            child.gradient[idx] = (child.gradient[idx] + rng.randint(-20, 21)) & 0xFF
        elif what == 4:  # stripes
            idx = rng.randint(0, 4)
            child.stripes[idx] = (child.stripes[idx] + rng.randint(-20, 21)) & 0xFF
        elif what == 5:  # threshold
            idx = rng.randint(0, 6)
            child.threshold[idx] = (child.threshold[idx] + rng.randint(-30, 31)) & 0xFF
        elif what == 6:  # symmetry
            child.symmetry = rng.randint(0, 5)
        elif what == 7:  # shapes
            idx = rng.randint(0, 4)
            child.shapes[idx] = (child.shapes[idx] + rng.randint(-20, 21)) & 0xFF

        # Multi-mutation for faster exploration
        for _ in range(strength - 1):
            child = child.mutate(rng, 1)

        return child

    def generate(self):
        """Generate 128×96 mono image from genome parameters."""
        img = np.zeros((H, W), dtype=np.float32)

        # Layer 1: pRNG base noise
        prng = CMWC(self.seed)
        noise_bytes = prng.generate(IMG_SIZE)
        for i, byte in enumerate(noise_bytes):
            row = i // BW
            col_base = (i % BW) * 8
            if row >= H: break
            for bit in range(8):
                if byte & (1 << (7 - bit)):
                    img[row, col_base + bit] = 1.0

        # Layer 2: tile masks (force tiles ON or OFF)
        for tile_idx in range(192):
            byte_idx = tile_idx // 8
            bit_idx = tile_idx % 8
            if byte_idx >= 24: break

            mask_bit = (self.tile_mask[byte_idx] >> bit_idx) & 1
            ty = (tile_idx // 16) * 8
            tx = (tile_idx % 16) * 8

            if mask_bit:
                # Force tile darker (OR with density pattern)
                density = 0.3
                for y in range(ty, min(ty + 8, H)):
                    for x in range(tx, min(tx + 8, W)):
                        if np.random.random() < density:
                            img[y, x] = 1.0

        # Layer 3: circle
        cx = self.circle[0] * W // 256
        cy = self.circle[1] * H // 256
        r = self.circle[2] * min(W, H) // 512 + 5
        fill = self.circle[3] / 255.0
        for y in range(H):
            for x in range(W):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if abs(dist - r) < 2:
                    img[y, x] = max(img[y, x], fill)
                if dist < r * 0.3:
                    img[y, x] = max(img[y, x], fill * 0.5)

        # Layer 4: gradient threshold
        angle = self.gradient[0] * np.pi / 128
        threshold = self.gradient[1] / 255.0
        strength = self.gradient[2] / 255.0 * 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for y in range(H):
            for x in range(W):
                grad_val = (cos_a * x / W + sin_a * y / H) * strength
                if grad_val > threshold:
                    img[y, x] = min(img[y, x] + 0.2, 1.0)

        # Layer 5: regional threshold
        for ry in range(2):
            for rx in range(3):
                region_thresh = self.threshold[ry * 3 + rx] / 255.0
                y0, y1 = ry * H // 2, (ry + 1) * H // 2
                x0, x1 = rx * W // 3, (rx + 1) * W // 3
                region = img[y0:y1, x0:x1]
                mask = region > region_thresh
                img[y0:y1, x0:x1] = mask.astype(np.float32)

        # Layer 6: symmetry
        if self.symmetry == 1:  # H-mirror (vertical axis)
            img = np.maximum(img, np.fliplr(img))
        elif self.symmetry == 2:  # V-mirror (horizontal axis)
            img = np.maximum(img, np.flipud(img))
        elif self.symmetry == 3:  # both
            img = np.maximum(img, np.fliplr(img))
            img = np.maximum(img, np.flipud(img))
        elif self.symmetry == 4:  # 4-fold (kaleidoscope)
            q = img[:H // 2, :W // 2]
            img[:H // 2, :W // 2] = q
            img[:H // 2, W // 2:] = np.fliplr(q)
            img[H // 2:, :W // 2] = np.flipud(q)
            img[H // 2:, W // 2:] = np.flipud(np.fliplr(q))

        return np.clip(img, 0, 1)


def image_to_tensor(img):
    """Convert generated image to CNN-ready tensor."""
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # 1×1×96×128
    t = t.repeat(1, 3, 1, 1)  # 1×3×96×128
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
    return t


def main():
    parser = argparse.ArgumentParser(description='Hybrid Image Generator Search')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--pop', type=int, default=500)
    parser.add_argument('--gens', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VGG16 for perceptual loss
    print("Loading VGG16...")
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare target
    if args.synthetic or args.target is None:
        print("Generating synthetic cat...")
        target_img = Image.new('L', (W, H), 200)
        draw = ImageDraw.Draw(target_img)
        cx, cy = W // 2, H // 2
        draw.ellipse([cx - 25, cy - 20, cx + 25, cy + 15], fill=40)
        draw.ellipse([cx - 18, cy - 30, cx - 8, cy - 18], fill=40)
        draw.ellipse([cx + 8, cy - 30, cx + 18, cy - 18], fill=40)
        draw.ellipse([cx - 10, cy - 8, cx - 4, cy - 2], fill=200)
        draw.ellipse([cx + 4, cy - 8, cx + 10, cy - 2], fill=200)
        draw.ellipse([cx - 2, cy + 2, cx + 2, cy + 5], fill=200)
        target = np.array(target_img, dtype=np.float32) / 255.0
        target = 1.0 - target  # invert (dark = filled)
    else:
        target_img = Image.open(args.target).convert('L').resize((W, H))
        target = np.array(target_img, dtype=np.float32) / 255.0

    # Compute target features
    target_tensor = image_to_tensor(target).to(device)
    target_tensor = normalize(target_tensor)
    with torch.no_grad():
        target_features = vgg(target_tensor)

    def evaluate(genome):
        img = genome.generate()
        tensor = image_to_tensor(img).to(device)
        tensor = normalize(tensor)
        with torch.no_grad():
            features = vgg(tensor)
            loss = F.mse_loss(features, target_features).item()
        return loss

    # Evolution
    print(f"Population: {args.pop}, Generations: {args.gens}")
    print(f"Genome: 57 searchable bytes (seed + masks + basis + threshold + symmetry)")
    rng = np.random.RandomState(42)

    population = [Genome(rng) for _ in range(args.pop)]
    scores = [evaluate(g) for g in population]
    best_loss = min(scores)
    best_genome = population[scores.index(best_loss)]
    print(f"Initial best: loss={best_loss:.4f}")

    t0 = time.time()
    for gen in range(args.gens):
        # Sort
        paired = sorted(zip(scores, population), key=lambda x: x[0])
        scores = [s for s, _ in paired]
        population = [g for _, g in paired]

        if scores[0] < best_loss:
            best_loss = scores[0]
            best_genome = population[0]

        # Keep top 20%, mutate rest
        keep = args.pop // 5
        new_pop = population[:keep]
        new_scores = scores[:keep]

        while len(new_pop) < args.pop:
            parent = population[rng.randint(0, keep)]
            strength = 1 + rng.randint(0, 3)
            child = parent.mutate(rng, strength)
            loss = evaluate(child)
            new_pop.append(child)
            new_scores.append(loss)

        population = new_pop
        scores = new_scores

        if gen % 20 == 0:
            elapsed = time.time() - t0
            print(f"Gen {gen:4d}: best={best_loss:.4f} gen_best={scores[0]:.4f} "
                  f"sym={best_genome.symmetry} ({elapsed:.0f}s)")

    # Save results
    outdir = "media/prng_images/hybrid"
    os.makedirs(outdir, exist_ok=True)

    best_img = best_genome.generate()
    Image.fromarray((best_img * 255).astype(np.uint8)).save(f"{outdir}/best_mono.png")

    # Side-by-side: target | generated
    from PIL import Image as PILImage
    side = PILImage.new('L', (W * 2 + 4, H), 128)
    side.paste(PILImage.fromarray((target * 255).astype(np.uint8)), (0, 0))
    side.paste(PILImage.fromarray((best_img * 255).astype(np.uint8)), (W + 4, 0))
    side.save(f"{outdir}/comparison.png")

    # Save multiple good results
    paired = sorted(zip(scores, population), key=lambda x: x[0])
    for i in range(min(10, len(paired))):
        loss, genome = paired[i]
        img = genome.generate()
        Image.fromarray((img * 255).astype(np.uint8)).save(
            f"{outdir}/rank{i:02d}_loss{loss:.4f}_sym{genome.symmetry}.png")

    print(f"\n=== RESULT ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Symmetry: {best_genome.symmetry}")
    print(f"Circle: cx={best_genome.circle[0]} cy={best_genome.circle[1]} r={best_genome.circle[2]}")
    print(f"Total bytes: 57 searchable (vs 10 for seed-only)")
    print(f"Saved to {outdir}/")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()

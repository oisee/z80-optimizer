#!/usr/bin/env python3
"""
pRNG Cat Search — find Z80 pRNG seeds that generate cat-like images.

Uses MobileNetV2 (pre-trained on ImageNet) as fitness function:
  pRNG seed → generate 128×96 mono image → CNN → P(cat) → maximize

Usage:
  python3 prng_cat_search.py [--target cat|face|dog] [--pop 1000] [--gens 100]
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import argparse
import time

# ImageNet class indices for animals
CAT_CLASSES = list(range(281, 286))  # tabby, tiger cat, persian, siamese, egyptian
DOG_CLASSES = list(range(151, 269))  # dogs
FACE_CLASSES = [983]  # not really a class, use custom approach

# Patrik Rak CMWC pRNG (Python version, matches Z80 exactly)
class CMWC:
    def __init__(self, seed: int):
        self.idx = 0
        self.carry = (seed >> 56) & 0xFF
        self.table = [(seed >> (i * 7)) & 0xFF for i in range(8)]
        for i in range(8):
            if self.table[i] == 0:
                self.table[i] = i + 1

    def next(self) -> int:
        self.idx = (self.idx + 1) & 7
        y = self.table[self.idx]
        t = y * 253 + self.carry
        self.carry = t >> 8
        x = (~t) & 0xFF
        self.table[self.idx] = x
        return x

    def generate(self, n: int) -> bytes:
        return bytes([self.next() for _ in range(n)])


def seed_to_image(seed: int, w=128, h=96, mirror=True) -> np.ndarray:
    """Generate 128×96 mono image from pRNG seed.

    If mirror=True: generate full noise, then OR with horizontal flip.
    Result: irregular but symmetric — looks more like faces/objects.
    """
    prng = CMWC(seed)
    data = prng.generate(w * h // 8)  # 1 bit per pixel, packed

    # Unpack bits to pixels
    img = np.zeros((h, w), dtype=np.float32)
    for i, byte in enumerate(data):
        row = i // (w // 8)
        col_base = (i % (w // 8)) * 8
        if row >= h:
            break
        for bit in range(8):
            if byte & (1 << (7 - bit)):
                img[row, col_base + bit] = 1.0

    if mirror:
        # OR with horizontal flip → symmetric but irregular
        flipped = np.fliplr(img)
        img = np.clip(img + flipped, 0, 1)

    return img


def image_to_grayscale(img: np.ndarray, block=4) -> np.ndarray:
    """Convert 1-bit image to grayscale by counting bits in blocks.

    block=2: 64×48, 5 gray levels (0-4 bits per 2×2)
    block=3: 42×32, 10 gray levels
    block=4: 32×24, 17 gray levels (0-16 bits per 4×4)

    This gives CNN something meaningful to classify!
    """
    h, w = img.shape
    gh, gw = h // block, w // block
    gray = np.zeros((gh, gw), dtype=np.float32)
    for y in range(gh):
        for x in range(gw):
            patch = img[y*block:(y+1)*block, x*block:(x+1)*block]
            gray[y, x] = patch.sum() / (block * block)
    return gray


def batch_seeds_to_images(seeds: list, w=128, h=96, mirror=True) -> torch.Tensor:
    """Convert batch of seeds to tensor of images.

    Pipeline: seed → 1-bit noise → mirror (OR flip) → grayscale (4×4 blocks) → 224×224 RGB
    """
    imgs = []
    for s in seeds:
        # Step 1: Generate 1-bit with symmetry
        mono = seed_to_image(s, w, h, mirror=mirror)
        # Step 2: Convert to grayscale via block averaging
        gray = image_to_grayscale(mono, block=4)  # 32×24, continuous values
        imgs.append(gray)

    # Stack and convert to 3-channel 224×224 for CNN
    batch = np.stack(imgs)  # (N, 24, 32)
    batch = torch.from_numpy(batch).unsqueeze(1)  # (N, 1, 24, 32)
    batch = batch.repeat(1, 3, 1, 1)  # (N, 3, 24, 32) — gray to RGB
    batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
    return batch


def main():
    parser = argparse.ArgumentParser(description='pRNG Cat/Dog/Face Search')
    parser.add_argument('--target', default='cat', choices=['cat', 'dog', 'face'])
    parser.add_argument('--pop', type=int, default=500, help='Population size')
    parser.add_argument('--gens', type=int, default=200, help='Generations')
    parser.add_argument('--batch', type=int, default=100, help='CNN batch size')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pre-trained MobileNetV2
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Target classes
    if args.target == 'cat':
        target_classes = CAT_CLASSES
        print(f"Target: CAT (ImageNet classes {target_classes})")
    elif args.target == 'dog':
        target_classes = DOG_CLASSES
        print(f"Target: DOG (ImageNet classes {target_classes[:5]}...{target_classes[-5:]})")
    else:
        target_classes = list(range(1000))  # any recognizable thing
        print("Target: anything recognizable")

    def quick_filter(seeds):
        """Stage 1: fast filter — check if image has interesting structure.
        Rejects pure noise quickly without CNN.
        Returns: (seed, structure_score) pairs that pass threshold."""
        candidates = []
        for s in seeds:
            mono = seed_to_image(s, mirror=True)
            gray = image_to_grayscale(mono, block=4)

            # Quick checks:
            # 1. Not too uniform (need contrast)
            std = gray.std()
            if std < 0.05 or std > 0.45:
                candidates.append((s, 0.0))
                continue

            # 2. Has some structure (autocorrelation)
            h, w = gray.shape
            h_corr = np.mean(np.abs(gray[:, 1:] - gray[:, :-1]))
            v_corr = np.mean(np.abs(gray[1:, :] - gray[:-1, :]))
            structure = 1.0 - (h_corr + v_corr) / 2  # higher = more structured

            # 3. Center is different from edges (face-like)
            center = gray[h//4:3*h//4, w//4:3*w//4].mean()
            edge = (gray[:h//4, :].mean() + gray[3*h//4:, :].mean()) / 2
            center_contrast = abs(center - edge)

            score = structure * 0.3 + center_contrast * 0.5 + std * 0.2
            candidates.append((s, score))

        return candidates

    def evaluate_batch(seeds):
        """Stage 2: CNN evaluation on pre-filtered candidates."""
        imgs = batch_seeds_to_images(seeds, mirror=True)

        scores = []
        for i in range(0, len(seeds), args.batch):
            batch = imgs[i:i + args.batch].to(device)
            batch = normalize(batch)

            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                target_prob = probs[:, target_classes].sum(dim=1)
                scores.extend(target_prob.cpu().numpy().tolist())

        return scores

    # Initialize random population
    print(f"\nPopulation: {args.pop}, Generations: {args.gens}")
    rng = np.random.RandomState(42)
    population = [int(rng.randint(0, 2**63)) for _ in range(args.pop)]
    scores = evaluate_batch(population)

    best_score = max(scores)
    best_seed = population[scores.index(best_score)]
    print(f"Initial best: score={best_score:.6f} seed=0x{best_seed:016X}")

    t0 = time.time()
    total_evals = args.pop

    for gen in range(args.gens):
        # Sort by score (descending)
        paired = list(zip(scores, population))
        paired.sort(reverse=True)
        scores, population = zip(*paired)
        scores, population = list(scores), list(population)

        # Keep top 20%, mutate to fill rest
        keep = args.pop // 5
        new_pop = population[:keep]
        new_scores = scores[:keep]

        # Generate mutations
        while len(new_pop) < args.pop:
            parent = population[rng.randint(0, keep)]
            # Mutate: flip 1-4 random bits
            child = parent
            n_flips = 1 + rng.randint(0, 4)
            for _ in range(n_flips):
                bit = rng.randint(0, 64)
                child ^= (1 << bit)
            new_pop.append(child)

        # Evaluate new members
        new_members = new_pop[keep:]
        new_member_scores = evaluate_batch(new_members)
        new_scores.extend(new_member_scores)

        population = new_pop
        scores = new_scores
        total_evals += len(new_members)

        # Track best
        gen_best_idx = scores.index(max(scores))
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_seed = population[gen_best_idx]

        if gen % 10 == 0:
            elapsed = time.time() - t0
            rate = total_evals / elapsed
            print(f"Gen {gen:4d}: best={best_score:.6f} "
                  f"gen_best={scores[0]:.6f} "
                  f"seed=0x{best_seed:016X} "
                  f"({rate:.0f} eval/s)")

    elapsed = time.time() - t0
    print(f"\n=== RESULT ===")
    print(f"Best score: {best_score:.6f} ({args.target} probability)")
    print(f"Best seed:  0x{best_seed:016X}")
    print(f"Total evaluations: {total_evals} in {elapsed:.1f}s ({total_evals/elapsed:.0f}/s)")

    # Show ASCII art of best result
    img = seed_to_image(best_seed)
    print(f"\nBest image (128×96):")
    for y in range(0, 96, 2):
        line = ""
        for x in range(0, 128, 2):
            line += "█" if img[y, x] > 0.5 else " "
        print(f"  {line}")

    # Save
    fname = f"cat_seed_{best_seed:016X}.bin"
    with open(fname, 'wb') as f:
        prng = CMWC(best_seed)
        f.write(prng.generate(128 * 96 // 8))
    print(f"Saved: {fname}")

    # Top-5 predictions for best image
    imgs = batch_seeds_to_images([best_seed]).to(device)
    imgs = normalize(imgs)
    with torch.no_grad():
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        top5 = probs[0].topk(5)

    # Load ImageNet labels
    try:
        labels = models.MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]
        print(f"\nTop-5 CNN predictions:")
        for i in range(5):
            idx = top5.indices[i].item()
            prob = top5.values[i].item()
            print(f"  {prob:.4f} — {labels[idx]} (class {idx})")
    except:
        print(f"\nTop-5 class indices: {top5.indices.tolist()}")
        print(f"Top-5 probabilities: {[f'{p:.4f}' for p in top5.values.tolist()]}")


if __name__ == '__main__':
    main()

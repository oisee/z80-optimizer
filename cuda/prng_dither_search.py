#!/usr/bin/env python3
"""
pRNG Dithered Target Search — find Z80 pRNG seeds whose output resembles a
dithered reference image, scored by VGG16 perceptual loss.

Approach (inspired by Introspec's BB demo):
  1. Load target grayscale image, resize to 128x96
  2. Floyd-Steinberg dither to 1-bit
  3. Apply OR horizontal mirror (same symmetry as pRNG output)
  4. Search for pRNG SEED whose mirrored output is closest to dithered target
  5. VGG16 conv3_3 feature-space MSE as perceptual loss

Island model: 10 independent populations that share top-3 seeds every 50 gens.
Hill climbing: mutate seed (flip 1-4 bits), keep if perceptual loss decreases.

Usage:
  python3 prng_dither_search.py --target cat.jpg --pop 200 --gens 200 --islands 10
  python3 prng_dither_search.py --synthetic  # generate synthetic cat test image

Results saved to media/prng_images/dithered/
"""

import sys
import os
import argparse
import time
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter

# Import CMWC, seed_to_image, image_to_grayscale from sibling script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prng_cat_search import CMWC, seed_to_image, image_to_grayscale

# Output directory
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'media', 'prng_images', 'dithered')


# ---------------------------------------------------------------------------
# Target image preparation
# ---------------------------------------------------------------------------

def generate_synthetic_cat(w=128, h=96) -> Image.Image:
    """Generate a simple synthetic cat-like image for testing.

    Draws an oval face, triangular ears, eyes, nose, and whiskers
    on a gradient background. Not art, but enough structure for the
    perceptual loss to latch onto.
    """
    img = Image.new('L', (w, h), 200)
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(h):
        v = int(180 + 40 * y / h)
        draw.line([(0, y), (w, y)], fill=v)

    cx, cy = w // 2, h // 2 + 4

    # Head (ellipse)
    draw.ellipse([cx - 30, cy - 24, cx + 30, cy + 24], fill=120, outline=60)

    # Ears (triangles)
    draw.polygon([(cx - 26, cy - 22), (cx - 18, cy - 40), (cx - 8, cy - 20)],
                 fill=100, outline=60)
    draw.polygon([(cx + 26, cy - 22), (cx + 18, cy - 40), (cx + 8, cy - 20)],
                 fill=100, outline=60)
    # Inner ears
    draw.polygon([(cx - 22, cy - 22), (cx - 17, cy - 34), (cx - 12, cy - 22)],
                 fill=140)
    draw.polygon([(cx + 22, cy - 22), (cx + 17, cy - 34), (cx + 12, cy - 22)],
                 fill=140)

    # Eyes
    draw.ellipse([cx - 16, cy - 8, cx - 6, cy + 2], fill=40)
    draw.ellipse([cx + 6, cy - 8, cx + 16, cy + 2], fill=40)
    # Eye highlights
    draw.ellipse([cx - 14, cy - 6, cx - 10, cy - 2], fill=220)
    draw.ellipse([cx + 8, cy - 6, cx + 12, cy - 2], fill=220)

    # Nose
    draw.polygon([(cx - 3, cy + 6), (cx + 3, cy + 6), (cx, cy + 10)], fill=60)

    # Mouth
    draw.arc([cx - 8, cy + 6, cx, cy + 16], start=0, end=180, fill=60)
    draw.arc([cx, cy + 6, cx + 8, cy + 16], start=0, end=180, fill=60)

    # Whiskers
    for dy in [-2, 2, 6]:
        draw.line([(cx - 30, cy + dy), (cx - 48, cy + dy - 4)], fill=60, width=1)
        draw.line([(cx + 30, cy + dy), (cx + 48, cy + dy - 4)], fill=60, width=1)

    # Slight blur to add grayscale gradients
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error-diffusion dither.

    Input:  float32 array in [0, 1], shape (H, W)
    Output: float32 array in {0, 1}, shape (H, W)
    """
    h, w = gray.shape
    buf = gray.astype(np.float64).copy()

    for y in range(h):
        for x in range(w):
            old = buf[y, x]
            new = 1.0 if old > 0.5 else 0.0
            buf[y, x] = new
            err = old - new
            if x + 1 < w:
                buf[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    buf[y + 1, x - 1] += err * 3 / 16
                buf[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * 1 / 16

    return buf.astype(np.float32)


def prepare_target(path: str, w=128, h=96) -> np.ndarray:
    """Load image, resize to w x h, dither to 1-bit, apply OR mirror.

    Returns float32 array in {0, 1}, shape (h, w).
    """
    img = Image.open(path).convert('L')
    img = img.resize((w, h), Image.LANCZOS)
    gray = np.array(img, dtype=np.float32) / 255.0

    # Floyd-Steinberg dither
    dithered = floyd_steinberg_dither(gray)

    # OR horizontal mirror (same as pRNG output symmetry)
    flipped = np.fliplr(dithered)
    mirrored = np.clip(dithered + flipped, 0, 1)

    return mirrored


# ---------------------------------------------------------------------------
# VGG16 perceptual feature extractor
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """Extract conv3_3 features from VGG16 for perceptual loss."""

    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # features[:16] = conv1 -> conv2 -> conv3_3 (before ReLU after conv3_3)
        self.features = vgg.features[:16].to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 224, 224) already normalized."""
        return self.features(x)

    def image_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """Convert (H, W) float32 image to normalized (1, 3, 224, 224) tensor."""
        t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = t.repeat(1, 3, 1, 1)  # gray -> RGB
        t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        t = self.normalize(t[0]).unsqueeze(0)  # normalize each channel
        return t.to(self.device)

    def batch_images_to_tensor(self, images: list) -> torch.Tensor:
        """Convert list of (H, W) float32 images to (N, 3, 224, 224) tensor."""
        tensors = []
        for img in images:
            t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
            tensors.append(t)
        batch = torch.stack(tensors)  # (N, 1, H, W)
        batch = batch.repeat(1, 3, 1, 1)  # (N, 3, H, W)
        batch = F.interpolate(batch, size=(224, 224), mode='bilinear',
                              align_corners=False)
        # Normalize each image in batch
        for i in range(batch.shape[0]):
            batch[i] = self.normalize(batch[i])
        return batch.to(self.device)


def perceptual_loss(feat_gen: torch.Tensor, feat_target: torch.Tensor) -> torch.Tensor:
    """MSE between feature maps — lower is better (more similar)."""
    return F.mse_loss(feat_gen, feat_target, reduction='none').mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def seed_to_grayscale(seed: int, w=128, h=96) -> np.ndarray:
    """Seed -> 1-bit mirrored image -> grayscale (block=4) -> (24, 32)."""
    mono = seed_to_image(seed, w, h, mirror=True)
    return image_to_grayscale(mono, block=4)


def hamming_distance_64(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit seeds."""
    return bin((a ^ b) & ((1 << 64) - 1)).count('1')


def mutate_seed(seed: int, rng: np.random.RandomState, n_flips_range=(1, 5)) -> int:
    """Flip 1-4 random bits in the seed."""
    child = seed
    n_flips = rng.randint(n_flips_range[0], n_flips_range[1])
    for _ in range(n_flips):
        bit = rng.randint(0, 64)
        child ^= (1 << bit)
    return child


# ---------------------------------------------------------------------------
# Island model search
# ---------------------------------------------------------------------------

def evaluate_seeds(seeds: list, vgg: VGGFeatureExtractor,
                   target_features: torch.Tensor,
                   batch_size: int = 64) -> np.ndarray:
    """Compute perceptual loss for a list of seeds.

    Returns: np.ndarray of losses (lower = better).
    """
    all_losses = []

    for i in range(0, len(seeds), batch_size):
        batch_seeds = seeds[i:i + batch_size]
        images = [seed_to_grayscale(s) for s in batch_seeds]
        tensor = vgg.batch_images_to_tensor(images)

        with torch.no_grad():
            feat = vgg(tensor)
            # Expand target features to match batch size
            target_exp = target_features.expand(feat.shape[0], -1, -1, -1)
            losses = perceptual_loss(feat, target_exp)
            all_losses.extend(losses.cpu().numpy().tolist())

    return np.array(all_losses)


def island_search(target_img: np.ndarray, vgg: VGGFeatureExtractor,
                  n_islands: int = 10, pop_size: int = 200,
                  n_gens: int = 200, share_every: int = 50,
                  share_top: int = 3, batch_size: int = 64,
                  seed: int = 42):
    """Island model hill climbing with perceptual loss.

    Each island maintains an independent population. Every `share_every`
    generations, the top `share_top` seeds from each island are broadcast
    to all others (replacing worst members).
    """
    rng = np.random.RandomState(seed)

    # Precompute target features
    print("Computing target features...")
    target_tensor = vgg.image_to_tensor(target_img)
    with torch.no_grad():
        target_features = vgg(target_tensor)
    print(f"  Target feature shape: {target_features.shape}")

    # Initialize islands with random seeds
    print(f"\nInitializing {n_islands} islands x {pop_size} population...")
    islands = []
    for isle in range(n_islands):
        pop = [int(rng.randint(0, 2**63)) for _ in range(pop_size)]
        losses = evaluate_seeds(pop, vgg, target_features, batch_size)
        islands.append({'pop': pop, 'losses': losses, 'rng': np.random.RandomState(seed + isle + 1)})

    global_best_loss = float('inf')
    global_best_seed = 0
    total_evals = n_islands * pop_size

    # Find initial best
    for isle in islands:
        idx = np.argmin(isle['losses'])
        if isle['losses'][idx] < global_best_loss:
            global_best_loss = isle['losses'][idx]
            global_best_seed = isle['pop'][idx]

    print(f"Initial best: loss={global_best_loss:.6f} seed=0x{global_best_seed:016X}")

    t0 = time.time()

    for gen in range(n_gens):
        for isle_idx, isle in enumerate(islands):
            pop = isle['pop']
            losses = isle['losses']
            irng = isle['rng']

            # Sort by loss (ascending — lower is better)
            order = np.argsort(losses)
            pop = [pop[i] for i in order]
            losses = losses[order]

            # Keep top 20%
            keep = max(pop_size // 5, 2)
            new_pop = pop[:keep]
            new_losses = list(losses[:keep])

            # Hill climbing: mutate from top seeds
            children = []
            while len(children) < pop_size - keep:
                parent_idx = irng.randint(0, keep)
                child = mutate_seed(pop[parent_idx], irng)
                children.append(child)

            # Evaluate children
            child_losses = evaluate_seeds(children, vgg, target_features, batch_size)
            total_evals += len(children)

            new_pop.extend(children)
            new_losses.extend(child_losses.tolist())

            isle['pop'] = new_pop
            isle['losses'] = np.array(new_losses)

            # Track global best
            idx = np.argmin(isle['losses'])
            if isle['losses'][idx] < global_best_loss:
                global_best_loss = isle['losses'][idx]
                global_best_seed = isle['pop'][idx]

        # Island sharing
        if (gen + 1) % share_every == 0 and n_islands > 1:
            # Collect top-k from each island
            top_seeds = []
            for isle in islands:
                order = np.argsort(isle['losses'])
                for j in range(min(share_top, len(order))):
                    top_seeds.append(isle['pop'][order[j]])

            # Inject into all islands (replace worst members)
            for isle in islands:
                order = np.argsort(isle['losses'])[::-1]  # worst first
                n_inject = min(len(top_seeds), len(order) // 4)
                for j in range(n_inject):
                    worst_idx = order[j]
                    isle['pop'][worst_idx] = top_seeds[j % len(top_seeds)]
                # Re-evaluate injected seeds
                inject_indices = list(order[:n_inject])
                inject_seeds = [isle['pop'][i] for i in inject_indices]
                inject_losses = evaluate_seeds(inject_seeds, vgg, target_features,
                                               batch_size)
                total_evals += len(inject_seeds)
                for j, idx in enumerate(inject_indices):
                    isle['losses'][idx] = inject_losses[j]

            print(f"  [Share] Broadcast {len(top_seeds)} seeds across {n_islands} islands")

        # Progress
        if gen % 10 == 0 or gen == n_gens - 1:
            elapsed = time.time() - t0
            rate = total_evals / max(elapsed, 0.001)
            island_bests = [np.min(isle['losses']) for isle in islands]
            print(f"Gen {gen:4d}: global_best={global_best_loss:.6f} "
                  f"island_bests=[{', '.join(f'{b:.4f}' for b in island_bests)}] "
                  f"seed=0x{global_best_seed:016X} "
                  f"({rate:.0f} eval/s)")

    elapsed = time.time() - t0
    print(f"\nSearch complete: {total_evals} evaluations in {elapsed:.1f}s "
          f"({total_evals / elapsed:.0f} eval/s)")

    # Collect all seeds from all islands
    all_seeds = []
    all_losses_list = []
    for isle in islands:
        all_seeds.extend(isle['pop'])
        all_losses_list.extend(isle['losses'].tolist())

    return all_seeds, all_losses_list, global_best_seed, global_best_loss


def select_diverse_top_k(seeds: list, losses: list, k: int = 20,
                         min_hamming: int = 8) -> list:
    """Select top-k seeds ensuring minimum Hamming distance between each pair.

    Greedy: take best, then skip any seed too close to already-selected ones.
    """
    paired = sorted(zip(losses, seeds))
    selected = []

    for loss, seed in paired:
        if len(selected) >= k:
            break
        # Check Hamming distance to all selected
        too_close = False
        for _, sel_seed in selected:
            if hamming_distance_64(seed, sel_seed) < min_hamming:
                too_close = True
                break
        if not too_close:
            selected.append((loss, seed))

    return selected


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_result_image(seed: int, target_img: np.ndarray, rank: int,
                      loss: float, outdir: str):
    """Save side-by-side comparison: target vs generated."""
    gen_mono = seed_to_image(seed, 128, 96, mirror=True)
    gen_gray = image_to_grayscale(gen_mono, block=4)

    # Scale both to displayable size
    scale = 4
    th, tw = target_img.shape
    gh, gw = gen_gray.shape

    # Target (dithered, already mirrored)
    target_pil = Image.fromarray((target_img * 255).astype(np.uint8))
    target_pil = target_pil.resize((tw * scale, th * scale), Image.NEAREST)

    # Generated (grayscale from blocks)
    gen_pil = Image.fromarray((gen_gray * 255).astype(np.uint8))
    gen_pil = gen_pil.resize((gw * scale, gh * scale), Image.NEAREST)

    # Generated (1-bit, full res)
    mono_pil = Image.fromarray((gen_mono * 255).astype(np.uint8))
    mono_pil = mono_pil.resize((128 * 2, 96 * 2), Image.NEAREST)

    # Side by side: target | grayscale | 1-bit
    combo_w = target_pil.width + gen_pil.width + mono_pil.width + 20
    combo_h = max(target_pil.height, gen_pil.height, mono_pil.height) + 20
    combo = Image.new('L', (combo_w, combo_h), 128)

    x = 5
    combo.paste(target_pil, (x, 5))
    x += target_pil.width + 5
    combo.paste(gen_pil, (x, 5))
    x += gen_pil.width + 5
    combo.paste(mono_pil, (x, 5))

    fname = os.path.join(outdir, f"rank{rank:02d}_loss{loss:.4f}_0x{seed:016X}.png")
    combo.save(fname)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='pRNG Dithered Target Search — VGG perceptual loss')
    parser.add_argument('--target', type=str, default=None,
                        help='Path to target image (JPG/PNG)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate and use a synthetic cat image')
    parser.add_argument('--pop', type=int, default=200,
                        help='Population per island (default: 200)')
    parser.add_argument('--gens', type=int, default=200,
                        help='Number of generations (default: 200)')
    parser.add_argument('--islands', type=int, default=10,
                        help='Number of islands (default: 10)')
    parser.add_argument('--share-every', type=int, default=50,
                        help='Share top seeds every N gens (default: 50)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Save top-K diverse seeds (default: 20)')
    parser.add_argument('--batch', type=int, default=64,
                        help='VGG batch size (default: 64)')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducibility')
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Target image ---
    if args.synthetic or args.target is None:
        print("Generating synthetic cat image...")
        target_pil = generate_synthetic_cat(128, 96)
        synth_path = os.path.join(OUTDIR, 'synthetic_cat.png')
        target_pil.save(synth_path)
        print(f"  Saved synthetic target: {synth_path}")

        # Dither
        gray = np.array(target_pil, dtype=np.float32) / 255.0
        dithered = floyd_steinberg_dither(gray)
        # OR mirror
        flipped = np.fliplr(dithered)
        target_img = np.clip(dithered + flipped, 0, 1)
    else:
        print(f"Loading target: {args.target}")
        target_img = prepare_target(args.target)

    # Save dithered target
    target_save = Image.fromarray((target_img * 255).astype(np.uint8))
    target_save.save(os.path.join(OUTDIR, 'target_dithered.png'))
    print(f"  Target shape: {target_img.shape}, "
          f"density: {target_img.mean():.3f}")

    # --- VGG feature extractor ---
    print("Loading VGG16 feature extractor (conv3_3)...")
    vgg = VGGFeatureExtractor(device=device)
    print("  VGG16 features[:16] loaded.")

    # --- Search ---
    print(f"\nIsland search: {args.islands} islands x {args.pop} pop x "
          f"{args.gens} gens")
    print(f"  Share top-3 every {args.share_every} gens")
    print(f"  Will save top-{args.top_k} diverse seeds\n")

    all_seeds, all_losses, best_seed, best_loss = island_search(
        target_img=target_img,
        vgg=vgg,
        n_islands=args.islands,
        pop_size=args.pop,
        n_gens=args.gens,
        share_every=args.share_every,
        batch_size=args.batch,
        seed=args.seed,
    )

    # --- Select diverse top-K ---
    print(f"\nSelecting top-{args.top_k} diverse seeds (min Hamming distance 8)...")
    top_seeds = select_diverse_top_k(all_seeds, all_losses, k=args.top_k,
                                     min_hamming=8)

    print(f"\n{'='*60}")
    print(f"  TOP-{len(top_seeds)} DIVERSE RESULTS")
    print(f"{'='*60}")

    for rank, (loss, seed) in enumerate(top_seeds):
        fname = save_result_image(seed, target_img, rank, loss, OUTDIR)
        print(f"  #{rank:2d}  loss={loss:.6f}  seed=0x{seed:016X}  -> {fname}")

    # Save seeds to text file
    seeds_file = os.path.join(OUTDIR, 'top_seeds.txt')
    with open(seeds_file, 'w') as f:
        f.write(f"# pRNG Dithered Target Search Results\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Islands: {args.islands}, Pop: {args.pop}, Gens: {args.gens}\n")
        f.write(f"# Device: {device}\n\n")
        for rank, (loss, seed) in enumerate(top_seeds):
            f.write(f"{rank:2d}  0x{seed:016X}  loss={loss:.6f}\n")
    print(f"\nSeeds saved: {seeds_file}")

    # ASCII art of best
    print(f"\nBest image (128x96, 2x downsampled):")
    img = seed_to_image(best_seed, mirror=True)
    for y in range(0, 96, 2):
        line = ""
        for x in range(0, 128, 2):
            line += "#" if img[y, x] > 0.5 else " "
        print(f"  {line}")

    # Save raw binary for ZX Spectrum loader
    bin_file = os.path.join(OUTDIR, f'best_0x{best_seed:016X}.bin')
    prng = CMWC(best_seed)
    with open(bin_file, 'wb') as f:
        f.write(prng.generate(128 * 96 // 8))
    print(f"\nRaw binary saved: {bin_file}")
    print(f"Best seed: 0x{best_seed:016X} (loss={best_loss:.6f})")


if __name__ == '__main__':
    main()

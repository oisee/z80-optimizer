#!/usr/bin/env python3
"""
make_heatmap.py — generate per-pixel importance weight map for LFSR-16 search.

Output: raw uint8 binary, W×H = 128×96 = 12288 bytes, row-major.
  0   = ignore completely
  1   = background (tiny contribution)
  255 = maximum importance (eyes, edges)

Modes (can combine):
  --edge          Edge-based weights from target image (no detector, any content)
  --face          Face detector (OpenCV Haar) → eyes/nose/mouth get high weight
  --combined      edge × face_boost  [default]

Usage:
  python3 make_heatmap.py --target frame.pgm --out frame.wmap [--edge|--face|--combined]
  python3 make_heatmap.py --target frame.pgm --out frame.wmap --vis heatmap.png
"""
import sys, os, struct, argparse
import numpy as np
from scipy.ndimage import gaussian_filter

W, H = 128, 96

# ── PGM loader ──────────────────────────────────────────────────────────────
def load_pgm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P5', f"Not P5 PGM: {magic}"
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'): break
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    return data, w, h

# ── edge weight ─────────────────────────────────────────────────────────────
def make_edge_weight(img_gray, sigma=4.0, floor=0.08):
    """
    Binary edge map: each pixel = fraction of 8 neighbors that differ.
    Blurred to create smooth gradient. Floor ensures background still counts.
    """
    binary = (img_gray > 127).astype(np.float32)
    # count differing neighbors (8-connected)
    edge = np.zeros_like(binary)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            shifted = np.roll(np.roll(binary, dy, axis=0), dx, axis=1)
            edge += (binary != shifted).astype(np.float32)
    edge /= 8.0  # normalize 0..1
    # blur to spread importance into surrounding region
    edge = gaussian_filter(edge, sigma=sigma)
    # add floor so background pixels still contribute a little
    edge = edge * (1.0 - floor) + floor
    return edge  # float32 [floor..1.0]

# ── face weight ──────────────────────────────────────────────────────────────
def make_face_weight(img_gray, base_weight, eye_boost=4.0, face_boost=2.5,
                     mouth_boost=2.0, sigma_eye=6.0, sigma_face=15.0):
    """
    Detect face + eye landmarks with OpenCV Haar cascades.
    Add Gaussian blobs at eyes/mouth with high weights.
    Falls back to centered Gaussian if no face detected.
    """
    try:
        import cv2
    except ImportError:
        print("  [face] cv2 not available, skipping face detection", file=sys.stderr)
        return base_weight

    # scale up for better detection (Haar works better on larger images)
    scale = 4
    big = cv2.resize(img_gray, (W*scale, H*scale), interpolation=cv2.INTER_LINEAR)

    cv2_data = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cv2_data + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2_data + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(big, scaleFactor=1.1, minNeighbors=3,
                                           minSize=(20*scale, 20*scale))

    weight = base_weight.copy()
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    if len(faces) == 0:
        print("  [face] no face detected — using centered Gaussian fallback", file=sys.stderr)
        # fallback: soft Gaussian centered at image center
        cx, cy = W/2, H*0.42
        r = gaussian_blob(xs, ys, cx, cy, sigma_x=W*0.28, sigma_y=H*0.35)
        weight = weight * (1.0 + r * (face_boost - 1.0))
        return np.clip(weight, 0, 1)

    for (fx, fy, fw, fh) in faces[:1]:  # use first/largest face
        # scale back to 128×96
        fx, fy, fw, fh = fx//scale, fy//scale, fw//scale, fh//scale
        print(f"  [face] detected at ({fx},{fy}) size {fw}×{fh}", file=sys.stderr)

        # face region boost
        face_cx, face_cy = fx + fw/2, fy + fh/2
        face_r = gaussian_blob(xs, ys, face_cx, face_cy,
                                sigma_x=fw*0.55, sigma_y=fh*0.6)
        weight = weight * (1.0 + face_r * (face_boost - 1.0))

        # detect eyes within face region (on big image)
        face_roi = big[fy*scale:(fy+fh)*scale, fx*scale:(fx+fw)*scale]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.05,
                                             minNeighbors=2, minSize=(8*scale, 8*scale))
        if len(eyes) >= 2:
            for (ex, ey, ew, eh) in eyes[:2]:
                ecx = fx + ex//scale + ew//(scale*2)
                ecy = fy + ey//scale + eh//(scale*2)
                print(f"  [eye]  at ({ecx:.0f},{ecy:.0f})", file=sys.stderr)
                eye_r = gaussian_blob(xs, ys, ecx, ecy,
                                      sigma_x=fw*0.14, sigma_y=fh*0.10)
                weight = weight + eye_r * (eye_boost - 1.0) * base_weight
        elif len(eyes) == 1:
            (ex, ey, ew, eh) = eyes[0]
            ecx = fx + ex//scale + ew//(scale*2)
            ecy = fy + ey//scale + eh//(scale*2)
            eye_r = gaussian_blob(xs, ys, ecx, ecy, sigma_x=fw*0.14, sigma_y=fh*0.10)
            weight = weight + eye_r * (eye_boost - 1.0) * base_weight
            print(f"  [eye]  1 eye at ({ecx:.0f},{ecy:.0f})", file=sys.stderr)
        else:
            # estimate eye positions from face geometry
            print("  [eye]  not detected — estimating from face box", file=sys.stderr)
            for ex_frac in [0.33, 0.67]:
                ecx = fx + fw * ex_frac
                ecy = fy + fh * 0.38
                eye_r = gaussian_blob(xs, ys, ecx, ecy, sigma_x=fw*0.12, sigma_y=fh*0.09)
                weight = weight + eye_r * (eye_boost - 1.0) * base_weight

        # mouth / chin zone
        mx, my = fx + fw*0.5, fy + fh*0.72
        mouth_r = gaussian_blob(xs, ys, mx, my, sigma_x=fw*0.22, sigma_y=fh*0.12)
        weight = weight + mouth_r * (mouth_boost - 1.0) * base_weight
        print(f"  [mouth] estimated at ({mx:.0f},{my:.0f})", file=sys.stderr)

    return np.clip(weight, 0, 1)

def gaussian_blob(xs, ys, cx, cy, sigma_x, sigma_y):
    return np.exp(-0.5 * (((xs-cx)/sigma_x)**2 + ((ys-cy)/sigma_y)**2))

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--out',    required=True)
    ap.add_argument('--vis',    default=None, help='save PNG visualization')
    ap.add_argument('--edge',   action='store_true')
    ap.add_argument('--face',   action='store_true')
    ap.add_argument('--combined', action='store_true')
    ap.add_argument('--edge-sigma', type=float, default=4.0)
    ap.add_argument('--floor',  type=float, default=0.08,
                    help='minimum weight for any pixel (0=hard ignore, 0.08=soft)')
    ap.add_argument('--eye-boost',   type=float, default=4.0)
    ap.add_argument('--face-boost',  type=float, default=2.5)
    ap.add_argument('--mouth-boost', type=float, default=2.0)
    args = ap.parse_args()

    # default: combined
    if not (args.edge or args.face or args.combined):
        args.combined = True

    img, w, h = load_pgm(args.target)
    if w != W or h != H:
        import cv2
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    print(f"Target: {args.target} ({w}×{h})")

    # always compute edge as base
    edge_w = make_edge_weight(img, sigma=args.edge_sigma, floor=args.floor)

    if args.edge:
        weight = edge_w
        print("Mode: edge only")
    elif args.face:
        weight = make_face_weight(img, edge_w,
                                   eye_boost=args.eye_boost,
                                   face_boost=args.face_boost,
                                   mouth_boost=args.mouth_boost)
        print("Mode: face only")
    else:  # combined (default)
        weight = make_face_weight(img, edge_w,
                                   eye_boost=args.eye_boost,
                                   face_boost=args.face_boost,
                                   mouth_boost=args.mouth_boost)
        print("Mode: combined (edge + face)")

    # normalize: keep relative weights, map to uint8 1..255
    weight = np.clip(weight, 0, None)
    wmax = weight.max()
    if wmax > 0:
        weight = weight / wmax
    weight = np.clip(weight * 254 + 1, 1, 255).astype(np.uint8)

    # save raw uint8
    weight.tofile(args.out)
    print(f"Weight map: {args.out}  ({W}×{H}, uint8, min={weight.min()}, max={weight.max()}, mean={weight.mean():.1f})")

    # visualization
    if args.vis:
        from PIL import Image
        import cv2 as cv
        wf = weight.astype(np.float32) / 255.0
        # heatmap: blue=low, red=high
        hm = (wf * 255).astype(np.uint8)
        hm_color = cv.applyColorMap(hm, cv.COLORMAP_JET)
        hm_color = cv.cvtColor(hm_color, cv.COLOR_BGR2RGB)
        orig_rgb = np.stack([img]*3, axis=2)
        # blend
        alpha = 0.55
        blend = (hm_color * alpha + orig_rgb * (1-alpha)).astype(np.uint8)
        # scale up
        blend_big = cv.resize(blend, (W*4, H*4), interpolation=cv.INTER_NEAREST)
        Image.fromarray(blend_big).save(args.vis)
        print(f"Visualization: {args.vis}")

if __name__ == '__main__':
    main()

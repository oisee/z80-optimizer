#!/usr/bin/env python3
"""
encode_anim.py — full pipeline: video/frames → LFSR-16 animation_flat JSON

Usage:
  # From video file
  python3 cuda/encode_anim.py --input video.mp4 --out output.json

  # From directory of PGM/PNG frames
  python3 cuda/encode_anim.py --input frames/ --out output.json

  # Full options
  python3 cuda/encode_anim.py \
    --input che.mp4 \
    --out data/my_anim.json \
    --budget 256 \
    --kf-budget 512 \
    --weighted \
    --every 5 \
    --max-frames 25 \
    --gpu 0 \
    --workdir /tmp/encode_work \
    --name "My Animation"

Output: animation_flat JSON, compatible with docs/renderer.html

Build dependency:
  nvcc -O3 -o cuda/prng_budget_search cuda/prng_budget_search.cu -lm
"""

import sys, os, json, argparse, subprocess, glob, shutil, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_DIR   = SCRIPT_DIR.parent
SEARCH_BIN = REPO_DIR / 'cuda' / 'prng_budget_search'
HEATMAP_PY = SCRIPT_DIR / 'make_heatmap.py'

W, H = 128, 96

# ── ffmpeg frame extraction ───────────────────────────────────────────────────
def extract_frames(input_path, out_dir, every=5, max_frames=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"select='not(mod(n,{every}))',scale={W}:{H},format=gray"
    if max_frames:
        vf += f",trim=end_frame={max_frames * every}"
    cmd = ['ffmpeg', '-y', '-i', str(input_path),
           '-vf', vf, '-vsync', 'vfr',
           str(out_dir / 'frame_%03d.pgm')]
    print(f"[ffmpeg] extracting frames (every {every}th, {W}×{H} gray)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1000:], file=sys.stderr)
        sys.exit(1)
    frames = sorted(out_dir.glob('frame_*.pgm'))
    if max_frames:
        frames = frames[:max_frames]
    print(f"[ffmpeg] {len(frames)} frames extracted")
    return frames

# ── copy/link existing frames from directory ──────────────────────────────────
def collect_frames(frames_dir, max_frames=None):
    frames = sorted(Path(frames_dir).glob('*.pgm'))
    if not frames:
        frames = sorted(Path(frames_dir).glob('*.png'))
    if max_frames:
        frames = frames[:max_frames]
    print(f"[frames] found {len(frames)} frames in {frames_dir}")
    return frames

# ── generate weight maps ──────────────────────────────────────────────────────
def make_weightmaps(frames, wmap_dir, verbose=False):
    wmap_dir.mkdir(parents=True, exist_ok=True)
    wmaps = []
    for i, f in enumerate(frames):
        wmap = wmap_dir / (f.stem + '.wmap')
        if wmap.exists():
            wmaps.append(wmap)
            continue
        cmd = [sys.executable, str(HEATMAP_PY),
               '--target', str(f), '--out', str(wmap)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            print(r.stderr.strip())
        else:
            sys.stdout.write(f"\r[heatmap] {i+1}/{len(frames)}")
            sys.stdout.flush()
        wmaps.append(wmap)
    if not verbose:
        print(f"\r[heatmap] {len(frames)}/{len(frames)} weight maps done")
    return wmaps

# ── run CUDA search for one frame ─────────────────────────────────────────────
def search_frame(target, init_canvas, wmap, out_json, out_pgm,
                 budget, is_keyframe, gpu, auto_bounce, extra_args):
    cmd = [str(SEARCH_BIN)]
    if is_keyframe:
        cmd += ['--keyframe']
    else:
        cmd += ['--delta']
        if init_canvas:
            cmd += ['--init-canvas', str(init_canvas)]
    cmd += ['--budget', str(budget),
            '--target', str(target),
            '--out', str(out_json),
            '--out-pgm', str(out_pgm),
            '--gpu', str(gpu)]
    if wmap and wmap.exists():
        cmd += ['--weight-map', str(wmap)]
    if auto_bounce and not is_keyframe:
        cmd += ['--auto-bounce']
    cmd += extra_args

    r = subprocess.run(cmd, capture_output=True, text=True)
    # extract final error and seeds from output
    seeds_used, error = '?', '?'
    for line in r.stdout.split('\n'):
        if line.startswith('Final:'):
            # "Final: 508 seeds, 2.409% error, 14.3s"
            parts = line.split()
            seeds_used = parts[1]
            error = parts[3].rstrip('%')
    return seeds_used, error, r.returncode == 0

# ── assemble animation_flat JSON ──────────────────────────────────────────────
def assemble(seed_jsons, label):
    all_seeds = []
    frame_starts = []
    frame_sizes = []

    for p in seed_jsons:
        with open(p) as f:
            data = json.load(f)
        records = data.get('seeds', [])
        frame_starts.append(len(all_seeds))
        frame_sizes.append(len(records))
        all_seeds.extend(records)

    return {
        'type': 'animation_flat',
        'label': label,
        'n_frames': len(frame_sizes),
        'total_seeds': len(all_seeds),
        'frame_starts': frame_starts,
        'frame_sizes': frame_sizes,
        'seeds': all_seeds,
    }

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Encode video/frames as LFSR-16 animation_flat JSON')
    ap.add_argument('--input',       required=True, help='Video file or directory of PGM frames')
    ap.add_argument('--out',         required=True, help='Output animation_flat JSON path')
    ap.add_argument('--budget',      type=int, default=128, help='Seeds per delta frame (default 128)')
    ap.add_argument('--kf-budget',   type=int, default=-1,  help='Seeds for keyframe (default: 2× delta budget)')
    ap.add_argument('--weighted',    action='store_true',   help='Use heatmap weight map (face/edge priority)')
    ap.add_argument('--every',       type=int, default=5,   help='Extract every Nth frame from video (default 5)')
    ap.add_argument('--max-frames',  type=int, default=None,help='Max frames to encode')
    ap.add_argument('--gpu',         type=int, default=0,   help='GPU device ID (default 0)')
    ap.add_argument('--auto-bounce', action='store_true',   help='Auto-pick blk for delta L0')
    ap.add_argument('--shrink',      type=float, default=-1,help='Area shrink per KF phase (default 0.90)')
    ap.add_argument('--workdir',     default=None,          help='Working directory (default: /tmp/encode_<name>)')
    ap.add_argument('--name',        default=None,          help='Animation label (default: input filename stem)')
    ap.add_argument('--keep-work',   action='store_true',   help='Keep workdir after encoding')
    ap.add_argument('--verbose',     action='store_true')
    args, extra = ap.parse_known_args()

    if not SEARCH_BIN.exists():
        print(f"ERROR: {SEARCH_BIN} not found. Build with:", file=sys.stderr)
        print(f"  nvcc -O3 -o cuda/prng_budget_search cuda/prng_budget_search.cu -lm", file=sys.stderr)
        sys.exit(1)

    name = args.name or Path(args.input).stem
    kf_budget = args.kf_budget if args.kf_budget > 0 else args.budget * 2
    workdir = Path(args.workdir) if args.workdir else Path(f'/tmp/encode_{name}')
    workdir.mkdir(parents=True, exist_ok=True)

    frames_dir  = workdir / 'frames'
    wmaps_dir   = workdir / 'wmaps'
    seeds_dir   = workdir / 'seeds'
    results_dir = workdir / 'results'
    seeds_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"=== encode_anim: {name} ===")
    print(f"  budget: kf={kf_budget} dt={args.budget}  weighted={args.weighted}  gpu={args.gpu}")

    # Step 1: get frames
    inp = Path(args.input)
    if inp.is_dir():
        frames = collect_frames(inp, args.max_frames)
    else:
        frames = extract_frames(inp, frames_dir, args.every, args.max_frames)

    n = len(frames)
    print(f"  encoding {n} frames")

    # Step 2: weight maps
    wmaps = [None] * n
    if args.weighted:
        wmaps = make_weightmaps(frames, wmaps_dir, args.verbose)

    # Step 3: CUDA search
    extra_args = extra[:]
    if args.shrink > 0:
        extra_args += ['--shrink', str(args.shrink)]

    seed_jsons = []
    prev_pgm   = None
    t0 = time.time()

    print(f"\n{'Fr':4}  {'mode':9}  {'seeds':6}  {'error':8}  {'elapsed':8}")
    print('-' * 45)

    for i, frame in enumerate(frames):
        fn = f"{i+1:03d}"
        out_json = seeds_dir  / f'seeds_{fn}.json'
        out_pgm  = results_dir / f'result_{fn}.pgm'
        is_kf    = (i == 0)
        budget   = kf_budget if is_kf else args.budget

        seeds_used, error, ok = search_frame(
            target=frame, init_canvas=prev_pgm,
            wmap=wmaps[i], out_json=out_json, out_pgm=out_pgm,
            budget=budget, is_keyframe=is_kf,
            gpu=args.gpu, auto_bounce=args.auto_bounce,
            extra_args=extra_args)

        mode = 'keyframe' if is_kf else 'delta'
        elapsed = time.time() - t0
        status = '' if ok else ' [WARN]'
        print(f"  {fn}   {mode:9}  {seeds_used:>6}  {error:>6}%   {elapsed:6.1f}s{status}")

        seed_jsons.append(out_json)
        prev_pgm = out_pgm

    # Step 4: assemble
    print(f"\n[assemble] {n} frames → {args.out}")
    anim = assemble(seed_jsons, label=name)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(anim, f, separators=(',', ':'))

    sz = out_path.stat().st_size
    total_time = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Output:  {out_path}  ({sz//1024}KB)")
    print(f"  Frames:  {anim['n_frames']}")
    print(f"  Seeds:   {anim['total_seeds']}")
    print(f"  Time:    {total_time:.1f}s  ({total_time/n:.1f}s/frame)")
    print(f"  Sizes:   {anim['frame_sizes']}")

    if not args.keep_work:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"  Workdir: {workdir}")

    print(f"\nPlay: open docs/renderer.html and drag-drop {out_path}")

if __name__ == '__main__':
    main()

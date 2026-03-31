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
    --budget 64 \
    --kf-budget 256 \
    --every 4 \
    --kf-every 100 \
    --kf-error 15.0 \
    --weighted \
    --gpu 0 \
    --name "My Animation"

Output: animation_flat JSON, compatible with docs/renderer.html

Keyframe insertion (--kf-every / --kf-error):
  A keyframe resets the canvas to black and re-encodes from scratch.
  --kf-every N   : insert keyframe every N frames unconditionally
  --kf-error X   : insert keyframe when delta error exceeds X%
  Both conditions are OR'd — either one triggers a keyframe.
  The JSON includes frame_types[] ('kf'/'dt') so the renderer can
  reset the canvas at keyframe boundaries.

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
    cmd = ['ffmpeg', '-y', '-i', str(input_path),
           '-vf', vf, '-vsync', 'vfr',
           str(out_dir / 'frame_%04d.pgm')]
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
    # --cp applies to delta frames only; strip it for keyframes
    filtered_extra = [a for a in extra_args if not (is_keyframe and a == '--cp')]
    cmd += filtered_extra

    r = subprocess.run(cmd, capture_output=True, text=True)
    seeds_used, error = '?', None
    for line in r.stdout.split('\n'):
        if line.startswith('Final:'):
            parts = line.split()
            seeds_used = parts[1]
            try:
                error = float(parts[3].rstrip('%'))
            except Exception:
                pass
    return seeds_used, error, r.returncode == 0

# ── assemble animation_flat JSON ──────────────────────────────────────────────
def _cp_expanded_size(record):
    """Number of expanded seed entries for a single JSON seed record.
    CP records expand to 1 carrier + len(ps) payloads; flat records stay as 1."""
    if isinstance(record, dict) and record.get('type') == 'cp':
        return 1 + len(record.get('ps', []))
    return 1

def assemble(seed_jsons, frame_types, label):
    all_seeds = []
    frame_starts = []
    frame_sizes = []

    # frame_starts/frame_sizes track EXPANDED seed indices so that the
    # renderer's frame boundaries remain correct after CP record expansion.
    expanded_total = 0
    for p in seed_jsons:
        with open(p) as f:
            data = json.load(f)
        records = data.get('seeds', [])
        frame_starts.append(expanded_total)
        expanded = sum(_cp_expanded_size(r) for r in records)
        frame_sizes.append(expanded)
        expanded_total += expanded
        all_seeds.extend(records)

    return {
        'type': 'animation_flat',
        'label': label,
        'n_frames': len(frame_sizes),
        'total_seeds': len(all_seeds),
        'frame_starts': frame_starts,
        'frame_sizes': frame_sizes,
        'frame_types': frame_types,   # 'kf' or 'dt' per frame
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
    ap.add_argument('--cp',          action='store_true',   help='Use carrier-payload (CP) mode for delta frames')
    ap.add_argument('--cp-seeds',    type=int, default=255, help='Carrier seed budget: 255=u8 (fast), 65535=u16 (thorough)')
    ap.add_argument('--cp-andN-lo',  type=int, default=3,   help='Carrier AND-N range low (default 3)')
    ap.add_argument('--cp-andN-hi',  type=int, default=8,   help='Carrier AND-N range high (default 8)')
    ap.add_argument('--cp-andN',     type=int, default=None,help='Fix carrier AND-N to single value (overrides lo/hi)')
    ap.add_argument('--shrink',      type=float, default=-1,help='Area shrink per KF phase (default 0.90)')
    ap.add_argument('--kf-every',    type=int, default=0,   help='Insert keyframe every N frames (0=off)')
    ap.add_argument('--kf-error',    type=float, default=0, help='Insert keyframe when delta error > X%% (0=off)')
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
    if args.kf_every > 0:
        print(f"  kf-every: {args.kf_every} frames")
    if args.kf_error > 0:
        print(f"  kf-error: >{args.kf_error}% triggers keyframe")

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
    if args.cp:
        extra_args += ['--cp']
        if args.cp_seeds != 255:
            extra_args += ['--cp-seeds', str(args.cp_seeds)]
        if args.cp_andN is not None:
            extra_args += ['--cp-andN', str(args.cp_andN)]
        else:
            if args.cp_andN_lo != 3:
                extra_args += ['--cp-andN-lo', str(args.cp_andN_lo)]
            if args.cp_andN_hi != 8:
                extra_args += ['--cp-andN-hi', str(args.cp_andN_hi)]

    seed_jsons  = []
    frame_types = []
    prev_pgm    = None
    t0 = time.time()
    kf_count = 0
    dt_count = 0

    print(f"\n{'Fr':>5}  {'type':8}  {'seeds':6}  {'error':8}  {'elapsed':8}  {'note'}")
    print('-' * 55)

    for i, frame in enumerate(frames):
        fn = f"{i+1:04d}"
        out_json = seeds_dir  / f'seeds_{fn}.json'
        out_pgm  = results_dir / f'result_{fn}.pgm'

        # Decide keyframe vs delta
        force_kf = (
            i == 0                                          # always kf for first frame
            or (args.kf_every > 0 and i % args.kf_every == 0)  # periodic
        )
        is_kf = force_kf
        note  = 'periodic' if (force_kf and i > 0 and args.kf_every > 0) else ''

        budget = kf_budget if is_kf else args.budget

        seeds_used, error, ok = search_frame(
            target=frame, init_canvas=(None if is_kf else prev_pgm),
            wmap=wmaps[i], out_json=out_json, out_pgm=out_pgm,
            budget=budget, is_keyframe=is_kf,
            gpu=args.gpu, auto_bounce=args.auto_bounce,
            extra_args=extra_args)

        # Adaptive keyframe: re-encode as kf if error too high.
        # Guard: only trigger once per scene — if previous frame was already
        # a kf and still bad, the scene is just dense; don't cascade.
        prev_was_kf = len(frame_types) > 0 and frame_types[-1] == 'kf'
        if (not is_kf and args.kf_error > 0
                and error is not None and error > args.kf_error
                and not prev_was_kf):
            note = f're-kf (err={error:.1f}%)'
            is_kf = True
            budget = kf_budget
            seeds_used, error, ok = search_frame(
                target=frame, init_canvas=None,
                wmap=wmaps[i], out_json=out_json, out_pgm=out_pgm,
                budget=budget, is_keyframe=True,
                gpu=args.gpu, auto_bounce=False,
                extra_args=extra_args)

        ftype = 'kf' if is_kf else 'dt'
        frame_types.append(ftype)
        if is_kf: kf_count += 1
        else:      dt_count += 1

        elapsed = time.time() - t0
        err_str = f"{error:.2f}%" if error is not None else '?'
        status  = '' if ok else ' [WARN]'
        print(f"  {fn}  {ftype:8}  {seeds_used:>6}  {err_str:>8}  {elapsed:6.1f}s  {note}{status}")

        seed_jsons.append(out_json)
        prev_pgm = out_pgm

    # Step 4: assemble
    print(f"\n[assemble] {n} frames ({kf_count} kf + {dt_count} dt) → {args.out}")
    anim = assemble(seed_jsons, frame_types, label=name)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(anim, f, separators=(',', ':'))

    sz = out_path.stat().st_size
    total_time = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Output:  {out_path}  ({sz//1024}KB)")
    print(f"  Frames:  {anim['n_frames']}  ({kf_count} kf + {dt_count} dt)")
    print(f"  Seeds:   {anim['total_seeds']}")
    print(f"  Time:    {total_time:.1f}s  ({total_time/n:.1f}s/frame)")
    print(f"  KF at:   {[i for i,t in enumerate(frame_types) if t=='kf'][:20]}{'...' if kf_count>20 else ''}")

    if not args.keep_work:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"  Workdir: {workdir}")

    print(f"\nPlay: open docs/renderer.html and drag-drop {out_path}")

if __name__ == '__main__':
    main()

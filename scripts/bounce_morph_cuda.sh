#!/bin/bash
# Free-bounce morphing via CUDA subprocess calls.
# Each step: bruteforce toward new target ON TOP of previous canvas.
# Switch target when quality is good enough.

CUDA=cuda/prng_segmented_search
OUTBASE=media/prng_images/morph_bounce_cuda
TARGETS=(
    media/prng_images/targets/che.pgm
    media/prng_images/targets/einstein_photo_bin.pgm
    media/prng_images/targets/monalisa.pgm
    media/prng_images/targets/raised_fist.pgm
    media/prng_images/targets/uncle_sam.pgm
    media/prng_images/targets/masked_protester.pgm
)
NAMES=(che einstein monalisa fist uncle_sam masked)
MODE=quadtree  # full coverage for best morphing
DENSITY=2      # 85 seeds per step

mkdir -p $OUTBASE

PREV_CANVAS=""
FRAME=0

for i in "${!TARGETS[@]}"; do
    target="${TARGETS[$i]}"
    name="${NAMES[$i]}"
    out="$OUTBASE/step_${FRAME}_${name}"

    echo "=== Step $FRAME: $name ==="

    if [ -z "$PREV_CANVAS" ]; then
        $CUDA --target "$target" --mode $MODE --density $DENSITY --output "$out"
    else
        $CUDA --target "$target" --canvas "$PREV_CANVAS" --mode $MODE --density $DENSITY --output "$out"
    fi

    PREV_CANVAS="$out/canvas.pgm"

    # Convert PGMs to PNGs
    for pgm in "$out"/*.pgm; do
        python3 -c "from PIL import Image; Image.open('$pgm').save('${pgm%.pgm}.png')" 2>/dev/null
    done

    FRAME=$((FRAME + 1))
done

echo
echo "=== Generating animation ==="

# Generate animated GIF from all level frames
python3 << PYEOF
import numpy as np
from PIL import Image
import glob, os

base = "$OUTBASE"
names = ["che", "einstein", "monalisa", "fist", "uncle_sam", "masked"]
pals = [((255,20,147),(255,255,0)),((64,224,208),(255,105,180)),
        ((255,165,0),(0,191,255)),((255,0,0),(0,0,0)),
        ((0,80,255),(255,255,255)),((0,200,0),(0,0,0))]

def load_pgm(p):
    with open(p,'rb') as f:
        assert f.readline().strip()==b'P5'
        while True:
            l=f.readline().strip()
            if not l.startswith(b'#'): break
        w,h=map(int,l.split()); int(f.readline().strip())
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(h,w)

def colorize(m,pi):
    ink,paper=pals[pi%len(pals)]
    o=np.zeros((*m.shape,3),dtype=np.uint8)
    for c in range(3): o[:,:,c]=np.where(m>=128,ink[c],paper[c])
    return o

gif_frames = []
for i, name in enumerate(names):
    d = f"{base}/step_{i}_{name}"
    levels = sorted(glob.glob(f"{d}/level*_err*.pgm"))
    for lf in levels:
        mono = load_pgm(lf)
        color = colorize(mono, i)
        gif_frames.append(Image.fromarray(color))

if gif_frames:
    # Hold last frame of each target longer
    durations = [150] * len(gif_frames)
    # Find last frame of each target and make it 500ms
    frame_idx = 0
    for i, name in enumerate(names):
        d = f"{base}/step_{i}_{name}"
        n = len(glob.glob(f"{d}/level*_err*.pgm"))
        if n > 0:
            durations[frame_idx + n - 1] = 800  # pause on completed face
        frame_idx += n

    gif_frames[0].save(f"{base}/bounce_morph.gif", save_all=True,
                       append_images=gif_frames[1:], duration=durations, loop=0)
    print(f"GIF: {len(gif_frames)} frames → {base}/bounce_morph.gif")

# Grid: final frame of each target
key = []
for i, name in enumerate(names):
    d = f"{base}/step_{i}_{name}"
    canvas = f"{d}/canvas.pgm"
    if os.path.exists(canvas):
        key.append(colorize(load_pgm(canvas), i))

if len(key) >= 6:
    h,w = key[0].shape[:2]
    grid = np.zeros((h*2,w*3,3),dtype=np.uint8)
    for i,k in enumerate(key[:6]):
        gy,gx=divmod(i,3)
        grid[gy*h:(gy+1)*h,gx*w:(gx+1)*w]=k
    Image.fromarray(grid).save(f"{base}/morph_grid.png")
    print(f"Grid: {base}/morph_grid.png")

# Playback strip (every level of every target)
if gif_frames:
    n = len(gif_frames)
    cols = min(n, 6)
    rows = (n+cols-1)//cols
    h,w = gif_frames[0].size[1], gif_frames[0].size[0]
    strip = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
    for i,f in enumerate(gif_frames):
        gy,gx=divmod(i,cols)
        strip[gy*h:(gy+1)*h, gx*w:(gx+1)*w] = np.array(f)
    Image.fromarray(strip).save(f"{base}/playback_grid.png")
    print(f"Playback: {base}/playback_grid.png ({rows}x{cols})")

PYEOF

echo "Done! Results in $OUTBASE/"

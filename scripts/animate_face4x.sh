#!/bin/bash
# Face4x Animation Pipeline
# Usage: ./scripts/animate_face4x.sh <video.mp4> <output_dir> [fps] [threshold]
#
# Extracts frames, binarizes, bruteforces each with face4x (213 seeds),
# generates animated GIF + grid + key strip.
#
# Each frame: 213 seeds × 2 bytes = 426 bytes
# Example: 25 frames = 10.4 KB total seed data

set -e

VIDEO="$1"
OUTDIR="$2"
FPS="${3:-5}"
THRESH="${4:-80}"

if [ -z "$VIDEO" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: $0 <video.mp4> <output_dir> [fps=5] [threshold=80]"
    echo ""
    echo "Extracts frames, binarizes (red channel >= threshold),"
    echo "bruteforces each with face4x recipe (213 seeds per frame)."
    echo ""
    echo "Output: animated.gif, all_grid.png, key_strip.png, frame_NNN/"
    exit 1
fi

mkdir -p "$OUTDIR/frames" "$OUTDIR/pgm" "$OUTDIR/results"

# Step 1: Extract frames
echo "=== Extracting frames at ${FPS}fps ==="
ffmpeg -i "$VIDEO" -vf "fps=${FPS},scale=128:96:flags=lanczos" \
    -y "$OUTDIR/frames/frame_%03d.png" 2>&1 | tail -3
NFRAMES=$(ls "$OUTDIR/frames/frame_"*.png 2>/dev/null | wc -l)
echo "Extracted: $NFRAMES frames"

# Step 2: Binarize (red channel threshold)
echo "=== Binarizing (red channel >= $THRESH) ==="
python3 -c "
from PIL import Image
import numpy as np, glob, os
d = '$OUTDIR'
for f in sorted(glob.glob(f'{d}/frames/frame_*.png')):
    img = Image.open(f).convert('RGB')
    r = np.array(img)[:,:,0]
    binary = (r >= $THRESH).astype(np.uint8) * 255
    name = os.path.basename(f).replace('.png', '.pgm')
    with open(f'{d}/pgm/{name}', 'wb') as out:
        out.write(f'P5\n128 96\n255\n'.encode())
        out.write(binary.tobytes())
print(f'Binarized {len(glob.glob(f\"{d}/pgm/*.pgm\"))} frames')
"

# Step 3: Generate face4x segment file
echo "=== Generating face4x segments ==="
python3 -c "
import numpy as np, os
W, H = 128, 96
face_regions = {
    'face_center':(24,8,80,64),'eyes_band':(16,20,96,24),
    'left_eye':(32,24,32,16),'right_eye':(64,24,32,16),
    'nose':(48,34,32,24),'mouth':(40,48,48,20),
    'forehead':(24,4,80,20),'chin':(40,62,48,16),
    'left_cheek':(16,28,24,36),'right_cheek':(88,28,24,36),
}
def subdivide(rx,ry,rw,rh,nx,ny,block):
    tiles=[]
    tw=(rw//nx//block)*block; th=(rh//ny//block)*block
    if tw<block: tw=block
    if th<block: th=block
    for ty in range(ny):
        for tx in range(nx):
            x=rx+tx*tw;y=ry+ty*th;w=min(tw,rx+rw-x);h=min(th,ry+rh-y)
            w=(w//block)*block;h=(h//block)*block
            if w>=block and h>=block: tiles.append((x,y,w,h,block))
    return tiles
scale=4; all_segs=[(0,0,W,H,8)]
for qy in range(2):
    for qx in range(2): all_segs.append((qx*64,qy*48,64,48,4))
for name in ['face_center','eyes_band','forehead']:
    rx,ry,rw,rh=face_regions[name]
    for t in subdivide(rx,ry,rw,rh,scale,1,4): all_segs.append(t)
for bg in [(0,0,40,32,2),(88,0,40,32,2),(0,64,48,32,2),(80,64,48,32,2)]: all_segs.append(bg)
for name in ['left_eye','right_eye','nose','mouth','left_cheek','right_cheek']:
    rx,ry,rw,rh=face_regions[name]
    for t in subdivide(rx,ry,rw,rh,scale,scale,2): all_segs.append(t)
fine=[(32,27,24,12),(72,27,24,12),(52,42,24,14),(44,52,40,12),(36,22,24,8),(68,22,24,8)]
for rx,ry,rw,rh in fine:
    for t in subdivide(rx,ry,rw,rh,scale,scale,1): all_segs.append(t)
path='/tmp/face4x_anim.txt'
with open(path,'w') as f:
    for s in all_segs:
        rx,ry,rw,rh,blk=s[:5];npx=max((rw//blk)*(rh//blk),1)
        f.write(f'{rx} {ry} {rw} {rh} {blk} {npx*3} 0\n')
n=len(all_segs); link=f'/tmp/face{n}x_segs.txt'
if os.path.exists(link): os.unlink(link)
os.symlink(path,link)
print(f'Face4x: {n} segments')
"

N=$(wc -l < /tmp/face4x_anim.txt)

# Step 4: Bruteforce each frame
echo "=== Bruteforcing $NFRAMES frames ($N segments each) ==="
for pgm in $(ls "$OUTDIR/pgm/"*.pgm | sort); do
    name=$(basename "$pgm" .pgm)
    out="$OUTDIR/results/$name"
    cuda/prng_segmented_search --target "$pgm" \
        --mode facefile --density $N --output "$out" 2>&1 > /dev/null
    err=$(ls "$out"/level*_err*.pgm 2>/dev/null | tail -1 | grep -o 'err[0-9]*' | grep -o '[0-9]*')
    python3 -c "from PIL import Image; Image.open('$out/canvas.pgm').save('$out/canvas.png')" 2>/dev/null
    echo "  $name: err=$err ($(echo "scale=1; $err*100/12288" | bc)%)"
done

# Step 5: Generate visuals
echo "=== Generating animation ==="
python3 << PYEOF
import numpy as np
from PIL import Image
import glob

def load_pgm(p):
    with open(p,'rb') as f:
        assert f.readline().strip()==b'P5'
        while True:
            l=f.readline().strip()
            if not l.startswith(b'#'): break
        w,h=map(int,l.split()); int(f.readline().strip())
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(h,w)

base = "$OUTDIR/results"
canvases = sorted(glob.glob(f"{base}/*/canvas.pgm"))
n = len(canvases)
if n == 0:
    print("No results found!")
    exit(1)

gif = []
for c in canvases:
    mono = load_pgm(c)
    color = np.zeros((96,128,3), dtype=np.uint8)
    color[:,:,0] = np.where(mono>=128, 220, 10)
    color[:,:,1] = np.where(mono>=128, 30, 0)
    gif.append(Image.fromarray(color))

gif[0].save(f"$OUTDIR/animated.gif", save_all=True,
            append_images=gif[1:], duration=200, loop=0)

h, w = 96, 128
# Grid (auto-size)
cols = min(n, 5)
rows = (n + cols - 1) // cols
grid = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
for i, g in enumerate(gif):
    gy, gx = divmod(i, cols)
    grid[gy*h:(gy+1)*h, gx*w:(gx+1)*w] = np.array(g)
Image.fromarray(grid).save(f"$OUTDIR/all_grid.png")

# Key strip (6 evenly spaced)
step = max(1, n // 6)
key = [gif[i] for i in range(0, n, step)][:6]
strip = np.zeros((h, w*len(key), 3), dtype=np.uint8)
for i, k in enumerate(key): strip[:, i*w:(i+1)*w] = np.array(k)
Image.fromarray(strip).save(f"$OUTDIR/key_strip.png")

print(f"Done: {n} frames, {n*213*2}B seeds ({n*213*2/1024:.1f}KB)")
print(f"  animated.gif, all_grid.png, key_strip.png")
PYEOF

echo "=== Complete: $OUTDIR/ ==="

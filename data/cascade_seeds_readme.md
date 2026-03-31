# Cascade AND-3→7 Seed List

Renderer data for reproducing the 6-frame cascade progression on any platform.

**Result:** 1.2% binary pixel error @ 1171 seeds. LFSR-16 only.

## Files

- `cascade_seeds.json` — 1171 seed records, ordered (apply in sequence)
- Snapshots at steps 21 / 149 / 213 / 405 / 597 / 1205: `result_cas_s*.png` in `docs/`

## Algorithm (pseudocode)

```python
# 1. LFSR-16
def lfsr16(state):
    bit = state & 1
    state >>= 1
    if bit: state ^= 0xB400
    return state

# 2. Make pattern buffer (768 blocks for a 128×96 / 32×24 grid)
def make_buf(seed, warmup, and_n):
    state = seed if seed != 0 else 1
    for _ in range(warmup):
        state = lfsr16(state)
    buf = []
    for _ in range(768):       # 32×24 blocks
        acc = 1
        for _ in range(and_n): # AND this many consecutive bits
            state = lfsr16(state)
            acc &= (state & 1)
        buf.append(acc)
    return buf                 # list of 768 values in {0,1}

# 3. Apply buffer onto canvas (XOR)
def apply_buf(canvas, buf, ox, oy, blk):
    # canvas: 128×96 binary image (list of 128*96 bits, or packed bytes)
    for by in range(24):
        for bx in range(32):
            if buf[by*32 + bx] == 0:
                continue
            for dy in range(blk):
                for dx in range(blk):
                    x = ox + bx*blk + dx
                    y = oy + by*blk + dy
                    if 0 <= x < 128 and 0 <= y < 96:
                        canvas[y][x] ^= 1   # XOR flip

# 4. Render all frames
canvas = [[0]*128 for _ in range(96)]

SNAPSHOTS = {21, 149, 213, 405, 597, 1205}
frames = {}

seeds = load_json("cascade_seeds.json")["seeds"]
for rec in seeds:
    buf = make_buf(rec["seed"], rec["warmup"], rec["and_n"])
    apply_buf(canvas, buf, rec["ox"], rec["oy"], rec["blk"])
    if rec["step"] in SNAPSHOTS:
        frames[rec["step"]] = [row[:] for row in canvas]  # snapshot
```

## Seed record format

```json
{
  "step":   1,        // sequential index (1-based), apply in order
  "seed":   450,      // LFSR-16 initial state (1..65535)
  "ox":     0,        // canvas X offset for this patch
  "oy":     0,        // canvas Y offset for this patch
  "blk":    4,        // block size in pixels (4, 2, or 1)
  "and_n":  3,        // how many LFSR bits to AND per block (3..7)
  "warmup": 0,        // LFSR steps to run before filling buffer
  "label":  "L0-AND3" // human-readable layer name
}
```

## Layer schedule

| Steps | AND degree | P(flip) | Role |
|-------|:----------:|:-------:|------|
| 1     | AND-3 | 1/8  | L0 blk=4 full screen — broad strokes |
| 2–5   | AND-3 | 1/8  | L1 blk=2 quadrants |
| 6–21  | AND-4 | 1/16 | L2 blk=1 16 patches |
| 22–149 | AND-5 | 1/32 | passes 1-8, medium corrections |
| 150–405 | AND-6 | 1/64 | passes 9-24, fine corrections |
| 406–1205 | AND-7 | 1/128 | passes 25-74, pointwise corrections |

## Frame snapshots

| Step | L_bin | Description |
|------|------:|-------------|
| 21   | 41.6% | L2-AND4 done — patches set |
| 149  | 28.6% | AND-5 done — silhouette visible |
| 213  | 24.5% | ≡ 2D spray budget — face readable |
| 405  | 16.2% | AND-6 done — portrait clear |
| 597  | 10.1% | ≡ quadtree budget — fine detail |
| 1205 |  1.2% | AND-7 done — near-perfect |

## JavaScript snippet

```js
function lfsr16(s) { return (s >> 1) ^ (s & 1 ? 0xB400 : 0); }

function makeBuf(seed, warmup, andN) {
  let s = seed || 1;
  for (let i = 0; i < warmup; i++) s = lfsr16(s);
  const buf = new Uint8Array(768);
  for (let i = 0; i < 768; i++) {
    let acc = 1;
    for (let k = 0; k < andN; k++) { s = lfsr16(s); acc &= (s & 1); }
    buf[i] = acc;
  }
  return buf;
}

function applyBuf(canvas, buf, ox, oy, blk) {
  for (let by = 0; by < 24; by++)
    for (let bx = 0; bx < 32; bx++) {
      if (!buf[by*32+bx]) continue;
      for (let dy = 0; dy < blk; dy++)
        for (let dx = 0; dx < blk; dx++) {
          const x = ox + bx*blk + dx, y = oy + by*blk + dy;
          if (x < 128 && y < 96) canvas[y*128+x] ^= 1;
        }
    }
}
```

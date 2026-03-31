# Foveal AND-Cascade — Canonical Snapshot

**Date:** 2026-03-31
**Result:** 0.06% binary pixel error, 939 seeds, 28.2s on RTX 4060 Ti (CUDA)

## What's here

| File | Description |
|------|-------------|
| `foveal_cascade_seeds.json` | 939 seed records — apply in order to reproduce |
| `result.pgm` | Final canvas (128×96 binary PGM) |
| `s0001.pgm` … `s1209.pgm` | Milestone snapshots |
| `prng_cascade_search.cu` | CUDA source that produced this result |
| `buf_foveal_cascade.go` | Go reference implementation |

## Reproduce

```bash
nvcc -O3 -o cascade_search prng_cascade_search.cu -lm
./cascade_search --target <your_target.pgm> --out my_seeds.json
```

Or see `docs/foveal_cascade.md` for full algorithm description.

## Key numbers

| Budget | Error |
|-------:|------:|
| 25 steps | 32.9% |
| 100 steps | 23.9% |
| 213 steps | 16.2% |
| 597 steps | 4.1% |
| 1209 steps | **0.06%** |

Applied 939/1209 seeds. 128 unique positions. 30ms/step on GPU.

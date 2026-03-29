# Segmented Hierarchical Search — Che Guevara

**Method:** Image split into progressively smaller rectangles, each brute-forced (65536 LFSR seeds).
XOR correction: each level fixes errors from previous levels.

## Target

![target](target.png)

## Progression (85 seeds = 170 bytes, density=3)

| Level 0: 1 seed, 8x8 (35.4%) | Level 1: +4 seeds, 4x4 (34.5%) |
|---|---|
| ![L0](level0_compare.png) | ![L1](level1_compare.png) |

| Level 2: +16 seeds, 2x2 (32.3%) | Level 3: +64 seeds, 1x1 (31.2%) |
|---|---|
| ![L2](level2_compare.png) | ![L3](level3_compare.png) |

## Segment Budget

| Level | Segments | Cumul | Block | Coverage | Error |
|-------|----------|-------|-------|----------|-------|
| 0 | 1 | 1 | 8x8 | whole image | 35.4% |
| 1 | 4 | 5 | 4x4 | 4 quadrants 64x48 | 34.5% |
| 2 | 16 | 21 | 2x2 | 16 tiles 32x24 | 32.3% |
| 3 | 64 | 85 | 1x1 | 64 tiles 16x12 | 31.2% |

Total: 85 seeds x 2 bytes = **170 bytes** data. Search time: <1 second.

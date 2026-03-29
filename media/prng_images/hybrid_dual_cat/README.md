# Dual-Layer Cat (quick run)

**Mode:** dual-layer, 4096 pop, 8 islands, 500 gens
**GPU:** RTX 4060 Ti, ~557K img/s
**Best fitness:** 0.0590

First dual-layer test. Proved 5.8x improvement over single-layer (0.284 -> 0.059).

## Target

![target](target.png)

## Final Result (target | generated)

![final](final_compare.png)

## Evolution

| Gen 100 (f=0.1017) | Gen 200 (f=0.0960) | Gen 300 (f=0.0851) |
|---|---|---|
| ![g100](gen0100_compare.png) | ![g200](gen0200_compare.png) | ![g300](gen0300_compare.png) |

| Gen 400 (f=0.0614) | Gen 500 (f=0.0590) |
|---|---|
| ![g400](gen0400_compare.png) | ![g500](gen0500_compare.png) |

## Best per checkpoint

| Gen | Fitness | Image |
|-----|---------|-------|
| 100 | 0.1017 | ![](gen0100_f0.1017.png) |
| 200 | 0.0960 | ![](gen0200_f0.0960.png) |
| 300 | 0.0851 | ![](gen0300_f0.0851.png) |
| 400 | 0.0614 | ![](gen0400_f0.0614.png) |
| 500 | 0.0590 | ![](gen0500_f0.0590.png) |

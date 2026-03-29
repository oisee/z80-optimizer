# Single-Layer Skull (baseline)

**Mode:** single-layer, 4096 pop, 8 islands, forced sym diversity, 2000 gens
**GPU:** RTX 4060 Ti (GPU1), ~948K img/s
**Best fitness:** 0.3448

Baseline single-layer skull search. Later superseded by dual-layer (2.3x better).

## Target

![target](target.png)

## Final Result (target | generated)

![final](final_compare.png)

## Evolution

| Gen 200 (f=0.3474) | Gen 400 (f=0.3471) | Gen 600 (f=0.3471) |
|---|---|---|
| ![g200](gen0200_compare.png) | ![g400](gen0400_compare.png) | ![g600](gen0600_compare.png) |

| Gen 1000 (f=0.3471) | Gen 1600 (f=0.3453) | Gen 2000 (f=0.3448) |
|---|---|---|
| ![g1000](gen1000_compare.png) | ![g1600](gen1600_compare.png) | ![g2000](gen2000_compare.png) |

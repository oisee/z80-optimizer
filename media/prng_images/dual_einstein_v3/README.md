# Einstein v3 — Aggressive Restarts

**Mode:** dual-layer + subtractive, 8192 pop, 16 islands, 10000 gens, restart-stall=50
**GPU:** RTX 4060 Ti, ~300K img/s (82M images total)
**Best fitness:** 0.1509

Aggressive island restarts (every 50 stalled gens) help escape local optima.

## Target

![target](target.png)

## Final (f=0.1509)

![final](final_compare.png)

## Evolution

| Gen 500 (f=0.1667) | Gen 1000 (f=0.1566) | Gen 2000 (f=0.1542) |
|---|---|---|
| ![g500](gen0500_compare.png) | ![g1000](gen1000_compare.png) | ![g2000](gen2000_compare.png) |

| Gen 5000 (f=0.1541) | Gen 8000 (f=0.1541) | Gen 10000 (f=0.1509) |
|---|---|---|
| ![g5000](gen5000_compare.png) | ![g8000](gen8000_compare.png) | ![g10000](gen10000_compare.png) |

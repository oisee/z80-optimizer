# pRNG Image Search Gallery

Z80 pRNG seeds that generate recognizable patterns from pure noise.
Each image = 8-byte SEED → Patrik Rak CMWC pRNG → 128×96 mono → OR horizontal flip → grayscale.

## Pipeline
```
SEED (8 bytes)
  → CMWC pRNG (×253, period ~2^66)
  → 1536 bytes (128×96 mono, 1 bit/pixel)
  → OR with horizontal flip (vertical symmetry)
  → 4×4 block average → 32×24 grayscale
  → MobileNetV2 (ImageNet) → class probability
```

## Search Methods
- **Exhaustive**: 4-byte seed, 4.3B candidates, ~2 min on RTX 4060 Ti
- **Hill climbing**: 8-byte seed, 1500 population × 300 generations × mutations
- **Multi-target**: 7 ImageNet classes searched simultaneously
- **Top-10**: best 10 seeds kept per target class

## Leaderboard (all targets, sorted by CNN score)

| Rank | Score | Target | Seed | Mono | Grayscale |
|------|-------|--------|------|------|-----------|
| 1 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/00_mono_0x7776B0B492A3C25E.png) | ![g](cat/00_gray_0x7776B0B492A3C25E.png) |
| 2 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/01_mono_0x7776B0B492A3C25E.png) | ![g](cat/01_gray_0x7776B0B492A3C25E.png) |
| 3 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/02_mono_0x7776B0B492A3C25E.png) | ![g](cat/02_gray_0x7776B0B492A3C25E.png) |
| 4 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/03_mono_0x7776B0B492A3C25E.png) | ![g](cat/03_gray_0x7776B0B492A3C25E.png) |
| 5 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/04_mono_0x7776B0B492A3C25E.png) | ![g](cat/04_gray_0x7776B0B492A3C25E.png) |
| 6 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/05_mono_0x7776B0B492A3C25E.png) | ![g](cat/05_gray_0x7776B0B492A3C25E.png) |
| 7 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/06_mono_0x7776B0B492A3C25E.png) | ![g](cat/06_gray_0x7776B0B492A3C25E.png) |
| 8 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/07_mono_0x7776B0B492A3C25E.png) | ![g](cat/07_gray_0x7776B0B492A3C25E.png) |
| 9 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/08_mono_0x7776B0B492A3C25E.png) | ![g](cat/08_gray_0x7776B0B492A3C25E.png) |
| 10 | 0.0172 | cat | `0x7776B0B492A3C25E` | ![m](cat/09_mono_0x7776B0B492A3C25E.png) | ![g](cat/09_gray_0x7776B0B492A3C25E.png) |
| 11 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/00_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/00_gray_0xB21E93F1CC4B8BC9.png) |
| 12 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/01_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/01_gray_0xB21E93F1CC4B8BC9.png) |
| 13 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/02_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/02_gray_0xB21E93F1CC4B8BC9.png) |
| 14 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/03_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/03_gray_0xB21E93F1CC4B8BC9.png) |
| 15 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/04_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/04_gray_0xB21E93F1CC4B8BC9.png) |
| 16 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/05_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/05_gray_0xB21E93F1CC4B8BC9.png) |
| 17 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/06_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/06_gray_0xB21E93F1CC4B8BC9.png) |
| 18 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/07_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/07_gray_0xB21E93F1CC4B8BC9.png) |
| 19 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/08_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/08_gray_0xB21E93F1CC4B8BC9.png) |
| 20 | 0.0171 | maze | `0xB21E93F1CC4B8BC9` | ![m](maze/09_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/09_gray_0xB21E93F1CC4B8BC9.png) |
| 21 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/00_mono_0x1297C68AA981FB60.png) | ![g](mask/00_gray_0x1297C68AA981FB60.png) |
| 22 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/01_mono_0x1297C68AA981FB60.png) | ![g](mask/01_gray_0x1297C68AA981FB60.png) |
| 23 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/02_mono_0x1297C68AA981FB60.png) | ![g](mask/02_gray_0x1297C68AA981FB60.png) |
| 24 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/03_mono_0x1297C68AA981FB60.png) | ![g](mask/03_gray_0x1297C68AA981FB60.png) |
| 25 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/04_mono_0x1297C68AA981FB60.png) | ![g](mask/04_gray_0x1297C68AA981FB60.png) |
| 26 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/05_mono_0x1297C68AA981FB60.png) | ![g](mask/05_gray_0x1297C68AA981FB60.png) |
| 27 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/06_mono_0x1297C68AA981FB60.png) | ![g](mask/06_gray_0x1297C68AA981FB60.png) |
| 28 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/07_mono_0x1297C68AA981FB60.png) | ![g](mask/07_gray_0x1297C68AA981FB60.png) |
| 29 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/08_mono_0x1297C68AA981FB60.png) | ![g](mask/08_gray_0x1297C68AA981FB60.png) |
| 30 | 0.0077 | mask | `0x1297C68AA981FB60` | ![m](mask/09_mono_0x1297C68AA981FB60.png) | ![g](mask/09_gray_0x1297C68AA981FB60.png) |

## Cat

Best score: 0.0172

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/00_mono_0x7776B0B492A3C25E.png) | ![g](cat/00_gray_0x7776B0B492A3C25E.png) |
| 2 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/01_mono_0x7776B0B492A3C25E.png) | ![g](cat/01_gray_0x7776B0B492A3C25E.png) |
| 3 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/02_mono_0x7776B0B492A3C25E.png) | ![g](cat/02_gray_0x7776B0B492A3C25E.png) |
| 4 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/03_mono_0x7776B0B492A3C25E.png) | ![g](cat/03_gray_0x7776B0B492A3C25E.png) |
| 5 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/04_mono_0x7776B0B492A3C25E.png) | ![g](cat/04_gray_0x7776B0B492A3C25E.png) |
| 6 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/05_mono_0x7776B0B492A3C25E.png) | ![g](cat/05_gray_0x7776B0B492A3C25E.png) |
| 7 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/06_mono_0x7776B0B492A3C25E.png) | ![g](cat/06_gray_0x7776B0B492A3C25E.png) |
| 8 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/07_mono_0x7776B0B492A3C25E.png) | ![g](cat/07_gray_0x7776B0B492A3C25E.png) |
| 9 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/08_mono_0x7776B0B492A3C25E.png) | ![g](cat/08_gray_0x7776B0B492A3C25E.png) |
| 10 | 0.0172 | `0x7776B0B492A3C25E` | ![m](cat/09_mono_0x7776B0B492A3C25E.png) | ![g](cat/09_gray_0x7776B0B492A3C25E.png) |

## Maze

Best score: 0.0171

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/00_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/00_gray_0xB21E93F1CC4B8BC9.png) |
| 2 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/01_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/01_gray_0xB21E93F1CC4B8BC9.png) |
| 3 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/02_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/02_gray_0xB21E93F1CC4B8BC9.png) |
| 4 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/03_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/03_gray_0xB21E93F1CC4B8BC9.png) |
| 5 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/04_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/04_gray_0xB21E93F1CC4B8BC9.png) |
| 6 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/05_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/05_gray_0xB21E93F1CC4B8BC9.png) |
| 7 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/06_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/06_gray_0xB21E93F1CC4B8BC9.png) |
| 8 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/07_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/07_gray_0xB21E93F1CC4B8BC9.png) |
| 9 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/08_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/08_gray_0xB21E93F1CC4B8BC9.png) |
| 10 | 0.0171 | `0xB21E93F1CC4B8BC9` | ![m](maze/09_mono_0xB21E93F1CC4B8BC9.png) | ![g](maze/09_gray_0xB21E93F1CC4B8BC9.png) |

## Mask

Best score: 0.0077

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/00_mono_0x1297C68AA981FB60.png) | ![g](mask/00_gray_0x1297C68AA981FB60.png) |
| 2 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/01_mono_0x1297C68AA981FB60.png) | ![g](mask/01_gray_0x1297C68AA981FB60.png) |
| 3 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/02_mono_0x1297C68AA981FB60.png) | ![g](mask/02_gray_0x1297C68AA981FB60.png) |
| 4 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/03_mono_0x1297C68AA981FB60.png) | ![g](mask/03_gray_0x1297C68AA981FB60.png) |
| 5 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/04_mono_0x1297C68AA981FB60.png) | ![g](mask/04_gray_0x1297C68AA981FB60.png) |
| 6 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/05_mono_0x1297C68AA981FB60.png) | ![g](mask/05_gray_0x1297C68AA981FB60.png) |
| 7 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/06_mono_0x1297C68AA981FB60.png) | ![g](mask/06_gray_0x1297C68AA981FB60.png) |
| 8 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/07_mono_0x1297C68AA981FB60.png) | ![g](mask/07_gray_0x1297C68AA981FB60.png) |
| 9 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/08_mono_0x1297C68AA981FB60.png) | ![g](mask/08_gray_0x1297C68AA981FB60.png) |
| 10 | 0.0077 | `0x1297C68AA981FB60` | ![m](mask/09_mono_0x1297C68AA981FB60.png) | ![g](mask/09_gray_0x1297C68AA981FB60.png) |

## Butterfly

Best score: 0.0032

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/00_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/00_gray_0x2E327A5B1E28ED96.png) |
| 2 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/01_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/01_gray_0x2E327A5B1E28ED96.png) |
| 3 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/02_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/02_gray_0x2E327A5B1E28ED96.png) |
| 4 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/03_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/03_gray_0x2E327A5B1E28ED96.png) |
| 5 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/04_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/04_gray_0x2E327A5B1E28ED96.png) |
| 6 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/05_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/05_gray_0x2E327A5B1E28ED96.png) |
| 7 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/06_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/06_gray_0x2E327A5B1E28ED96.png) |
| 8 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/07_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/07_gray_0x2E327A5B1E28ED96.png) |
| 9 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/08_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/08_gray_0x2E327A5B1E28ED96.png) |
| 10 | 0.0032 | `0x2E327A5B1E28ED96` | ![m](butterfly/09_mono_0x2E327A5B1E28ED96.png) | ![g](butterfly/09_gray_0x2E327A5B1E28ED96.png) |

## Spider_Web

Best score: 0.0025

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/00_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/00_gray_0x6DF2B5E4D04A8B8B.png) |
| 2 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/01_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/01_gray_0x6DF2B5E4D04A8B8B.png) |
| 3 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/02_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/02_gray_0x6DF2B5E4D04A8B8B.png) |
| 4 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/03_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/03_gray_0x6DF2B5E4D04A8B8B.png) |
| 5 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/04_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/04_gray_0x6DF2B5E4D04A8B8B.png) |
| 6 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/05_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/05_gray_0x6DF2B5E4D04A8B8B.png) |
| 7 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/06_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/06_gray_0x6DF2B5E4D04A8B8B.png) |
| 8 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/07_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/07_gray_0x6DF2B5E4D04A8B8B.png) |
| 9 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/08_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/08_gray_0x6DF2B5E4D04A8B8B.png) |
| 10 | 0.0025 | `0x6DF2B5E4D04A8B8B` | ![m](spider_web/09_mono_0x6DF2B5E4D04A8B8B.png) | ![g](spider_web/09_gray_0x6DF2B5E4D04A8B8B.png) |

## Starfish

Best score: 0.0010

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/00_mono_0x5CC19777153A41FD.png) | ![g](starfish/00_gray_0x5CC19777153A41FD.png) |
| 2 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/01_mono_0x5CC19777153A41FD.png) | ![g](starfish/01_gray_0x5CC19777153A41FD.png) |
| 3 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/02_mono_0x5CC19777153A41FD.png) | ![g](starfish/02_gray_0x5CC19777153A41FD.png) |
| 4 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/03_mono_0x5CC19777153A41FD.png) | ![g](starfish/03_gray_0x5CC19777153A41FD.png) |
| 5 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/04_mono_0x5CC19777153A41FD.png) | ![g](starfish/04_gray_0x5CC19777153A41FD.png) |
| 6 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/05_mono_0x5CC19777153A41FD.png) | ![g](starfish/05_gray_0x5CC19777153A41FD.png) |
| 7 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/06_mono_0x5CC19777153A41FD.png) | ![g](starfish/06_gray_0x5CC19777153A41FD.png) |
| 8 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/07_mono_0x5CC19777153A41FD.png) | ![g](starfish/07_gray_0x5CC19777153A41FD.png) |
| 9 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/08_mono_0x5CC19777153A41FD.png) | ![g](starfish/08_gray_0x5CC19777153A41FD.png) |
| 10 | 0.0010 | `0x5CC19777153A41FD` | ![m](starfish/09_mono_0x5CC19777153A41FD.png) | ![g](starfish/09_gray_0x5CC19777153A41FD.png) |

## Jellyfish

Best score: 0.0003

| Rank | Score | Seed | Mono | Grayscale |
|------|-------|------|------|-----------|
| 1 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/00_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/00_gray_0xC6CF1C693BB89CDE.png) |
| 2 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/01_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/01_gray_0xC6CF1C693BB89CDE.png) |
| 3 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/02_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/02_gray_0xC6CF1C693BB89CDE.png) |
| 4 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/03_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/03_gray_0xC6CF1C693BB89CDE.png) |
| 5 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/04_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/04_gray_0xC6CF1C693BB89CDE.png) |
| 6 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/05_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/05_gray_0xC6CF1C693BB89CDE.png) |
| 7 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/06_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/06_gray_0xC6CF1C693BB89CDE.png) |
| 8 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/07_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/07_gray_0xC6CF1C693BB89CDE.png) |
| 9 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/08_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/08_gray_0xC6CF1C693BB89CDE.png) |
| 10 | 0.0003 | `0xC6CF1C693BB89CDE` | ![m](jellyfish/09_mono_0xC6CF1C693BB89CDE.png) | ![g](jellyfish/09_gray_0xC6CF1C693BB89CDE.png) |

## Original Seeds (pre-search)

| Name | Seed | CNN Top-1 | Mono | Grayscale |
|------|------|-----------|------|-----------|
| chainmail | `0x5CF45186D99C20C8` | chain mail (7%) | ![m](prng_chainmail_mono.png) | ![g](prng_chainmail_gray.png) |
| deadbeef | `0xDEADBEEFCAFEBABE` | — | ![m](prng_deadbeef_mono.png) | ![g](prng_deadbeef_gray.png) |
| random1 | `0x1234567890ABCDEF` | — | ![m](prng_random1_mono.png) | ![g](prng_random1_gray.png) |

## Technical Notes

### Why scores are low (1-2%)
MobileNetV2 is trained on photos, not 1-bit noise patterns.
Even with symmetry, the patterns are abstract textures, not recognizable objects.
Better approaches (TODO):
1. **Dithered target**: convert photo → Floyd-Steinberg dither → find pRNG seed matching dithered image
2. **Feature matching**: compare VGG intermediate features (perceptual loss) instead of class probability
3. **Multi-scale symmetry**: mirror on 2+ axes, radial symmetry for more structure
4. **Guided generation**: use gradient of CNN loss to inform seed mutation direction

### Hardware
- GPU0 (RTX 4060 Ti 16GB): multi-target search
- GPU1 (RTX 4060 Ti 16GB): dedicated cat search (2000 pop × 500 gen)
- Total overnight: ~12 hours, ~10B seed evaluations

### Inspired by
- **Introspec** — BB (Big Brother) 256-byte ZX Spectrum intro
- **.ded^RMDA (Maxim Muchkaev)** — Hole #17 enigma, CALL-chain rendering
- **Mona** (Atari 256b) — the original 'draw with noise' concept

### Tools
- `cuda/prng_cat_search.py` — CNN-guided search (PyTorch + MobileNetV2)
- `cuda/z80_image_search.cu` — Pure CUDA brute-force
- `cuda/z80_prng_search.cu` — Generic pRNG seed search
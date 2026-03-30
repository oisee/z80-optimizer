# Foveal Gallery — Block-Scan LFSR

4 iconic faces × 2 modes (face-aware 36 seeds vs quadtree 597 seeds).
Block-scan LFSR: 1 bit = 1 block (deterministic coverage).

## Warhol Pop-Art Grid

| Face | Face-aware (36 seeds, 72B) | Quadtree (597 seeds, 1194B) |
|------|---|---|
| **Che Guevara** | ![face](che_face/warhol.png) 36.9% | ![qt](che_quadtree/warhol.png) 19.3% |
| **Marilyn Monroe** | ![face](marilyn_t55_face/warhol.png) 36.2% | ![qt](marilyn_t55_quadtree/warhol.png) 19.2% |
| **Mona Lisa** | ![face](monalisa_face/warhol.png) 37.8% | ![qt](monalisa_quadtree/warhol.png) 19.0% |
| **Einstein** | ![face](einstein_photo_bin_face/warhol.png) 34.4% | ![qt](einstein_photo_bin_quadtree/warhol.png) 19.3% |

## Progressive Layers (Quadtree)

### Che Guevara

| L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- |
| ![L0](che_quadtree/level0_compare.png) | ![L1](che_quadtree/level1_compare.png) | ![L2](che_quadtree/level2_compare.png) | ![L3](che_quadtree/level3_compare.png) | ![L4](che_quadtree/level4_compare.png) | ![L5](che_quadtree/level5_compare.png) |

### Marilyn Monroe

| L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- |
| ![L0](marilyn_t55_quadtree/level0_compare.png) | ![L1](marilyn_t55_quadtree/level1_compare.png) | ![L2](marilyn_t55_quadtree/level2_compare.png) | ![L3](marilyn_t55_quadtree/level3_compare.png) | ![L4](marilyn_t55_quadtree/level4_compare.png) | ![L5](marilyn_t55_quadtree/level5_compare.png) |

### Mona Lisa

| L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- |
| ![L0](monalisa_quadtree/level0_compare.png) | ![L1](monalisa_quadtree/level1_compare.png) | ![L2](monalisa_quadtree/level2_compare.png) | ![L3](monalisa_quadtree/level3_compare.png) | ![L4](monalisa_quadtree/level4_compare.png) | ![L5](monalisa_quadtree/level5_compare.png) |

### Einstein

| L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- |
| ![L0](einstein_photo_bin_quadtree/level0_compare.png) | ![L1](einstein_photo_bin_quadtree/level1_compare.png) | ![L2](einstein_photo_bin_quadtree/level2_compare.png) | ![L3](einstein_photo_bin_quadtree/level3_compare.png) | ![L4](einstein_photo_bin_quadtree/level4_compare.png) | ![L5](einstein_photo_bin_quadtree/level5_compare.png) |

## Progressive Layers (Face-Aware)

### Che Guevara

| L0 | L1 | L2 | L3 |
| --- | --- | --- | --- |
| ![L0](che_face/level0_compare.png) | ![L1](che_face/level1_compare.png) | ![L2](che_face/level2_compare.png) | ![L3](che_face/level3_compare.png) |

### Marilyn Monroe

| L0 | L1 | L2 | L3 |
| --- | --- | --- | --- |
| ![L0](marilyn_t55_face/level0_compare.png) | ![L1](marilyn_t55_face/level1_compare.png) | ![L2](marilyn_t55_face/level2_compare.png) | ![L3](marilyn_t55_face/level3_compare.png) |

### Mona Lisa

| L0 | L1 | L2 | L3 |
| --- | --- | --- | --- |
| ![L0](monalisa_face/level0_compare.png) | ![L1](monalisa_face/level1_compare.png) | ![L2](monalisa_face/level2_compare.png) | ![L3](monalisa_face/level3_compare.png) |

### Einstein

| L0 | L1 | L2 | L3 |
| --- | --- | --- | --- |
| ![L0](einstein_photo_bin_face/level0_compare.png) | ![L1](einstein_photo_bin_face/level1_compare.png) | ![L2](einstein_photo_bin_face/level2_compare.png) | ![L3](einstein_photo_bin_face/level3_compare.png) |

## Results Summary

| Face | Face-aware (72B) | Quadtree (1194B) | Ratio |
|------|-----------------|-----------------|-------|
| Che Guevara | 36.9% | 19.3% | 17× more data for 48% less error |
| Marilyn Monroe | 36.2% | 19.2% | 17× more data for 47% less error |
| Mona Lisa | 37.8% | 19.0% | 17× more data for 50% less error |
| Einstein | 34.4% | 19.3% | 17× more data for 44% less error |

## Key Insight

Block-scan LFSR (1 bit = 1 block) gives deterministic coverage.
Face-aware concentrates 80% of fine detail on eyes/mouth/nose.
Quadtree wins on raw error (full pixel coverage) but face-aware
produces more recognizable faces at 1/16th the data budget.

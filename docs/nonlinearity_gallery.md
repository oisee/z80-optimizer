# Buffer Nonlinearity Gallery

**Question:** Can direct buffer (LFSR-16 → 1 bit/block) match point-spray quality?
**Target:** Che Guevara face, 128×96 binary. **Best reference:** point-spray face4x = 26.5% @ 213 seeds.

Full analysis: [experiment_buffer_nonlinearity.md](experiment_buffer_nonlinearity.md)

---

## Side-by-side (512×384 each, 4× nearest-neighbor upscale)

| Target | Linear direct buf | 1D point-spray |
|:------:|:-----------------:|:--------------:|
| ![target](result_target.png) | ![linear](result_linear_buf.png) | ![1dspray](result_1dspray_buf.png) |
| **original** | **L_bin=42.2%** | **L_bin=40.2%** |
| — | 21 eff. seeds, oscillates | 26 eff. seeds, P≈0.39 |

| AND-2 nonlinear | Majority-3 | **2D point-spray face4x** |
|:---------------:|:----------:|:-------------------------:|
| ![and2](result_and2_buf.png) | ![majority](result_majority_buf.png) | ![spray](result_spray_face4x.png) |
| **L_bin=36.3%** | **L_bin=39.1%** | **L_bin≈26.5%** |
| 58 eff. seeds, P=0.25 | 15 eff. seeds, P=0.5 | 213 seeds, monotone ↓ |

---

## Nonlinearity Ladder

| Method | P(flip) | Degree | Eff.seeds | L_bin@213 | Converges? |
|--------|:-------:|:------:|:---------:|:---------:|:----------:|
| Linear (1 bit) | 0.50 | 1 | 21 | 42.2% | ✗ oscillates |
| Majority-3 | **0.50** | 3 | 15 | 39.1% | ✗ plateau |
| 1D spray (N=576, LFSR-16) | ≈0.39 | 576† | 26 | 40.2% | ✗ plateau |
| AND-2 | **0.25** | 2 | 58 | 36.3% | ✗ plateau |
| AND-3 | **0.125** | 3 | TBD | TBD | ? |
| **2D spray (LFSR-32)** | ≈0.39 | N+‡ | **213+** | **26.5%** | **✓ monotone** |

† correlated: 576 consecutive LFSR-16 steps, limited diversity
‡ near-independent xy from different bit-slices of 32-bit state

---

## Key Finding: Sparsity > Degree

**Unexpected:** Majority-3 (P=0.5, cubic) has *fewer* effective seeds than AND-2 (P=0.25, quadratic).

With P=0.5 patterns: XOR of two seeds ≈ cancels → effective seed count stays low.
With P=0.25 (AND-2): patterns are sparser → less mutual cancellation → more seeds contribute.

**What makes 2D point-spray work:**
1. Near-independent (x,y) from different 32-bit LFSR slices (`state>>0` vs `state>>16`)
2. Overcoverage collisions create high-degree polynomial patterns
3. Each seed explores different region of pattern space → no plateau

---

## Combined comparison (all methods)

![all](buf_comparison.png)

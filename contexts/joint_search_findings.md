# Joint-2 Search: Proof of Concept Results

## Finding: Joint optimization beats greedy by 4%+

Greedy (current): find best seed_A, lock, find best seed_B given A.
  ROI error = 184 pixels

Joint-2 (sampled 2048²): find best (seed_A, seed_B) pair simultaneously.
  ROI error = 176 pixels (−4.3%, and still searching)

On LEFT EYE region only (32×16 @ 2×2 + 24×10 @ 1×1 overlapping).
Full 65536² search would likely find even better.

## Why it works

Greedy lock seed_A before seeing seed_B. If seed_A creates a pattern
that's hard for seed_B to correct (e.g. noise in the overlap zone),
the joint result is suboptimal.

Joint-2 tests ALL combinations — finds (A,B) where A creates a pattern
that B can EASILY correct, and vice versa.

## Architecture for CUDA kernel

```
__global__ void joint2_kernel(
    uint8_t* canvas,    // current canvas (all other seeds applied)
    uint8_t* target,    // target image
    // Region A:
    int rxA, ryA, rwA, rhA, blkA,
    // Region B:  
    int rxB, ryB, rwB, rhB, blkB,
    // Output:
    uint32_t* errors    // 65536 × 65536 → need reduction
) {
    // Thread (i, j): test seedA=i, seedB=j
    int seedA = blockIdx.x;   // 0..65535
    int seedB = threadIdx.x + blockIdx.y * blockDim.x;  // 0..65535
    
    // Copy canvas to local
    uint8_t local[PACKED_SIZE];
    memcpy(local, canvas, PACKED_SIZE);
    
    // Apply seedA to region A
    draw_segment(local, seedA, 0, rxA, ryA, rwA, rhA, blkA, ...);
    // Apply seedB to region B  
    draw_segment(local, seedB, 1, rxB, ryB, rwB, rhB, blkB, ...);
    
    // Count error in ROI (union of A and B)
    uint32_t err = count_roi_error(local, target, roi_x, roi_y, roi_w, roi_h);
    
    // Atomic min to find global best
    atomicMin(&global_best_err, err);
    // Store per-thread error for finding best (seedA, seedB) pair
}
```

Launch: 65536 blocks × 256 threads = 16M threads
Each thread tests one (seedA, seedB) pair.
256 passes to cover all 65536 seedB values.

Time estimate: 65536 × 256 passes × ~100 instructions / (4000 cores × 2.5GHz)
  = ~43 seconds per pair (matches earlier estimate)

## Optimization schedule

1. Phase 1: Greedy pass (0.5 sec) — current
2. Phase 2: Identify ~20 critical overlapping pairs
3. Phase 3: Joint-2 each pair on GPU (20 × 43s = 14 min)
4. Phase 4: Re-greedy remaining seeds (0.5 sec)

Expected improvement: 4-8% error reduction, SAME 213 seeds / 426 bytes.

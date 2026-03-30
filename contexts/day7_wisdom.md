# Day 7 Wisdom — March 30, 2026

## Key Discoveries

### Foveal Image Search
- Face-aware segmentation: 24-213 seeds, attention-weighted on eyes/nose/mouth
- Scales linearly: ~5% improvement per 2× seeds
- Mondrian random placement beats golden ratio at low seed budgets
- Block-scan LFSR: 1 bit = 1 block (deterministic vs random point-spray)

### XOR Morphing
- Cumulative canvas: each target bruteforced ON TOP of previous
- Reverse-pyramid V-shape: 1→2→4→8→4→2→1 (dissolve then reconverge)
- Error profile: 18% → 46% (peak dissolve) → 18% (new face locks)
- Per-target pop-art palette with polarity detection

### Target Polarity
- White-face targets (Che, Lenin, Einstein): invert before bruteforce
- All targets should have black=face, white=bg for consistent palette
- Smart colorize: black→dark ink, white→light paper

### Animated Portrait (1K intro concept)
- Static background: 426B (213 seeds, face4x)
- Animated center: 5 frames × ~16B = 80B (center region seeds only)
- Total: ~706B code+data — fits 1K intro
- XOR cycling: apply/undo center seeds for animation frames

### Bad Apple
- 33 keyframes bruteforced in 16 seconds, 5.5KB total
- 1194 bytes/frame (full quadtree) — too much for video
- Delta approach: ~20-40 bytes/frame between similar frames
- Full video estimate: 44KB at 10fps — fits 128K Spectrum
- Agon project has full pipeline: 6572 frames, tile analysis, diff encoding

### u32 Multi-Convention
- HLIX > DEHL for all arithmetic (SHL=30T vs 34T, ADD=30T vs 54T)
- HLH'L' wins 14/19 ×K benchmarks (EXX=4T save)
- DEHL never wins — our current convention is suboptimal
- SHA-256 could use 4×32 vars in registers (no RAM spills)

### Real Photo Targets
- Marilyn Monroe (1953 Wikipedia), Karl Marx, Uncle Sam (I Want You)
- Lenin trafaret from /mnt/safe (oktjabrjatskij znachok style)
- Einstein with tongue (Arthur Sasse 1951)
- Frida Kahlo, Charlie Chaplin (bowler hat)

## Files Created
- media/prng_images/morph_v5/ — correct polarity morphing
- media/prng_images/foveal_gallery/ — full face-aware gallery
- media/prng_images/badapple/ — Bad Apple proof-of-concept
- data/u32_conventions.json — 3 conventions comparison
- contexts/philipp_reply_v2.md — draft reply to SDCC maintainer

## Numbers
- Morphing: 5 faces, 37 frames, ~18% each, pop-art palette
- Bad Apple: 33 keyframes, 5.5KB, 12.7% error
- Face-aware 4×: 213 seeds (426B), 26.5% Che
- div8 v3: avg 79T, carry_compare 26T for K≥128

## Joint-2 Search (Day 7 continued)

### Key Discovery: Joint seed optimization beats greedy by 15-68%
- Greedy: find best seed_A, lock, find best seed_B → suboptimal pair
- Joint-2: test ALL 65536² = 4.3B (seed_A, seed_B) pairs → globally optimal
- Left eye: 500→156 err (−68%), Right eye: 495→155 err (−67%), Nose: 598→210 err (−65%)
- Same seeds, same bytes — just BETTER combinations
- CUDA kernel: grid(65536,256) × block(256) = 4.3B threads, ~132s per pair

### Why it works
- LFSR generates ~50% random pattern per seed
- Greedy locks seed_A without seeing seed_B → may create hard-to-correct pattern
- Joint-2 finds (A,B) where XOR(A) XOR(B) is closest to target delta
- XOR(A,B) creates patterns neither A nor B produces alone

### Independent vs overlapping pairs
- Independent (left_eye ≠ right_eye): can optimize in parallel, lock separately
- Overlapping (eye_2x2 + eye_1x1): must optimize jointly (this is what joint-2 does)
- 99% of same-level segments are independent (ухо ≠ глаз)

### Convergence guarantee
- Each LFSR seed has slight bias (not exactly 50%)
- Among 65536 seeds, one correlates with needed correction
- Each layer guaranteed to improve (GPU finds best-of-65536)
- Multi-scale (8→4→2→1) refines progressively
- Joint-2 finds even better combinations greedy misses

### Files
- cuda/joint2_search.cu — CUDA joint-2 kernel (working, tested)
- contexts/joint_search_findings.md — architecture notes

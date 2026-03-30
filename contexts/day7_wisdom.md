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

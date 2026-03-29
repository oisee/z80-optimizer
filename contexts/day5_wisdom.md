# Day 5 Wisdom — March 29, 2026

## Key Discoveries

### Dual-Layer Image Generator (BREAKTHROUGH)
- 5-layer architecture: 3 additive OR + 2 subtractive AND NOT
- Layer A (H-mirror) + B (no-sym) + C (detail) additive, D (H-mirror) + E (no-sym) subtractive
- Subtractive layers carve holes: eyes, mouth, nose from white mass
- Cat: 4.9% error from 128 bytes genome. Skull: 14.7%
- 557K img/s on RTX 4060 Ti (100× faster than Python)
- Key insight: faces need H-mirror (eyes symmetric) but NOT V-mirror (forehead ≠ chin)
- Island model + stall restart essential for escaping local optima

### Segmented Hierarchical LFSR (NEW APPROACH)
- Image split into progressively smaller rectangles
- Each segment brute-forced independently: 65536 LFSR seeds per segment
- Level 0: 1 seed whole image 8×8 → L1: 4 quadrants 4×4 → L2: 16 tiles 2×2 → L3: 64 tiles 1×1
- **Guaranteed convergence**: each level can only reduce error
- Che Guevara 15% error with 6 levels (597 seeds = 1194 bytes)
- Che Guevara 31% error with 4 levels (85 seeds = 170 bytes) — fits 256b intro!
- User's insight: brute-force segments not whole image → exponentially more effective

### Introspec BB Algorithm Analysis (SOURCE CODE!)
- Received bb_brute_search_2.zip from Introspec (April 2014)
- **24-bit Galois LFSR** (NOT 32-bit!): FG24a(8) + FG24b(16), polynomial 0xDB
- **2×2 pixel XOR plots** on Spectrum screen (256×192)
- 66 layers, layer N draws N×2 random points (decreasing count)
- **3 weighted masks** for fitness: mask0 (all), mask1 (medium), mask2 (face=4× weight)
- High byte of LFSR **carries between layers** (not reset)
- diff = error0 + error1 + error2 + error2 (face weighted 4×)
- CUDA port: 0.9s per s0 (66 layers × 65536 seeds) vs DAYS on CPU
- Full 256 s0 sweep in ~4 minutes on 2 GPUs
- Results still noisy — single greedy pass insufficient; Introspec used multi-pass + manual tuning

### u32 Arithmetic Library (COMPLETE)
- ADC HL,rr EXISTS on Z80! (ED 4A/5A/6A/7A, 15T) — key for u32
- SBC HL,rr also exists (ED 42/52/62/72, 15T)
- SHL32: ADD HL,HL; EX DE,HL; ADC HL,HL; EX DE,HL = 34T, 4 ops — PROVEN OPTIMAL
- SHR32: SRL D; RR E; RR H; RR L = 32T, 4 ops — PROVEN OPTIMAL
- SAR32: SRA D; RR E; RR H; RR L = 32T, 4 ops — PROVEN OPTIMAL
- ADD32 from stack: 54T, 6 ops (POP BC; ADD HL,BC; POP BC; EX DE,HL; ADC HL,BC; EX DE,HL)
- NEG32: 57T, 12 ops — LD A,0 (NOT XOR A) preserves carry between bytes
- SEXT16_32: RLA; SBC A,A; LD D,A; LD E,A = 24T — sign via carry trick
- XOR32/AND32: 100T byte-by-byte (no native 16-bit logic ops)
- ROTR32: 32-40T branch variant (for SHA-256)
- SHA-256 feasibility: ~800T/round × 64 = 51K T-states = 15ms @3.5MHz

### divmod8 Strategy
- GPU brute-force too slow (21^6 × 256 initB per K)
- Better: analytical multiply-and-shift: A÷K = A × M >> S
- div3 = ×171>>9 already proven EXACT
- Use existing mul8 table (254/254) + shift chain
- Format: div8 = mul8[magic_M] + SRL×S

## Files Created/Modified

### New
- `cuda/prng_hybrid_gpu.cu` — CUDA dual-layer evolutionary generator (5 layers, islands)
- `cuda/prng_layered_search.cu` — Layered LFSR search (Mona/BB style, 64-128 layers)
- `cuda/prng_segmented_search.cu` — Hierarchical segmented LFSR (multi-resolution)
- `cuda/bb_search.cu` — CUDA port of Introspec's BB brute-force algorithm
- `cuda/che_intro.asm` — ZX Spectrum intro attempt (351 bytes, needs LFSR fix)
- `data/u32_ops.json` — Complete u32 arithmetic library (13 operations)
- `data/div8_all.jsonl` — Partial div8 results (2/254)
- `media/prng_images/targets/che.pgm` — Che Guevara binary target (128×96)
- `media/prng_images/targets/einstein_real.pgm` — Einstein real photo target
- `media/prng_images/dual_cat_long/` — Best cat result (f=0.049)
- `media/prng_images/dual_skull_long/` — Best skull result (f=0.147)
- `media/prng_images/segmented_che_v2/` — Best Che (15%, 6 levels)
- `media/prng_images/bb_putin/` — Introspec's original target + CUDA results
- `media/prng_images/bb_che_p4/` — Che via BB algorithm
- `media/prng_images/dual_einstein_v4/` — Einstein with subtractive layers
- `media/prng_images/README.md` — Complete gallery (Hall of Fame, 29 experiments)
- `contexts/day5_wisdom.md` — This file

### Modified
- `media/prng_images/README.md` — Full rewrite with all methods compared

### Commits: 45816cc → f46d6d2 (day 5, ~15 commits)

## What We Told Colleagues
- MinZ (ju6yy047): u32_ops.json (13 ops, proven), ADC HL,rr confirmation, SHL32/SHR32 optimal, divmod8 strategy (analytical mul-and-shift), SHA-256 feasibility (15ms/block)
- MinZ acknowledged: mul16 integrated (7.7× speedup), u32 loader ready, scalar overloading done

## Overnight / Background
- div8 brute-force running (slow, 2/254 found)
- BB search completed for Putin (s0=0..255) and Che (p=2 and p=4)

## Key Decisions
- V-mirror removed from image generator (faces NOT vertically symmetric)
- Segmented approach > whole-image LFSR (user insight: brute-force smaller regions)
- divmod8: analytical multiply-and-shift > GPU brute-force (21^6 too large)
- u32 on Z80: DEHL convention, EXX for spill (not working data)

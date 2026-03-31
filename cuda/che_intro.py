#!/usr/bin/env python3
"""
Che Guevara 256-byte ZX Spectrum intro — hand-assembled.

Generates 128x96 image from 85 LFSR seeds using hierarchical XOR.
Uses ZX Spectrum screen memory (0x4000-0x57FF).

Output: che_intro.bin (raw binary, load at 0x8000)
Run: mze --target spectrum che_intro.bin
"""

import struct
import sys
import os

# ============================================================
# Seed table from segmented search (85 seeds, levels 0-3)
# ============================================================
# Read seeds from the search output
seed_file = "media/prng_images/segmented_che/seeds.txt"
seeds = []
if os.path.exists(seed_file):
    with open(seed_file) as f:
        for line in f:
            if line.startswith("seg"):
                parts = line.split()
                seed_hex = parts[2]  # 0xABCD
                seeds.append(int(seed_hex, 16))
    print(f"Loaded {len(seeds)} seeds from {seed_file}")
else:
    print(f"WARNING: {seed_file} not found, using dummy seeds")
    seeds = [0x1234] * 85

# Segment specs matching our CUDA search
# Level 0: 1 seg, whole image, 8x8, pts=576 (192*3)
# Level 1: 4 segs, quadrants, 4x4, pts=288
# Level 2: 16 segs, tiles, 2x2, pts=144
# Level 3: 64 segs, tiles, 1x1, pts=36
segments = []

# Level 0
segments.append({"rx":0, "ry":0, "rw":128, "rh":96, "blk":8, "pts":576})
# Level 1
for qy in range(2):
    for qx in range(2):
        segments.append({"rx":qx*64, "ry":qy*48, "rw":64, "rh":48, "blk":4, "pts":288})
# Level 2
for ty in range(4):
    for tx in range(4):
        segments.append({"rx":tx*32, "ry":ty*24, "rw":32, "rh":24, "blk":2, "pts":144})
# Level 3
for ty in range(8):
    for tx in range(8):
        segments.append({"rx":tx*16, "ry":ty*12, "rw":16, "rh":12, "blk":1, "pts":36})

assert len(segments) == 85, f"Expected 85 segments, got {len(segments)}"

# ============================================================
# Z80 hand-assembler helpers
# ============================================================
code = bytearray()
labels = {}
fixups = []  # (offset, label_name, type) for forward references
ORG = 0x8000

def emit(*args):
    for b in args:
        code.append(b & 0xFF)

def emit16(val):
    emit(val & 0xFF, (val >> 8) & 0xFF)

def here():
    return ORG + len(code)

def label(name):
    labels[name] = here()

def jr_fixup(name):
    """Emit JR placeholder, fix up later"""
    fixups.append((len(code), name, "jr"))
    emit(0x00)  # placeholder

def jp_fixup(name):
    """Emit JP placeholder"""
    fixups.append((len(code), name, "jp"))
    emit16(0x0000)

def resolve():
    for offset, name, typ in fixups:
        if name not in labels:
            raise ValueError(f"Unresolved label: {name}")
        target = labels[name]
        if typ == "jr":
            rel = target - (ORG + offset + 1)  # JR is relative to NEXT instruction
            if rel < -128 or rel > 127:
                raise ValueError(f"JR out of range: {name} at {offset}, rel={rel}")
            code[offset] = rel & 0xFF
        elif typ == "jp":
            code[offset] = target & 0xFF
            code[offset + 1] = (target >> 8) & 0xFF

# ============================================================
# ZX Spectrum screen: 128x96 mapped to 256x192
# We plot at 2x scale: pixel (x,y) → screen pixels (2x, 2y)..(2x+1, 2y+1)
# Screen address for row Y: complex interleaving
# For Y in 0..191:
#   hi = 0x40 | ((Y & 0xC0) >> 3) | ((Y & 7))  — wait, standard formula:
#   addr = 0x4000 + ((Y & 0x38) << 2) + ((Y & 0xC0) << 5) + ((Y & 7) << 8) + (X >> 3)
#   bit = 7 - (X & 7)
#
# But 128x96 at 2x = 256x192, so screen_x = x*2, screen_y = y*2
# Each pixel becomes 2x2 block.
#
# Simplification: for the intro we use 1:1 mapping in the top-left 128x96 area.
# Screen X = 0..127 (left half), Screen Y = 0..95 (top 96 lines)
# ============================================================

# Actually, for simplicity let's map to 1:1 in the left half of screen.
# 128 pixels wide = 16 bytes per row. XOR at byte offset.

# ============================================================
# Generate Z80 code
# ============================================================

# Entry point: clear screen, then draw all segments
# LD HL, 0x4000; LD DE, 0x4001; LD BC, 0x17FF; LD (HL), 0; LDIR
emit(0x21); emit16(0x4000)  # LD HL, 0x4000
emit(0x11); emit16(0x4001)  # LD DE, 0x4001
emit(0x01); emit16(0x17FF)  # LD BC, 0x17FF
emit(0x36, 0x00)            # LD (HL), 0
emit(0xED, 0xB0)            # LDIR
# 13 bytes

# Also clear attributes to white on black
emit(0x21); emit16(0x5800)  # LD HL, 0x5800
emit(0x11); emit16(0x5801)  # LD DE, 0x5801
emit(0x01); emit16(0x02FF)  # LD BC, 0x02FF
emit(0x36, 0x38)            # LD (HL), 0x38 (white ink on black paper)
emit(0xED, 0xB0)            # LDIR
# 13 bytes, total = 26

# IX = pointer to segment table
emit(0xDD, 0x21); emit16(0x0000)  # LD IX, seed_table (fixup later)
seed_table_fixup_offset = len(code) - 2

# IY = segment count
emit(0xFD, 0x21); emit16(len(seeds))  # LD IY, 85

label("seg_loop")
# Load seed from (IX) into DEHL: seed << 16 | (seg_id * 13 + 0xBEEF)
# For simplicity: D=seed_hi, E=seed_lo, H=seg_id_derived_hi, L=seg_id_derived_lo
# Actually let's just use seed as DE, and compute HL from segment index

# Load seed into DE
emit(0xDD, 0x56, 0x01)  # LD D, (IX+1)
emit(0xDD, 0x5E, 0x00)  # LD E, (IX+0)

# HL = arbitrary init based on segment position (we'll use IX offset as proxy)
# For each segment we do N iterations of LFSR + XOR plot
# The CUDA code uses: state = (seed << 16) | (seg_id * 7 + 0x1337)
# On Z80: DE = seed, HL = seg_id * 7 + 0x1337

# We need a seg_id counter. Use B' (shadow B) or stack variable.
# Simpler: push a counter on stack.

# Load segment metadata: block_size and num_points are hardcoded per level.
# For 256b we can't afford per-segment metadata. Instead:
# Use implicit structure: first 1 seed = L0, next 4 = L1, next 16 = L2, next 64 = L3

# SIMPLIFICATION for 256b: all segments use same num_points and block_size=1
# This loses the multi-resolution but saves tons of code.
# Each seed XORs ~48 random pixels into its 16x12 tile.

# Actually let's do the simplest possible thing that works:
# For each of 85 seeds:
#   Init LFSR from seed
#   Plot N points, each: LFSR step, extract (x,y), XOR pixel
#   Where (x,y) is constrained to... hmm, we need segment boundaries.

# For 256b this is too complex. Let's use the BB approach instead:
# ALL seeds operate on the FULL screen. No segments.
# Just 85 iterations, each with its own seed, plotting ~64 XOR points.

# This is simpler and proven (BB uses 62 seeds on full screen).

# Reset code
code.clear()
labels.clear()
fixups.clear()

# ============ SIMPLIFIED BB-STYLE 256b INTRO ============
# 85 seeds, each draws ~48 XOR points on 128x96 area (left half of Spectrum screen)
# Screen layout: rows 0-95, columns 0-15 (128 pixels = 16 bytes)

NSEEDS = len(seeds)
POINTS_PER_SEED = 48

# Clear screen
emit(0x21); emit16(0x4000)  # LD HL, 0x4000
emit(0x11); emit16(0x4001)  # LD DE, 0x4001
emit(0x01); emit16(0x17FF)  # LD BC, 0x17FF
emit(0x36, 0x00)            # LD (HL), 0
emit(0xED, 0xB0)            # LDIR  = 13 bytes

# Set attributes: white on black
emit(0x21); emit16(0x5800)  # LD HL, 0x5800
emit(0x11); emit16(0x5801)  # LD DE, 0x5801
emit(0x01); emit16(0x02FF)  # LD BC, 0x02FF
emit(0x36, 0x38)            # LD (HL), 0x38
emit(0xED, 0xB0)            # LDIR  = 13 bytes, total = 26

# IX = seed table pointer
emit(0xDD, 0x21)            # LD IX, seed_table
seed_table_fixup = len(code)
emit16(0x0000)              # placeholder = 4 bytes, total = 30

# C = seed counter
emit(0x0E, NSEEDS)          # LD C, 85 = 2 bytes, total = 32

label("outer")
# Load seed into DE: D=(IX+1), E=(IX+0)
emit(0xDD, 0x56, 0x01)      # LD D, (IX+1) = 3 bytes
emit(0xDD, 0x5E, 0x00)      # LD E, (IX+0) = 3 bytes

# HL = 0x1337 (fixed init for low half of LFSR state)
emit(0x21); emit16(0x1337)  # LD HL, 0x1337 = 3 bytes, total = 41

# Warm up LFSR: 8 steps
emit(0x06, 0x08)            # LD B, 8
label("warmup")
# LFSR step inline: SRL D; RR E; RR H; RR L; JR NC, skip; XOR poly
emit(0xCB, 0x3A)            # SRL D
emit(0xCB, 0x1B)            # RR E
emit(0xCB, 0x1C)            # RR H
emit(0xCB, 0x1D)            # RR L
emit(0x30)                  # JR NC, +skip (skip XOR)
jr_fixup("warmup_skip")
emit(0x7A)                  # LD A, D
emit(0xEE, 0xB4)            # XOR 0xB4
emit(0x57)                  # LD D, A
emit(0x7B)                  # LD A, E
emit(0xEE, 0xBC)            # XOR 0xBC
emit(0x5F)                  # LD E, A
emit(0x7C)                  # LD A, H
emit(0xEE, 0xD3)            # XOR 0xD3
emit(0x67)                  # LD H, A
emit(0x7D)                  # LD A, L
emit(0xEE, 0x5C)            # XOR 0x5C
emit(0x6F)                  # LD L, A
label("warmup_skip")
emit(0x10)                  # DJNZ warmup
jr_fixup("warmup")

# B = points per seed
emit(0x06, POINTS_PER_SEED) # LD B, 48

label("inner")
# LFSR step (same code, need to call/inline)
# To save bytes, we'll use a CALL to shared LFSR routine
emit(0xCD)                  # CALL lfsr_step
jp_fixup("lfsr_step")

# Extract X (0..127) from bits 0-6 of L, Y (0..95) from bits 7-13
# X = L AND 0x7F
emit(0x7D)                  # LD A, L
emit(0xE6, 0x7F)            # AND 0x7F  → A = X (0..127)
# Save X
emit(0xC5)                  # PUSH BC (save counters)

# Compute screen address for pixel at (A=X, Y from H low bits)
# Y = H AND 0x5F → mod 96... actually let's use: Y = (H XOR E) AND 0x7F, mod 96
# Simpler: Y = ((DEHL >> 7) & 0x7F) % 96
# Or just: use different state bits
# A = X (0..127), we need Y separately

# Y = (H & 0x7F) if < 96, else wrap
emit(0xF5)                  # PUSH AF (save X)
emit(0x7C)                  # LD A, H
emit(0xE6, 0x7F)            # AND 0x7F → 0..127

# A mod 96: if A >= 96, subtract 96
emit(0xFE, 96)              # CP 96
emit(0x38, 0x02)            # JR C, +2 (skip if < 96)
emit(0xD6, 96)              # SUB 96

# Now A = Y (0..95). Compute Spectrum screen address.
# Standard formula: addr = 0x4000 + ((Y&7)<<8) + ((Y&0x38)<<2) + ((Y&0xC0)<<5) + (X>>3)
# But Y is only 0..95, so Y&0xC0 = 0 for Y<64, = 64 for Y>=64

# Simplified: store Y in reg, compute address
# Let's use a lookup or direct computation

# Y in A. Save it.
emit(0x57)                  # LD D, A  (D = Y)
emit(0xF1)                  # POP AF   (A = X)
emit(0x5F)                  # LD E, A  (E = X)

# Compute byte offset: X >> 3
emit(0xCB, 0x3F)            # SRL A
emit(0xCB, 0x3F)            # SRL A
emit(0xCB, 0x3F)            # SRL A    → A = X/8 (column byte)

# Screen address: H = 0x40 | (Y&7) | ((Y&0x38)>>3)<<5
# Actually: high byte = 0x40 + (Y>>6)*8 + (Y&7)
#           low byte  = ((Y&0x38)>>3)*32 + X/8
# Simpler formula for Y < 96:
#   hi = 0x40 + (Y & 7)  + ((Y & 0xC0) >> 3)
#   lo = ((Y >> 3) & 7) * 32 + (X >> 3)
# Wait, standard Spectrum screen:
#   addr = 0x4000 | ((Y & 0xC0) << 5) | ((Y & 0x38) << 2) | ((Y & 7) << 8) | (X >> 3)
# For Y < 64: (Y & 0xC0) = 0
#   addr = 0x4000 | ((Y & 0x38) << 2) | ((Y & 7) << 8) | (X >> 3)
# For Y 64..95: (Y & 0xC0) = 64, (64 << 5) = 0x800
#   addr = 0x4800 | ((Y & 0x38) << 2) | ((Y & 7) << 8) | (X >> 3)

# Let's compute in HL:
# A = X/8 already. Save to L.
emit(0x6F)                  # LD L, A  (L = X/8)

# D = Y. Compute high byte.
emit(0x7A)                  # LD A, D  (A = Y)
emit(0xE6, 0x07)            # AND 7    → Y & 7
emit(0xF6, 0x40)            # OR 0x40  → 0x40 | (Y&7)
emit(0x67)                  # LD H, A  → H = high byte (partial)

# Add (Y & 0x38) << 2 to L  → that's ((Y>>3)&7) * 32
emit(0x7A)                  # LD A, D  (A = Y again)
emit(0xE6, 0x38)            # AND 0x38
emit(0x07)                  # RLCA
emit(0x07)                  # RLCA     → (Y&0x38)<<2, but this might overflow into H
# Actually (Y & 0x38) << 2 = 0..0xE0, add to L which is 0..15 → might overflow
emit(0xB5)                  # OR L     → combine
emit(0x6F)                  # LD L, A

# Handle Y >= 64: add 0x08 to H
emit(0x7A)                  # LD A, D
emit(0xE6, 0x40)            # AND 0x40
emit(0x28, 0x02)            # JR Z, +2  (skip if Y < 64)
emit(0xCB, 0xFC)            # SET 3, H  → H |= 0x08 (third=0x800)

# Now HL = screen address. XOR the pixel.
# Bit position = 7 - (X & 7). X is in E.
emit(0x7B)                  # LD A, E  (A = X)
emit(0xE6, 0x07)            # AND 7
emit(0x47)                  # LD B, A  → B = X & 7

# Create bitmask: 0x80 >> B
emit(0x3E, 0x80)            # LD A, 0x80
emit(0xB0)                  # OR B (test if B=0... no, we need shift)
# Actually: shift A right by B positions
emit(0x78)                  # LD A, B
emit(0xB7)                  # OR A (set flags)
emit(0x3E, 0x80)            # LD A, 0x80
emit(0x28, 0x04)            # JR Z, +4 (skip shift if B=0)
label("shift_loop")
emit(0xCB, 0x3F)            # SRL A
emit(0x10)                  # DJNZ shift_loop
jr_fixup("shift_loop")

# XOR pixel
emit(0xAE)                  # XOR (HL)
emit(0x77)                  # LD (HL), A

# Restore DEHL (LFSR state) — we clobbered D,E,H,L for screen calc!
# Problem: we need to preserve LFSR state across screen plotting.
# Solution: save DEHL on stack before plotting.

# OOPS — we need to restructure. DEHL is our LFSR state AND we use DEHL for screen calc.
# Fix: save LFSR state, do screen calc, restore.

# This adds bytes. Let's restructure using shadow registers.
# EXX saves/restores BC,DE,HL. We can use EXX to swap.

print("WARNING: Code too complex for hand-assembly, need restructure")
print(f"Code so far: {len(code)} bytes")

# Let me restart with a cleaner design using EXX

code.clear()
labels.clear()
fixups.clear()

# ============ CLEAN VERSION: use EXX for LFSR state ============
# LFSR state in shadow DE'HL'
# Main regs for screen calculation
# B = point counter, C = seed counter
# IX = seed table pointer

# Clear screen
emit(0x21); emit16(0x4000)
emit(0x11); emit16(0x4001)
emit(0x01); emit16(0x17FF)
emit(0x36, 0x00)
emit(0xED, 0xB0)            # 13 bytes

# Set attributes
emit(0x21); emit16(0x5800)
emit(0x11); emit16(0x5801)
emit(0x01); emit16(0x02FF)
emit(0x36, 0x38)
emit(0xED, 0xB0)            # 13 bytes, total=26

# IX = seed table
emit(0xDD, 0x21)
seed_table_fixup = len(code)
emit16(0x0000)              # total=30

emit(0x0E, NSEEDS)          # LD C, 85; total=32

label("outer")
# Load seed: D'=(IX+1), E'=(IX+0), H'=0x13, L'=0x37
emit(0xD9)                  # EXX (switch to shadow)
emit(0xDD, 0x56, 0x01)      # LD D, (IX+1)
emit(0xDD, 0x5E, 0x00)      # LD E, (IX+0)
emit(0x21); emit16(0x1337)  # LD HL, 0x1337

# Warmup: 8 LFSR steps
emit(0x06, 0x08)            # LD B, 8
label("warmup")
emit(0xCB, 0x3A)            # SRL D
emit(0xCB, 0x1B)            # RR E
emit(0xCB, 0x1C)            # RR H
emit(0xCB, 0x1D)            # RR L
emit(0x30, 0x0E)            # JR NC, +14 (skip XOR)
emit(0x7A); emit(0xEE, 0xB4); emit(0x57)  # A=D; XOR B4; D=A
emit(0x7B); emit(0xEE, 0xBC); emit(0x5F)  # A=E; XOR BC; E=A
emit(0x7C); emit(0xEE, 0xD3); emit(0x67)  # A=H; XOR D3; H=A
emit(0x7D); emit(0xEE, 0x5C); emit(0x6F)  # A=L; XOR 5C; L=A
emit(0x10)                   # DJNZ warmup
jr_fixup("warmup")

emit(0xD9)                  # EXX (back to main)

# B = point counter
emit(0x06, POINTS_PER_SEED) # LD B, 48

label("inner")
# LFSR step in shadow regs
emit(0xD9)                  # EXX
emit(0xCB, 0x3A)            # SRL D
emit(0xCB, 0x1B)            # RR E
emit(0xCB, 0x1C)            # RR H
emit(0xCB, 0x1D)            # RR L
emit(0x30, 0x0E)            # JR NC, +14
emit(0x7A); emit(0xEE, 0xB4); emit(0x57)
emit(0x7B); emit(0xEE, 0xBC); emit(0x5F)
emit(0x7C); emit(0xEE, 0xD3); emit(0x67)
emit(0x7D); emit(0xEE, 0x5C); emit(0x6F)

# Extract X, Y from LFSR state (still in shadow)
# X = L & 0x7F (0..127)
emit(0x7D)                  # LD A, L
emit(0xE6, 0x7F)            # AND 0x7F → X
emit(0x4F)                  # LD C, A  → C' = X (using shadow C temporarily!)
# Y = H & 0x7F, mod 96
emit(0x7C)                  # LD A, H
emit(0xE6, 0x7F)            # AND 0x7F
emit(0xFE, 96)              # CP 96
emit(0x38, 0x02)            # JR C, +2
emit(0xD6, 96)              # SUB 96
# A = Y (0..95), C' = X
emit(0x57)                  # LD D, A  → D' = Y (temp, clobbers LFSR D!)

# PROBLEM: we're clobbering LFSR D with Y!
# Need to save LFSR D first.
print("ERROR: clobbering LFSR state. Need stack save or different approach.")
print(f"Current code: {len(code)} bytes")
print("Switching to stack-based approach...")

# ============ FINAL VERSION: PUSH/POP LFSR state ============
code.clear()
labels.clear()
fixups.clear()

# Strategy:
# LFSR state on stack: PUSH DE; PUSH HL after each step
# Screen calc in main regs
# This costs ~8 bytes per iteration but is clean

# Actually simplest: just use a memory buffer for LFSR state
# lfsr_state: DS 4 ; at fixed address

# OR: use IY for something...

# SIMPLEST APPROACH: Separate LFSR step and screen plot.
# LFSR state in DEHL permanently.
# Screen calc: save DEHL on stack, calc, plot, restore DEHL.
# PUSH DE; PUSH HL; <calc+plot>; POP HL; POP DE
# Cost: 4 PUSHes/POPs = 4 bytes per point overhead

# Clear screen
emit(0x21); emit16(0x4000)
emit(0x11); emit16(0x4001)
emit(0x01); emit16(0x17FF)
emit(0x36, 0x00)
emit(0xED, 0xB0)             # 13

# Attributes
emit(0x21); emit16(0x5800)
emit(0x11); emit16(0x5801)
emit(0x01); emit16(0x02FF)
emit(0x36, 0x38)
emit(0xED, 0xB0)             # 13, total=26

# IX = seed table
emit(0xDD, 0x21)
seed_table_fixup = len(code)
emit16(0x0000)               # 4, total=30

# C = number of seeds remaining
emit(0x0E, NSEEDS)           # 2, total=32

label("outer")
# Load seed → DE, init HL=0x1337
emit(0xDD, 0x56, 0x01)       # LD D, (IX+1)
emit(0xDD, 0x5E, 0x00)       # LD E, (IX+0)
emit(0x21); emit16(0x1337)   # LD HL, 0x1337  = 9, total=41

# Push C (seed counter), use B for points
emit(0xC5)                    # PUSH BC = 1, total=42

# Warmup: 8 LFSR steps (inline tiny version)
emit(0x06, 0x08)             # LD B, 8
label("warmup")
emit(0xCD)                   # CALL lfsr_step
jp_fixup("lfsr_step")
emit(0x10)                   # DJNZ warmup
jr_fixup("warmup")           # total ~= 49

# B = points per seed
emit(0x06, POINTS_PER_SEED)  # LD B, 48

label("inner")
emit(0xCD)                   # CALL lfsr_step
jp_fixup("lfsr_step")

# Save LFSR state
emit(0xD5)                   # PUSH DE
emit(0xE5)                   # PUSH HL
emit(0xC5)                   # PUSH BC (point counter)

# Extract X = L & 0x7F, Y = H mod 96
emit(0x7D)                   # LD A, L
emit(0xE6, 0x7F)             # AND 0x7F → X
emit(0x5F)                   # LD E, A  (E = X)

emit(0x7C)                   # LD A, H
emit(0xE6, 0x7F)             # AND 0x7F
emit(0xFE, 96)               # CP 96
emit(0x38, 0x02)             # JR C, +2
emit(0xD6, 96)               # SUB 96
emit(0x57)                   # LD D, A  (D = Y)

# Compute Spectrum screen address
# X in E (0..127), Y in D (0..95)
# Column byte = X >> 3
emit(0x7B)                   # LD A, E
emit(0xCB, 0x3F)             # SRL A
emit(0xCB, 0x3F)             # SRL A
emit(0xCB, 0x3F)             # SRL A
emit(0x6F)                   # LD L, A  (L = X/8)

# High byte: 0x40 | (Y&7) [+ 0x08 if Y>=64]
emit(0x7A)                   # LD A, D  (Y)
emit(0xE6, 0x07)             # AND 7
emit(0xF6, 0x40)             # OR 0x40
emit(0x67)                   # LD H, A

# Low byte += (Y & 0x38) << 2
emit(0x7A)                   # LD A, D
emit(0xE6, 0x38)             # AND 0x38
emit(0x07)                   # RLCA
emit(0x07)                   # RLCA
emit(0xB5)                   # OR L
emit(0x6F)                   # LD L, A

# Third: Y >= 64
emit(0x7A)                   # LD A, D
emit(0xFE, 64)               # CP 64
emit(0x38, 0x02)             # JR C, +2
emit(0xCB, 0xFC)             # SET 7, H ... wait, 0x08 to H = bit 3
# Actually SET 3,H = CB FC... let me check
# SET 3, H = CB DC. Bit 3 of H.
# H = 0x40..0x47. Setting bit 3 → 0x48..0x4F = +0x800 offset. Correct!

# Fix: SET 3, H
code[-1] = 0xDC  # CB DC = SET 3, H

# XOR pixel: bit = 7 - (X & 7)
emit(0x7B)                   # LD A, E (X)
emit(0xE6, 0x07)             # AND 7
emit(0x47)                   # LD B, A  (shift count)
emit(0x3E, 0x80)             # LD A, 0x80
emit(0xB0)                   # OR B ... no, need to test B=0
# Shift right B times
emit(0x78)                   # LD A, B
emit(0xB7)                   # OR A
emit(0x3E, 0x80)             # LD A, 0x80
emit(0x28, 0x04)             # JR Z, +4 (no shift needed)
emit(0x47)                   # LD B, A ... wait, B is shift count
# Ugh, let me redo. B=shift count, A=0x80.
# If B=0, skip. Else loop B times: SRL A; DJNZ

# Simpler approach:
code_len_before = len(code)
# Redo last few bytes
code = code[:code_len_before - 8]  # back up
# B still has X&7 from before... no, we overwrote B
# Let me just do it cleanly:
emit(0x7B)                   # LD A, E (X)
emit(0xE6, 0x07)             # AND 7   → shift count
emit(0x28, 0x06)             # JR Z, no_shift (if count=0, bit=0x80)
emit(0x47)                   # LD B, A
emit(0x3E, 0x80)             # LD A, 0x80
label("srl_loop")
emit(0xCB, 0x3F)             # SRL A
emit(0x10)                   # DJNZ srl_loop
jr_fixup("srl_loop")
emit(0x18, 0x02)             # JR +2 (skip no_shift)
label("no_shift")
emit(0x3E, 0x80)             # LD A, 0x80

# XOR with screen
emit(0xAE)                   # XOR (HL)
emit(0x77)                   # LD (HL), A

# Restore state
emit(0xC1)                   # POP BC
emit(0xE1)                   # POP HL
emit(0xD1)                   # POP DE

# Loop points
emit(0x10)                   # DJNZ inner
jr_fixup("inner")

# Next seed: advance IX by 2
emit(0xDD, 0x23)             # INC IX
emit(0xDD, 0x23)             # INC IX

# Restore seed counter
emit(0xC1)                   # POP BC (C = seed counter)
emit(0x0D)                   # DEC C
emit(0x20)                   # JR NZ, outer
jr_fixup("outer")

# Done: DI; HALT
emit(0xF3)                   # DI
emit(0x76)                   # HALT

# LFSR subroutine
label("lfsr_step")
emit(0xCB, 0x3A)             # SRL D
emit(0xCB, 0x1B)             # RR E
emit(0xCB, 0x1C)             # RR H
emit(0xCB, 0x1D)             # RR L
emit(0xD0)                   # RET NC (bit was 0, no XOR needed)
emit(0x7A); emit(0xEE, 0xB4); emit(0x57)  # D ^= B4
emit(0x7B); emit(0xEE, 0xBC); emit(0x5F)  # E ^= BC
emit(0x7C); emit(0xEE, 0xD3); emit(0x67)  # H ^= D3
emit(0x7D); emit(0xEE, 0x5C); emit(0x6F)  # L ^= 5C
emit(0xC9)                   # RET

# Resolve fixups
# Seed table will follow immediately after code
seed_table_addr = here()
code[seed_table_fixup] = seed_table_addr & 0xFF
code[seed_table_fixup + 1] = (seed_table_addr >> 8) & 0xFF

resolve()

code_size = len(code)
print(f"\nCode: {code_size} bytes")
print(f"Seeds: {NSEEDS * 2} bytes")
print(f"Total: {code_size + NSEEDS * 2} bytes")

# Append seed table
for s in seeds[:NSEEDS]:
    emit(s & 0xFF, (s >> 8) & 0xFF)

total = len(code)
print(f"Final binary: {total} bytes")

if total <= 256:
    print("✅ FITS IN 256 BYTES!")
elif total <= 512:
    print(f"⚠ Fits in 512 bytes ({total}/512)")
else:
    print(f"❌ Too large: {total} bytes")

# Write binary
outpath = "cuda/che_intro.bin"
with open(outpath, "wb") as f:
    f.write(code)
print(f"Written to {outpath}")
print(f"Run: mze --target spectrum {outpath}")

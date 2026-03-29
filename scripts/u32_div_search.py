#!/usr/bin/env python3
"""
Composite u32 division search across multiple accumulator conventions.

For each convention (DEHL, HLIX, HLH'L'), search for optimal u32÷K
using composite operations as atomic building blocks.

Pool of composite ops (per convention):
  SHL32   — shift accumulator left (×2)
  SHR32   — shift accumulator right (÷2)
  SAVE    — copy main acc to temp acc
  RESTORE — copy temp acc to main acc
  ADD     — main += temp
  SUB     — main -= temp
  SWAP    — exchange main ↔ temp

For small K (÷2..÷255), quotient might fit in 8 or 16 bits.
For u32÷10 (itoa): result is u32, need full 32-bit output.

Strategy: multiply-and-shift at u32 level.
  u32÷K ≈ (u32 × M) >> S  where M and S chosen per K.
  But u32×M requires 64-bit intermediate — decompose!

Alternative for small K: repeated subtraction structure.
  u32÷10 = shift-and-subtract patterns.

Key insight: we search DECOMPOSITIONS, not individual Z80 ops.
"""

import sys
from itertools import product as cartesian

# === Cost Tables per Convention ===

COSTS = {
    "DEHL":   {"shl": 34, "shr": 32, "add": 54, "sub": 58, "save": 28, "restore": 28, "swap": 56},
    "HLIX":   {"shl": 30, "shr": 28, "add": 30, "sub": 34, "save": 24, "restore": 24, "swap": 48},
    "HLH'L'": {"shl": 30, "shr": 28, "add": 30, "sub": 34, "save":  4, "restore":  4, "swap":  8},
}

# === Simulator ===

class U32State:
    """Two 32-bit accumulators + one 8-bit."""
    __slots__ = ['main', 'temp', 'a']
    def __init__(self, val):
        self.main = val & 0xFFFFFFFF
        self.temp = 0
        self.a = 0

def make_ops():
    """Return list of (name, func, cost_key)."""
    ops = []

    def shl(s):
        s.main = (s.main << 1) & 0xFFFFFFFF
    ops.append(("SHL", shl, "shl"))

    def shr(s):
        s.main = s.main >> 1
    ops.append(("SHR", shr, "shr"))

    def save(s):
        s.temp = s.main
    ops.append(("SAVE", save, "save"))

    def restore(s):
        s.main = s.temp
    ops.append(("RESTORE", restore, "restore"))

    def add(s):
        s.main = (s.main + s.temp) & 0xFFFFFFFF
    ops.append(("ADD", add, "add"))

    def sub(s):
        s.main = (s.main - s.temp) & 0xFFFFFFFF
    ops.append(("SUB", sub, "sub"))

    def swap(s):
        s.main, s.temp = s.temp, s.main
    ops.append(("SWAP", swap, "swap"))

    # Extract high byte (for narrowing division result)
    def extract_h8(s):
        s.a = (s.main >> 24) & 0xFF
    ops.append(("EXTR_H8", extract_h8, "shr"))  # ~same cost as SHR

    def extract_h16(s):
        s.main = (s.main >> 16) & 0xFFFF
    ops.append(("EXTR_H16", extract_h16, "shr"))

    return ops

OPS = make_ops()

def verify(seq, k, output="main", n_samples=1000):
    """Verify a composite sequence computes ÷K correctly.

    Tests n_samples random-ish inputs covering edge cases.
    """
    import random

    # Test set: edge cases + random samples
    test_vals = [0, 1, k-1, k, k+1, k*2, k*2-1, k*2+1,
                 255, 256, 65535, 65536, 0xFFFFFF, 0xFFFFFFFF,
                 k*100, k*255, k*1000]

    random.seed(42)
    test_vals.extend(random.randint(0, 0xFFFFFFFF) for _ in range(n_samples))
    test_vals = [v & 0xFFFFFFFF for v in test_vals]

    for val in test_vals:
        s = U32State(val)
        for op_idx in seq:
            OPS[op_idx][1](s)

        result = s.main if output == "main" else s.a
        expected = val // k

        if output == "a":
            expected = expected & 0xFF
        elif output == "main":
            expected = expected & 0xFFFFFFFF

        if result != expected:
            return False
    return True

def search_div(k, max_depth=8, conv="HLH'L'"):
    """Search for u32÷K using composite ops."""
    costs = COSTS[conv]
    n_ops = len(OPS)
    best = None
    best_cost = 999999

    for depth in range(1, max_depth + 1):
        found_at_depth = False
        for seq in cartesian(range(n_ops), repeat=depth):
            # Quick cost check
            cost = sum(costs[OPS[i][2]] for i in seq)
            if cost >= best_cost:
                continue

            # Prune obviously bad sequences
            names = [OPS[i][0] for i in seq]
            # Must start with SHL or SAVE (otherwise just passing through)
            if names[0] not in ("SHL", "SHR", "SAVE"):
                continue
            # Can't RESTORE before SAVE
            save_seen = False
            skip = False
            for n in names:
                if n == "SAVE": save_seen = True
                if n in ("RESTORE", "ADD", "SUB", "SWAP") and not save_seen:
                    skip = True
                    break
            if skip:
                continue

            if verify(seq, k, n_samples=200):
                # Found! Verify more thoroughly
                if verify(seq, k, n_samples=5000):
                    cost = sum(costs[OPS[i][2]] for i in seq)
                    if cost < best_cost:
                        best_cost = cost
                        best = (seq, names, cost, depth)
                        found_at_depth = True

        if found_at_depth:
            print(f"  depth={depth}: {best[1]} = {best[2]}T", file=sys.stderr)

        # If we found something and next depth is much more expensive, stop
        if best and depth >= 6:
            break

    return best

def main():
    print("=== u32÷K Composite Search ===\n")

    key_divisors = [2, 3, 4, 5, 7, 8, 9, 10, 16, 100, 128, 256, 1000]

    for conv in ["HLH'L'", "HLIX", "DEHL"]:
        print(f"\n--- Convention: {conv} ---")
        costs = COSTS[conv]

        for k in key_divisors:
            print(f"\nu32÷{k} ({conv}):", file=sys.stderr)
            result = search_div(k, max_depth=7, conv=conv)

            if result:
                seq, names, cost, depth = result
                print(f"  div{k}: {' → '.join(names)} = {cost}T (depth {depth})")
            else:
                print(f"  div{k}: not found at max_depth=7")

if __name__ == "__main__":
    # Quick test: just ÷2, ÷4, ÷8, ÷10 on best convention
    print("Quick search: u32÷K on HLH'L' convention\n")
    conv = "HLH'L'"
    costs = COSTS[conv]

    for k in [2, 3, 4, 5, 7, 8, 10, 16, 100, 256]:
        print(f"u32÷{k}:", end=" ", flush=True)
        result = search_div(k, max_depth=7, conv=conv)
        if result:
            seq, names, cost, depth = result
            print(f"{' → '.join(names)} = {cost}T")
        else:
            print("not found")

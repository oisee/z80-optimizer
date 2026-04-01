#!/usr/bin/env python3
"""Generate mul8_library.asm from mulopt8_clobber.json.

Emits standalone entries for each constant k=2..255.
Fall-through chains are only used where the shorter sequence is a genuine
SUFFIX of the longer one (detected automatically).

Bug that motivated this script: a previous generator reversed all sequences
and then found "common prefixes" of the reversed seqs (= common suffixes in
original order). It then emitted the reversed sequences as ASM, producing
code that executes instructions in wrong order and gives wrong results for
all non-power-of-2 constants. This script emits sequences in correct order.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
JSON_IN = REPO / "data" / "mulopt8_clobber.json"
ASM_OUT = REPO / "data" / "mul8_library.asm"


def load_table(path):
    with open(path) as f:
        entries = json.load(f)
    return {e["k"]: e for e in entries}


def verify_sequence(ops, k):
    """Simulate ops on all 256 inputs, return True if A_out == A_in * k (mod 256)."""
    def exec_seq(ops, a_in):
        a, b, carry = a_in & 0xFF, 0, False
        a_s, b_s, carry_s = 0, 0, False  # shadow regs (EX AF,AF' / EXX)
        for op in ops:
            if op == "ADD A,A":
                r = a + a
                carry = r > 0xFF
                a = r & 0xFF
            elif op == "ADD A,B":
                r = a + b
                carry = r > 0xFF
                a = r & 0xFF
            elif op == "SUB B":
                r = a - b
                carry = r < 0
                a = r & 0xFF
            elif op == "LD B,A":
                b = a
            elif op == "SLA A":
                carry = bool(a & 0x80)
                a = (a << 1) & 0xFF
            elif op == "SRL A":
                carry = bool(a & 0x01)
                a = (a >> 1) & 0xFF
            elif op == "RLA":
                new_carry = bool(a & 0x80)
                a = ((a << 1) | (1 if carry else 0)) & 0xFF
                carry = new_carry
            elif op == "RLCA":
                new_carry = bool(a & 0x80)
                a = ((a << 1) | (1 if new_carry else 0)) & 0xFF
                carry = new_carry
            elif op == "ADC A,B":
                r = a + b + (1 if carry else 0)
                carry = r > 0xFF
                a = r & 0xFF
            elif op == "ADC A,A":
                r = a + a + (1 if carry else 0)
                carry = r > 0xFF
                a = r & 0xFF
            elif op == "SBC A,B":
                r = a - b - (1 if carry else 0)
                carry = r < 0
                a = r & 0xFF
            elif op == "SBC A,A":
                r = a - a - (1 if carry else 0)
                carry = r < 0
                a = r & 0xFF
            elif op == "OR A":
                carry = False
            elif op == "RRCA":
                new_carry = bool(a & 0x01)
                a = ((a >> 1) | (0x80 if new_carry else 0)) & 0xFF
                carry = new_carry
            elif op == "RRA":
                new_carry = bool(a & 0x01)
                a = ((a >> 1) | (0x80 if carry else 0)) & 0xFF
                carry = new_carry
            elif op == "NEG":
                carry = a != 0
                a = (-a) & 0xFF
            elif op == "EX AF,AF'":
                a, a_s = a_s, a
                carry, carry_s = carry_s, carry
            elif op == "EXX":
                b, b_s = b_s, b
            else:
                raise ValueError(f"Unknown op: {op!r}")
        return a

    for a_in in range(256):
        expected = (a_in * k) & 0xFF
        got = exec_seq(ops, a_in)
        if got != expected:
            return False, a_in, expected, got
    return True, None, None, None


def find_suffix_chains(by_k):
    """Find pairs where ops[k_short] is a genuine suffix of ops[k_long].
    Returns dict: k_long -> k_short (longest shared suffix wins).
    Only the deepest chain is kept (no multi-level for simplicity).
    """
    suffix_map = {}  # k_long -> k_short
    for k_long, e_long in by_k.items():
        best_suffix_k = None
        best_suffix_len = 0
        ops_long = e_long["ops"]
        for k_short, e_short in by_k.items():
            if k_short == k_long:
                continue
            ops_short = e_short["ops"]
            n = len(ops_short)
            if n >= len(ops_long):
                continue
            if ops_long[-n:] == ops_short and n > best_suffix_len:
                best_suffix_k = k_short
                best_suffix_len = n
        if best_suffix_k is not None:
            suffix_map[k_long] = best_suffix_k
    return suffix_map


def build_chains(by_k, suffix_map):
    """Build multi-level fall-through chains.

    Returns list of chains, where each chain is a list of k values in
    emission order (longest first, shortest last). The last element is the
    shared tail that gets the RET.

    Example: suffix_map = {240: 255, 254: 255}
    → chain [240, 254, 255] (all three share the ×255 tail at the bottom)
    """
    tail_set = set(suffix_map.values())
    # Find roots: constants that have a tail but are NOT someone else's tail
    # (i.e., nobody points to them)
    roots = [k for k in suffix_map if k not in tail_set]

    chains = []
    tail_to_heads = {}  # tail_k → list of k that point to it
    for k, tail_k in suffix_map.items():
        tail_to_heads.setdefault(tail_k, []).append(k)

    # Build one chain per unique tail value
    for tail_k in sorted(tail_set):
        heads = sorted(tail_to_heads[tail_k])
        # Order: longer sequences first (emit their unique prefixes, then fall to tail)
        heads_sorted = sorted(heads, key=lambda k: len(by_k[k]["ops"]), reverse=True)
        chains.append(heads_sorted + [tail_k])

    return chains


def emit_library(by_k, suffix_map):
    chains = build_chains(by_k, suffix_map)

    # Map k → chain_index for lookup
    in_chain = {}
    for chain in chains:
        for k in chain:
            in_chain[k] = chain

    header = """\
; Z80 Optimal Constant Multiply Library
; Generated by scripts/gen_mul8_library.py from mulopt8_clobber.json
; (GPU brute-force, provably optimal sequences)
;
; {n_constants} constants (k=2..255), sequences in CORRECT execution order.
; {n_chains} fall-through chains (genuine suffix relationships only).
;
; Input:  A = value to multiply
; Output: A = A * K (mod 256)
; Clobber: B (for most), F always
;
; Usage:  LD A, (value)
;         CALL mul_42
;         ; A now contains value * 42
;
; Repository: https://github.com/oisee/z80-optimizer
; License: MIT
;
"""

    emitted = set()
    group_lines = []

    def emit_chain(chain):
        """Emit a multi-level fall-through chain."""
        ks = chain
        chain_label = " → ".join(f"×{k}" for k in ks)
        group_lines.append(f"; --- Chain: {chain_label} ---")
        for i, k in enumerate(ks):
            entry = by_k[k]
            clobbers = entry.get("clobber", ["B", "F"])
            group_lines.append(
                f"mul_{k}:  ; ×{k}  ({entry['length']} insts, {entry['tstates']}T)"
            )
            # Unique prefix = ops not shared with next entry in chain
            if i + 1 < len(ks):
                next_k = ks[i + 1]
                next_ops = by_k[next_k]["ops"]
                # unique prefix = first (len(ops) - len(next_ops)) ops
                unique_ops = entry["ops"][: len(entry["ops"]) - len(next_ops)]
            else:
                unique_ops = entry["ops"]  # last/only entry: emit all ops
            for op in unique_ops:
                group_lines.append(f"    {op}")
            emitted.add(k)
        group_lines.append("    RET")
        group_lines.append("")

    def emit_standalone(k):
        entry = by_k[k]
        group_lines.append(
            f"mul_{k}:  ; ×{k}  ({entry['length']} insts, {entry['tstates']}T)"
        )
        for op in entry["ops"]:
            group_lines.append(f"    {op}")
        group_lines.append("    RET")
        group_lines.append("")
        emitted.add(k)

    # Emit chains first (they contain multiple labels)
    chain_ks = set()
    for chain in chains:
        for k in chain:
            chain_ks.add(k)

    for chain in chains:
        emit_chain(chain)

    # Emit remaining standalone entries in k order
    for k in sorted(by_k.keys()):
        if k not in emitted:
            emit_standalone(k)

    n_chains = len(chains)
    n_constants = len(by_k)
    filled_header = header.format(n_constants=n_constants, n_chains=n_chains)
    return filled_header + "\n".join(group_lines) + "\n"


def main():
    print(f"Loading {JSON_IN}...", file=sys.stderr)
    by_k = load_table(JSON_IN)
    print(f"  {len(by_k)} entries loaded", file=sys.stderr)

    # Verify all sequences
    bad = 0
    for k, e in sorted(by_k.items()):
        ok, a_in, expected, got = verify_sequence(e["ops"], k)
        if not ok:
            print(f"  VERIFY FAIL k={k}: ops={e['ops']}, input={a_in}, expected={expected}, got={got}", file=sys.stderr)
            bad += 1
    if bad == 0:
        print(f"  All {len(by_k)} sequences verified correct.", file=sys.stderr)
    else:
        print(f"  {bad} sequences FAILED verification!", file=sys.stderr)
        sys.exit(1)

    suffix_map = find_suffix_chains(by_k)
    print(f"  {len(suffix_map)} genuine fall-through chains found:", file=sys.stderr)
    for k_long, k_short in sorted(suffix_map.items()):
        print(f"    ×{k_long} → ×{k_short}", file=sys.stderr)

    asm = emit_library(by_k, suffix_map)

    with open(ASM_OUT, "w") as f:
        f.write(asm)
    print(f"Written {ASM_OUT} ({len(asm)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()

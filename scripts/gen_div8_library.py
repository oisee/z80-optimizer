#!/usr/bin/env python3
"""Generate div8_library.asm from div8_optimal.json.

111 genuine suffix chains, mostly ending at div_32/div_16/div_8/div_4/div_2
(multiply-and-shift sequences share common SRL A tails).

Pseudo-ops expanded to real Z80:
  SWAP_HL  → LD A,L / LD L,H / LD H,A   (swap H and L; T-states: +8T vs. bare)
  SUB HL,BC → OR A / SBC HL,BC            (subtract BC from HL, carry-free)
  (OR A clears carry before each SBC HL,BC; safe since sequences never underflow)

Convention:
  Input:  A = dividend (0..255)
  Output: A = floor(A / K)
  Clobber: B, C, H, L, F (varies by sequence — see per-entry comment)
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
JSON_IN = REPO / "data" / "div8_optimal.json"
ASM_OUT = REPO / "data" / "div8_library.asm"


def load_table(path):
    with open(path) as f:
        d = json.load(f)
    return {e["k"]: e for e in d["entries"]}


def exec_div8(ops, a_in):
    """Simulate div8 ops. Returns A after sequence."""
    a = a_in & 0xFF
    b, c, h, l = 0, 0, 0, 0
    carry = False

    def hl_val():
        return (h << 8) | l

    def set_hl(v):
        nonlocal h, l
        v &= 0xFFFF
        h, l = v >> 8, v & 0xFF

    for op in ops:
        if op == "SRL A":
            carry = bool(a & 1)
            a >>= 1
        elif op == "LD H,0":
            h = 0
        elif op == "LD L,A":
            l = a
        elif op == "LD A,H":
            a = h
        elif op == "LD A,L":
            a = l
        elif op == "LD C,A":
            c = a
        elif op == "LD B,A":
            b = a
        elif op.startswith("LD B,"):
            b = int(op[5:])
        elif op == "ADD HL,HL":
            v = hl_val() * 2
            carry = v > 0xFFFF
            set_hl(v)
        elif op == "ADD HL,BC":
            v = hl_val() + ((b << 8) | c)
            carry = v > 0xFFFF
            set_hl(v)
        elif op == "SUB HL,BC":
            # Subtract BC from HL without carry (carry-free semantics in optimizer)
            v = hl_val() - ((b << 8) | c)
            carry = v < 0
            set_hl(v)
        elif op == "SWAP_HL":
            # Swap H and L: HL = (L<<8)|H
            h, l = l, h
        elif op == "OR A":
            carry = False
        elif op == "AND 1":
            a &= 1
            carry = False
        elif op == "ADD A,B":
            r = a + b
            carry = r > 0xFF
            a = r & 0xFF
        elif op == "ADC A,B":
            r = a + b + (1 if carry else 0)
            carry = r > 0xFF
            a = r & 0xFF
        elif op == "SBC A,A":
            r = a - a - (1 if carry else 0)
            carry = r < 0
            a = r & 0xFF
        else:
            raise ValueError(f"Unknown div8 op: {op!r}")
    return a


def verify_sequence(ops, k):
    """Verify div8 sequence for all 256 inputs. Returns (ok, bad_input, expected, got)."""
    for a_in in range(256):
        expected = a_in // k
        got = exec_div8(ops, a_in)
        if got != expected:
            return False, a_in, expected, got
    return True, None, None, None


def expand_op(op):
    """Expand pseudo-ops to real Z80 instruction lists."""
    if op == "SWAP_HL":
        return ["LD A,L", "LD L,H", "LD H,A"]
    elif op == "SUB HL,BC":
        return ["OR A", "SBC HL,BC"]
    else:
        return [op]


def expand_ops(ops):
    """Expand all pseudo-ops in a sequence."""
    result = []
    for op in ops:
        result.extend(expand_op(op))
    return result


def find_suffix_chains(by_k):
    """Find pairs where ops[k_short] is a genuine suffix of ops[k_long].
    Uses ORIGINAL (unexpanded) ops for suffix detection — expansion is length-preserving
    per unique op, so we detect chains in the original representation then expand.
    Returns dict: k_long -> k_short (longest shared suffix wins).
    """
    suffix_map = {}
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
    """Build valid linear fall-through chains by following suffix pointers.
    Sorted longest-first so longer chains own shared tails.
    """
    pointed_to = set(suffix_map.values())
    roots = sorted(k for k in suffix_map if k not in pointed_to)

    chains = []
    for root in roots:
        chain = [root]
        cur = root
        while cur in suffix_map:
            cur = suffix_map[cur]
            chain.append(cur)
        if len(chain) >= 2:
            chains.append(chain)

    chains.sort(key=lambda c: -len(c))
    return chains


def emit_library(by_k, suffix_map):
    chains = build_chains(by_k, suffix_map)

    emitted = set()
    group_lines = []

    n_constants = len(by_k)
    n_chains = len(chains)
    n_unverified = sum(1 for e in by_k.values() if e.get("unverified"))

    caution_line = (
        f"; WARNING: {n_unverified} entries failed Z80 simulation (tagged CAUTION below, excluded from chains).\n"
        if n_unverified else ""
    )

    header = f"""\
; Z80 Optimal Constant Division Library (8-bit: A / K → A, floor)
; Generated by scripts/gen_div8_library.py from div8_optimal.json
; (Analytical multiply-and-shift + GPU-verified sequences)
;
; {n_constants} constants (k=2..255), sequences in CORRECT execution order.
; {n_chains} fall-through chains (genuine suffix relationships only, verified entries only).
{caution_line}
;
; Convention:
;   Input:  A = dividend (unsigned 8-bit, 0..255)
;   Output: A = floor(A / K)
;   Clobber: varies (B, C, H, L, F) — see per-entry comment
;
; Usage:
;   LD A, (dividend)
;   CALL div_42
;   ; A = dividend / 42 (floor)
;
; Pseudo-op expansions:
;   SWAP_HL  → LD A,L / LD L,H / LD H,A  (put L into H, zero L: HL = A<<8)
;   SUB HL,BC → OR A / SBC HL,BC          (carry-free HL -= BC)
;
; Note: T-state counts are for original optimizer sequences.
; Expanded pseudo-ops add ~4-8T per occurrence vs. the listed T-states.
;
; Repository: https://github.com/oisee/z80-optimizer
; License: MIT
;
"""

    def clobber_str(e):
        return ", ".join(e.get("clobbers", ["A", "F"]))

    def emit_ops_for_chain_member(k, unique_ops_orig):
        """Emit expanded ops for a chain member's unique prefix."""
        for op in unique_ops_orig:
            expanded = expand_op(op)
            if len(expanded) == 1:
                group_lines.append(f"    {expanded[0]}")
            else:
                group_lines.append(f"    ; {op} (expanded):")
                for eop in expanded:
                    group_lines.append(f"    {eop}")

    def emit_chain(chain):
        ks = chain
        chain_label = " → ".join(f"÷{k}" for k in ks)
        group_lines.append(f"; --- Chain: {chain_label} ---")
        for i, k in enumerate(ks):
            entry = by_k[k]
            group_lines.append(
                f"div_{k}:  ; ÷{k}  ({entry['length']} insts, {entry['tstates']}T"
                f" | clobbers: {clobber_str(entry)})"
            )
            if i + 1 < len(ks):
                next_k = ks[i + 1]
                next_ops = by_k[next_k]["ops"]
                unique_ops = entry["ops"][: len(entry["ops"]) - len(next_ops)]
            else:
                unique_ops = entry["ops"]
            emit_ops_for_chain_member(k, unique_ops)
            emitted.add(k)
        group_lines.append("    RET")
        group_lines.append("")

    def emit_jp_prefix(k, chain):
        """Emit unique prefix of k in chain + JP to already-emitted tail.
        Finds the deepest already-emitted entry in chain and jumps to it.
        """
        entry = by_k[k]
        # Find the suffix entry to JP to (first already-emitted member of chain after k)
        k_idx = chain.index(k)
        jp_target = None
        for j in range(k_idx + 1, len(chain)):
            if chain[j] in emitted:
                jp_target = chain[j]
                break
        if jp_target is None:
            emit_standalone(k)
            return
        target_ops = by_k[jp_target]["ops"]
        unique_ops = entry["ops"][: len(entry["ops"]) - len(target_ops)]
        group_lines.append(
            f"div_{k}:  ; ÷{k}  ({entry['length']} insts, {entry['tstates']}T"
            f" | clobbers: {clobber_str(entry)})"
        )
        emit_ops_for_chain_member(k, unique_ops)
        group_lines.append(f"    JRS div_{jp_target}")
        group_lines.append("")
        emitted.add(k)

    def emit_standalone(k):
        entry = by_k[k]
        if entry.get("unverified"):
            a_in, expected, got = entry["verify_fail"]
            group_lines.append(
                f"; CAUTION: div_{k} — Z80 simulation FAILED (a_in={a_in}: got {got}, expected {expected})"
            )
            group_lines.append(f"; Method: {entry.get('method','?')} — sequence may be incorrect for this input.")
        group_lines.append(
            f"div_{k}:  ; ÷{k}  ({entry['length']} insts, {entry['tstates']}T"
            f" | clobbers: {clobber_str(entry)})"
        )
        for op in entry["ops"]:
            expanded = expand_op(op)
            if len(expanded) == 1:
                group_lines.append(f"    {expanded[0]}")
            else:
                group_lines.append(f"    ; {op} (expanded):")
                for eop in expanded:
                    group_lines.append(f"    {eop}")
        group_lines.append("    RET")
        group_lines.append("")
        emitted.add(k)

    for chain in chains:
        # Find the first already-emitted member of this chain
        first_emitted_idx = next(
            (i for i, k in enumerate(chain) if k in emitted), None
        )
        if first_emitted_idx is None:
            emit_chain(chain)
        elif first_emitted_idx == 0:
            pass  # entire chain already emitted
        else:
            # Emit unique prefixes with JP to already-emitted tail
            for i in range(first_emitted_idx):
                k = chain[i]
                if k not in emitted:
                    emit_jp_prefix(k, chain)

    for k in sorted(by_k.keys()):
        if k not in emitted:
            emit_standalone(k)

    return header + "\n".join(group_lines) + "\n"


def main():
    print(f"Loading {JSON_IN}...", file=sys.stderr)
    by_k = load_table(JSON_IN)
    print(f"  {len(by_k)} entries loaded", file=sys.stderr)

    # Verify all sequences; tag unverified entries (don't abort)
    bad_ks = set()
    for k, e in sorted(by_k.items()):
        ok, a_in, expected, got = verify_sequence(e["ops"], k)
        if not ok:
            print(f"  VERIFY FAIL k={k}: a_in={a_in}, expected={expected}, got={got} (method={e.get('method','?')})", file=sys.stderr)
            bad_ks.add(k)
            by_k[k]["unverified"] = True
            by_k[k]["verify_fail"] = (a_in, expected, got)
    n_good = len(by_k) - len(bad_ks)
    if bad_ks:
        print(f"  {len(bad_ks)} entries FAILED Z80 simulation (tagged CAUTION, excluded from chains).", file=sys.stderr)
        print(f"  {n_good} entries verified correct.", file=sys.stderr)
    else:
        print(f"  All {len(by_k)} sequences verified correct.", file=sys.stderr)

    # Only verified entries participate in fall-through chains
    verified_by_k = {k: e for k, e in by_k.items() if not e.get("unverified")}
    suffix_map = find_suffix_chains(verified_by_k)
    print(f"  {len(suffix_map)} genuine fall-through chains found.", file=sys.stderr)

    chains = build_chains(verified_by_k, suffix_map)
    long_chains = [(c, len(c)) for c in chains if len(c) >= 3]
    long_chains.sort(key=lambda x: -x[1])
    for chain, depth in long_chains[:8]:
        print(f"    {'→'.join(f'÷{k}' for k in chain)}", file=sys.stderr)
    if len(long_chains) > 8:
        print(f"    ... ({len(long_chains)} chains of depth ≥3)", file=sys.stderr)

    asm = emit_library(by_k, suffix_map)

    with open(ASM_OUT, "w") as f:
        f.write(asm)
    print(f"Written {ASM_OUT} ({len(asm)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()

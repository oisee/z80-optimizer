#!/usr/bin/env python3
"""Generate mul8_library.asm from mulopt8_clobber.json + chains_mul8_mod256.json.

Strategy for fall-through chains:
  1. Abstract-chain sequences (from chains_mul8_mod256.json, translated to Z80):
     For k where the abstract chain uses only {save=LD B,A, dbl=ADD A,A,
     add(0)=ADD A,B, sub(0)=SUB B} AND has equal T-states to the GPU-optimal
     sequence, we use the abstract-chain translation. These ops are all
     carry-INDEPENDENT, so "mul_2k = [ADD A,A] + mul_k_body" is always a
     genuine Z80 suffix — the existing suffix detector finds doubling families
     automatically (mul_56→mul_28→mul_14→mul_7, etc.)

  2. GPU-optimal sequences (from mulopt8_clobber.json):
     Used for all other k (complex chains with NEG, carry-dependent ADC/SBC/RLA,
     or where the abstract chain is suboptimal). These still form whatever
     literal-suffix chains exist (e.g., ×240→×254→×255 via NEG).

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
JSON_OPT = REPO / "data" / "mulopt8_clobber.json"
JSON_CHAINS = REPO / "data" / "chains_mul8_mod256.json"
ASM_OUT = REPO / "data" / "mul8_library.asm"

# Abstract-chain op → Z80 opcode + T-states
ABSTRACT_OPS = {
    "save":   ("LD B,A",  4),
    "dbl":    ("ADD A,A", 4),
    "add(0)": ("ADD A,B", 4),
    "sub(0)": ("SUB B",   4),
}


def load_table(path):
    with open(path) as f:
        entries = json.load(f)
    return {e["k"]: e for e in entries}


def translate_abstract(ops):
    """Translate abstract chain ops to Z80 opcode list, or None if untranslatable."""
    result = []
    for op in ops:
        if op not in ABSTRACT_OPS:
            return None
        result.append(ABSTRACT_OPS[op][0])
    return result


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


def build_best_sequences(opt_table, chains_table):
    """For each k, choose abstract-chain or GPU-optimal sequence.

    Prefers abstract-chain translation when:
      - All ops translate (only {save,dbl,add(0),sub(0)})
      - T-states equal to GPU-optimal
    This gives carry-independent sequences that form natural suffix chains.
    """
    by_k = {}
    for k in sorted(opt_table.keys()):
        opt = opt_table[k]
        chosen_ops = opt["ops"]
        chosen_ts = opt["tstates"]
        chosen_src = "gpu"

        if k in chains_table:
            abstract_ops = chains_table[k]["ops"]
            z80_ops = translate_abstract(abstract_ops)
            if z80_ops is not None:
                abstract_ts = len(z80_ops) * 4
                if abstract_ts == opt["tstates"]:
                    chosen_ops = z80_ops
                    chosen_ts = abstract_ts
                    chosen_src = "abstract"

        by_k[k] = {
            "k": k,
            "ops": chosen_ops,
            "length": len(chosen_ops),
            "tstates": chosen_ts,
            "clobber": opt.get("clobber", ["B", "F"]),
            "src": chosen_src,
        }
    return by_k


def find_suffix_chains(by_k):
    """Find pairs where ops[k_short] is a genuine suffix of ops[k_long].
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

    Each chain is k0 → k1 → ... → kn where suffix_map[ki] = k_{i+1}.
    Only entries NOT pointed to by anyone else are chain roots.
    Chains are sorted longest-first so the emitter can let longer chains
    "own" shared tails; shorter chains whose tail is already emitted fall
    back to standalone emission.
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

    # Longest first so they "own" shared tails during emission
    chains.sort(key=lambda c: -len(c))
    return chains


def emit_library(by_k, suffix_map, n_abstract):
    chains = build_chains(by_k, suffix_map)

    in_chain = {}
    for chain in chains:
        for k in chain:
            in_chain[k] = chain

    n_constants = len(by_k)
    n_chains = len(chains)

    header = f"""\
; Z80 Optimal Constant Multiply Library (8-bit: A × K → A mod 256)
; Generated by scripts/gen_mul8_library.py
; Sources: mulopt8_clobber.json (GPU brute-force) + chains_mul8_mod256.json (abstract chains)
;
; {n_constants} constants (k=2..255), sequences in CORRECT execution order.
; {n_chains} fall-through chains (genuine suffix relationships only).
; {n_abstract} entries use abstract-chain translation (carry-independent, enables doubling chains).
;
; Input:  A = value to multiply
; Output: A = A * K (mod 256)
; Clobber: B (for most), F always
;
; Usage:  LD A, (value)
;         CALL mul_42
;         ; A now contains value * 42
;
; Note: abstract-chain sequences (LD B,A / ADD A,A / ADD A,B / SUB B only) are
; carry-independent and form doubling families: mul_56→mul_28→mul_14→mul_7, etc.
; GPU-optimal sequences (using ADC/SBC/RLA/RLCA/NEG) may be a few T-states
; better for isolated calls but cannot chain.
;
; Repository: https://github.com/oisee/z80-optimizer
; License: MIT
;
"""

    emitted = set()
    group_lines = []

    def emit_chain(chain):
        ks = chain
        chain_label = " → ".join(f"×{k}" for k in ks)
        group_lines.append(f"; --- Chain: {chain_label} ---")
        for i, k in enumerate(ks):
            entry = by_k[k]
            src_tag = f" [{entry['src']}]" if entry["src"] == "abstract" else ""
            group_lines.append(
                f"mul_{k}:  ; ×{k}  ({entry['length']} insts, {entry['tstates']}T{src_tag})"
            )
            if i + 1 < len(ks):
                next_k = ks[i + 1]
                next_ops = by_k[next_k]["ops"]
                unique_ops = entry["ops"][: len(entry["ops"]) - len(next_ops)]
            else:
                unique_ops = entry["ops"]
            for op in unique_ops:
                group_lines.append(f"    {op}")
            emitted.add(k)
        group_lines.append("    RET")
        group_lines.append("")

    def emit_jp_prefix(k, chain):
        """Emit unique prefix of k in chain + JP to already-emitted tail."""
        entry = by_k[k]
        src_tag = f" [{entry['src']}]" if entry["src"] == "abstract" else ""
        k_idx = chain.index(k)
        jp_target = next((chain[j] for j in range(k_idx + 1, len(chain)) if chain[j] in emitted), None)
        if jp_target is None:
            emit_standalone(k)
            return
        target_ops = by_k[jp_target]["ops"]
        unique_ops = entry["ops"][: len(entry["ops"]) - len(target_ops)]
        group_lines.append(
            f"mul_{k}:  ; ×{k}  ({entry['length']} insts, {entry['tstates']}T{src_tag})"
        )
        for op in unique_ops:
            group_lines.append(f"    {op}")
        group_lines.append(f"    JP mul_{jp_target}")
        group_lines.append("")
        emitted.add(k)

    def emit_standalone(k):
        entry = by_k[k]
        src_tag = f" [{entry['src']}]" if entry["src"] == "abstract" else ""
        group_lines.append(
            f"mul_{k}:  ; ×{k}  ({entry['length']} insts, {entry['tstates']}T{src_tag})"
        )
        for op in entry["ops"]:
            group_lines.append(f"    {op}")
        group_lines.append("    RET")
        group_lines.append("")
        emitted.add(k)

    # Emit chains longest-first; if a chain's tail is already emitted, use JP prefix.
    for chain in chains:
        first_emitted_idx = next(
            (i for i, k in enumerate(chain) if k in emitted), None
        )
        if first_emitted_idx is None:
            emit_chain(chain)
        elif first_emitted_idx == 0:
            pass  # entire chain already emitted
        else:
            for i in range(first_emitted_idx):
                k = chain[i]
                if k not in emitted:
                    emit_jp_prefix(k, chain)

    for k in sorted(by_k.keys()):
        if k not in emitted:
            emit_standalone(k)

    return header + "\n".join(group_lines) + "\n"


def main():
    print(f"Loading {JSON_OPT}...", file=sys.stderr)
    opt_table = load_table(JSON_OPT)
    print(f"  {len(opt_table)} GPU-optimal entries loaded", file=sys.stderr)

    print(f"Loading {JSON_CHAINS}...", file=sys.stderr)
    with open(JSON_CHAINS) as f:
        chains_raw = json.load(f)
    chains_table = {e["k"]: e for e in chains_raw}
    print(f"  {len(chains_table)} abstract chain entries loaded", file=sys.stderr)

    by_k = build_best_sequences(opt_table, chains_table)
    n_abstract = sum(1 for e in by_k.values() if e["src"] == "abstract")
    print(f"  {n_abstract} entries use abstract-chain translation (equally optimal)", file=sys.stderr)

    # Verify all chosen sequences
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
    print(f"  {len(suffix_map)} genuine fall-through chains found.", file=sys.stderr)

    # Show chain summary by family
    chains = build_chains(by_k, suffix_map)
    long_chains = [(c, len(c)) for c in chains if len(c) >= 3]
    long_chains.sort(key=lambda x: -x[1])
    for chain, depth in long_chains[:10]:
        print(f"    {'→'.join(f'×{k}' for k in chain)}", file=sys.stderr)
    if len(long_chains) > 10:
        print(f"    ... ({len(long_chains)} chains of depth ≥3)", file=sys.stderr)

    asm = emit_library(by_k, suffix_map, n_abstract)

    with open(ASM_OUT, "w") as f:
        f.write(asm)
    print(f"Written {ASM_OUT} ({len(asm)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()

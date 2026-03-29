#!/usr/bin/env python3
"""Composite-operation search for optimal div8.

Instead of searching 21 individual Z80 ops at depth 18, we search
composite "macro" operations. div10 becomes depth ~4 instead of 18.

Key insight: valid div sequences have a rigid structure:
  [SAVE?] → PREAMBLE → MUL16[M] → [ADD_H_saved?] → EXTRACT_H → SRL×N

So we don't need arbitrary-depth search — we enumerate the STRUCTURE:
  For each M=2..255:
    For each S=8..19:
      Check: (A * M) >> S == A // K ?
  For each R=1..254 (where M = 256+R):
    For each S=8..19:
      Check: (A * (256+R)) >> S == A // K ?

This is O(254 × 12 × 256) per K — trivial.

But we ALSO try non-standard compositions the analytical approach missed:
  - Two mul16 in sequence: MUL16[M1] → MUL16[M2] → shift
    (computes A * M1 * M2 >> S — might find shorter for some K)
  - MUL16[M] → extract → MUL8-style further transforms
  - Shift-then-multiply: SRL first, then mul16
"""

import json
import sys

def load_mul16():
    with open("data/mulopt16_complete.json") as f:
        return {e["k"]: e for e in json.load(f)}

MUL16 = load_mul16()

def verify(k, func):
    """Verify func(a) == a // k for all a in 0..255."""
    for a in range(256):
        if func(a) != a // k:
            return False
    return True

def search_all():
    """Exhaustive structured search for all K=2..255."""

    with open("data/div8_optimal.json") as f:
        analytical = {e["k"]: e for e in json.load(f)["entries"]}

    results = []
    improvements = 0

    for k in range(2, 256):
        best_t = analytical[k]["tstates"]
        best_ops = analytical[k]["length"]
        best_method = analytical[k].get("method", "?")
        best_desc = None
        found_improvement = False

        # ── Method 1: Standard mul-shift (same as analytical, but double-check) ──
        for m in range(2, 256):
            if m not in MUL16:
                continue
            mul_entry = MUL16[m]
            mul_t = mul_entry["tstates"]
            mul_len = mul_entry["length"]

            for s in range(8, 20):
                if verify(k, lambda a, m=m, s=s: (a * m) >> s):
                    # Cost: LD H,0 (7T) + LD L,A (4T) + mul16 + LD A,H (4T) + SRL×(s-8)
                    extra_srl = s - 8
                    total_t = 7 + 4 + mul_t + 4 + 8 * extra_srl
                    total_ops = 2 + mul_len + 1 + extra_srl
                    if total_t < best_t:
                        best_t = total_t
                        best_ops = total_ops
                        best_desc = f"STANDARD: A×{m}>>{s}, {total_ops}ops {total_t}T (was {analytical[k]['tstates']}T)"
                        found_improvement = True
                    break  # smallest S wins for this M

        # ── Method 2: Add-256 (M = 256+R) ──
        for r in range(1, 255):
            if r not in MUL16:
                continue
            mul_entry = MUL16[r]
            mul_t = mul_entry["tstates"]
            mul_len = mul_entry["length"]
            b_clobbered = "B" in mul_entry.get("clobber", [])

            for s in range(8, 20):
                m = 256 + r
                if verify(k, lambda a, m=m, s=s: (a * m) >> s):
                    # Cost depends on clobber
                    if not b_clobbered:
                        # SAVE_B + PREAMBLE + mul16[R] + ADD_H_B + SRL×(s-8)
                        preamble_t = 4 + 7 + 4  # LD B,A + LD H,0 + LD L,A
                        add_t = 4 + 4  # LD A,H + ADD A,B
                        preamble_ops = 3
                        add_ops = 2
                    else:
                        d_clobbered = "D" in mul_entry.get("clobber", [])
                        if not d_clobbered:
                            preamble_t = 4 + 7 + 4
                            add_t = 4 + 4
                            preamble_ops = 3
                            add_ops = 2
                        else:
                            preamble_t = 8 + 7 + 4  # LD IXL,A + LD H,0 + LD L,A
                            add_t = 4 + 8  # LD A,H + ADD A,IXL
                            preamble_ops = 3
                            add_ops = 2

                    extra_srl = s - 8
                    total_t = preamble_t + mul_t + add_t + 8 * extra_srl
                    total_ops = preamble_ops + mul_len + add_ops + extra_srl

                    if total_t < best_t:
                        best_t = total_t
                        best_ops = total_ops
                        best_desc = f"ADD256: A×{m}>>{s} (R={r}), {total_ops}ops {total_t}T (was {analytical[k]['tstates']}T)"
                        found_improvement = True
                    break

        # ── Method 3: Pre-shift then multiply ──
        # For some K: (A >> P) * M >> S = A // K
        # This works when K = 2^P * Q and we can divide by Q after pre-shifting
        for pre_shift in range(1, 8):
            for m in range(2, 256):
                if m not in MUL16:
                    continue
                mul_entry = MUL16[m]
                mul_t = mul_entry["tstates"]
                mul_len = mul_entry["length"]

                for s in range(8, 20):
                    if verify(k, lambda a, ps=pre_shift, m=m, s=s: ((a >> ps) * m) >> s):
                        # Cost: SRL×P + LD H,0 + LD L,A + mul16 + LD A,H + SRL×(s-8)
                        extra_srl = s - 8
                        total_t = 8 * pre_shift + 7 + 4 + mul_t + 4 + 8 * extra_srl
                        total_ops = pre_shift + 2 + mul_len + 1 + extra_srl

                        if total_t < best_t:
                            best_t = total_t
                            best_ops = total_ops
                            best_desc = f"PRESHIFT: (A>>{pre_shift})×{m}>>{s}, {total_ops}ops {total_t}T (was {analytical[k]['tstates']}T)"
                            found_improvement = True
                        break  # smallest S
                if found_improvement and best_desc and "PRESHIFT" in best_desc:
                    break
            if found_improvement and best_desc and "PRESHIFT" in best_desc:
                break

        # ── Method 4: Round-down with correction ──
        # q = ((A*M)>>8 + ((A - (A*M)>>8) >> 1)) >> (S-9)
        # Hacker's Delight method for when simple mul-shift needs M > 2^N
        for m in range(2, 256):
            if m not in MUL16:
                continue
            mul_entry = MUL16[m]
            mul_t = mul_entry["tstates"]
            mul_len = mul_entry["length"]
            b_clobbered = "B" in mul_entry.get("clobber", [])

            for s in range(9, 20):
                def round_down_func(a, m=m, s=s):
                    t = (a * m) >> 8
                    return (t + ((a - t) >> 1)) >> (s - 9)

                if verify(k, round_down_func):
                    # Cost: SAVE + PREAMBLE + mul16 + correction + SRL×extra
                    # correction: LD A,saved; SUB H; SRL A; ADD A,H = 4+4+8+4 = 20T, 4 ops
                    if not b_clobbered:
                        save_t = 4  # LD B,A
                        correction_t = 4 + 4 + 8 + 4  # LD A,B; SUB H; SRL A; ADD A,H
                    else:
                        save_t = 4  # LD D,A
                        correction_t = 4 + 4 + 8 + 4
                    preamble_t = 7 + 4  # LD H,0; LD L,A
                    extra_srl = s - 9
                    total_t = save_t + preamble_t + mul_t + correction_t + 8 * extra_srl
                    total_ops = 1 + 2 + mul_len + 4 + extra_srl

                    if total_t < best_t:
                        best_t = total_t
                        best_ops = total_ops
                        best_desc = f"ROUNDDOWN: A×{m}>>{s} + correction, {total_ops}ops {total_t}T (was {analytical[k]['tstates']}T)"
                        found_improvement = True
                    break

        # ── Method 5: Double-multiply (two sequential mul16) ──
        # HL = A * M1 * M2 then >> S
        # This might find shorter paths where M1*M2 has a cheaper decomposition
        # Only try if analytical is expensive (>120T)
        if analytical[k]["tstates"] > 120:
            for m1 in [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128]:
                if m1 not in MUL16:
                    continue
                for m2 in range(2, 256):
                    if m2 not in MUL16:
                        continue
                    m_eff = m1 * m2
                    if m_eff > 65535:
                        continue
                    for s in range(8, 24):
                        if verify(k, lambda a, me=m_eff, s=s: (a * me) >> s):
                            mul1 = MUL16[m1]
                            mul2 = MUL16[m2]
                            total_t = 7 + 4 + mul1["tstates"] + mul2["tstates"] + 4 + 8 * max(0, s - 8)
                            total_ops = 2 + mul1["length"] + mul2["length"] + 1 + max(0, s - 8)

                            if total_t < best_t:
                                best_t = total_t
                                best_ops = total_ops
                                best_desc = f"DOUBLE_MUL: A×{m1}×{m2}>>{s} (eff {m_eff}), {total_ops}ops {total_t}T (was {analytical[k]['tstates']}T)"
                                found_improvement = True
                            break

        if found_improvement:
            improvements += 1
            print(f"div{k}: {best_desc}")

        results.append({
            "k": k,
            "best_tstates": best_t,
            "best_ops": best_ops,
            "analytical_tstates": analytical[k]["tstates"],
            "analytical_ops": analytical[k]["length"],
            "improved": found_improvement,
            "desc": best_desc
        })

    # Summary
    print(f"\n=== Summary ===")
    print(f"Improvements found: {improvements}/254")

    if improvements > 0:
        print(f"\nAll improvements:")
        for r in results:
            if r["improved"]:
                delta = r["analytical_tstates"] - r["best_tstates"]
                print(f"  div{r['k']}: {r['analytical_tstates']}T → {r['best_tstates']}T (save {delta}T) — {r['desc']}")

    # Confirmation stats
    confirmed = sum(1 for r in results if not r["improved"])
    print(f"Confirmed optimal: {confirmed}/254")

    total_analytical = sum(r["analytical_tstates"] for r in results)
    total_best = sum(r["best_tstates"] for r in results)
    print(f"Total T-states: analytical={total_analytical}, best={total_best}, saved={total_analytical - total_best}")

if __name__ == "__main__":
    search_all()

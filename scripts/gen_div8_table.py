#!/usr/bin/env python3
"""Generate div8_optimal.json: analytical multiply-and-shift division table.

For each K=2..255, finds magic M and shift S such that:
  floor(A * M / 2^S) = floor(A / K)  for all A in 0..255

Then composes the Z80 sequence from mul16 table:
  LD H,0; LD L,A     (preamble: HL = A as u16)
  mul16[M] ops        (HL = A * M, 16-bit)
  LD A,H              (get high byte = product >> 8)
  SRL A × (S-8)       (finish the shift)

Special cases:
  K = power of 2: just SRL A repeated (no multiply needed)
"""

import json
import sys
import math

def find_magic(k):
    """Find (M, S) pair for unsigned 8-bit div by K.

    We need: floor(A * M / 2^S) = floor(A / K) for all A in 0..255.
    M must be in 2..255 (so we can use mul16 table).
    S must be >= 8 (so result is in high byte or above).

    Strategy: try formula first, then exhaustive scan of all (S, M).
    Prefer smallest S (fewest SRL instructions), then smallest M.
    """
    # Try formula first for each S
    for s in range(8, 20):
        m = math.ceil((1 << s) / k)
        if 2 <= m <= 255:
            if _verify(k, m, s):
                return (m, s)

    # Exhaustive: try ALL M for each S
    for s in range(8, 20):
        for m in range(2, 256):
            if _verify(k, m, s):
                return (m, s)

    return None


def find_magic_extended(k):
    """Extended search: allows M up to 510 using add-shift trick.

    For M > 255, we decompose: A*M = A*(256+R) = A*256 + A*R = (A<<8) + A*R
    In 16-bit: we can compute A*R via mul16[R], then add A<<8 (just add A to H).

    Also tries: floor-based formula with correction step.
    """
    # First try standard method
    result = find_magic(k)
    if result:
        return ("standard", result)

    # Try M in 256..510 (decomposed as 256+R, R=1..254)
    for s in range(8, 20):
        for r in range(1, 255):
            m = 256 + r
            ok = True
            for a in range(256):
                if (a * m) >> s != a // k:
                    ok = False
                    break
            if ok:
                return ("add256", (r, s))

    # Try add-half correction: q = ((A*M)>>8 + A) >> (S-7)
    # This handles the "round-down" case from Hacker's Delight
    for s in range(9, 20):
        for m in range(2, 256):
            ok = True
            for a in range(256):
                # t = (A*M) >> 8; q = (t + ((A - t) >> 1)) >> (s - 9)
                t = (a * m) >> 8
                q = (t + ((a - t) >> 1)) >> (s - 9)
                if q != a // k:
                    ok = False
                    break
            if ok:
                return ("round_down", (m, s))

    return None


def _verify(k, m, s):
    for a in range(256):
        if (a * m) >> s != a // k:
            return False
    return True

def is_power_of_2(k):
    return k > 0 and (k & (k - 1)) == 0

def log2(k):
    n = 0
    while (1 << n) < k:
        n += 1
    return n

def main():
    # Load mul16 table
    with open("data/mulopt16_complete.json") as f:
        mul16_data = json.load(f)

    mul16 = {}
    for entry in mul16_data:
        mul16[entry["k"]] = entry

    results = []
    missing_mul = []
    no_magic = []

    for k in range(2, 256):
        # Special case: power of 2
        if is_power_of_2(k):
            n = log2(k)
            ops = ["SRL A"] * n
            results.append({
                "k": k,
                "method": "shift",
                "magic_m": None,
                "shift_s": n,
                "preamble": [],
                "mul_ops": [],
                "postamble": ops,
                "ops": ops,
                "length": n,
                "tstates": 8 * n,  # SRL A = 8T each
                "bytes": 2 * n,    # SRL A = CB 3F = 2 bytes each
                "clobbers": ["A", "F"],
                "verified": True,
                "notes": f"div{k} = {n}× SRL A"
            })
            continue

        # Find magic multiply-and-shift
        ext_result = find_magic_extended(k)
        if ext_result is None:
            no_magic.append(k)
            continue

        method_type, params = ext_result

        if method_type == "standard":
            m, s = params
            if m not in mul16:
                missing_mul.append((k, m, s))
                continue

            mul_entry = mul16[m]
            preamble = ["LD H,0", "LD L,A"]
            preamble_t = 7 + 4  # LD H,n=7T, LD L,A=4T

            mul_ops = mul_entry["ops"]
            mul_t = mul_entry["tstates"]

            extra_shifts = s - 8
            postamble = ["LD A,H"]  # 4T
            postamble_t = 4
            for _ in range(extra_shifts):
                postamble.append("SRL A")
                postamble_t += 8

            all_ops = preamble + mul_ops + postamble
            total_t = preamble_t + mul_t + postamble_t

            clobber_set = {"A", "F", "H", "L"}
            for c in mul_entry.get("clobber", []):
                clobber_set.add(c)

            results.append({
                "k": k,
                "method": "mul_shift",
                "magic_m": m,
                "shift_s": s,
                "preamble": preamble,
                "mul_ops": mul_ops,
                "postamble": postamble,
                "ops": all_ops,
                "length": len(all_ops),
                "tstates": total_t,
                "mul_length": mul_entry["length"],
                "mul_tstates": mul_t,
                "clobbers": sorted(clobber_set),
                "verified": True,
                "notes": f"div{k} = A×{m}>>{s}"
            })

        elif method_type == "add256":
            r, s = params
            # A*(256+R) = A*256 + A*R. In HL: mul16[R] gives HL=A*R, then ADD A to H
            if r not in mul16:
                missing_mul.append((k, r, s))
                continue

            mul_entry = mul16[r]
            # Preamble: save A, set HL=A
            preamble = ["LD B,A", "LD H,0", "LD L,A"]  # save A in B
            preamble_t = 4 + 7 + 4  # 15T

            mul_ops = mul_entry["ops"]
            mul_t = mul_entry["tstates"]

            # After mul16[R]: HL = A*R. Now add A*256 = add A to H
            # B still has original A (if mul doesn't clobber B)
            b_clobbered = "B" in mul_entry.get("clobber", [])

            # After mul16[R]: H has high byte of A*R. Add original A to get
            # high byte of A*(256+R). Then shift right to get quotient.
            # Optimization: midamble leaves result in A, skip redundant LD H,A; LD A,H
            if b_clobbered:
                preamble = ["LD D,A", "LD H,0", "LD L,A"]
                preamble_t = 4 + 7 + 4

                d_clobbered = "D" in mul_entry.get("clobber", [])
                if d_clobbered:
                    preamble = ["LD IXL,A", "LD H,0", "LD L,A"]
                    preamble_t = 8 + 7 + 4
                    midamble = ["LD A,H", "ADD A,IXL"]
                    midamble_t = 4 + 8
                else:
                    midamble = ["LD A,H", "ADD A,D"]
                    midamble_t = 4 + 4
            else:
                midamble = ["LD A,H", "ADD A,B"]
                midamble_t = 4 + 4  # 8T

            # A now has high byte of A*(256+R). Shift right by (S-8).
            extra_shifts = s - 8
            postamble = []
            postamble_t = 0
            for _ in range(extra_shifts):
                postamble.append("SRL A")
                postamble_t += 8

            all_ops = preamble + mul_ops + midamble + postamble
            total_t = preamble_t + mul_t + midamble_t + postamble_t

            clobber_set = {"A", "F", "H", "L"}
            for c in mul_entry.get("clobber", []):
                clobber_set.add(c)
            if not b_clobbered:
                clobber_set.add("B")
            elif "D" not in mul_entry.get("clobber", []):
                clobber_set.add("D")
            else:
                clobber_set.add("IXL")

            results.append({
                "k": k,
                "method": "mul_add256_shift",
                "magic_m": 256 + r,
                "magic_r": r,
                "shift_s": s,
                "preamble": preamble,
                "mul_ops": mul_ops,
                "midamble": midamble,
                "postamble": postamble,
                "ops": all_ops,
                "length": len(all_ops),
                "tstates": total_t,
                "mul_length": mul_entry["length"],
                "mul_tstates": mul_t,
                "clobbers": sorted(clobber_set),
                "verified": True,
                "notes": f"div{k} = A×{256+r}>>{s} (via A×{r} + A<<8)"
            })

        elif method_type == "round_down":
            m, s = params
            if m not in mul16:
                missing_mul.append((k, m, s))
                continue

            mul_entry = mul16[m]
            # Hacker's Delight round-down method:
            # t = (A*M) >> 8
            # q = (t + ((A - t) >> 1)) >> (s - 9)
            #
            # Z80 implementation:
            # LD B,A; LD H,0; LD L,A          (save A in B, HL=A)
            # mul16[M]                          (HL = A*M)
            # LD A,B; SUB H; SRL A; ADD A,H    (A = ((B - H) >> 1) + H = (A-t)/2 + t)
            # SRL A × (s-9)                    (finish shift)

            preamble = ["LD B,A", "LD H,0", "LD L,A"]
            preamble_t = 4 + 7 + 4  # 15T

            mul_ops = mul_entry["ops"]
            mul_t = mul_entry["tstates"]

            b_clobbered = "B" in mul_entry.get("clobber", [])
            if b_clobbered:
                # Use D instead
                preamble = ["LD D,A", "LD H,0", "LD L,A"]
                preamble_t = 4 + 7 + 4
                d_clobbered = "D" in mul_entry.get("clobber", [])
                if d_clobbered:
                    preamble = ["LD IXL,A", "LD H,0", "LD L,A"]
                    preamble_t = 8 + 7 + 4
                    correction = ["LD A,IXL", "SUB H", "SRL A", "ADD A,H"]
                    correction_t = 8 + 4 + 8 + 4  # 24T
                else:
                    correction = ["LD A,D", "SUB H", "SRL A", "ADD A,H"]
                    correction_t = 4 + 4 + 8 + 4  # 20T
            else:
                correction = ["LD A,B", "SUB H", "SRL A", "ADD A,H"]
                correction_t = 4 + 4 + 8 + 4  # 20T

            extra_shifts = s - 9
            postamble = []
            postamble_t = 0
            for _ in range(extra_shifts):
                postamble.append("SRL A")
                postamble_t += 8

            all_ops = preamble + mul_ops + correction + postamble
            total_t = preamble_t + mul_t + correction_t + postamble_t

            clobber_set = {"A", "F", "H", "L"}
            for c in mul_entry.get("clobber", []):
                clobber_set.add(c)
            if not b_clobbered:
                clobber_set.add("B")

            results.append({
                "k": k,
                "method": "round_down",
                "magic_m": m,
                "shift_s": s,
                "preamble": preamble,
                "mul_ops": mul_ops,
                "correction": correction,
                "postamble": postamble,
                "ops": all_ops,
                "length": len(all_ops),
                "tstates": total_t,
                "mul_length": mul_entry["length"],
                "mul_tstates": mul_t,
                "clobbers": sorted(clobber_set),
                "verified": True,
                "notes": f"div{k} = round_down(A×{m}, {s})"
            })

    # Verify all results by simulating the actual math
    verify_errors = []
    for r in results:
        k = r["k"]
        for a in range(256):
            actual = a // k

            if r["method"] == "shift":
                computed = a >> r["shift_s"]
            elif r["method"] in ("mul_shift", "mul_add256_shift"):
                computed = (a * r["magic_m"]) >> r["shift_s"]
            elif r["method"] == "round_down":
                m, s = r["magic_m"], r["shift_s"]
                t = (a * m) >> 8
                computed = (t + ((a - t) >> 1)) >> (s - 9)
            else:
                computed = -1

            if computed != actual:
                verify_errors.append((k, a, computed, actual, r["method"]))
                r["verified"] = False
                break

    # Summary
    print(f"Results: {len(results)}/254 divisors", file=sys.stderr)
    if missing_mul:
        print(f"Missing mul16 entries: {len(missing_mul)}", file=sys.stderr)
        for k, m, s in missing_mul[:10]:
            print(f"  div{k} needs mul16[{m}] (shift {s})", file=sys.stderr)
    if no_magic:
        print(f"No magic found: {len(no_magic)} — {no_magic[:20]}", file=sys.stderr)
    if verify_errors:
        print(f"Verification FAILED for {len(verify_errors)} entries!", file=sys.stderr)
        for k, a, exp, act in verify_errors[:5]:
            print(f"  div{k}: A={a}, got {exp}, expected {act}", file=sys.stderr)

    # Stats
    shifts_only = sum(1 for r in results if r["method"] == "shift")
    mul_shifts = sum(1 for r in results if r["method"] == "mul_shift")
    print(f"  Power-of-2 (shift only): {shifts_only}", file=sys.stderr)
    print(f"  Multiply-and-shift: {mul_shifts}", file=sys.stderr)

    if mul_shifts > 0:
        ts = [r["tstates"] for r in results if r["method"] == "mul_shift"]
        print(f"  T-states range: {min(ts)}-{max(ts)} (avg {sum(ts)/len(ts):.0f})", file=sys.stderr)

    # Write output
    output = {
        "description": "Optimal u8 division sequences for Z80 via multiply-and-shift",
        "method": "A÷K = (A×M) >> S, composed from mul16 table + SRL chain",
        "convention": "Input: A = dividend. Output: A = quotient = floor(A/K). Clobbers: H,L + mul clobbers.",
        "total": len(results),
        "coverage": f"{len(results)}/254",
        "entries": results
    }

    with open("data/div8_optimal.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten to data/div8_optimal.json", file=sys.stderr)

if __name__ == "__main__":
    main()

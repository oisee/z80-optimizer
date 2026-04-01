# Z80 Library Unit Test Design Proposal

**Date:** 2026-04-01  
**Status:** draft / proposal for mze/mzx integration

---

## Problem Statement

The mul8/mul16/div8 library sequences have discovered bugs:

1. **`div8_optimal.json` k=5**: ops implement `(n×105)>>10 ≈ n/10`, not `n/5`. **251/256 inputs wrong** (e.g. n=137 → got 14, want 27).
2. **k=10**: correctly implements `(n>>1)×103>>9 ≈ n/10`. ✓
3. **mul8 carry dependency**: sequences using `RLA`/`RLCA` include the incoming carry flag in the result. The GPU searcher (`z80_mulopt_fast.cu`) tested only with `B=0, CY=0` — results are incorrect if called with `CY≠0`.

The root cause: GPU brute-force verified sequences in isolation with clean initial state. Real code can call these routines with arbitrary `B` and `CY`.

---

## Proposed Test CLI: `mzx --test`

Extend `mzx` with a deterministic test mode:

```bash
# Single test: set registers, run entry point, check expected
mzx --run mul8_library.bin --entry mul_15 \
    --set a=10 --set b=0 --set f=0 \
    --exp a=150
# exit 0 = pass, exit 1 = fail

# Exhaustive sweep: all A=0..255
mzx --run mul8_library.bin --entry mul_15 \
    --set b=0 --set f=0 \
    --sweep a=0..255 \
    --exp "a == (a_in * 15) & 0xFF"
# exit 0 if all 256 pass

# Full carry test: A=0..255 × CY=0,1 (512 inputs)
mzx --run mul8_library.bin --entry mul_15 \
    --set b=0 \
    --sweep a=0..255,f=0..1 \
    --exp "a == (a_in * 15) & 0xFF"
```

### Register naming

| Flag | Meaning |
|------|---------|
| `--set r=V` | Set register `r` to value `V` before run |
| `--sweep r=lo..hi` | Iterate register over range |
| `--exp r=V` | Assert register `r` equals `V` after halt |
| `--exp "expr"` | Evaluate expression (uses `a_in`, `b_in`, etc. for pre-call values) |
| `--entry label` | Jump to label (resolved from `.sym` file or label in binary) |
| `--json` | Output JSON report instead of exit code |

---

## Test Generation from JSON Tables

Script: `scripts/gen_mzx_tests.py`

```python
# Generate test cases from mulopt8_clobber.json
for k in range(2, 256):
    entry = mul8_table[k]
    # Test: exhaustive A=0..255, B=0, CY=0
    tests.append({
        "entry": f"mul_{k}",
        "sweep": {"a": range(256)},
        "set": {"b": 0, "f": 0},   # CY=0
        "exp": "a == (a_in * k) & 0xFF"
    })
    # Test: carry sensitivity (if ops contain RLA/RLCA → must pass with CY=1 too)
    if any("RLA" in op or "RLCA" in op for op in entry["ops"]):
        tests.append({
            "entry": f"mul_{k}",
            "note": "carry_sensitive",
            "sweep": {"a": range(256), "f": [0, 1]},  # CY=0 and CY=1
            "set": {"b": 0},
            "exp": "a == (a_in * k) & 0xFF"
        })
```

---

## Carry Sensitivity Audit

Sequences using `RLA` or `RLCA` are carry-sensitive and must either:

**Option A** (safe): Replace `RLCA` → `ADD A,A`, `RLA` → `ADD A,A` in the generator  
- `ADD A,A` = `RLCA` but ignores incoming CY (4T, same cost)
- `SUB B` = `SBC A,B` but ignores incoming CY (4T, same cost)

**Option B** (documented): Mark sequences as `"carry_in": "must_be_0"` in JSON, emit `AND A` or `OR A` (resets CY, 4T) as preamble in library generator

Recommendation: **Option A** — fix the ops in the JSON tables. The optimizer found these sequences assuming CY=0 anyway; replacing RLA→ADD A,A gives identical results under that assumption and is safe for all CY.

Scan from current tables:
```bash
python3 -c "
import json
data = json.load(open('data/mulopt8_clobber.json'))
for k,entry in data.items():
    ops = entry.get('ops', [])
    if any('RLA' in op for op in ops):
        print(f'k={k}: carry-sensitive ops: {[o for o in ops if \"RL\" in o]}')
"
```

---

## div8 k=5 Fix

The k=5 entry computes `(n×105)>>10 ≈ n/10` (wrong). Correct formula:

```
n/5 ≈ (n × 205) >> 10    [205/1024 = 0.2002...]
```

Or equivalently (preshift approach, same T-states):
```
n/5 ≈ (n × 103) >> 9     [103/512 = 0.2012...]  -- slightly less accurate
```

Re-run GPU search for k=5:
```bash
cuda/z80_divmod_fast --div 5 --max-len 12 --json
```

Verify fix in 1 line:
```python
assert all((205 * n) >> 10 == n // 5 for n in range(256))  # True
```

---

## CI Integration

```makefile
# Makefile target
test-mul8:
    python3 scripts/gen_mzx_tests.py --table data/mulopt8_clobber.json --out /tmp/mul8_tests.json
    mzx --run data/mul8_library.bin --batch /tmp/mul8_tests.json --json > /tmp/mul8_results.json
    python3 scripts/check_results.py /tmp/mul8_results.json

test-div8:
    python3 scripts/gen_mzx_tests.py --table data/div8_optimal.json --out /tmp/div8_tests.json
    mzx --run data/div8_library.bin --batch /tmp/div8_tests.json --json > /tmp/div8_results.json
    python3 scripts/check_results.py /tmp/div8_results.json
```

---

## Summary of Bugs Found (April 1, 2026)

| File | Bug | Severity |
|------|-----|----------|
| `div8_optimal.json` k=5 | ops implement n/10 not n/5, 251/256 wrong | **CRITICAL** |
| `z80_mulopt_fast.cu` L158-159 | tests only B=0, CY=0; carry-sensitive ops pass incorrectly | HIGH |
| `mul8_library.asm` (RLA-using sequences) | incorrect if called with CY≠0 | MEDIUM |

Thanks: Joaquín Ferrero for finding and reporting these. Filed as issues on z80-optimizer.

---

## Validation Approach (until mzx test mode exists)

Quick Python validator using our own Z80 simulator (`pkg/cpu`):
```bash
# Go test: exhaustive verify all 254 constants, all 256 inputs, B=0..255, CY=0..1
CGO_ENABLED=0 ~/go/bin/go1.24.3 test ./pkg/mulopt/ -run TestMul8Exhaustive -v
```

Or standalone:
```bash
# Use existing z80opt verify-jsonl infrastructure  
python3 scripts/verify_library.py --lib data/mulopt8_clobber.json --all-inputs --all-carry
```

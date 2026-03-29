#!/bin/bash
# Run divmod brute-force for all K=2..255
# Output: data/divmod8_results.jsonl (one JSON per line)
# Usage: bash cuda/run_divmod_all.sh [--max-len 8] [--gpu-id 0]

MAX_LEN=${1:-8}
OUTDIR="data"
mkdir -p "$OUTDIR"

DIVOUT="$OUTDIR/div8_results.jsonl"
MODOUT="$OUTDIR/mod8_results.jsonl"
DMOUT="$OUTDIR/divmod8_results.jsonl"

> "$DIVOUT"
> "$MODOUT"
> "$DMOUT"

echo "=== div8 K=2..255, max-len=$MAX_LEN ==="
for K in $(seq 2 255); do
    RESULT=$(./cuda/z80_divmod_fast --div $K --json --max-len $MAX_LEN 2>/dev/null)
    if [ -n "$RESULT" ]; then
        echo "$RESULT" >> "$DIVOUT"
        if [ $((K % 10)) -eq 0 ]; then
            echo "  div K=$K done"
        fi
    else
        echo "  div K=$K: no result at len $MAX_LEN"
    fi
done
echo "div8: $(wc -l < $DIVOUT) results → $DIVOUT"

echo "=== mod8 K=2..255 ==="
for K in $(seq 2 255); do
    RESULT=$(./cuda/z80_divmod_fast --mod $K --json --max-len $MAX_LEN 2>/dev/null)
    if [ -n "$RESULT" ]; then
        echo "$RESULT" >> "$MODOUT"
        if [ $((K % 10)) -eq 0 ]; then
            echo "  mod K=$K done"
        fi
    fi
done
echo "mod8: $(wc -l < $MODOUT) results → $MODOUT"

echo "=== divmod8 K=2..255 ==="
for K in $(seq 2 255); do
    RESULT=$(./cuda/z80_divmod_fast --divmod $K --json --max-len $MAX_LEN 2>/dev/null)
    if [ -n "$RESULT" ]; then
        echo "$RESULT" >> "$DMOUT"
        if [ $((K % 10)) -eq 0 ]; then
            echo "  divmod K=$K done"
        fi
    fi
done
echo "divmod8: $(wc -l < $DMOUT) results → $DMOUT"

echo "=== DONE ==="
wc -l "$DIVOUT" "$MODOUT" "$DMOUT"

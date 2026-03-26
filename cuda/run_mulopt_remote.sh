#!/bin/bash
# Self-contained mulopt search for remote GPU (e.g. RTX 2070)
#
# Usage:
#   1. Copy this file + z80_mulopt_fast.cu to the remote machine
#   2. ssh remote 'bash run_mulopt_remote.sh'
#
# Or from the z80-optimizer repo:
#   scp cuda/z80_mulopt_fast.cu cuda/run_mulopt_remote.sh i5:~/
#   ssh i5 'bash run_mulopt_remote.sh'

set -e

MAX_LEN=${1:-9}  # default: search length 9

echo "=== Z80 mulopt remote search ==="
echo "Max length: $MAX_LEN"
echo ""

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found."
    echo "Install: sudo apt install nvidia-cuda-toolkit"
    echo "Or: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA: $(nvcc --version 2>&1 | grep release)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo ""

# Build
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/z80_mulopt_fast.cu"
BIN="$SCRIPT_DIR/z80_mulopt_fast"

if [ ! -f "$SRC" ]; then
    SRC="$HOME/z80_mulopt_fast.cu"
fi
if [ ! -f "$SRC" ]; then
    echo "ERROR: z80_mulopt_fast.cu not found"
    exit 1
fi

echo "Building z80_mulopt_fast (14-op reduced pool)..."
nvcc -O3 -o "$BIN" "$SRC"
echo "Built: $BIN"
echo ""

# Run
OUTFILE="mulopt_len${MAX_LEN}_results.jsonl"
echo "Starting search (max-len $MAX_LEN, 14 ops)..."
echo "Results: $OUTFILE"
echo "14 ops → 14^$MAX_LEN = $(python3 -c "print(f'{14**$MAX_LEN:.2e}')" 2>/dev/null || echo '?') candidates per constant"
echo ""

"$BIN" --max-len "$MAX_LEN" --json > "$OUTFILE" 2>&1

echo ""
echo "Done! Results in $OUTFILE"
FOUND=$(grep -c '"length"' "$OUTFILE" 2>/dev/null || echo 0)
echo "Constants solved: $FOUND / 254"

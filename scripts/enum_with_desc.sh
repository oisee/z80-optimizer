#!/bin/bash
# Pipe regalloc-enum through GPU server, output {"desc": ..., "result": ...} pairs
# Usage: ./scripts/enum_with_desc.sh [--max-vregs N] [--gpu 0]

MAX_VREGS=${1:-4}
GPU=${2:-0}

ENUM=./regalloc-enum
SERVER=./cuda/z80_regalloc

# Generate patterns to temp file
TMPDIR=$(mktemp -d)
$ENUM --max-vregs "$MAX_VREGS" > "$TMPDIR/descs.jsonl" 2>/dev/null

# Run through GPU server
CUDA_VISIBLE_DEVICES=$GPU $SERVER --server < "$TMPDIR/descs.jsonl" > "$TMPDIR/results.jsonl" 2>/dev/null

# Merge desc + result line by line
paste -d'\n' "$TMPDIR/descs.jsonl" "$TMPDIR/results.jsonl" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
for i in range(0, len(lines), 2):
    if i+1 < len(lines):
        desc = json.loads(lines[i])
        result = json.loads(lines[i+1])
        print(json.dumps({'desc': desc, 'result': result}))
"

rm -rf "$TMPDIR"

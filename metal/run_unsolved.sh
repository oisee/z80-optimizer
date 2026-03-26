#!/bin/bash
# Run Metal mulopt for unsolved constants only (len-8), merge with previous results
set -e

PREV=${1:-/tmp/mulopt7.json}
OUT=${2:-/tmp/mulopt8_metal.json}
MAX_LEN=${3:-8}

# Get unsolved list
UNSOLVED=$(python3 -c "
import json, sys
with open('$PREV') as f:
    solved = {r['k'] for r in json.load(f)}
for k in range(2, 256):
    if k not in solved:
        print(k)
")

TOTAL=$(echo "$UNSOLVED" | wc -l | tr -d ' ')
echo "Running $TOTAL unsolved constants at len-$MAX_LEN via Metal..."

TMPDIR=$(mktemp -d)
I=0
for K in $UNSOLVED; do
    I=$((I+1))
    printf "\r%d/%d (x%d)..." "$I" "$TOTAL" "$K" >&2
    /tmp/metal_mulopt --k "$K" --max-len "$MAX_LEN" --json 2>/dev/null > "$TMPDIR/$K.json" || true
done
echo "" >&2

# Merge: previous + new results
python3 -c "
import json, glob, os, sys

with open('$PREV') as f:
    results = {r['k']: r for r in json.load(f)}

for f in glob.glob('$TMPDIR/*.json'):
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            for r in data:
                if r.get('k') and r['k'] not in results:
                    results[r['k']] = r
        elif isinstance(data, dict) and data.get('k'):
            if data['k'] not in results:
                results[data['k']] = data
    except: pass

out = sorted(results.values(), key=lambda r: r['k'])
json.dump(out, open('$OUT', 'w'), indent=2)
print(f'Merged: {len(out)}/254 solved -> $OUT', file=sys.stderr)
"

rm -rf "$TMPDIR"

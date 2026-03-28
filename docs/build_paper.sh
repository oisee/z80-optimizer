#!/bin/bash
# Build regalloc paper: Markdown → PDF/EPUB with rendered Mermaid diagrams
# Usage: bash docs/build_paper.sh

set -e
cd "$(dirname "$0")"

INPUT="regalloc_paper.md"
OUTPUT_PDF="regalloc_paper.pdf"
OUTPUT_EPUB="regalloc_paper.epub"
DIAGRAMS_DIR="/tmp/regalloc_diagrams"
PUPPETEER_CFG="/tmp/puppeteer.json"

echo "=== Building Register Allocation Paper ==="

# Puppeteer config for mmdc (no-sandbox for headless env)
cat > "$PUPPETEER_CFG" << 'EOF'
{ "args": ["--no-sandbox", "--disable-setuid-sandbox"] }
EOF

# Step 1: Extract and render Mermaid diagrams
mkdir -p "$DIAGRAMS_DIR"
echo "Extracting Mermaid diagrams..."

# Extract ```mermaid blocks, render each to SVG
python3 << 'PYEOF'
import re, subprocess, os

diagrams_dir = os.environ.get("DIAGRAMS_DIR", "/tmp/regalloc_diagrams")
input_file = os.environ.get("INPUT", "regalloc_paper.md")

with open(input_file) as f:
    content = f.read()

# Find all mermaid blocks
pattern = r'```mermaid\n(.*?)```'
matches = list(re.finditer(pattern, content, re.DOTALL))
print(f"Found {len(matches)} Mermaid diagrams")

modified = content
for i, m in enumerate(reversed(matches)):  # reverse to preserve positions
    diagram_code = m.group(1).strip()
    svg_name = f"diagram_{len(matches)-1-i:02d}.svg"
    svg_path = os.path.join(diagrams_dir, svg_name)

    # Write temp .mmd file
    mmd_path = os.path.join(diagrams_dir, f"temp_{i}.mmd")
    with open(mmd_path, 'w') as f:
        f.write(diagram_code)

    # Render with mmdc
    result = subprocess.run(
        ["mmdc", "-i", mmd_path, "-o", svg_path,
         "-p", "/tmp/puppeteer.json", "-b", "white", "-w", "800"],
        capture_output=True, text=True
    )

    if os.path.exists(svg_path):
        # Replace mermaid block with image reference
        img_ref = f"![Diagram {len(matches)-1-i}]({svg_path})"
        modified = modified[:m.start()] + img_ref + modified[m.end():]
        print(f"  [{len(matches)-1-i}] OK → {svg_name}")
    else:
        print(f"  [{len(matches)-1-i}] FAILED: {result.stderr[:100]}")
        # Keep as code block but change language to text
        modified = modified[:m.start()] + "```\n" + diagram_code + "\n```" + modified[m.end():]

# Write modified markdown
output_md = os.path.join(diagrams_dir, "paper_with_images.md")
with open(output_md, 'w') as f:
    f.write(modified)
print(f"Written: {output_md}")
PYEOF

PROCESSED_MD="$DIAGRAMS_DIR/paper_with_images.md"

if [ ! -f "$PROCESSED_MD" ]; then
    echo "Diagram processing failed, using original markdown"
    PROCESSED_MD="$INPUT"
fi

# Step 2: Build PDF
echo "Building PDF..."
pandoc "$PROCESSED_MD" \
    -o "$OUTPUT_PDF" \
    --pdf-engine=pdflatex \
    -V geometry:margin=2.5cm \
    -V fontsize=11pt \
    -V documentclass=article \
    -V title="Register Allocation as a Solved Game" \
    -V author="Alice \& Claude" \
    -V date="March 2026" \
    --toc \
    --highlight-style=tango \
    -V colorlinks=true \
    -V linkcolor=blue \
    2>&1 || echo "PDF build failed (LaTeX issues with SVG?), trying without images..."

if [ ! -f "$OUTPUT_PDF" ]; then
    echo "Fallback: build PDF from original markdown (no rendered diagrams)"
    pandoc "$INPUT" \
        -o "$OUTPUT_PDF" \
        --pdf-engine=pdflatex \
        -V geometry:margin=2.5cm \
        -V fontsize=11pt \
        -V documentclass=article \
        -V title="Register Allocation as a Solved Game" \
        -V author="Alice \& Claude" \
        -V date="March 2026" \
        --toc \
        --highlight-style=tango \
        2>&1
fi

# Step 3: Build EPUB
echo "Building EPUB..."
pandoc "$INPUT" \
    -o "$OUTPUT_EPUB" \
    --metadata title="Register Allocation as a Solved Game" \
    --metadata author="Alice & Claude" \
    --metadata date="2026-03-28" \
    --toc \
    2>&1

echo "=== Done ==="
ls -lh "$OUTPUT_PDF" "$OUTPUT_EPUB" 2>/dev/null

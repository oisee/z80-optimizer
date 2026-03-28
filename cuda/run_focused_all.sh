#!/bin/bash
# Launch focused sequential search on all GPUs
# i7 GPU0: log2_f3.5(3ops), bin2bcd(4ops), log2_x28(7ops), cbrt(8ops)
# i7 GPU1: recip(7ops), popcnt(7ops), log2_f4.4(10ops)
# i5:      same binary, bcd2bin(11ops), sqrt_f4.4(11ops), sqrt_f3.5(12ops)
#
# All targets are in ONE binary — each GPU runs the full pipeline sequentially.
# Split: GPU0 gets odds, GPU1 gets evens, i5 gets the big ones.

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Launching focused search on i7 GPU0 ==="
CUDA_VISIBLE_DEVICES=0 nohup "$DIR/z80_focused" --gpu 0 \
    > /tmp/focused_gpu0.txt 2>/tmp/focused_gpu0.log &
echo "PID: $!"

echo "=== Launching focused search on i7 GPU1 ==="
CUDA_VISIBLE_DEVICES=1 nohup "$DIR/z80_focused" --gpu 0 \
    > /tmp/focused_gpu1.txt 2>/tmp/focused_gpu1.log &
echo "PID: $!"

echo "=== Copying to i5 ==="
scp "$DIR/z80_focused" i5:~/z80_focused 2>/dev/null
ssh i5 "nohup ~/z80_focused --gpu 0 > /tmp/focused_i5.txt 2>/tmp/focused_i5.log &" 2>/dev/null
echo "i5 launched"

echo ""
echo "Monitor: tail -f /tmp/focused_gpu0.log"
echo "Results: cat /tmp/focused_gpu0.txt /tmp/focused_gpu1.txt"
echo "i5:      ssh i5 'cat /tmp/focused_i5.txt'"

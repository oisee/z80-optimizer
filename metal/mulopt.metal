// mulopt.metal — Metal compute shader for Z80 constant multiply search
//
// Port of cuda/z80_mulopt_fast.cu for Apple Silicon GPUs.
// Same 14-op reduced pool, same QuickCheck + full verification.

#include <metal_stdlib>
using namespace metal;

#define NUM_OPS 14

constant uchar opCost[NUM_OPS] = {
    4,4,4,4,4,4,4,4, // ADD/ADC/SBC/LD
    8,                // SRL A
    4,4,4,4,          // RLA/RRA/RLCA/RRCA
    8                 // NEG
};

static inline void exec_op(uchar op, thread uchar &a, thread uchar &b, thread bool &carry) {
    ushort r; uchar bit;
    switch (op) {
    case 0: // ADD A,A
        r = (ushort)a + a; carry = r > 0xFF; a = (uchar)r; break;
    case 1: // ADD A,B
        r = (ushort)a + b; carry = r > 0xFF; a = (uchar)r; break;
    case 2: // SUB B
        carry = (a < b); a = a - b; break;
    case 3: // LD B,A
        b = a; break;
    case 4: // ADC A,B
        r = (ushort)a + b + (carry ? 1 : 0); carry = r > 0xFF; a = (uchar)r; break;
    case 5: // ADC A,A
        r = (ushort)a + a + (carry ? 1 : 0); carry = r > 0xFF; a = (uchar)r; break;
    case 6: { // SBC A,B
        uchar c = carry ? 1 : 0;
        carry = ((short)a - (short)b - c) < 0;
        a = a - b - c; break;
    }
    case 7: { // SBC A,A
        uchar c = carry ? 1 : 0;
        carry = c > 0;
        a = -c; break;
    }
    case 8: // SRL A
        carry = (a & 1) != 0; a = a >> 1; break;
    case 9: // RLA
        bit = carry ? 1 : 0; carry = (a & 0x80) != 0; a = (a << 1) | bit; break;
    case 10: // RRA
        bit = carry ? 0x80 : 0; carry = (a & 1) != 0; a = (a >> 1) | bit; break;
    case 11: // RLCA
        carry = (a & 0x80) != 0; a = (a << 1) | (a >> 7); break;
    case 12: // RRCA
        carry = (a & 1) != 0; a = (a >> 1) | (a << 7); break;
    case 13: // NEG
        carry = (a != 0); a = (uchar)(0 - a); break;
    }
}

static inline uchar run_seq(thread uchar *ops, int len, uchar input) {
    uchar a = input, b = 0;
    bool carry = false;
    for (int i = 0; i < len; i++) exec_op(ops[i], a, b, carry);
    return a;
}

// Args buffer layout: { k (uchar), seqLen (int), offset (ulong), count (ulong) }
struct MuloptArgs {
    uint k;
    int seqLen;
    ulong offset;
    ulong count;
};

kernel void mulopt_kernel(
    constant MuloptArgs &args [[buffer(0)]],
    device atomic_uint *bestScore [[buffer(1)]],
    device atomic_uint *bestIdxLo [[buffer(2)]],
    device atomic_uint *bestIdxHi [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= args.count) return;

    ulong seqIdx = args.offset + tid;
    uchar ops[12];
    ulong tmp = seqIdx;
    for (int i = args.seqLen - 1; i >= 0; i--) {
        ops[i] = (uchar)(tmp % NUM_OPS);
        tmp /= NUM_OPS;
    }

    uchar k = (uchar)args.k;

    // QuickCheck: 4 test values
    if (run_seq(ops, args.seqLen, 1) != (uchar)(1 * k)) return;
    if (run_seq(ops, args.seqLen, 2) != (uchar)(2 * k)) return;
    if (run_seq(ops, args.seqLen, 127) != (uchar)(127 * k)) return;
    if (run_seq(ops, args.seqLen, 255) != (uchar)(255 * k)) return;

    // Full verification
    for (int input = 0; input < 256; input++) {
        if (run_seq(ops, args.seqLen, (uchar)input) != (uchar)(input * k)) return;
    }

    // Compute cost
    uint cost = 0;
    for (int i = 0; i < args.seqLen; i++) cost += opCost[ops[i]];
    uint score = ((uint)args.seqLen << 16) | cost;

    // atomic_min on score; if we win, store index
    uint old = atomic_fetch_min_explicit(bestScore, score, memory_order_relaxed);
    if (score <= old) {
        atomic_store_explicit(bestIdxLo, (uint)(seqIdx & 0xFFFFFFFF), memory_order_relaxed);
        atomic_store_explicit(bestIdxHi, (uint)(seqIdx >> 32), memory_order_relaxed);
    }
}

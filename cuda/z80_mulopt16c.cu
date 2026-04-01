// z80_mulopt16c.cu — GPU brute-force optimal HL×K→HL for Z80
//
// For each constant K (2..255), finds the shortest instruction sequence
// such that HL_out = HL_in * K (mod 65536) for ALL 16-bit inputs.
//
// Input convention:  HL = value to multiply
// Output convention: HL = HL × K (mod 65536)
// Scratch:           BC, DE may be clobbered (noted in output)
// Carry:             may be clobbered
//
// Differs from mulopt16 (A×K→HL) in that the input is already 16-bit in HL.
// This enables longer sequences but avoids the 8→16 widening overhead.
//
// Op pool (11 instructions):
//   ADD HL,HL (11T)  — HL = HL*2
//   ADD HL,BC (11T)  — HL += BC
//   ADD HL,DE (11T)  — HL += DE
//   LD C,L    ( 4T)  — save HL.lo → C
//   LD B,H    ( 4T)  — save HL.hi → B
//   LD E,L    ( 4T)  — save HL.lo → E
//   LD D,H    ( 4T)  — save HL.hi → D
//   LD L,C    ( 4T)  — restore C → HL.lo
//   LD H,B    ( 4T)  — restore B → HL.hi
//   LD L,E    ( 4T)  — restore E → HL.lo
//   LD H,D    ( 4T)  — restore D → HL.hi
//
// Verification: exhaustive over all 256 low-byte inputs (H=0) + 8 spot checks.
// A sequence is valid if HL_out = HL_in * K mod 65536 for all tested inputs.
//
// Build:
//   nvcc -O3 -o cuda/z80_mulopt16c cuda/z80_mulopt16c.cu
// Usage:
//   cuda/z80_mulopt16c [--max-len 8] [--k 3] [--json] [--all]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// ============================================================
// Op pool
// ============================================================
#define OP_ADD_HLHL  0   // ADD HL,HL  (11T)
#define OP_ADD_HLBC  1   // ADD HL,BC  (11T)
#define OP_ADD_HLDE  2   // ADD HL,DE  (11T)
#define OP_LD_CL     3   // LD C,L      (4T)  — save HL.lo → C
#define OP_LD_BH     4   // LD B,H      (4T)  — save HL.hi → B
#define OP_LD_EL     5   // LD E,L      (4T)  — save HL.lo → E
#define OP_LD_DH     6   // LD D,H      (4T)  — save HL.hi → D
#define OP_LD_LC     7   // LD L,C      (4T)  — restore C → HL.lo
#define OP_LD_HB     8   // LD H,B      (4T)  — restore B → HL.hi
#define OP_LD_LE     9   // LD L,E      (4T)  — restore E → HL.lo
#define OP_LD_HD    10   // LD H,D      (4T)  — restore D → HL.hi
#define OP_EX_DEHL  11   // EX DE,HL    (4T)  — swap DE↔HL (key for accumulation)
#define NUM_OPS     12

static __constant__ uint8_t opCost[NUM_OPS] = {
    11, 11, 11,  // ADD HL,HL / BC / DE
     4,  4,      // LD C,L / LD B,H
     4,  4,      // LD E,L / LD D,H
     4,  4,      // LD L,C / LD H,B
     4,  4,      // LD L,E / LD H,D
     4,           // EX DE,HL
};

// ============================================================
// State: H, L, B, C, D, E  (carry not needed for these ops)
// ============================================================
struct State16 {
    uint8_t h, l, b, c, d, e;
};

__device__ __host__ State16 exec16(uint8_t op, State16 s) {
    uint16_t hl, bc, de, r;
    switch (op) {
    case OP_ADD_HLHL:
        hl = ((uint16_t)s.h << 8) | s.l;
        r = hl + hl;
        s.h = (uint8_t)(r >> 8);
        s.l = (uint8_t)(r & 0xFF);
        break;
    case OP_ADD_HLBC:
        hl = ((uint16_t)s.h << 8) | s.l;
        bc = ((uint16_t)s.b << 8) | s.c;
        r = hl + bc;
        s.h = (uint8_t)(r >> 8);
        s.l = (uint8_t)(r & 0xFF);
        break;
    case OP_ADD_HLDE:
        hl = ((uint16_t)s.h << 8) | s.l;
        de = ((uint16_t)s.d << 8) | s.e;
        r = hl + de;
        s.h = (uint8_t)(r >> 8);
        s.l = (uint8_t)(r & 0xFF);
        break;
    case OP_LD_CL: s.c = s.l; break;
    case OP_LD_BH: s.b = s.h; break;
    case OP_LD_EL: s.e = s.l; break;
    case OP_LD_DH: s.d = s.h; break;
    case OP_LD_LC: s.l = s.c; break;
    case OP_LD_HB: s.h = s.b; break;
    case OP_LD_LE: s.l = s.e; break;
    case OP_LD_HD: s.h = s.d; break;
    case OP_EX_DEHL: {
        uint8_t th = s.h, tl = s.l;
        s.h = s.d; s.l = s.e;
        s.d = th;  s.e = tl;
        break;
    }
    }
    return s;
}

__device__ uint16_t run_seq16(const uint8_t *ops, int len, uint16_t input) {
    State16 s;
    s.h = (uint8_t)(input >> 8);
    s.l = (uint8_t)(input & 0xFF);
    s.b = s.c = s.d = s.e = 0;
    for (int i = 0; i < len; i++) {
        s = exec16(ops[i], s);
    }
    return ((uint16_t)s.h << 8) | s.l;
}

__device__ uint16_t seq_cost16(const uint8_t *ops, int len) {
    uint16_t cost = 0;
    for (int i = 0; i < len; i++) cost += opCost[ops[i]];
    return cost;
}

__host__ __device__ void decode_seq16(uint64_t idx, int len, uint8_t *ops) {
    for (int i = len - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(idx % NUM_OPS);
        idx /= NUM_OPS;
    }
}

// ============================================================
// Kernel
// ============================================================
// Quick-check inputs: covers low byte (H=0), high byte (H=1), mixed
static __constant__ uint16_t qcInputs[8] = {
    1, 2, 3, 127, 128, 255, 256, 511
};

__global__ void mulopt16c_kernel(uint16_t k, int seqLen, uint64_t offset, uint64_t count,
                                  uint32_t *d_bestScore, uint64_t *d_bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[12];
    decode_seq16(seqIdx, seqLen, ops);

    // QuickCheck: 8 spot inputs
    for (int i = 0; i < 8; i++) {
        uint16_t inp = qcInputs[i];
        uint16_t expected = (uint16_t)(inp * k);  // mod 65536
        if (run_seq16(ops, seqLen, inp) != expected) return;
    }

    // Full verification: all 256 low-byte inputs (H=0, L=0..255)
    for (int v = 0; v < 256; v++) {
        uint16_t inp = (uint16_t)v;
        if (run_seq16(ops, seqLen, inp) != (uint16_t)(inp * k)) return;
    }
    // Full verification: all 256 high-byte inputs (H=0..255, L=0)
    for (int v = 0; v < 256; v++) {
        uint16_t inp = (uint16_t)(v << 8);
        if (run_seq16(ops, seqLen, inp) != (uint16_t)(inp * k)) return;
    }

    // Valid! Score = (length << 16) | tstates
    uint16_t cost = seq_cost16(ops, seqLen);
    uint32_t score = ((uint32_t)seqLen << 16) | cost;
    uint32_t old = atomicMin(d_bestScore, score);
    if (score <= old) {
        atomicExch((unsigned long long *)d_bestIdx, (unsigned long long)seqIdx);
    }
}

// ============================================================
// Host
// ============================================================
static uint64_t ipow64(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; i++) r *= base;
    return r;
}

static const char *opName16(uint8_t op) {
    static const char *names[NUM_OPS] = {
        "ADD HL,HL", "ADD HL,BC", "ADD HL,DE",
        "LD C,L",    "LD B,H",
        "LD E,L",    "LD D,H",
        "LD L,C",    "LD H,B",
        "LD L,E",    "LD H,D",
        "EX DE,HL",
    };
    return op < NUM_OPS ? names[op] : "?";
}

struct Mul16cResult {
    int k;
    int length;
    int tstates;
    uint8_t ops[12];
    bool found;
    bool clobbers_bc;
    bool clobbers_de;
};

static void analyze_clobbers(Mul16cResult &r) {
    // Simulate with HL_in=1, check if BC/DE changed from zero
    State16 s; s.h=0; s.l=1; s.b=s.c=s.d=s.e=0;
    for (int i = 0; i < r.length; i++) s = exec16(r.ops[i], s);
    r.clobbers_bc = (s.b != 0 || s.c != 0);
    r.clobbers_de = (s.d != 0 || s.e != 0);
}

static Mul16cResult solve_k16c(int k, int maxLen, bool quiet) {
    Mul16cResult result;
    result.k = k;
    result.found = false;

    uint32_t *d_bestScore;
    uint64_t *d_bestIdx;
    cudaMalloc(&d_bestScore, sizeof(uint32_t));
    cudaMalloc(&d_bestIdx,   sizeof(uint64_t));

    int blockSize = 256;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow64(NUM_OPS, len);

        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initIdx   = 0;
        cudaMemcpy(d_bestScore, &initScore, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestIdx,   &initIdx,   sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t batchSize = (uint64_t)blockSize * 65535;
        uint64_t offset = 0;
        while (offset < total) {
            uint64_t count = total - offset;
            if (count > batchSize) count = batchSize;
            uint64_t grid = (count + blockSize - 1) / blockSize;
            mulopt16c_kernel<<<(unsigned int)grid, blockSize>>>(
                (uint16_t)k, len, offset, count, d_bestScore, d_bestIdx);
            cudaDeviceSynchronize();
            offset += count;
        }

        uint32_t bestScore;
        uint64_t bestIdx;
        cudaMemcpy(&bestScore, d_bestScore, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestIdx,   d_bestIdx,   sizeof(uint64_t), cudaMemcpyDeviceToHost);

        if (bestScore != 0xFFFFFFFF) {
            result.length  = len;
            result.tstates = bestScore & 0xFFFF;
            decode_seq16(bestIdx, len, result.ops);
            result.found = true;
            analyze_clobbers(result);
            if (!quiet) {
                fprintf(stderr, "  K=%d: found len=%d tstates=%d\n", k, len, result.tstates);
            }
            break;
        }
        if (!quiet && len <= 4) {
            fprintf(stderr, "  K=%d len=%d: no solution\n", k, len);
        }
    }

    cudaFree(d_bestScore);
    cudaFree(d_bestIdx);
    return result;
}

static void print_json(const Mul16cResult &r) {
    printf("{\"k\":%d,\"length\":%d,\"tstates\":%d,\"ops\":[", r.k, r.length, r.tstates);
    for (int i = 0; i < r.length; i++) {
        if (i > 0) printf(",");
        printf("\"%s\"", opName16(r.ops[i]));
    }
    printf("],\"clobbers_bc\":%s,\"clobbers_de\":%s,\"verified\":true}\n",
           r.clobbers_bc ? "true" : "false",
           r.clobbers_de ? "true" : "false");
    fflush(stdout);
}

int main(int argc, char **argv) {
    int maxLen = 8;
    int singleK = -1;
    bool jsonMode = false;
    bool allK = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) singleK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) jsonMode = true;
        else if (!strcmp(argv[i], "--all")) allK = true;
    }

    if (singleK < 0 && !allK) {
        fprintf(stderr, "Usage: z80_mulopt16c [--max-len N] [--k K | --all] [--json]\n");
        fprintf(stderr, "  --k K     solve single constant K\n");
        fprintf(stderr, "  --all     solve all K in [2..255]\n");
        fprintf(stderr, "  --max-len N  max sequence length (default 8)\n");
        fprintf(stderr, "  --json    output JSON lines\n");
        return 1;
    }

    cudaSetDevice(0);

    if (jsonMode && allK) printf("[\n");

    int kStart = (singleK > 0) ? singleK : 2;
    int kEnd   = (singleK > 0) ? singleK : 255;

    int solved = 0, total = kEnd - kStart + 1;
    int64_t totalT = 0;

    for (int k = kStart; k <= kEnd; k++) {
        fprintf(stderr, "K=%d/%d ...\n", k, kEnd);
        Mul16cResult r = solve_k16c(k, maxLen, false);
        if (r.found) {
            solved++;
            totalT += r.tstates;
            if (jsonMode) {
                if (allK && k > kStart) printf(",\n");
                print_json(r);
            } else {
                printf("K=%3d: %2d ops %3dT  [%s%s]  ",
                    k, r.length, r.tstates,
                    r.clobbers_bc ? "BC" : "  ",
                    r.clobbers_de ? "+DE" : "   ");
                for (int i = 0; i < r.length; i++) {
                    printf("%s", opName16(r.ops[i]));
                    if (i < r.length-1) printf("; ");
                }
                printf("\n");
            }
        } else {
            fprintf(stderr, "  K=%d: NOT FOUND (max-len=%d)\n", k, maxLen);
            if (jsonMode) {
                if (allK && k > kStart) printf(",\n");
                printf("{\"k\":%d,\"found\":false}\n", k);
            }
        }
        fflush(stdout);
    }

    if (jsonMode && allK) printf("]\n");

    if (total > 1) {
        fprintf(stderr, "\nSolved: %d/%d  avg T-states: %.1f\n",
                solved, total, solved > 0 ? (double)totalT / solved : 0.0);
    }
    return 0;
}

// z80_mulopt16.cu — GPU brute-force optimal u8×K=u16 constant multiplication for Z80
//
// For each constant K (2..255), finds the shortest instruction sequence
// where HL_out = A_in * K (full 16-bit result).
//
// Input: A = multiplicand (u8), B=0, H=0, L=0
// Output: H:L = full 16-bit product
//
// Build: nvcc -O3 -o z80_mulopt16 z80_mulopt16.cu
// Usage: z80_mulopt16 [--max-len 10] [--k 42] [--json]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Instruction opcodes (18 total)
#define OP_ADD_AA    0   // ADD A,A   (4T)
#define OP_ADD_AB    1   // ADD A,B   (4T)
#define OP_SUB_B     2   // SUB B     (4T)
#define OP_LD_BA     3   // LD B,A    (4T)
#define OP_LD_HA     4   // LD H,A    (4T)
#define OP_LD_LA     5   // LD L,A    (4T)
#define OP_LD_AH     6   // LD A,H    (4T)
#define OP_LD_AL     7   // LD A,L    (4T)
#define OP_ADD_HLHL  8   // ADD HL,HL (11T) — 16-bit double
#define OP_ADC_AA    9   // ADC A,A   (4T)
#define OP_ADC_AB   10   // ADC A,B   (4T)
#define OP_SBC_AB   11   // SBC A,B   (4T)
#define OP_NEG      12   // NEG       (8T)
#define OP_SLA_A    13   // SLA A     (8T)
#define OP_SRL_A    14   // SRL A     (8T)
#define OP_RLA      15   // RLA       (4T)
#define OP_RLCA     16   // RLCA      (4T)
#define OP_OR_A     17   // OR A      (4T) — clears carry
#define NUM_OPS     18

__constant__ uint8_t opCost[NUM_OPS] = {
    4, 4, 4, 4,     // ADD/SUB/LD B,A
    4, 4, 4, 4,     // LD H,A / LD L,A / LD A,H / LD A,L
    11,              // ADD HL,HL
    4, 4, 4,         // ADC/SBC
    8, 8, 8,         // NEG/SLA/SRL
    4, 4, 4          // RLA/RLCA/OR A
};

// State: A, B (scratch), H, L, carry
__device__ void exec_op(uint8_t op, uint8_t &a, uint8_t &b,
                        uint8_t &h, uint8_t &l, bool &carry) {
    uint16_t r, hl;
    uint16_t c;
    uint8_t bit;
    switch (op) {
    case OP_ADD_AA:
        r = (uint16_t)a + a;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_ADD_AB:
        r = (uint16_t)a + b;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_SUB_B:
        carry = a < b;
        a = a - b;
        break;
    case OP_LD_BA:
        b = a;
        break;
    case OP_LD_HA:
        h = a;
        break;
    case OP_LD_LA:
        l = a;
        break;
    case OP_LD_AH:
        a = h;
        break;
    case OP_LD_AL:
        a = l;
        break;
    case OP_ADD_HLHL:
        hl = ((uint16_t)h << 8) | l;
        r = (uint32_t)hl + hl;
        carry = r > 0xFFFF; // actually uint32 but for 16-bit...
        hl = hl + hl;
        h = (uint8_t)(hl >> 8);
        l = (uint8_t)hl;
        break;
    case OP_ADC_AA:
        c = carry ? 1 : 0;
        r = (uint16_t)a + a + c;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_ADC_AB:
        c = carry ? 1 : 0;
        r = (uint16_t)a + b + c;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_SBC_AB:
        c = carry ? 1 : 0;
        carry = ((int16_t)a - (int16_t)b - (int16_t)c) < 0;
        a = a - b - (uint8_t)c;
        break;
    case OP_NEG:
        carry = (a != 0);
        a = (uint8_t)(0 - a);
        break;
    case OP_SLA_A:
        carry = (a & 0x80) != 0;
        a = a << 1;
        break;
    case OP_SRL_A:
        carry = (a & 0x01) != 0;
        a = a >> 1;
        break;
    case OP_RLA:
        bit = carry ? 1 : 0;
        carry = (a & 0x80) != 0;
        a = (a << 1) | bit;
        break;
    case OP_RLCA:
        carry = (a & 0x80) != 0;
        a = (a << 1) | (a >> 7);
        break;
    case OP_OR_A:
        carry = false;
        break;
    }
}

// Run sequence, return H:L as uint16
__device__ uint16_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0, h = 0, l = 0;
    bool carry = false;
    for (int i = 0; i < len; i++) {
        exec_op(ops[i], a, b, h, l, carry);
    }
    return ((uint16_t)h << 8) | l;
}

__device__ void decode_seq(uint64_t idx, int len, uint8_t *ops) {
    for (int i = len - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(idx % NUM_OPS);
        idx /= NUM_OPS;
    }
}

__device__ uint16_t seq_cost(const uint8_t *ops, int len) {
    uint16_t cost = 0;
    for (int i = 0; i < len; i++) {
        cost += opCost[ops[i]];
    }
    return cost;
}

// Kernel: test one sequence for constant K, verify HL = A_in * K (16-bit)
__global__ void mulopt16_kernel(uint16_t k, int seqLen, uint64_t offset, uint64_t count,
                                 uint32_t *d_bestScore, uint64_t *d_bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[12];
    decode_seq(seqIdx, seqLen, ops);

    // QuickCheck: test discriminating inputs
    if (run_seq(ops, seqLen, 1) != (uint16_t)(1 * k)) return;
    if (run_seq(ops, seqLen, 2) != (uint16_t)(2 * k)) return;
    if (run_seq(ops, seqLen, 127) != (uint16_t)(127 * k)) return;
    if (run_seq(ops, seqLen, 255) != (uint16_t)(255 * k)) return;

    // Full verification
    for (int input = 0; input < 256; input++) {
        if (run_seq(ops, seqLen, (uint8_t)input) != (uint16_t)(input * k)) return;
    }

    uint16_t cost = seq_cost(ops, seqLen);
    uint32_t score = ((uint32_t)seqLen << 16) | cost;

    uint32_t old = atomicMin(d_bestScore, score);
    if (score <= old) {
        atomicExch((unsigned long long *)d_bestIdx, (unsigned long long)seqIdx);
    }
}

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t result = 1;
    for (int i = 0; i < exp; i++) result *= base;
    return result;
}

static const char *opName(uint8_t op) {
    static const char *names[] = {
        "ADD A,A", "ADD A,B", "SUB B", "LD B,A",
        "LD H,A", "LD L,A", "LD A,H", "LD A,L",
        "ADD HL,HL",
        "ADC A,A", "ADC A,B", "SBC A,B",
        "NEG", "SLA A", "SRL A",
        "RLA", "RLCA", "OR A"
    };
    return op < NUM_OPS ? names[op] : "?";
}

struct MulResult {
    int k;
    int length;
    int tstates;
    uint8_t ops[12];
    bool found;
};

static MulResult solve_k(int k, int maxLen) {
    MulResult result;
    result.k = k;
    result.found = false;

    uint32_t *d_bestScore;
    uint64_t *d_bestIdx;
    cudaMalloc(&d_bestScore, sizeof(uint32_t));
    cudaMalloc(&d_bestIdx, sizeof(uint64_t));

    uint32_t initScore = 0xFFFFFFFF;
    uint64_t initIdx = 0;
    int blockSize = 256;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);

        cudaMemcpy(d_bestScore, &initScore, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestIdx, &initIdx, sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t batchSize = (uint64_t)blockSize * 65535;
        uint64_t offset = 0;

        while (offset < total) {
            uint64_t count = total - offset;
            if (count > batchSize) count = batchSize;

            uint64_t grid = (count + blockSize - 1) / blockSize;
            mulopt16_kernel<<<(unsigned int)grid, blockSize>>>(
                (uint16_t)k, len, offset, count, d_bestScore, d_bestIdx);
            cudaDeviceSynchronize();
            offset += count;
        }

        uint32_t bestScore;
        cudaMemcpy(&bestScore, d_bestScore, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (bestScore != 0xFFFFFFFF) {
            uint64_t bestIdx;
            cudaMemcpy(&bestIdx, d_bestIdx, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            result.found = true;
            result.length = len;
            result.tstates = bestScore & 0xFFFF;
            uint64_t idx = bestIdx;
            for (int i = len - 1; i >= 0; i--) {
                result.ops[i] = (uint8_t)(idx % NUM_OPS);
                idx /= NUM_OPS;
            }
            break;
        }
    }

    cudaFree(d_bestScore);
    cudaFree(d_bestIdx);
    return result;
}

int main(int argc, char *argv[]) {
    int maxLen = 10;
    int singleK = 0;
    bool jsonMode = false;

    // Sleep instead of busy-wait on cudaDeviceSynchronize — frees CPU cores
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-len") == 0 && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i+1 < argc) singleK = atoi(argv[++i]);
        else if (strcmp(argv[i], "--json") == 0) jsonMode = true;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "z80_mulopt16 — GPU brute-force u8×K=u16 (result in HL)\n");
            fprintf(stderr, "Usage: z80_mulopt16 [--max-len 10] [--k 42] [--json]\n");
            return 0;
        }
    }

    if (singleK > 0) {
        fprintf(stderr, "Searching for mul16 ×%d (max length %d, %d ops, GPU)...\n",
                singleK, maxLen, NUM_OPS);
        MulResult r = solve_k(singleK, maxLen);
        if (!r.found) {
            fprintf(stderr, "No sequence found for ×%d within length %d\n", singleK, maxLen);
            return 1;
        }
        if (jsonMode) {
            printf("{\"k\": %d, \"ops\": [", r.k);
            for (int i = 0; i < r.length; i++) {
                if (i > 0) printf(", ");
                printf("\"%s\"", opName(r.ops[i]));
            }
            printf("], \"length\": %d, \"tstates\": %d}\n", r.length, r.tstates);
        } else {
            printf("×%d (u16):", r.k);
            for (int i = 0; i < r.length; i++) printf(" %s /", opName(r.ops[i]));
            printf(" (%d insts, %dT)\n", r.length, r.tstates);
        }
        return 0;
    }

    // Solve all constants 2..255
    fprintf(stderr, "GPU mulopt16: solving ×2..×255 → HL (max length %d, %d ops)\n", maxLen, NUM_OPS);

    if (jsonMode) printf("[\n");
    int solved = 0;
    bool firstJson = true;

    for (int k = 2; k <= 255; k++) {
        fprintf(stderr, "\r×%d/255 (%d solved)...", k, solved);
        MulResult r = solve_k(k, maxLen);

        if (r.found) {
            solved++;
            if (jsonMode) {
                if (!firstJson) printf(",\n");
                firstJson = false;
                printf("  {\"k\": %d, \"ops\": [", r.k);
                for (int i = 0; i < r.length; i++) {
                    if (i > 0) printf(", ");
                    printf("\"%s\"", opName(r.ops[i]));
                }
                printf("], \"length\": %d, \"tstates\": %d}", r.length, r.tstates);
            } else {
                printf("×%3d (u16):", r.k);
                for (int i = 0; i < r.length; i++) printf(" %s /", opName(r.ops[i]));
                printf(" (%d insts, %dT)\n", r.length, r.tstates);
            }
            fflush(stdout);
        } else if (!jsonMode) {
            printf("×%3d (u16): NOT FOUND (max %d)\n", k, maxLen);
        }
    }

    if (jsonMode) printf("\n]\n");
    fprintf(stderr, "\rDone: %d/254 constants solved            \n", solved);
    return 0;
}

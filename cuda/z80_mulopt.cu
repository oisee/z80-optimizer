// z80_mulopt.cu — GPU brute-force optimal constant multiplication for Z80
//
// For each constant K (2..255), finds the shortest instruction sequence
// where A_out = A_in * K (mod 256).
//
// Each GPU thread evaluates one instruction sequence. Thread index encodes
// the sequence (base-NumOps digits). QuickCheck with 4 test values rejects
// >99% of candidates before full 256-input verification.
//
// Build: nvcc -O3 -o z80_mulopt z80_mulopt.cu
// Usage: z80_mulopt [--max-len 8] [--k 42] [--json]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Instruction opcodes
#define OP_ADD_AA   0   // ADD A,A  (4T)
#define OP_ADD_AB   1   // ADD A,B  (4T)
#define OP_SUB_B    2   // SUB B    (4T)
#define OP_LD_BA    3   // LD B,A   (4T)
#define OP_SLA_A    4   // SLA A    (8T)
#define OP_SRL_A    5   // SRL A    (8T)
#define OP_EX_AF    6   // EX AF,AF'(4T)
#define OP_EXX      7   // EXX      (4T)
#define OP_ADC_AB   8   // ADC A,B  (4T)
#define OP_ADC_AA   9   // ADC A,A  (4T)
#define OP_SBC_AB  10   // SBC A,B  (4T)
#define OP_SBC_AA  11   // SBC A,A  (4T)
#define OP_OR_A    12   // OR A     (4T)
#define NUM_OPS    13

__constant__ uint8_t opCost[NUM_OPS] = {4,4,4,4,8,8,4,4,4,4,4,4,4};

// Execute one instruction. State: a, b, carry, aS (shadow A), bS (shadow B), carryS
__device__ void exec_op(uint8_t op, uint8_t &a, uint8_t &b, bool &carry,
                        uint8_t &aS, uint8_t &bS, bool &carryS) {
    uint16_t r;
    uint16_t c;
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
        r = (uint16_t)a - b;
        carry = (a < b);
        a = (uint8_t)r;
        break;
    case OP_LD_BA:
        b = a;
        break;
    case OP_SLA_A:
        carry = (a & 0x80) != 0;
        a = a << 1;
        break;
    case OP_SRL_A:
        carry = (a & 0x01) != 0;
        a = a >> 1;
        break;
    case OP_EX_AF: {
        uint8_t ta = a; a = aS; aS = ta;
        bool tc = carry; carry = carryS; carryS = tc;
        break;
    }
    case OP_EXX: {
        uint8_t tb = b; b = bS; bS = tb;
        break;
    }
    case OP_ADC_AB:
        c = carry ? 1 : 0;
        r = (uint16_t)a + b + c;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_ADC_AA:
        c = carry ? 1 : 0;
        r = (uint16_t)a + a + c;
        carry = r > 0xFF;
        a = (uint8_t)r;
        break;
    case OP_SBC_AB:
        c = carry ? 1 : 0;
        r = (uint16_t)a - b - c;
        carry = (int16_t)((int16_t)a - (int16_t)b - (int16_t)c) < 0;
        a = (uint8_t)r;
        break;
    case OP_SBC_AA:
        c = carry ? 1 : 0;
        r = (uint16_t)a - a - c;
        carry = c > 0; // 0-0-c < 0 iff c=1
        a = (uint8_t)r;
        break;
    case OP_OR_A:
        carry = false;
        break;
    }
}

// Run a sequence on input, return final A
__device__ uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0, aS = 0, bS = 0;
    bool carry = false, carryS = false;
    for (int i = 0; i < len; i++) {
        exec_op(ops[i], a, b, carry, aS, bS, carryS);
    }
    return a;
}

// Decode thread index to instruction sequence (base NUM_OPS)
__device__ void decode_seq(uint64_t idx, int len, uint8_t *ops) {
    for (int i = len - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(idx % NUM_OPS);
        idx /= NUM_OPS;
    }
}

// Compute T-state cost
__device__ uint16_t seq_cost(const uint8_t *ops, int len) {
    uint16_t cost = 0;
    for (int i = 0; i < len; i++) {
        cost += opCost[ops[i]];
    }
    return cost;
}

// Kernel: test one sequence for constant K
// d_bestCost stores (length << 16) | tstates — so shorter length always wins,
// ties broken by lower T-states.
__global__ void mulopt_kernel(uint8_t k, int seqLen, uint64_t offset, uint64_t count,
                               uint32_t *d_bestScore, uint64_t *d_bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t seqIdx = offset + tid;
    uint8_t ops[12]; // max length
    decode_seq(seqIdx, seqLen, ops);

    // QuickCheck: test 4 discriminating inputs
    if (run_seq(ops, seqLen, 1) != (uint8_t)(1 * k)) return;
    if (run_seq(ops, seqLen, 2) != (uint8_t)(2 * k)) return;
    if (run_seq(ops, seqLen, 127) != (uint8_t)(127 * k)) return;
    if (run_seq(ops, seqLen, 255) != (uint8_t)(255 * k)) return;

    // Full verification: all 256 inputs
    for (int input = 0; input < 256; input++) {
        if (run_seq(ops, seqLen, (uint8_t)input) != (uint8_t)(input * k)) return;
    }

    // Valid! Compute score = tstates (length is fixed per launch)
    uint16_t cost = seq_cost(ops, seqLen);
    uint32_t score = ((uint32_t)seqLen << 16) | cost;

    uint32_t old = atomicMin(d_bestScore, score);
    if (score <= old) {
        atomicExch((unsigned long long *)d_bestIdx, (unsigned long long)seqIdx);
    }
}

// Host: compute NUM_OPS^len
static uint64_t ipow(uint64_t base, int exp) {
    uint64_t result = 1;
    for (int i = 0; i < exp; i++) result *= base;
    return result;
}

static const char *opName(uint8_t op) {
    static const char *names[] = {
        "ADD A,A", "ADD A,B", "SUB B", "LD B,A", "SLA A", "SRL A",
        "EX AF,AF'", "EXX", "ADC A,B", "ADC A,A", "SBC A,B", "SBC A,A", "OR A"
    };
    return op < NUM_OPS ? names[op] : "?";
}

static int hostOpCost[] = {4,4,4,4,8,8,4,4,4,4,4,4,4};

struct MulResult {
    int k;
    int length;
    int tstates;
    uint8_t ops[12];
    bool found;
};

// Solve one constant K
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
    cudaMemcpy(d_bestScore, &initScore, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestIdx, &initIdx, sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blockSize = 256;

    // Search each length, stop at first length that finds a solution
    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);

        // Reset for this length
        cudaMemcpy(d_bestScore, &initScore, sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint64_t batchSize = (uint64_t)blockSize * 65535;
        uint64_t offset = 0;

        while (offset < total) {
            uint64_t count = total - offset;
            if (count > batchSize) count = batchSize;

            uint64_t grid = (count + blockSize - 1) / blockSize;
            mulopt_kernel<<<(unsigned int)grid, blockSize>>>(
                (uint8_t)k, len, offset, count, d_bestScore, d_bestIdx);
            cudaDeviceSynchronize();
            offset += count;
        }

        // Check if we found a solution at this length
        uint32_t bestScore;
        cudaMemcpy(&bestScore, d_bestScore, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (bestScore != 0xFFFFFFFF) {
            uint64_t bestIdx;
            cudaMemcpy(&bestIdx, d_bestIdx, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            // Decode on host
            result.found = true;
            result.length = len;
            result.tstates = bestScore & 0xFFFF;
            uint64_t idx = bestIdx;
            for (int i = len - 1; i >= 0; i--) {
                result.ops[i] = (uint8_t)(idx % NUM_OPS);
                idx /= NUM_OPS;
            }
            break; // shortest length found
        }
    }

    cudaFree(d_bestScore);
    cudaFree(d_bestIdx);
    return result;
}

int main(int argc, char *argv[]) {
    int maxLen = 8;
    int singleK = 0;
    bool jsonMode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-len") == 0 && i+1 < argc) {
            maxLen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i+1 < argc) {
            singleK = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            jsonMode = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "z80_mulopt — GPU brute-force optimal constant multiplication\n");
            fprintf(stderr, "Usage: z80_mulopt [--max-len 8] [--k 42] [--json]\n");
            return 0;
        }
    }

    if (singleK > 0) {
        fprintf(stderr, "Searching for mul×%d (max length %d, %d ops, GPU)...\n",
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
            printf("×%d:", r.k);
            for (int i = 0; i < r.length; i++) printf(" %s", opName(r.ops[i]));
            printf(" (%d insts, %dT)\n", r.length, r.tstates);
        }
        return 0;
    }

    // Solve all constants 2..255
    fprintf(stderr, "GPU mulopt: solving ×2..×255 (max length %d, %d ops)\n", maxLen, NUM_OPS);

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
                printf("×%3d:", r.k);
                for (int i = 0; i < r.length; i++) printf(" %s", opName(r.ops[i]));
                printf(" (%d insts, %dT)\n", r.length, r.tstates);
            }
            fflush(stdout);
        } else if (!jsonMode) {
            printf("×%3d: NOT FOUND (max %d)\n", k, maxLen);
        }
    }

    if (jsonMode) printf("\n]\n");
    fprintf(stderr, "\rDone: %d/254 constants solved            \n", solved);
    return 0;
}

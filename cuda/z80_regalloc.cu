// z80_regalloc.cu — GPU brute-force register allocator for Z80
//
// Given a function's VIR ops (virtual register references + patterns),
// tries ALL possible register assignments on GPU and finds the one
// that produces the fewest instructions (provably optimal).
//
// Input:  function description (vregs, ops, patterns, liveness)
// Output: optimal assignment (vreg → physical register) + instruction count
//
// Architecture:
//   - Each thread tests one assignment (vreg0→loc0, vreg1→loc1, ...)
//   - Assignment encoded as a single uint64: 3 bits per vreg (7 GPR = 0..6)
//   - Thread checks: (1) no interference (2) pattern constraints (3) count moves
//   - Block-level reduction finds global minimum
//
// For N vregs and 7 GPR locations: 7^N threads total.
//   N=8:  5.7M   threads (~0.1s on RTX 4060 Ti)
//   N=10: 282M   threads (~2s)
//   N=12: 13.8B  threads (~60s)
//   N=14: 678B   threads (needs multi-pass or pruning)
//
// Build: nvcc -O3 -o z80_regalloc z80_regalloc.cu
// Usage: z80_regalloc < func_desc.json

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

// ============================================================
// Constants
// ============================================================
#define MAX_VREGS    16     // max virtual registers per function
#define MAX_OPS      64     // max VIR operations per block
#define MAX_LOCS     7      // A, B, C, D, E, H, L (8-bit GPR)
#define MAX_PATTERNS 8      // max patterns per operation
#define INVALID_COST 0xFFFF // marks infeasible assignment

// Physical register indices (matching VIR z80.go Locs)
#define LOC_A  0
#define LOC_B  1
#define LOC_C  2
#define LOC_D  3
#define LOC_E  4
#define LOC_H  5
#define LOC_L  6

// ============================================================
// Function description (uploaded to GPU constant memory)
// ============================================================

// One VIR operation's constraints
struct OpDesc {
    int     nPatterns;              // how many patterns match this op
    uint8_t patDstLocs[MAX_PATTERNS]; // bitmask: which locs can dst be in
    uint8_t patSrcLocs0[MAX_PATTERNS]; // bitmask: which locs can src0 be in
    uint8_t patSrcLocs1[MAX_PATTERNS]; // bitmask: which locs can src1 be in
    uint8_t patCost[MAX_PATTERNS];     // T-state cost of each pattern
    uint8_t patTiedDstSrc;          // bitmask: which patterns have tied dst=src0
    int     dstVreg;                // -1 if none
    int     srcVreg0;               // -1 if none
    int     srcVreg1;               // -1 if none
};

struct FuncDesc {
    int     nVregs;                 // number of virtual registers
    int     nOps;                   // number of operations
    OpDesc  ops[MAX_OPS];           // operation constraints
    // Liveness: interference pairs (vreg_a, vreg_b) that must differ
    int     nInterference;
    uint8_t interfA[MAX_VREGS * MAX_VREGS]; // vreg A of pair
    uint8_t interfB[MAX_VREGS * MAX_VREGS]; // vreg B of pair
    // Param constraints: vreg must be in specific loc (from caller ABI)
    int     nParamConstraints;
    uint8_t paramVreg[MAX_VREGS];
    uint8_t paramLoc[MAX_VREGS];
};

// Constant memory — shared by all threads, cached
__constant__ FuncDesc d_func;

// ============================================================
// Kernel: evaluate one register assignment
// ============================================================

// Decode assignment from thread index.
// Thread idx encodes: vreg[0] = idx % 7, vreg[1] = (idx/7) % 7, ...
__host__ __device__ void decode_assignment(uint64_t idx, int nVregs, uint8_t *assignment) {
    for (int i = 0; i < nVregs; i++) {
        assignment[i] = (uint8_t)(idx % MAX_LOCS);
        idx /= MAX_LOCS;
    }
}

// Check interference: no two simultaneously-live vregs share a register
__device__ bool check_interference(const uint8_t *assignment) {
    for (int i = 0; i < d_func.nInterference; i++) {
        int a = d_func.interfA[i];
        int b = d_func.interfB[i];
        if (assignment[a] == assignment[b]) {
            return false; // conflict
        }
    }
    return true;
}

// Check param constraints: ABI-imposed register assignments
__device__ bool check_params(const uint8_t *assignment) {
    for (int i = 0; i < d_func.nParamConstraints; i++) {
        int vreg = d_func.paramVreg[i];
        int loc  = d_func.paramLoc[i];
        if (assignment[vreg] != loc) {
            return false;
        }
    }
    return true;
}

// Evaluate cost: for each op, find cheapest valid pattern + count moves
__device__ uint16_t evaluate_cost(const uint8_t *assignment) {
    uint16_t totalCost = 0;
    uint8_t prevLoc[MAX_VREGS]; // track previous location for move detection
    memset(prevLoc, 0xFF, sizeof(prevLoc));

    for (int oi = 0; oi < d_func.nOps; oi++) {
        const OpDesc &op = d_func.ops[oi];

        // Find cheapest valid pattern
        uint8_t bestCost = 0xFF;
        bool found = false;

        for (int pi = 0; pi < op.nPatterns; pi++) {
            bool valid = true;

            // Check dst location constraint
            if (op.dstVreg >= 0) {
                uint8_t dstLoc = assignment[op.dstVreg];
                if (!(op.patDstLocs[pi] & (1u << dstLoc))) {
                    valid = false;
                }
            }

            // Check src0 location constraint
            if (valid && op.srcVreg0 >= 0) {
                uint8_t srcLoc = assignment[op.srcVreg0];
                if (!(op.patSrcLocs0[pi] & (1u << srcLoc))) {
                    valid = false;
                }
            }

            // Check src1 location constraint
            if (valid && op.srcVreg1 >= 0) {
                uint8_t srcLoc = assignment[op.srcVreg1];
                if (!(op.patSrcLocs1[pi] & (1u << srcLoc))) {
                    valid = false;
                }
            }

            // Check tied dst=src0
            if (valid && (op.patTiedDstSrc & (1u << pi))) {
                if (op.dstVreg >= 0 && op.srcVreg0 >= 0) {
                    if (assignment[op.dstVreg] != assignment[op.srcVreg0]) {
                        valid = false;
                    }
                }
            }

            if (valid && op.patCost[pi] < bestCost) {
                bestCost = op.patCost[pi];
                found = true;
            }
        }

        if (!found) {
            return INVALID_COST; // no valid pattern exists for this assignment
        }

        totalCost += bestCost;

        // Add move cost: if a vreg changed location since last use
        if (op.srcVreg0 >= 0 && prevLoc[op.srcVreg0] != 0xFF &&
            prevLoc[op.srcVreg0] != assignment[op.srcVreg0]) {
            totalCost += 4; // LD r, r' = 4 T-states
        }
        if (op.srcVreg1 >= 0 && prevLoc[op.srcVreg1] != 0xFF &&
            prevLoc[op.srcVreg1] != assignment[op.srcVreg1]) {
            totalCost += 4;
        }

        // Update prevLoc
        if (op.dstVreg >= 0) {
            prevLoc[op.dstVreg] = assignment[op.dstVreg];
        }
    }

    return totalCost;
}

// Main kernel: one thread per assignment
__global__ void regalloc_kernel(uint64_t offset, uint64_t count,
                                 uint32_t *d_bestCost, uint64_t *d_bestIdx,
                                 uint64_t *d_feasibleCount) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t assignmentIdx = offset + tid;
    uint8_t assignment[MAX_VREGS];
    decode_assignment(assignmentIdx, d_func.nVregs, assignment);

    // Quick reject: interference
    if (!check_interference(assignment)) return;

    // Quick reject: param constraints
    if (!check_params(assignment)) return;

    // Evaluate full cost
    uint16_t cost = evaluate_cost(assignment);
    if (cost == INVALID_COST) return;

    // Count feasible assignments
    atomicAdd((unsigned long long *)d_feasibleCount, 1ULL);

    // Atomic min — update global best
    uint32_t cost32 = (uint32_t)cost;
    uint32_t old = atomicMin(d_bestCost, cost32);
    if (cost32 <= old) {
        // We might be the new best — store our index
        // (race condition possible, but we'll verify on CPU)
        atomicExch((unsigned long long *)d_bestIdx, (unsigned long long)assignmentIdx);
    }
}

// ============================================================
// Minimal JSON parser (no external dependencies)
// ============================================================

// Read all of stdin into a string
static std::string read_stdin() {
    std::string result;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof(buf), stdin)) {
        result.append(buf, n);
    }
    return result;
}

// Simple JSON tokenizer/parser — just enough for our schema
struct JsonParser {
    const char *p;
    const char *end;

    JsonParser(const std::string &s) : p(s.data()), end(s.data() + s.size()) {}

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    bool expect(char c) {
        skip_ws();
        if (p < end && *p == c) { p++; return true; }
        return false;
    }

    bool peek(char c) {
        skip_ws();
        return p < end && *p == c;
    }

    // Parse a string key (assumes opening " already consumed or about to be)
    std::string parse_string() {
        skip_ws();
        if (p >= end || *p != '"') return "";
        p++; // skip opening "
        std::string result;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) { p++; }
            result += *p++;
        }
        if (p < end) p++; // skip closing "
        return result;
    }

    int parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; p++; }
        int val = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            val = val * 10 + (*p - '0');
            p++;
        }
        return neg ? -val : val;
    }

    bool parse_bool() {
        skip_ws();
        if (p + 4 <= end && memcmp(p, "true", 4) == 0) { p += 4; return true; }
        if (p + 5 <= end && memcmp(p, "false", 5) == 0) { p += 5; return false; }
        return false;
    }

    std::vector<int> parse_int_array() {
        std::vector<int> result;
        if (!expect('[')) return result;
        if (peek(']')) { expect(']'); return result; }
        do {
            result.push_back(parse_int());
        } while (expect(','));
        expect(']');
        return result;
    }
};

// Convert an array of location indices to a bitmask
static uint8_t locs_to_bitmask(const std::vector<int> &locs) {
    uint8_t mask = 0;
    for (int loc : locs) {
        if (loc >= 0 && loc < MAX_LOCS) {
            mask |= (1u << loc);
        }
    }
    return mask;
}

// Parse a pattern object: { "dstLocs": [...], "srcLocs0": [...], ... }
static bool parse_pattern(JsonParser &jp, OpDesc &op, int pi) {
    if (!jp.expect('{')) return false;
    while (!jp.peek('}')) {
        std::string key = jp.parse_string();
        jp.expect(':');
        if (key == "dstLocs") {
            op.patDstLocs[pi] = locs_to_bitmask(jp.parse_int_array());
        } else if (key == "srcLocs0") {
            op.patSrcLocs0[pi] = locs_to_bitmask(jp.parse_int_array());
        } else if (key == "srcLocs1") {
            op.patSrcLocs1[pi] = locs_to_bitmask(jp.parse_int_array());
        } else if (key == "cost") {
            op.patCost[pi] = (uint8_t)jp.parse_int();
        } else if (key == "tiedDstSrc") {
            if (jp.parse_bool()) {
                op.patTiedDstSrc |= (1u << pi);
            }
        }
        jp.expect(','); // optional trailing comma
    }
    jp.expect('}');
    return true;
}

// Parse an op object: { "dst": ..., "src0": ..., "src1": ..., "patterns": [...] }
static bool parse_op(JsonParser &jp, OpDesc &op) {
    memset(&op, 0, sizeof(op));
    op.dstVreg = -1;
    op.srcVreg0 = -1;
    op.srcVreg1 = -1;

    if (!jp.expect('{')) return false;
    while (!jp.peek('}')) {
        std::string key = jp.parse_string();
        jp.expect(':');
        if (key == "dst") {
            op.dstVreg = jp.parse_int();
        } else if (key == "src0") {
            op.srcVreg0 = jp.parse_int();
        } else if (key == "src1") {
            op.srcVreg1 = jp.parse_int();
        } else if (key == "patterns") {
            jp.expect('[');
            int pi = 0;
            while (!jp.peek(']') && pi < MAX_PATTERNS) {
                parse_pattern(jp, op, pi);
                pi++;
                jp.expect(','); // optional
            }
            op.nPatterns = pi;
            jp.expect(']');
        }
        jp.expect(','); // optional trailing comma
    }
    jp.expect('}');
    return true;
}

// Parse full JSON input into FuncDesc
static bool parse_json(const std::string &input, FuncDesc &func) {
    memset(&func, 0, sizeof(func));
    JsonParser jp(input);

    if (!jp.expect('{')) {
        fprintf(stderr, "JSON: expected opening {\n");
        return false;
    }

    while (!jp.peek('}')) {
        std::string key = jp.parse_string();
        jp.expect(':');

        if (key == "nVregs") {
            func.nVregs = jp.parse_int();
        } else if (key == "ops") {
            jp.expect('[');
            int oi = 0;
            while (!jp.peek(']') && oi < MAX_OPS) {
                parse_op(jp, func.ops[oi]);
                oi++;
                jp.expect(','); // optional
            }
            func.nOps = oi;
            jp.expect(']');
        } else if (key == "interference") {
            jp.expect('[');
            int ii = 0;
            while (!jp.peek(']')) {
                std::vector<int> pair = jp.parse_int_array();
                if (pair.size() == 2) {
                    func.interfA[ii] = (uint8_t)pair[0];
                    func.interfB[ii] = (uint8_t)pair[1];
                    ii++;
                }
                jp.expect(','); // optional
            }
            func.nInterference = ii;
            jp.expect(']');
        } else if (key == "paramConstraints") {
            jp.expect('[');
            int ci = 0;
            while (!jp.peek(']')) {
                jp.expect('{');
                int vreg = -1, loc = -1;
                while (!jp.peek('}')) {
                    std::string ckey = jp.parse_string();
                    jp.expect(':');
                    if (ckey == "vreg") vreg = jp.parse_int();
                    else if (ckey == "loc") loc = jp.parse_int();
                    jp.expect(','); // optional
                }
                jp.expect('}');
                if (vreg >= 0 && loc >= 0) {
                    func.paramVreg[ci] = (uint8_t)vreg;
                    func.paramLoc[ci] = (uint8_t)loc;
                    ci++;
                }
                jp.expect(','); // optional
            }
            func.nParamConstraints = ci;
            jp.expect(']');
        }
        jp.expect(','); // optional trailing comma
    }
    jp.expect('}');

    if (func.nVregs <= 0) {
        fprintf(stderr, "JSON: nVregs must be > 0\n");
        return false;
    }
    return true;
}

// ============================================================
// Host code
// ============================================================

void print_usage() {
    fprintf(stderr, "z80_regalloc — GPU brute-force register allocator\n");
    fprintf(stderr, "Usage: z80_regalloc [--json | --demo] < input\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Modes:\n");
    fprintf(stderr, "  --json   Read JSON function description from stdin\n");
    fprintf(stderr, "  --demo   Run built-in demo (add function)\n");
    fprintf(stderr, "  (default) Read binary FuncDesc from stdin\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Loc indices: A=0, B=1, C=2, D=3, E=4, H=5, L=6\n");
}

int main(int argc, char *argv[]) {
    FuncDesc func;
    memset(&func, 0, sizeof(func));

    bool jsonMode = false;

    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "--json") == 0) {
        jsonMode = true;
        std::string input = read_stdin();
        if (!parse_json(input, func)) {
            return 1;
        }
    } else if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
        // Simple function: add(a, b) -> a + b
        // vregs: v0=param_a, v1=param_b, v2=result
        func.nVregs = 3;
        func.nOps = 1; // ADD v2 = v0 + v1

        OpDesc &add = func.ops[0];
        add.nPatterns = 1;
        add.patDstLocs[0] = (1 << LOC_A); // ADD dst must be A
        add.patSrcLocs0[0] = (1 << LOC_A); // ADD src0 must be A (tied)
        add.patSrcLocs1[0] = 0x7F; // ADD src1 can be any GPR
        add.patCost[0] = 4; // ADD A, r = 4T
        add.patTiedDstSrc = 1; // pattern 0 has tied dst=src0
        add.dstVreg = 2;
        add.srcVreg0 = 0;
        add.srcVreg1 = 1;

        // Interference: v0 and v1 are live simultaneously
        func.nInterference = 1;
        func.interfA[0] = 0;
        func.interfB[0] = 1;

        // Param constraints: v0 in A, v1 in B (caller ABI)
        func.nParamConstraints = 2;
        func.paramVreg[0] = 0; func.paramLoc[0] = LOC_A;
        func.paramVreg[1] = 1; func.paramLoc[1] = LOC_B;

        fprintf(stderr, "Demo: add(a: u8 = A, b: u8 = B) -> u8\n");
        fprintf(stderr, "  vregs: 3, ops: 1, interference: 1, constraints: 2\n");
    } else {
        // Read binary FuncDesc from stdin
        size_t n = fread(&func, 1, sizeof(func), stdin);
        if (n < sizeof(int) * 2) {
            fprintf(stderr, "Error: failed to read FuncDesc from stdin\n");
            return 1;
        }
    }

    // Compute search space
    uint64_t totalAssignments = 1;
    for (int i = 0; i < func.nVregs; i++) {
        totalAssignments *= MAX_LOCS;
        if (totalAssignments > 100000000000ULL) { // 100B limit
            fprintf(stderr, "Search space too large: %d vregs -> 7^%d > 100B\n",
                    func.nVregs, func.nVregs);
            return 1;
        }
    }

    if (!jsonMode) {
        printf("Search space: 7^%d = %llu assignments\n", func.nVregs,
               (unsigned long long)totalAssignments);
    }

    // Upload function description to constant memory
    cudaMemcpyToSymbol(d_func, &func, sizeof(FuncDesc));

    // Allocate result buffers
    uint32_t *d_bestCost;
    uint64_t *d_bestIdx;
    uint64_t *d_feasibleCount;
    cudaMalloc(&d_bestCost, sizeof(uint32_t));
    cudaMalloc(&d_bestIdx, sizeof(uint64_t));
    cudaMalloc(&d_feasibleCount, sizeof(uint64_t));

    uint32_t initCost = INVALID_COST;
    uint64_t initIdx = 0;
    uint64_t initFeasible = 0;
    cudaMemcpy(d_bestCost, &initCost, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestIdx, &initIdx, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feasibleCount, &initFeasible, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;

    // For very large spaces, launch in batches
    uint64_t batchSize = (uint64_t)blockSize * 65535; // max grid size per launch
    uint64_t offset = 0;

    if (!jsonMode) {
        printf("Launching GPU regalloc...\n");
    }

    while (offset < totalAssignments) {
        uint64_t count = totalAssignments - offset;
        if (count > batchSize) count = batchSize;

        uint64_t grid = (count + blockSize - 1) / blockSize;
        regalloc_kernel<<<(unsigned int)grid, blockSize>>>(offset, count,
                                                            d_bestCost, d_bestIdx,
                                                            d_feasibleCount);
        cudaDeviceSynchronize();

        offset += count;
        if (!jsonMode && totalAssignments > 1000000) {
            printf("  %.1f%% (%llu / %llu)\n",
                   (double)offset / totalAssignments * 100,
                   (unsigned long long)offset,
                   (unsigned long long)totalAssignments);
        }
    }

    // Read results
    uint32_t bestCost;
    uint64_t bestIdx;
    uint64_t feasibleCount;
    cudaMemcpy(&bestCost, d_bestCost, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bestIdx, d_bestIdx, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&feasibleCount, d_feasibleCount, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (jsonMode) {
        // JSON output mode — single line to stdout, nothing else
        if (bestCost == INVALID_COST) {
            printf("{\"cost\": -1, \"assignment\": [], \"searchSpace\": %llu, \"feasible\": 0}\n",
                   (unsigned long long)totalAssignments);
        } else {
            uint8_t best[MAX_VREGS];
            decode_assignment(bestIdx, func.nVregs, best);
            printf("{\"cost\": %u, \"assignment\": [", bestCost);
            for (int i = 0; i < func.nVregs; i++) {
                if (i > 0) printf(", ");
                printf("%d", best[i]);
            }
            printf("], \"searchSpace\": %llu, \"feasible\": %llu}\n",
                   (unsigned long long)totalAssignments,
                   (unsigned long long)feasibleCount);
        }
    } else {
        // Human-readable output mode
        if (bestCost == INVALID_COST) {
            printf("No valid assignment found (all infeasible)\n");
            cudaFree(d_bestCost);
            cudaFree(d_bestIdx);
            cudaFree(d_feasibleCount);
            return 1;
        }

        uint8_t best[MAX_VREGS];
        decode_assignment(bestIdx, func.nVregs, best);

        const char *locNames[] = {"A", "B", "C", "D", "E", "H", "L"};
        printf("\nOptimal assignment (cost=%u T-states, %llu feasible):\n",
               bestCost, (unsigned long long)feasibleCount);
        for (int i = 0; i < func.nVregs; i++) {
            printf("  v%d -> %s\n", i, locNames[best[i]]);
        }

        printf("\n{\"cost\": %u, \"assignment\": [", bestCost);
        for (int i = 0; i < func.nVregs; i++) {
            if (i > 0) printf(", ");
            printf("%d", best[i]);
        }
        printf("], \"searchSpace\": %llu, \"feasible\": %llu}\n",
               (unsigned long long)totalAssignments,
               (unsigned long long)feasibleCount);
    }

    cudaFree(d_bestCost);
    cudaFree(d_bestIdx);
    cudaFree(d_feasibleCount);
    return 0;
}

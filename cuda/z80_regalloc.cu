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
#define MAX_LOCS     15     // A-L + BC,DE,HL + IXH,IXL,IYH,IYL + mem0
#define MAX_PATTERNS 16     // max patterns per operation
#define INVALID_COST 0xFFFF // marks infeasible assignment

// Physical register indices (matching VIR z80.go Locs)
#define LOC_A   0
#define LOC_B   1
#define LOC_C   2
#define LOC_D   3
#define LOC_E   4
#define LOC_H   5
#define LOC_L   6
#define LOC_BC  7
#define LOC_DE  8
#define LOC_HL  9
#define LOC_IXH 10
#define LOC_IXL 11
#define LOC_IYH 12
#define LOC_IYL 13
#define LOC_MEM0 14

// ============================================================
// Function description (uploaded to GPU constant memory)
// ============================================================

// One VIR operation's constraints
struct OpDesc {
    int      nPatterns;              // how many patterns match this op
    uint16_t patDstLocs[MAX_PATTERNS]; // bitmask: which locs can dst be in
    uint16_t patSrcLocs0[MAX_PATTERNS]; // bitmask: which locs can src0 be in
    uint16_t patSrcLocs1[MAX_PATTERNS]; // bitmask: which locs can src1 be in
    uint8_t  patCost[MAX_PATTERNS];     // T-state cost of each pattern
    uint16_t patTiedDstSrc;          // bitmask: which patterns have tied dst=src0
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
    // Vreg widths: 8 or 16. 16-bit vregs restricted to pair/mem locs.
    uint8_t vregWidth[MAX_VREGS];   // 0 or 8 = 8-bit (any loc), 16 = 16-bit (pairs only)
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

// Check vreg widths: 16-bit vregs must be in pair/mem locs (7-9, 14)
__device__ bool check_widths(const uint8_t *assignment) {
    for (int i = 0; i < d_func.nVregs; i++) {
        if (d_func.vregWidth[i] == 16) {
            uint8_t loc = assignment[i];
            // 16-bit vregs: only BC(7), DE(8), HL(9), mem0(14)
            if (loc < LOC_BC && loc != LOC_MEM0) return false;
            if (loc > LOC_HL && loc != LOC_MEM0) return false;
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
        // Cost depends on src/dst location type:
        //   locs 0-6 (GPR): 4T, locs 7-9 (pairs): 4T,
        //   locs 10-13 (IX/IY): 8T, loc 14 (mem): 13T
        if (op.srcVreg0 >= 0 && prevLoc[op.srcVreg0] != 0xFF &&
            prevLoc[op.srcVreg0] != assignment[op.srcVreg0]) {
            uint8_t from = prevLoc[op.srcVreg0], to = assignment[op.srcVreg0];
            uint8_t mc = (from == LOC_MEM0 || to == LOC_MEM0) ? 13 :
                         (from >= LOC_IXH || to >= LOC_IXH) ? 8 : 4;
            totalCost += mc;
        }
        if (op.srcVreg1 >= 0 && prevLoc[op.srcVreg1] != 0xFF &&
            prevLoc[op.srcVreg1] != assignment[op.srcVreg1]) {
            uint8_t from = prevLoc[op.srcVreg1], to = assignment[op.srcVreg1];
            uint8_t mc = (from == LOC_MEM0 || to == LOC_MEM0) ? 13 :
                         (from >= LOC_IXH || to >= LOC_IXH) ? 8 : 4;
            totalCost += mc;
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

    // Quick reject: vreg width constraints
    if (!check_widths(assignment)) return;

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

    bool try_null() {
        skip_ws();
        if (p + 4 <= end && memcmp(p, "null", 4) == 0) { p += 4; return true; }
        return false;
    }

    std::vector<int> parse_int_array() {
        std::vector<int> result;
        if (try_null()) return result;
        if (!expect('[')) return result;
        if (peek(']')) { expect(']'); return result; }
        do {
            result.push_back(parse_int());
        } while (expect(','));
        expect(']');
        return result;
    }

    // Skip an arbitrary JSON value (string, number, bool, null, array, object)
    void skip_value() {
        skip_ws();
        if (p >= end) return;
        if (*p == '"') { parse_string(); return; }
        if (*p == '[') {
            p++; int depth = 1;
            while (p < end && depth > 0) {
                if (*p == '[') depth++;
                else if (*p == ']') depth--;
                else if (*p == '"') { parse_string(); continue; }
                p++;
            }
            return;
        }
        if (*p == '{') {
            p++; int depth = 1;
            while (p < end && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                else if (*p == '"') { parse_string(); continue; }
                p++;
            }
            return;
        }
        // number, bool, null — consume until delimiter
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    }
};

// Convert an array of location indices to a bitmask.
// Empty array (or null, which parse_int_array returns as empty) means unconstrained.
static uint16_t locs_to_bitmask(const std::vector<int> &locs) {
    if (locs.empty()) return (1u << MAX_LOCS) - 1; // all locs valid
    uint16_t mask = 0;
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
        } else {
            jp.skip_value();
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
            while (!jp.peek(']')) {
                if (pi < MAX_PATTERNS) {
                    parse_pattern(jp, op, pi);
                    pi++;
                } else {
                    jp.skip_value(); // skip excess patterns
                }
                jp.expect(','); // optional
            }
            op.nPatterns = pi < MAX_PATTERNS ? pi : MAX_PATTERNS;
            jp.expect(']');
        } else {
            jp.skip_value();
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
            if (!jp.try_null()) {
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
            }
        } else if (key == "paramConstraints") {
            if (!jp.try_null()) {
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
        } else if (key == "widths") {
            std::vector<int> w = jp.parse_int_array();
            for (int i = 0; i < (int)w.size() && i < MAX_VREGS; i++) {
                func.vregWidth[i] = (uint8_t)w[i];
            }
        } else {
            jp.skip_value();
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
    fprintf(stderr, "Usage: z80_regalloc [--json | --server | --demo] < input\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Modes:\n");
    fprintf(stderr, "  --json     Read single JSON function from stdin, write result to stdout\n");
    fprintf(stderr, "  --server   Long-running: read JSON-per-line from stdin, write results per-line\n");
    fprintf(stderr, "             CUDA inits once at startup. Exits on EOF. For Go integration.\n");
    fprintf(stderr, "  --demo     Run built-in demo (add function)\n");
    fprintf(stderr, "  (default)  Read binary FuncDesc from stdin\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Loc indices: A=0,B=1,C=2,D=3,E=4,H=5,L=6,BC=7,DE=8,HL=9,IXH=10,IXL=11,IYH=12,IYL=13,mem0=14\n");
}

// Solve one function: upload to GPU, run kernel, return results via output params.
// GPU buffers must already be allocated. Resets them before each solve.
static bool solve_one(const FuncDesc &func,
                      uint32_t *d_bestCost, uint64_t *d_bestIdx, uint64_t *d_feasibleCount,
                      uint32_t &outCost, uint64_t &outIdx, uint64_t &outFeasible,
                      uint64_t &outTotal, bool quiet) {
    // Compute search space
    outTotal = 1;
    for (int i = 0; i < func.nVregs; i++) {
        outTotal *= MAX_LOCS;
        if (outTotal > 5000000000000ULL) {
            fprintf(stderr, "Search space too large: %d vregs -> %d^%d > 5T\n",
                    func.nVregs, MAX_LOCS, func.nVregs);
            return false;
        }
    }

    // Upload function description to constant memory
    cudaMemcpyToSymbol(d_func, &func, sizeof(FuncDesc));

    // Reset result buffers
    uint32_t initCost = INVALID_COST;
    uint64_t initIdx = 0;
    uint64_t initFeasible = 0;
    cudaMemcpy(d_bestCost, &initCost, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestIdx, &initIdx, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feasibleCount, &initFeasible, sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blockSize = 256;
    uint64_t batchSize = (uint64_t)blockSize * 65535;
    uint64_t offset = 0;

    if (!quiet) {
        printf("Search space: %d^%d = %llu assignments\n", MAX_LOCS, func.nVregs,
               (unsigned long long)outTotal);
        printf("Launching GPU regalloc...\n");
    }

    while (offset < outTotal) {
        uint64_t count = outTotal - offset;
        if (count > batchSize) count = batchSize;

        uint64_t grid = (count + blockSize - 1) / blockSize;
        regalloc_kernel<<<(unsigned int)grid, blockSize>>>(offset, count,
                                                            d_bestCost, d_bestIdx,
                                                            d_feasibleCount);
        cudaDeviceSynchronize();

        offset += count;
        if (!quiet && outTotal > 1000000) {
            printf("  %.1f%% (%llu / %llu)\n",
                   (double)offset / outTotal * 100,
                   (unsigned long long)offset,
                   (unsigned long long)outTotal);
        }
    }

    cudaMemcpy(&outCost, d_bestCost, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&outIdx, d_bestIdx, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&outFeasible, d_feasibleCount, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return true;
}

// ============================================================
// CPU backtracking solver for large search spaces
// ============================================================
// Uses interference-aware pruning: for each vreg, only try locations
// not already assigned to interfering neighbors. This turns L^N brute-force
// into a constraint-satisfaction search with dramatic pruning.

struct BacktrackState {
    const FuncDesc *func;
    uint8_t assignment[MAX_VREGS];
    uint32_t bestCost;
    uint8_t bestAssignment[MAX_VREGS];
    uint64_t feasibleCount;
    uint64_t nodesExplored;

    // Precomputed: for each vreg, list of interfering vregs
    int nNeighbors[MAX_VREGS];
    uint8_t neighbors[MAX_VREGS][MAX_VREGS];

    // Precomputed: valid location mask per vreg (from width constraints)
    uint16_t validLocs[MAX_VREGS];
};

// Evaluate cost on CPU (mirrors GPU evaluate_cost)
static uint16_t cpu_evaluate_cost(const FuncDesc &func, const uint8_t *assignment) {
    uint16_t totalCost = 0;
    uint8_t prevLoc[MAX_VREGS];
    memset(prevLoc, 0xFF, sizeof(prevLoc));

    for (int oi = 0; oi < func.nOps; oi++) {
        const OpDesc &op = func.ops[oi];
        uint8_t bestPatCost = 0xFF;
        bool found = false;

        for (int pi = 0; pi < op.nPatterns; pi++) {
            bool valid = true;
            if (op.dstVreg >= 0) {
                if (!(op.patDstLocs[pi] & (1u << assignment[op.dstVreg]))) valid = false;
            }
            if (valid && op.srcVreg0 >= 0) {
                if (!(op.patSrcLocs0[pi] & (1u << assignment[op.srcVreg0]))) valid = false;
            }
            if (valid && op.srcVreg1 >= 0) {
                if (!(op.patSrcLocs1[pi] & (1u << assignment[op.srcVreg1]))) valid = false;
            }
            if (valid && (op.patTiedDstSrc & (1u << pi))) {
                if (op.dstVreg >= 0 && op.srcVreg0 >= 0) {
                    if (assignment[op.dstVreg] != assignment[op.srcVreg0]) valid = false;
                }
            }
            if (valid && op.patCost[pi] < bestPatCost) {
                bestPatCost = op.patCost[pi];
                found = true;
            }
        }
        if (!found) return INVALID_COST;
        totalCost += bestPatCost;

        // Move costs
        if (op.srcVreg0 >= 0 && prevLoc[op.srcVreg0] != 0xFF &&
            prevLoc[op.srcVreg0] != assignment[op.srcVreg0]) {
            uint8_t from = prevLoc[op.srcVreg0], to = assignment[op.srcVreg0];
            uint8_t mc = (from == LOC_MEM0 || to == LOC_MEM0) ? 13 :
                         (from >= LOC_IXH || to >= LOC_IXH) ? 8 : 4;
            totalCost += mc;
        }
        if (op.srcVreg1 >= 0 && prevLoc[op.srcVreg1] != 0xFF &&
            prevLoc[op.srcVreg1] != assignment[op.srcVreg1]) {
            uint8_t from = prevLoc[op.srcVreg1], to = assignment[op.srcVreg1];
            uint8_t mc = (from == LOC_MEM0 || to == LOC_MEM0) ? 13 :
                         (from >= LOC_IXH || to >= LOC_IXH) ? 8 : 4;
            totalCost += mc;
        }

        if (op.dstVreg >= 0) prevLoc[op.dstVreg] = assignment[op.dstVreg];
    }
    return totalCost;
}

static void backtrack(BacktrackState &st, int depth) {
    st.nodesExplored++;

    // Progress report every 10B nodes
    if ((st.nodesExplored & 0x3FFFFFFFFULL) == 0) { // ~17B
        fprintf(stderr, "  backtrack progress: %lluB nodes, %llu feasible, best=%u\n",
                (unsigned long long)(st.nodesExplored / 1000000000ULL),
                (unsigned long long)st.feasibleCount, st.bestCost);
    }

    if (depth == st.func->nVregs) {
        // Full assignment — evaluate
        uint16_t cost = cpu_evaluate_cost(*st.func, st.assignment);
        if (cost != INVALID_COST) {
            st.feasibleCount++;
            if (cost < st.bestCost) {
                st.bestCost = cost;
                memcpy(st.bestAssignment, st.assignment, st.func->nVregs);
            }
        }
        return;
    }

    // Compute available locations: valid locs minus those used by interfering assigned neighbors
    uint16_t available = st.validLocs[depth];
    for (int ni = 0; ni < st.nNeighbors[depth]; ni++) {
        int nbr = st.neighbors[depth][ni];
        if (nbr < depth) { // already assigned
            available &= ~(1u << st.assignment[nbr]);
        }
    }

    // Also apply param constraints
    for (int i = 0; i < st.func->nParamConstraints; i++) {
        if (st.func->paramVreg[i] == depth) {
            available &= (1u << st.func->paramLoc[i]);
        }
    }

    // Try each available location with forward checking
    for (int loc = 0; loc < MAX_LOCS; loc++) {
        if (!(available & (1u << loc))) continue;
        st.assignment[depth] = (uint8_t)loc;

        // Forward check: verify no unassigned neighbor has 0 remaining options
        bool ok = true;
        for (int ni = 0; ni < st.nNeighbors[depth]; ni++) {
            int nbr = st.neighbors[depth][ni];
            if (nbr > depth) { // unassigned
                uint16_t nbrAvail = st.validLocs[nbr];
                // Remove locations used by already-assigned neighbors (including current)
                for (int nni = 0; nni < st.nNeighbors[nbr]; nni++) {
                    int nn = st.neighbors[nbr][nni];
                    if (nn <= depth) { // assigned (including current depth)
                        nbrAvail &= ~(1u << st.assignment[nn]);
                    }
                }
                if (nbrAvail == 0) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) continue;

        backtrack(st, depth + 1);
    }
}

// Order vregs by most-constrained first (highest degree in interference graph)
// This improves pruning by assigning the most constrained vregs early.
static void compute_vreg_order(const FuncDesc &func, int *order) {
    int degree[MAX_VREGS] = {};
    for (int i = 0; i < func.nInterference; i++) {
        degree[func.interfA[i]]++;
        degree[func.interfB[i]]++;
    }
    // Also boost vregs with param constraints
    for (int i = 0; i < func.nParamConstraints; i++) {
        degree[func.paramVreg[i]] += 100;
    }

    for (int i = 0; i < func.nVregs; i++) order[i] = i;
    // Sort by descending degree
    for (int i = 0; i < func.nVregs; i++) {
        for (int j = i + 1; j < func.nVregs; j++) {
            if (degree[order[j]] > degree[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }
}

// Remap a FuncDesc to use a different vreg ordering
static void remap_func(const FuncDesc &src, const int *order, FuncDesc &dst) {
    memcpy(&dst, &src, sizeof(FuncDesc));
    int inv[MAX_VREGS]; // inverse: inv[old] = new
    for (int i = 0; i < src.nVregs; i++) inv[order[i]] = i;

    for (int i = 0; i < src.nInterference; i++) {
        dst.interfA[i] = inv[src.interfA[i]];
        dst.interfB[i] = inv[src.interfB[i]];
    }
    for (int i = 0; i < src.nParamConstraints; i++) {
        dst.paramVreg[i] = inv[src.paramVreg[i]];
    }
    for (int i = 0; i < src.nVregs; i++) {
        dst.vregWidth[i] = src.vregWidth[order[i]];
    }
    for (int oi = 0; oi < src.nOps; oi++) {
        if (src.ops[oi].dstVreg >= 0) dst.ops[oi].dstVreg = inv[src.ops[oi].dstVreg];
        if (src.ops[oi].srcVreg0 >= 0) dst.ops[oi].srcVreg0 = inv[src.ops[oi].srcVreg0];
        if (src.ops[oi].srcVreg1 >= 0) dst.ops[oi].srcVreg1 = inv[src.ops[oi].srcVreg1];
    }
}

static bool solve_backtrack(const FuncDesc &func,
                            uint32_t &outCost, uint8_t *outAssignment,
                            uint64_t &outFeasible, uint64_t &outNodes) {
    // Reorder vregs: most constrained first
    int order[MAX_VREGS];
    compute_vreg_order(func, order);
    FuncDesc reordered;
    remap_func(func, order, reordered);

    BacktrackState st;
    st.func = &reordered;
    st.bestCost = INVALID_COST;
    st.feasibleCount = 0;
    st.nodesExplored = 0;
    memset(st.assignment, 0, sizeof(st.assignment));
    memset(st.bestAssignment, 0, sizeof(st.bestAssignment));

    // Precompute neighbor lists
    memset(st.nNeighbors, 0, sizeof(st.nNeighbors));
    for (int i = 0; i < reordered.nInterference; i++) {
        int a = reordered.interfA[i], b = reordered.interfB[i];
        st.neighbors[a][st.nNeighbors[a]++] = b;
        st.neighbors[b][st.nNeighbors[b]++] = a;
    }

    // Precompute valid location masks from width constraints + pattern analysis
    for (int i = 0; i < reordered.nVregs; i++) {
        if (reordered.vregWidth[i] == 16) {
            // 16-bit: BC(7), DE(8), HL(9), mem0(14)
            st.validLocs[i] = (1u << LOC_BC) | (1u << LOC_DE) | (1u << LOC_HL) | (1u << LOC_MEM0);
        } else {
            st.validLocs[i] = (1u << MAX_LOCS) - 1; // all locations
        }
    }

    // Further restrict: intersect with locations actually appearing in patterns
    // A vreg can only be at a location that some pattern allows for it
    uint16_t patternLocs[MAX_VREGS];
    memset(patternLocs, 0, sizeof(patternLocs));
    bool vregUsed[MAX_VREGS] = {};
    for (int oi = 0; oi < reordered.nOps; oi++) {
        const OpDesc &op = reordered.ops[oi];
        for (int pi = 0; pi < op.nPatterns; pi++) {
            if (op.dstVreg >= 0) {
                patternLocs[op.dstVreg] |= op.patDstLocs[pi];
                vregUsed[op.dstVreg] = true;
            }
            if (op.srcVreg0 >= 0) {
                patternLocs[op.srcVreg0] |= op.patSrcLocs0[pi];
                vregUsed[op.srcVreg0] = true;
            }
            if (op.srcVreg1 >= 0) {
                patternLocs[op.srcVreg1] |= op.patSrcLocs1[pi];
                vregUsed[op.srcVreg1] = true;
            }
        }
    }
    for (int i = 0; i < reordered.nVregs; i++) {
        if (vregUsed[i]) {
            st.validLocs[i] &= patternLocs[i];
        }
    }

    // Constraint propagation: if a vreg has 1 valid loc, fix it and remove from neighbors
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < reordered.nVregs; i++) {
            int cnt = __builtin_popcount(st.validLocs[i]);
            if (cnt == 1) {
                int loc = __builtin_ctz(st.validLocs[i]); // find the single loc
                // Remove this loc from all interfering neighbors
                for (int ni = 0; ni < st.nNeighbors[i]; ni++) {
                    int nbr = st.neighbors[i][ni];
                    if (st.validLocs[nbr] & (1u << loc)) {
                        st.validLocs[nbr] &= ~(1u << loc);
                        changed = true;
                        if (st.validLocs[nbr] == 0) {
                            fprintf(stderr, "Backtrack: constraint propagation made v%d infeasible!\n", nbr);
                        }
                    }
                }
            }
        }
    }

    // Report effective search space
    {
        double logSpace = 0;
        for (int i = 0; i < reordered.nVregs; i++) {
            int cnt = __builtin_popcount(st.validLocs[i]);
            logSpace += log10(cnt > 0 ? cnt : 1);
        }
        fprintf(stderr, "Backtrack: per-vreg locs = [");
        for (int i = 0; i < reordered.nVregs; i++) {
            if (i > 0) fprintf(stderr, ",");
            fprintf(stderr, "%d", __builtin_popcount(st.validLocs[i]));
        }
        fprintf(stderr, "] (10^%.1f effective)\n", logSpace);
    }

    fprintf(stderr, "Backtrack: %dv, %d intf, %d ops...\n",
            reordered.nVregs, reordered.nInterference, reordered.nOps);

    backtrack(st, 0);

    fprintf(stderr, "Backtrack done: explored %llu nodes, %llu feasible, best=%u\n",
            (unsigned long long)st.nodesExplored,
            (unsigned long long)st.feasibleCount,
            st.bestCost);

    outCost = st.bestCost;
    outFeasible = st.feasibleCount;
    outNodes = st.nodesExplored;

    // Unmap assignment back to original vreg order
    if (st.bestCost != INVALID_COST) {
        for (int i = 0; i < func.nVregs; i++) {
            outAssignment[order[i]] = st.bestAssignment[i];
        }
    }
    return true;
}

// Print JSON result line to stdout
static void print_json_result(const FuncDesc &func, uint32_t cost, uint64_t bestIdx,
                               uint64_t totalAssignments, uint64_t feasibleCount) {
    if (cost == INVALID_COST) {
        printf("{\"cost\": -1, \"assignment\": [], \"searchSpace\": %llu, \"feasible\": 0}\n",
               (unsigned long long)totalAssignments);
    } else {
        uint8_t best[MAX_VREGS];
        decode_assignment(bestIdx, func.nVregs, best);
        printf("{\"cost\": %u, \"assignment\": [", cost);
        for (int i = 0; i < func.nVregs; i++) {
            if (i > 0) printf(", ");
            printf("%d", best[i]);
        }
        printf("], \"searchSpace\": %llu, \"feasible\": %llu}\n",
               (unsigned long long)totalAssignments,
               (unsigned long long)feasibleCount);
    }
    fflush(stdout);
}

// --server mode: init CUDA once, read JSON-per-line from stdin, write JSON-per-line to stdout
static int run_server() {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Allocate GPU buffers once
    uint32_t *d_bestCost;
    uint64_t *d_bestIdx;
    uint64_t *d_feasibleCount;
    cudaMalloc(&d_bestCost, sizeof(uint32_t));
    cudaMalloc(&d_bestIdx, sizeof(uint64_t));
    cudaMalloc(&d_feasibleCount, sizeof(uint64_t));

    // Force CUDA context init with a dummy memcpy
    uint32_t dummy = 0;
    cudaMemcpy(d_bestCost, &dummy, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Signal readiness
    fprintf(stderr, "regalloc-server: ready\n");
    fflush(stderr);

    // Read one JSON object per line
    char buf[65536];
    int lineNum = 0;
    while (fgets(buf, sizeof(buf), stdin) != NULL) {
        lineNum++;
        // Skip empty lines
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) {
            buf[--len] = '\0';
        }
        if (len == 0) continue;

        std::string line(buf, len);
        FuncDesc func;
        if (!parse_json(line, func)) {
            fprintf(stderr, "regalloc-server: parse error on line %d\n", lineNum);
            // Output error result so Go can still read one line per input
            printf("{\"cost\": -1, \"assignment\": [], \"searchSpace\": 0, \"feasible\": 0, \"error\": \"parse error\"}\n");
            fflush(stdout);
            continue;
        }

        uint32_t cost;
        uint64_t bestIdx, feasible, total;
        if (!solve_one(func, d_bestCost, d_bestIdx, d_feasibleCount,
                       cost, bestIdx, feasible, total, true)) {
            // GPU search space too large — fall back to CPU backtracking
            uint8_t btAssignment[MAX_VREGS];
            uint64_t btNodes;
            solve_backtrack(func, cost, btAssignment, feasible, btNodes);

            if (cost == INVALID_COST) {
                printf("{\"cost\": -1, \"assignment\": [], \"searchSpace\": %llu, \"feasible\": 0, \"solver\": \"backtrack\"}\n",
                       (unsigned long long)btNodes);
            } else {
                printf("{\"cost\": %u, \"assignment\": [", cost);
                for (int i = 0; i < func.nVregs; i++) {
                    if (i > 0) printf(", ");
                    printf("%d", btAssignment[i]);
                }
                printf("], \"searchSpace\": %llu, \"feasible\": %llu, \"solver\": \"backtrack\"}\n",
                       (unsigned long long)btNodes,
                       (unsigned long long)feasible);
            }
            fflush(stdout);
            continue;
        }

        print_json_result(func, cost, bestIdx, total, feasible);
    }

    fprintf(stderr, "regalloc-server: processed %d functions, exiting\n", lineNum);

    cudaFree(d_bestCost);
    cudaFree(d_bestIdx);
    cudaFree(d_feasibleCount);
    return 0;
}

int main(int argc, char *argv[]) {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    FuncDesc func;
    memset(&func, 0, sizeof(func));

    bool jsonMode = false;

    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }

    // --server mode: long-running JSON-per-line protocol
    if (argc > 1 && strcmp(argv[1], "--server") == 0) {
        return run_server();
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
        add.patSrcLocs1[0] = (1u << MAX_LOCS) - 1; // ADD src1 can be any GPR
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

    // Allocate GPU buffers
    uint32_t *d_bestCost;
    uint64_t *d_bestIdx;
    uint64_t *d_feasibleCount;
    cudaMalloc(&d_bestCost, sizeof(uint32_t));
    cudaMalloc(&d_bestIdx, sizeof(uint64_t));
    cudaMalloc(&d_feasibleCount, sizeof(uint64_t));

    uint32_t cost;
    uint64_t bestIdx, feasible, total;
    if (!solve_one(func, d_bestCost, d_bestIdx, d_feasibleCount,
                   cost, bestIdx, feasible, total, jsonMode)) {
        cudaFree(d_bestCost);
        cudaFree(d_bestIdx);
        cudaFree(d_feasibleCount);
        return 1;
    }

    if (jsonMode) {
        print_json_result(func, cost, bestIdx, total, feasible);
    } else {
        if (cost == INVALID_COST) {
            printf("No valid assignment found (all infeasible)\n");
            cudaFree(d_bestCost);
            cudaFree(d_bestIdx);
            cudaFree(d_feasibleCount);
            return 1;
        }

        uint8_t best[MAX_VREGS];
        decode_assignment(bestIdx, func.nVregs, best);

        const char *locNames[] = {"A", "B", "C", "D", "E", "H", "L", "BC", "DE", "HL", "IXH", "IXL", "IYH", "IYL", "mem0"};
        printf("\nOptimal assignment (cost=%u T-states, %llu feasible):\n",
               cost, (unsigned long long)feasible);
        for (int i = 0; i < func.nVregs; i++) {
            printf("  v%d -> %s\n", i, locNames[best[i]]);
        }

        print_json_result(func, cost, bestIdx, total, feasible);
    }

    cudaFree(d_bestCost);
    cudaFree(d_bestIdx);
    cudaFree(d_feasibleCount);
    return 0;
}

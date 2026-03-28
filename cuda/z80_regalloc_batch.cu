// z80_regalloc_batch.cu — GPU exhaustive register allocator for batch of functions
// Input: JSON batch of functions with interference graphs + constraints
// Output: optimal register assignment + cost for each function
//
// Build: nvcc -O3 -o z80_regalloc_batch z80_regalloc_batch.cu
// Usage: ./z80_regalloc_batch < batch.json > results.json

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

// Max limits per function
#define MAX_VREGS 16
#define MAX_LOCS  9    // A,B,C,D,E,H,L,F,SP (or custom)
#define MAX_EDGES 64
#define MAX_OPS   64
#define MAX_FIXED 8
#define MAX_SRCS  3

// Location names for output
static const char *locNames[] = {"A","F","B","C","D","E","H","L","SP"};

// Op types affecting cost
#define OP_ADD   0
#define OP_SUB   1
#define OP_MUL   2   // needs A
#define OP_DIV   3   // needs A
#define OP_CMP   4   // needs A
#define OP_LOAD  5   // LD dst,src
#define OP_STORE 6
#define OP_SHIFT 7   // needs A (or CB prefix for any)
#define OP_LOGIC 8   // AND/OR/XOR — needs A
#define OP_CALL  9
#define OP_NEG   10  // needs A
#define OP_MOVE  11  // LD r,r'
#define NUM_OP_TYPES 12

// Per-function descriptor (host-side, copied to GPU)
struct FuncDesc {
    int nVregs;
    int nLocs;
    int nEdges;
    int nOps;
    int nFixed;

    // Interference edges: vregs that can't share a location
    int edges[MAX_EDGES][2];

    // Operations: op type, dst vreg, src vregs, tied flag
    struct Op {
        int type;
        int dst;
        int srcs[MAX_SRCS];
        int nSrcs;
        int tied;  // dst must be same loc as srcs[0]
    } ops[MAX_OPS];

    // Fixed constraints: vreg → must be in specific location
    int fixedVreg[MAX_FIXED];
    int fixedLoc[MAX_FIXED];
};

// Cost model: cost of performing op_type when dst is in loc_d, src in loc_s
// Returns extra move cost (0 if natural, >0 if needs LD)
__device__ int op_cost(int opType, int locDst, int locSrc0, int locSrc1, int tied) {
    // A=0, F=1, B=2, C=3, D=4, E=5, H=6, L=7, SP=8

    int cost = 0;

    switch (opType) {
    case OP_ADD: case OP_SUB: case OP_CMP: case OP_LOGIC:
        // ALU ops: dst MUST be A (loc 0). src can be any reg.
        if (locDst != 0) cost += 4;  // LD A,dst needed
        // No extra cost for src — ADD A,r works for all r
        break;

    case OP_MUL: case OP_DIV:
        // Multiply/divide: needs A, result in A (mul8: A×reg, cost from table)
        if (locDst != 0) cost += 4;  // move to A
        if (locSrc0 == 0) cost += 4; // src can't be A (same as dst), need move
        break;

    case OP_SHIFT:
        // SRL/SLA/etc with CB prefix works on any reg (8T)
        // Without CB: RLCA/RRCA/RLA/RRA only work on A (4T)
        if (locDst == 0) cost += 0;  // fast path: 4T
        else cost += 4;              // CB prefix: 8T vs 4T = +4T
        break;

    case OP_NEG:
        // NEG only works on A
        if (locDst != 0) cost += 8;  // LD A,r + NEG + LD r,A
        break;

    case OP_MOVE: case OP_LOAD:
        // LD r,r' — 4T for any reg pair
        cost += 0;
        break;

    case OP_CALL:
        // CALL: return value in A. If dst != A, need move after.
        if (locDst != 0) cost += 4;
        break;

    default:
        break;
    }

    // Tied constraint: if dst must equal src[0] location
    if (tied && locDst != locSrc0) {
        cost += 8;  // save + restore around op
    }

    return cost;
}

// GPU kernel: evaluate all assignments for one function
__global__ void eval_assignments(
    int nVregs, int nLocs,
    const int *edges, int nEdges,       // [nEdges×2]
    const int *opTypes, const int *opDsts, const int *opSrc0s,
    const int *opSrc1s, const int *opTied, int nOps,
    const int *fixedVreg, const int *fixedLoc, int nFixed,
    uint32_t *bestCost, uint32_t *bestAssign,
    uint64_t totalAssignments)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= totalAssignments) return;

    // Decode assignment: tid → loc[0..nVregs-1]
    int assign[MAX_VREGS];
    uint64_t tmp = tid;
    for (int i = nVregs - 1; i >= 0; i--) {
        assign[i] = tmp % nLocs;
        tmp /= nLocs;
    }

    // Check fixed constraints
    for (int f = 0; f < nFixed; f++) {
        if (assign[fixedVreg[f]] != fixedLoc[f]) return;  // infeasible
    }

    // Check interference: no two connected vregs share a location
    for (int e = 0; e < nEdges; e++) {
        if (assign[edges[e * 2]] == assign[edges[e * 2 + 1]]) return;  // conflict
    }

    // Compute cost
    int totalCost = 0;
    for (int o = 0; o < nOps; o++) {
        int locD = assign[opDsts[o]];
        int locS0 = (opSrc0s[o] >= 0) ? assign[opSrc0s[o]] : -1;
        int locS1 = (opSrc1s[o] >= 0) ? assign[opSrc1s[o]] : -1;
        totalCost += op_cost(opTypes[o], locD, locS0, locS1, opTied[o]);
    }

    // Atomic min: pack cost + assignment index
    uint32_t packed = ((uint32_t)totalCost << 16) | (uint32_t)(tid & 0xFFFF);
    atomicMin(bestCost, packed);

    // If we're the best so far, store full assignment
    uint32_t cur = *bestCost;
    if ((cur >> 16) == (uint32_t)totalCost) {
        // Encode assignment as bits: 4 bits per vreg (up to 16 locs)
        uint32_t enc = 0;
        for (int i = 0; i < nVregs && i < 8; i++) {
            enc |= ((uint32_t)assign[i] & 0xF) << (i * 4);
        }
        atomicExch(bestAssign, enc);
    }
}

// ============================================================
// Minimal JSON parser (enough for our format)
// ============================================================

static char *readAll(FILE *f) {
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    return buf;
}

static const char *skipWS(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }

static const char *expectChar(const char *p, char c) {
    p = skipWS(p);
    if (*p == c) return p + 1;
    fprintf(stderr, "Expected '%c', got '%c'\n", c, *p);
    return NULL;
}

static const char *parseString(const char *p, char *out, int maxLen) {
    p = skipWS(p);
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < maxLen - 1) out[i++] = *p++;
    out[i] = 0;
    if (*p == '"') p++;
    return p;
}

static const char *parseInt(const char *p, int *out) {
    p = skipWS(p);
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    *out = neg ? -v : v;
    return p;
}

// Parse one function from JSON (simplified parser)
static const char *parseFunc(const char *p, FuncDesc *fd, char *name) {
    memset(fd, 0, sizeof(*fd));
    fd->nLocs = MAX_LOCS;

    p = expectChar(p, '{');
    while (p && *skipWS(p) != '}') {
        char key[64];
        p = parseString(p, key, sizeof(key));
        p = expectChar(p, ':');

        if (!strcmp(key, "name")) {
            p = parseString(p, name, 128);
        } else if (!strcmp(key, "nVregs")) {
            p = parseInt(p, &fd->nVregs);
        } else if (!strcmp(key, "nLocs")) {
            p = parseInt(p, &fd->nLocs);
        } else if (!strcmp(key, "edges")) {
            p = expectChar(p, '[');
            fd->nEdges = 0;
            while (*skipWS(p) != ']') {
                p = expectChar(p, '[');
                p = parseInt(p, &fd->edges[fd->nEdges][0]);
                p = expectChar(p, ',');
                p = parseInt(p, &fd->edges[fd->nEdges][1]);
                p = expectChar(p, ']');
                fd->nEdges++;
                if (*skipWS(p) == ',') p = skipWS(p) + 1;
            }
            p = expectChar(p, ']');
        } else if (!strcmp(key, "ops")) {
            p = expectChar(p, '[');
            fd->nOps = 0;
            while (*skipWS(p) != ']') {
                p = expectChar(p, '{');
                FuncDesc::Op *op = &fd->ops[fd->nOps];
                op->nSrcs = 0; op->tied = 0; op->dst = -1;
                op->srcs[0] = op->srcs[1] = op->srcs[2] = -1;
                while (*skipWS(p) != '}') {
                    char k2[32];
                    p = parseString(p, k2, sizeof(k2));
                    p = expectChar(p, ':');
                    if (!strcmp(k2, "op")) {
                        char opName[32];
                        p = parseString(p, opName, sizeof(opName));
                        if (!strcmp(opName,"add")) op->type = OP_ADD;
                        else if (!strcmp(opName,"sub")) op->type = OP_SUB;
                        else if (!strcmp(opName,"mul")) op->type = OP_MUL;
                        else if (!strcmp(opName,"div")) op->type = OP_DIV;
                        else if (!strcmp(opName,"cmp")||!strcmp(opName,"cmp_lt")||!strcmp(opName,"cmp_eq")) op->type = OP_CMP;
                        else if (!strcmp(opName,"shift")||!strcmp(opName,"shr")||!strcmp(opName,"shl")) op->type = OP_SHIFT;
                        else if (!strcmp(opName,"and")||!strcmp(opName,"or")||!strcmp(opName,"xor")) op->type = OP_LOGIC;
                        else if (!strcmp(opName,"neg")) op->type = OP_NEG;
                        else if (!strcmp(opName,"call")) op->type = OP_CALL;
                        else if (!strcmp(opName,"load")||!strcmp(opName,"ld")) op->type = OP_LOAD;
                        else op->type = OP_MOVE;
                    } else if (!strcmp(k2, "dst")) {
                        p = parseInt(p, &op->dst);
                    } else if (!strcmp(k2, "srcs")) {
                        p = expectChar(p, '[');
                        while (*skipWS(p) != ']') {
                            int s; p = parseInt(p, &s);
                            op->srcs[op->nSrcs++] = s;
                            if (*skipWS(p) == ',') p = skipWS(p) + 1;
                        }
                        p = expectChar(p, ']');
                    } else if (!strcmp(k2, "tied")) {
                        p = skipWS(p);
                        if (*p == '"') {
                            char v[16]; p = parseString(p, v, sizeof(v));
                            if (!strcmp(v, "true")) op->tied = 1;
                        } else if (!strncmp(p, "true", 4)) {
                            op->tied = 1; p += 4;
                        } else if (!strncmp(p, "false", 5)) {
                            op->tied = 0; p += 5;
                        }
                    }
                    if (*skipWS(p) == ',') p = skipWS(p) + 1;
                }
                p = expectChar(p, '}');
                fd->nOps++;
                if (*skipWS(p) == ',') p = skipWS(p) + 1;
            }
            p = expectChar(p, ']');
        } else if (!strcmp(key, "fixed")) {
            p = expectChar(p, '{');
            fd->nFixed = 0;
            while (*skipWS(p) != '}') {
                char vregStr[16]; p = parseString(p, vregStr, sizeof(vregStr));
                p = expectChar(p, ':');
                char locStr[16]; p = parseString(p, locStr, sizeof(locStr));
                fd->fixedVreg[fd->nFixed] = atoi(vregStr);
                for (int i = 0; i < MAX_LOCS; i++)
                    if (!strcmp(locStr, locNames[i])) fd->fixedLoc[fd->nFixed] = i;
                fd->nFixed++;
                if (*skipWS(p) == ',') p = skipWS(p) + 1;
            }
            p = expectChar(p, '}');
        } else {
            // Skip unknown value
            p = skipWS(p);
            if (*p == '"') { char tmp[256]; p = parseString(p, tmp, sizeof(tmp)); }
            else if (*p == '[') { int depth=1; p++; while(depth>0){if(*p=='[')depth++;if(*p==']')depth--;p++;} }
            else if (*p == '{') { int depth=1; p++; while(depth>0){if(*p=='{')depth++;if(*p=='}')depth--;p++;} }
            else { while(*p && *p!=',' && *p!='}') p++; }
        }
        if (*skipWS(p) == ',') p = skipWS(p) + 1;
    }
    p = expectChar(p, '}');
    return p;
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

int main(int argc, char *argv[]) {
    int gpuId = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--gpu") && i+1 < argc) gpuId = atoi(argv[++i]);

    cudaSetDevice(gpuId);

    // Read input
    char *json = readAll(stdin);
    const char *p = skipWS(json);

    printf("[\n");
    int funcIdx = 0;

    // Parse array of functions
    p = expectChar(p, '[');
    while (p && *skipWS(p) != ']') {
        FuncDesc fd;
        char name[128] = "unknown";
        p = parseFunc(p, &fd, name);
        if (*skipWS(p) == ',') p = skipWS(p) + 1;

        if (fd.nVregs > MAX_VREGS) {
            fprintf(stderr, "%s: %d vregs > MAX_VREGS %d, skipping\n", name, fd.nVregs, MAX_VREGS);
            if (funcIdx > 0) printf(",\n");
            printf("  {\"name\":\"%s\",\"status\":\"skipped\",\"reason\":\"too many vregs\"}", name);
            funcIdx++;
            continue;
        }

        uint64_t totalAssign = ipow(fd.nLocs, fd.nVregs);
        fprintf(stderr, "[%d] %s: %d vregs, %d locs, %d edges, %d ops → %llu assignments\n",
                funcIdx, name, fd.nVregs, fd.nLocs, fd.nEdges, fd.nOps,
                (unsigned long long)totalAssign);

        // Upload data to GPU
        int *d_edges, *d_opTypes, *d_opDsts, *d_opSrc0s, *d_opSrc1s, *d_opTied;
        int *d_fixedVreg, *d_fixedLoc;
        uint32_t *d_bestCost, *d_bestAssign;

        int h_edges[MAX_EDGES * 2];
        for (int i = 0; i < fd.nEdges; i++) { h_edges[i*2] = fd.edges[i][0]; h_edges[i*2+1] = fd.edges[i][1]; }

        int h_opTypes[MAX_OPS], h_opDsts[MAX_OPS], h_opSrc0s[MAX_OPS], h_opSrc1s[MAX_OPS], h_opTied[MAX_OPS];
        for (int i = 0; i < fd.nOps; i++) {
            h_opTypes[i] = fd.ops[i].type;
            h_opDsts[i] = fd.ops[i].dst;
            h_opSrc0s[i] = fd.ops[i].srcs[0];
            h_opSrc1s[i] = fd.ops[i].srcs[1];
            h_opTied[i] = fd.ops[i].tied;
        }

        cudaMalloc(&d_edges, fd.nEdges * 2 * sizeof(int));
        cudaMalloc(&d_opTypes, fd.nOps * sizeof(int));
        cudaMalloc(&d_opDsts, fd.nOps * sizeof(int));
        cudaMalloc(&d_opSrc0s, fd.nOps * sizeof(int));
        cudaMalloc(&d_opSrc1s, fd.nOps * sizeof(int));
        cudaMalloc(&d_opTied, fd.nOps * sizeof(int));
        cudaMalloc(&d_fixedVreg, fd.nFixed * sizeof(int));
        cudaMalloc(&d_fixedLoc, fd.nFixed * sizeof(int));
        cudaMalloc(&d_bestCost, sizeof(uint32_t));
        cudaMalloc(&d_bestAssign, sizeof(uint32_t));

        cudaMemcpy(d_edges, h_edges, fd.nEdges * 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_opTypes, h_opTypes, fd.nOps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_opDsts, h_opDsts, fd.nOps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_opSrc0s, h_opSrc0s, fd.nOps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_opSrc1s, h_opSrc1s, fd.nOps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_opTied, h_opTied, fd.nOps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fixedVreg, fd.fixedVreg, fd.nFixed * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fixedLoc, fd.fixedLoc, fd.nFixed * sizeof(int), cudaMemcpyHostToDevice);

        uint32_t initCost = 0xFFFFFFFF;
        uint32_t initAssign = 0;
        cudaMemcpy(d_bestCost, &initCost, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestAssign, &initAssign, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Launch
        int bs = 256;
        uint64_t batch = (uint64_t)bs * 65535;
        for (uint64_t off = 0; off < totalAssign; off += batch) {
            uint64_t cnt = totalAssign - off;
            if (cnt > batch) cnt = batch;
            // Pass offset via grid offset trick: add to tid in kernel
            // Actually, kernel uses global tid directly. Just launch enough blocks.
        }
        // Simpler: one big launch
        uint64_t nBlocks = (totalAssign + bs - 1) / bs;
        if (nBlocks > 2147483647ULL) {
            // Too many blocks, split into batches
            fprintf(stderr, "  %llu blocks, batching...\n", (unsigned long long)nBlocks);
            for (uint64_t off = 0; off < totalAssign; off += batch) {
                uint64_t cnt = totalAssign - off;
                if (cnt > batch) cnt = batch;
                uint32_t nb = (uint32_t)((cnt + bs - 1) / bs);
                eval_assignments<<<nb, bs>>>(
                    fd.nVregs, fd.nLocs,
                    d_edges, fd.nEdges,
                    d_opTypes, d_opDsts, d_opSrc0s, d_opSrc1s, d_opTied, fd.nOps,
                    d_fixedVreg, d_fixedLoc, fd.nFixed,
                    d_bestCost, d_bestAssign, totalAssign);
                cudaDeviceSynchronize();
            }
        } else {
            eval_assignments<<<(uint32_t)nBlocks, bs>>>(
                fd.nVregs, fd.nLocs,
                d_edges, fd.nEdges,
                d_opTypes, d_opDsts, d_opSrc0s, d_opSrc1s, d_opTied, fd.nOps,
                d_fixedVreg, d_fixedLoc, fd.nFixed,
                d_bestCost, d_bestAssign, totalAssign);
            cudaDeviceSynchronize();
        }

        // Read results
        uint32_t bestCostVal, bestAssignVal;
        cudaMemcpy(&bestCostVal, d_bestCost, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestAssignVal, d_bestAssign, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Output
        if (funcIdx > 0) printf(",\n");
        if (bestCostVal == 0xFFFFFFFF) {
            printf("  {\"name\":\"%s\",\"status\":\"infeasible\"}", name);
            fprintf(stderr, "  → INFEASIBLE\n");
        } else {
            int cost = bestCostVal >> 16;
            int assign[MAX_VREGS];
            for (int i = 0; i < fd.nVregs && i < 8; i++)
                assign[i] = (bestAssignVal >> (i * 4)) & 0xF;

            printf("  {\"name\":\"%s\",\"status\":\"optimal\",\"cost\":%d,\"assignment\":{", name, cost);
            for (int i = 0; i < fd.nVregs; i++) {
                if (i > 0) printf(",");
                printf("\"%d\":\"%s\"", i, assign[i] < MAX_LOCS ? locNames[assign[i]] : "?");
            }
            printf("}}");
            fprintf(stderr, "  → cost=%d: ", cost);
            for (int i = 0; i < fd.nVregs; i++)
                fprintf(stderr, "r%d=%s ", i, assign[i] < MAX_LOCS ? locNames[assign[i]] : "?");
            fprintf(stderr, "\n");
        }

        cudaFree(d_edges); cudaFree(d_opTypes); cudaFree(d_opDsts);
        cudaFree(d_opSrc0s); cudaFree(d_opSrc1s); cudaFree(d_opTied);
        cudaFree(d_fixedVreg); cudaFree(d_fixedLoc);
        cudaFree(d_bestCost); cudaFree(d_bestAssign);
        funcIdx++;
    }

    printf("\n]\n");
    free(json);
    fprintf(stderr, "Done: %d functions processed\n", funcIdx);
    return 0;
}

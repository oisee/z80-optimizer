// z80_shuffle.cu — Brute-force optimal register shuffle sequences
// Given source and target register assignments, find shortest move sequence.
// Used for island boundary stitching in register allocation.
//
// Ops: LD r,r' (42 combinations for 7 GPR) + EX DE,HL + PUSH rr / POP rr
// State: 7 registers (A,B,C,D,E,H,L)
//
// Build: nvcc -O3 -o z80_shuffle z80_shuffle.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 7 GPR: A=0, B=1, C=2, D=3, E=4, H=5, L=6
#define NREGS 7

// Ops: LD dst,src for all dst≠src pairs = 42 + EX DE,HL + EX AF,AF' = 44
// But keep it simple: just LD r,r' (42) + EX DE,HL (1) = 43 ops
#define NUM_OPS 43

static const char *regNames[] = {"A","B","C","D","E","H","L"};

struct ShuffleOp {
    uint8_t type; // 0=LD dst,src  1=EX DE,HL
    uint8_t dst, src;
};

static ShuffleOp ops[NUM_OPS];
static int nOps = 0;

static void initOps() {
    // LD r,r' for all 42 pairs
    for (int d = 0; d < NREGS; d++)
        for (int s = 0; s < NREGS; s++)
            if (d != s) {
                ops[nOps++] = {0, (uint8_t)d, (uint8_t)s};
            }
    // EX DE,HL
    ops[nOps++] = {1, 0, 0};
}

static void applyOp(uint8_t regs[NREGS], int opIdx) {
    ShuffleOp &op = ops[opIdx];
    if (op.type == 0) { // LD dst,src
        regs[op.dst] = regs[op.src];
    } else { // EX DE,HL
        uint8_t td = regs[3], te = regs[4];
        regs[3] = regs[5]; regs[4] = regs[6];
        regs[5] = td; regs[6] = te;
    }
}

static const char *opName(int opIdx) {
    static char buf[32];
    ShuffleOp &op = ops[opIdx];
    if (op.type == 0) {
        snprintf(buf, sizeof(buf), "LD %s,%s", regNames[op.dst], regNames[op.src]);
    } else {
        snprintf(buf, sizeof(buf), "EX DE,HL");
    }
    return buf;
}

static int opCost(int opIdx) {
    return ops[opIdx].type == 0 ? 4 : 4; // LD r,r' = 4T, EX DE,HL = 4T
}

// Check if regs matches target
static bool matches(const uint8_t regs[NREGS], const uint8_t target[NREGS], int nActive) {
    for (int i = 0; i < nActive; i++)
        if (regs[i] != target[i]) return false;
    return true;
}

// BFS search for shortest shuffle
typedef struct {
    int ops[8];
    int len;
    int cost;
} Solution;

static Solution solve(const uint8_t src[NREGS], const uint8_t tgt[NREGS], int nActive, int maxLen) {
    Solution best = {{}, maxLen + 1, 9999};

    // DFS with iterative deepening
    int opsSeq[8];
    
    for (int depth = 0; depth <= maxLen; depth++) {
        // Try all sequences of length 'depth'
        // For depth ≤ 4: 43^4 = 3.4M — instant
        
        uint8_t regs[NREGS];
        
        if (depth == 0) {
            if (matches(src, tgt, nActive)) {
                best.len = 0; best.cost = 0;
                return best;
            }
            continue;
        }
        
        // Enumerate all sequences
        int indices[8] = {};
        uint64_t total = 1;
        for (int i = 0; i < depth; i++) total *= nOps;
        
        for (uint64_t seq = 0; seq < total; seq++) {
            // Decode
            uint64_t tmp = seq;
            for (int i = depth - 1; i >= 0; i--) {
                indices[i] = tmp % nOps;
                tmp /= nOps;
            }
            
            // Apply
            memcpy(regs, src, NREGS);
            for (int i = 0; i < depth; i++)
                applyOp(regs, indices[i]);
            
            if (matches(regs, tgt, nActive)) {
                int cost = 0;
                for (int i = 0; i < depth; i++) cost += opCost(indices[i]);
                if (depth < best.len || (depth == best.len && cost < best.cost)) {
                    best.len = depth;
                    best.cost = cost;
                    memcpy(best.ops, indices, depth * sizeof(int));
                }
                return best; // first found at this depth = shortest
            }
        }
    }
    return best;
}

int main(int argc, char *argv[]) {
    initOps();
    int maxLen = 4;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
    }
    
    // Demo: common shuffles needed at island boundaries
    // Each vreg has a "color" (0-6 = which register it's in)
    // Shuffle = permute registers to match target assignment
    
    printf("=== Register Shuffle Brute-Force (43 ops, max-len %d) ===\n\n", maxLen);
    
    // Test cases: common boundary patterns
    struct { const char *name; uint8_t src[7]; uint8_t tgt[7]; int n; } tests[] = {
        {"A↔B swap",      {0,1,2,3,4,5,6}, {1,0,2,3,4,5,6}, 7},
        {"A→B→C rotate",  {0,1,2,3,4,5,6}, {2,0,1,3,4,5,6}, 7},
        {"DE↔HL swap",    {0,1,2,3,4,5,6}, {0,1,2,5,6,3,4}, 7},
        {"A→H, H→A",     {0,1,2,3,4,5,6}, {5,1,2,3,4,0,6}, 7},
        {"All rotate +1", {0,1,2,3,4,5,6}, {6,0,1,2,3,4,5}, 7},
        {"BC↔DE swap",    {0,1,2,3,4,5,6}, {0,3,4,1,2,5,6}, 7},
        {"A→L, L→H, H→A",{0,1,2,3,4,5,6}, {5,1,2,3,4,6,0}, 7},
    };
    
    for (int t = 0; t < 7; t++) {
        Solution sol = solve(tests[t].src, tests[t].tgt, tests[t].n, maxLen);
        printf("%-20s", tests[t].name);
        if (sol.len <= maxLen) {
            printf("%d insts, %dT:", sol.len, sol.cost);
            for (int i = 0; i < sol.len; i++)
                printf(" %s", opName(sol.ops[i]));
            printf("\n");
        } else {
            printf("NOT FOUND at len %d\n", maxLen);
        }
    }
    
    return 0;
}

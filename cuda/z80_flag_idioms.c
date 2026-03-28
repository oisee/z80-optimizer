// Brute-force flag materialization idioms for Z80
// Finds shortest branchless sequences for all flag↔register conversions
// Build: gcc -O3 -march=native -o z80_flag_idioms z80_flag_idioms.c -lpthread
// Usage: ./z80_flag_idioms [max-depth]  (default 6)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DEPTH 8

// Op pool: single-byte + key 2-byte Z80 instructions for flag manipulation
// Each op: modifies A, Z, CY (and possibly B)
#define NUM_OPS 26

typedef struct {
    uint8_t a;
    uint8_t b;  // B register (temp)
    int z;      // Z flag (1=set, 0=clear)
    int cy;     // CY flag
} State;

static inline State run_op(State s, int op) {
    uint16_t r;
    switch (op) {
    case 0:  // SCF: set carry
        s.cy = 1; break;
    case 1:  // CCF: complement carry
        s.cy = !s.cy; break;
    case 2:  // CPL: complement A (no flag change except H,N)
        s.a = ~s.a; break;
    case 3:  // NEG: A = -A
        s.cy = (s.a != 0); s.a = (uint8_t)(0 - s.a);
        s.z = (s.a == 0); break;
    case 4:  // RLCA: rotate left, bit7→CY and bit0
        s.cy = (s.a >> 7) & 1;
        s.a = ((s.a << 1) | (s.a >> 7)) & 0xFF; break;
    case 5:  // RRCA: rotate right, bit0→CY and bit7
        s.cy = s.a & 1;
        s.a = ((s.a >> 1) | (s.a << 7)) & 0xFF; break;
    case 6:  // RLA: rotate left through carry
        r = (s.a << 1) | s.cy;
        s.cy = (s.a >> 7) & 1;
        s.a = r & 0xFF; break;
    case 7:  // RRA: rotate right through carry
        r = (s.a >> 1) | (s.cy << 7);
        s.cy = s.a & 1;
        s.a = r & 0xFF; break;
    case 8:  // ADD A,A
        r = (uint16_t)s.a + s.a;
        s.cy = r > 0xFF; s.a = r & 0xFF;
        s.z = (s.a == 0); break;
    case 9:  // ADC A,A
        r = (uint16_t)s.a + s.a + s.cy;
        s.cy = r > 0xFF; s.a = r & 0xFF;
        s.z = (s.a == 0); break;
    case 10: // SUB A (A -= A = 0)
        s.a = 0; s.z = 1; s.cy = 0; break;
    case 11: // SBC A,A: A = CY ? 0xFF : 0x00
        { int cc = s.cy; s.a = cc ? 0xFF : 0x00;
          s.cy = cc; s.z = !cc; } break;
    case 12: // AND A: Z = (A==0)
        s.z = (s.a == 0); s.cy = 0; break;
    case 13: // OR A: Z = (A==0)
        s.z = (s.a == 0); s.cy = 0; break;
    case 14: // XOR A: A=0, Z=1, CY=0
        s.a = 0; s.z = 1; s.cy = 0; break;
    case 15: // INC A
        s.a++; s.z = (s.a == 0); break;  // CY unaffected!
    case 16: // DEC A
        s.a--; s.z = (s.a == 0); break;  // CY unaffected!
    case 17: // LD A,0
        s.a = 0; break;  // no flag changes
    case 18: // LD A,1
        s.a = 1; break;
    case 19: // CP 0: Z = (A==0), CY = 0
        s.z = (s.a == 0); s.cy = 0; break;
    case 20: // CP 1: Z = (A==1), CY = (A < 1)
        s.z = (s.a == 1); s.cy = (s.a < 1); break;
    case 21: // ADC A,0
        { uint16_t t = (uint16_t)s.a + s.cy;
          s.cy = t > 0xFF; s.a = t & 0xFF;
          s.z = (s.a == 0); } break;
    case 22: // AND 1
        s.a &= 1; s.z = (s.a == 0); s.cy = 0; break;
    case 23: // OR 1
        s.a |= 1; s.z = 0; s.cy = 0; break;
    case 24: // XOR 1
        s.a ^= 1; s.z = (s.a == 0); s.cy = 0; break;
    case 25: // XOR 0xFF (= CPL but sets flags)
        s.a ^= 0xFF; s.z = (s.a == 0); s.cy = 0; break;
    }
    return s;
}

static const char *opNames[] = {
    "SCF", "CCF", "CPL", "NEG", "RLCA", "RRCA", "RLA", "RRA",
    "ADD A,A", "ADC A,A", "SUB A", "SBC A,A", "AND A", "OR A", "XOR A",
    "INC A", "DEC A", "LD A,0", "LD A,1", "CP 0", "CP 1",
    "ADC A,0", "AND 1", "OR 1", "XOR 1", "XOR 0xFF"
};

// T-states for each op
static const int opTstates[] = {
    4, 4, 4, 8, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 7, 7, 7, 7,
    7, 7, 7, 7, 7
};

// Bytes for each op
static const int opBytes[] = {
    1, 1, 1, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 2, 2, 2,
    2, 2, 2, 2, 2
};

// Clobber info: does this op modify A?
static const int clobberA[] = {
    0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 1,
    1, 1, 1, 1, 0, 0,
    1, 1, 1, 1, 1
};

// Conversion types
typedef enum {
    Z_TO_A01,      // Z→A: Z=1→A=1, Z=0→A=0 (bool 0/1)
    Z_TO_AFF,      // Z→A: Z=1→A=0xFF, Z=0→A=0x00 (bool 0/FF)
    Z_TO_CY,       // Z→CY
    CY_TO_A01,     // CY→A (0/1)
    CY_TO_AFF,     // CY→A (0/FF)
    CY_TO_Z,       // CY→Z
    A01_TO_Z,      // A(0/1)→Z: A=0→Z=1, A=1→Z=0
    A01_TO_CY,     // A(0/1)→CY: A=0→CY=0, A=1→CY=1
    AFF_TO_Z,      // A(0/FF)→Z: A=0→Z=1, A=0xFF→Z=0
    AFF_TO_CY,     // A(0/FF)→CY: A=0→CY=0, A=0xFF→CY=1
    A01_TO_AFF,    // A(0/1)→A(0/FF)
    AFF_TO_A01,    // A(0/FF)→A(0/1)
    NUM_CONVERSIONS
} Conversion;

static const char *convNames[] = {
    "Z→A(0/1)", "Z→A(0/FF)", "Z→CY",
    "CY→A(0/1)", "CY→A(0/FF)", "CY→Z",
    "A(0/1)→Z", "A(0/1)→CY",
    "A(0/FF)→Z", "A(0/FF)→CY",
    "A(0/1)→A(0/FF)", "A(0/FF)→A(0/1)"
};

// For each conversion: define 2 test cases (input0, input1)
// Each test: initial state → expected output
typedef struct {
    State init;     // initial state
    // What to check in output:
    int check_a;    // -1 = don't check, else expected A
    int check_z;    // -1 = don't check, else expected Z
    int check_cy;   // -1 = don't check, else expected CY
} TestCase;

static TestCase tests[NUM_CONVERSIONS][2];

static void setup_tests(void) {
    // Z→A(0/1): Z=0→A=0, Z=1→A=1. A is unknown/garbage initially.
    // Test with A=0x42 (garbage) to ensure sequence doesn't depend on A
    tests[Z_TO_A01][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=0, .check_z=-1, .check_cy=-1}; // Z=0→A=0
    tests[Z_TO_A01][1] = (TestCase){{0x42, 0, 1, 0}, .check_a=1, .check_z=-1, .check_cy=-1}; // Z=1→A=1

    tests[Z_TO_AFF][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=0x00, .check_z=-1, .check_cy=-1};
    tests[Z_TO_AFF][1] = (TestCase){{0x42, 0, 1, 0}, .check_a=0xFF, .check_z=-1, .check_cy=-1};

    tests[Z_TO_CY][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=-1, .check_z=-1, .check_cy=0};
    tests[Z_TO_CY][1] = (TestCase){{0x42, 0, 1, 0}, .check_a=-1, .check_z=-1, .check_cy=1};

    // CY→A: CY=0→A=0, CY=1→A=1. A is garbage.
    tests[CY_TO_A01][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=0, .check_z=-1, .check_cy=-1};
    tests[CY_TO_A01][1] = (TestCase){{0x42, 0, 0, 1}, .check_a=1, .check_z=-1, .check_cy=-1};

    tests[CY_TO_AFF][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=0x00, .check_z=-1, .check_cy=-1};
    tests[CY_TO_AFF][1] = (TestCase){{0x42, 0, 0, 1}, .check_a=0xFF, .check_z=-1, .check_cy=-1};

    tests[CY_TO_Z][0] = (TestCase){{0x42, 0, 0, 0}, .check_a=-1, .check_z=0, .check_cy=-1};
    tests[CY_TO_Z][1] = (TestCase){{0x42, 0, 0, 1}, .check_a=-1, .check_z=1, .check_cy=-1};

    // A(0/1)→Z: A=0→Z=1 (zero), A=1→Z=0 (nonzero)
    tests[A01_TO_Z][0] = (TestCase){{0, 0, 0, 0}, .check_a=-1, .check_z=1, .check_cy=-1};
    tests[A01_TO_Z][1] = (TestCase){{1, 0, 0, 0}, .check_a=-1, .check_z=0, .check_cy=-1};

    // A(0/1)→CY: A=0→CY=0, A=1→CY=1
    tests[A01_TO_CY][0] = (TestCase){{0, 0, 0, 0}, .check_a=-1, .check_z=-1, .check_cy=0};
    tests[A01_TO_CY][1] = (TestCase){{1, 0, 0, 0}, .check_a=-1, .check_z=-1, .check_cy=1};

    // A(0/FF)→Z: A=0→Z=1, A=0xFF→Z=0
    tests[AFF_TO_Z][0] = (TestCase){{0x00, 0, 0, 0}, .check_a=-1, .check_z=1, .check_cy=-1};
    tests[AFF_TO_Z][1] = (TestCase){{0xFF, 0, 0, 0}, .check_a=-1, .check_z=0, .check_cy=-1};

    // A(0/FF)→CY: A=0→CY=0, A=0xFF→CY=1
    tests[AFF_TO_CY][0] = (TestCase){{0x00, 0, 0, 0}, .check_a=-1, .check_z=-1, .check_cy=0};
    tests[AFF_TO_CY][1] = (TestCase){{0xFF, 0, 0, 0}, .check_a=-1, .check_z=-1, .check_cy=1};

    // A(0/1)→A(0/FF): A=0→A=0, A=1→A=0xFF
    tests[A01_TO_AFF][0] = (TestCase){{0, 0, 0, 0}, .check_a=0x00, .check_z=-1, .check_cy=-1};
    tests[A01_TO_AFF][1] = (TestCase){{1, 0, 0, 0}, .check_a=0xFF, .check_z=-1, .check_cy=-1};

    // A(0/FF)→A(0/1): A=0→A=0, A=0xFF→A=1
    tests[AFF_TO_A01][0] = (TestCase){{0x00, 0, 0, 0}, .check_a=0, .check_z=-1, .check_cy=-1};
    tests[AFF_TO_A01][1] = (TestCase){{0xFF, 0, 0, 0}, .check_a=1, .check_z=-1, .check_cy=-1};
}

static int check_result(State s, TestCase *tc) {
    if (tc->check_a >= 0 && s.a != (uint8_t)tc->check_a) return 0;
    if (tc->check_z >= 0 && s.z != tc->check_z) return 0;
    if (tc->check_cy >= 0 && s.cy != tc->check_cy) return 0;
    return 1;
}

// Additional robustness: test with multiple garbage initial values
static int test_robust(int conv, uint8_t *ops, int len) {
    // Test cases with different garbage A values (for flag-input conversions)
    uint8_t garbageA[] = {0x00, 0x42, 0x80, 0xFF, 0x01, 0x7F};
    int numGarbage = 6;

    for (int tc = 0; tc < 2; tc++) {
        TestCase *t = &tests[conv][tc];

        // For conversions where A is the input, only test with the specified A
        if (conv >= A01_TO_Z) {
            State s = t->init;
            for (int i = 0; i < len; i++) s = run_op(s, ops[i]);
            if (!check_result(s, t)) return 0;
        } else {
            // For flag-input conversions, test with various garbage A values
            for (int g = 0; g < numGarbage; g++) {
                State s = t->init;
                s.a = garbageA[g];  // garbage A
                for (int i = 0; i < len; i++) s = run_op(s, ops[i]);
                if (!check_result(s, t)) return 0;
            }
            // Also test with various garbage CY (for Z-input conversions)
            if (conv <= Z_TO_CY) {
                for (int g = 0; g < 2; g++) {
                    State s = t->init;
                    s.cy = g;  // test both CY states
                    s.a = 0x42;
                    for (int i = 0; i < len; i++) s = run_op(s, ops[i]);
                    if (!check_result(s, t)) return 0;
                }
            }
        }
    }
    return 1;
}

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

int main(int argc, char *argv[]) {
    int maxDepth = 6;
    if (argc > 1) maxDepth = atoi(argv[1]);

    setup_tests();

    printf("Z80 Flag Materialization Idiom Search\n");
    printf("Pool: %d ops, max depth: %d\n\n", NUM_OPS, maxDepth);

    // For each conversion, find shortest sequence
    for (int conv = 0; conv < NUM_CONVERSIONS; conv++) {
        printf("=== %s ===\n", convNames[conv]);
        int found = 0;
        int bestLen = maxDepth + 1;
        int bestTstates = 9999;
        uint8_t bestOps[MAX_DEPTH];
        int bestClobA = 1;

        // Also track best non-clobber-A version
        int bestLen_nc = maxDepth + 1;
        int bestTstates_nc = 9999;
        uint8_t bestOps_nc[MAX_DEPTH];

        for (int len = 1; len <= maxDepth && len <= bestLen + 1; len++) {
            uint64_t total = ipow(NUM_OPS, len);

            for (uint64_t idx = 0; idx < total; idx++) {
                uint8_t ops[MAX_DEPTH];
                uint64_t tmp = idx;
                for (int i = len - 1; i >= 0; i--) {
                    ops[i] = tmp % NUM_OPS;
                    tmp /= NUM_OPS;
                }

                if (!test_robust(conv, ops, len)) continue;

                // Calculate cost
                int tstates = 0, bytes = 0, clob = 0;
                for (int i = 0; i < len; i++) {
                    tstates += opTstates[ops[i]];
                    bytes += opBytes[ops[i]];
                    if (clobberA[ops[i]]) clob = 1;
                }

                // Best overall
                if (len < bestLen || (len == bestLen && tstates < bestTstates)) {
                    bestLen = len; bestTstates = tstates; bestClobA = clob;
                    memcpy(bestOps, ops, len);
                    found = 1;

                    printf("  len=%d %dT %dB%s:", len, tstates, bytes, clob?" [clobA]":"");
                    for (int i = 0; i < len; i++) printf(" %s;", opNames[ops[i]]);
                    printf("\n");
                }

                // Best without clobbering A (for A→flag conversions)
                if (!clob && (len < bestLen_nc || (len == bestLen_nc && tstates < bestTstates_nc))) {
                    bestLen_nc = len; bestTstates_nc = tstates;
                    memcpy(bestOps_nc, ops, len);

                    if (conv >= A01_TO_CY || conv == Z_TO_CY || conv == CY_TO_Z) {
                        printf("  len=%d %dT %dB [preserveA]:", len, tstates, bytes);
                        for (int i = 0; i < len; i++) printf(" %s;", opNames[ops[i]]);
                        printf("\n");
                    }
                }
            }
        }

        if (found) {
            printf("  BEST: %d instr, %dT%s\n\n", bestLen, bestTstates, bestClobA?" (clobbers A)":"");
        } else {
            printf("  NOT FOUND at depth %d\n\n", maxDepth);
        }
    }

    return 0;
}

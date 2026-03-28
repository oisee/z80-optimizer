// Branchless primitives: ABS, MIN, MAX, CLAMP + verification
// Build: gcc -O3 -o z80_branchless z80_branchless.c

#include <stdio.h>
#include <stdint.h>

// === ABS(A) signed, branchless ===
// abs(A) = (A XOR mask) - mask, where mask = sign_extend(bit7)
// Z80: LD B,A; RLCA; SBC A,A; LD C,A; XOR B; SUB C
// 6 instr, 24T, 6 bytes, clobbers B,C
uint8_t branchless_abs(uint8_t a) {
    uint8_t b = a;            // LD B,A
    // RLCA: bit7 → CY
    int cy = (a >> 7) & 1;
    // SBC A,A: mask
    uint8_t mask = cy ? 0xFF : 0x00;
    uint8_t c = mask;         // LD C,A
    uint8_t r = b ^ mask;     // XOR B
    r = r - c;                // SUB C
    return r;
}

// === MIN(A,B) unsigned, branchless ===
// LD C,A; SUB B; SBC A,A; LD D,A; LD A,C; XOR B; AND D; XOR B
// 8 instr, 32T, clobbers C,D
uint8_t branchless_min(uint8_t a, uint8_t b) {
    uint8_t c = a;            // LD C,A
    int cy = (a < b);         // SUB B (CY = A < B)
    uint8_t mask = cy ? 0xFF : 0x00;  // SBC A,A
    uint8_t d = mask;         // LD D,A
    uint8_t r = c;            // LD A,C
    r ^= b;                   // XOR B
    r &= d;                   // AND D
    r ^= b;                   // XOR B → CY ? C : B = (A<B) ? A : B
    return r;
}

// === MAX(A,B) unsigned, branchless ===
// LD C,A; SUB B; SBC A,A; LD D,A; LD A,B; XOR C; AND D; XOR C
// 8 instr, 32T, clobbers C,D
uint8_t branchless_max(uint8_t a, uint8_t b) {
    uint8_t c = a;
    int cy = (a < b);         // SUB B
    uint8_t mask = cy ? 0xFF : 0x00;
    uint8_t d = mask;
    uint8_t r = b;            // LD A,B (swap: select B when CY, else C)
    r ^= c;                   // XOR C
    r &= d;
    r ^= c;                   // → CY ? B : C = (A<B) ? B : A
    return r;
}

// === CLAMP(A, lo, hi) unsigned ===
// = MAX(lo, MIN(A, hi))
// MIN(A,hi): LD C,A; SUB hi_reg; SBC A,A; LD D,A; LD A,C; XOR hi_reg; AND D; XOR hi_reg
// Then MAX with lo: same pattern
// 16 instr total, 64T, or optimize:

// Optimized: two CMOVs chained
// Assume lo in E, hi in B
// Step 1: MIN(A, B) → clamp high
// Step 2: MAX(result, E) → clamp low
uint8_t branchless_clamp(uint8_t a, uint8_t lo, uint8_t hi) {
    // MIN(A, hi)
    uint8_t t = branchless_min(a, hi);
    // MAX(t, lo)
    return branchless_max(t, lo);
}

// === Signed MIN/MAX ===
// For signed: CY after SUB doesn't give signed comparison
// Need: XOR 0x80 to convert signed→unsigned, then compare
// Or: SUB B; then check overflow... complex
// Simplest: XOR 0x80 both operands, unsigned MIN, XOR 0x80 result
// LD C,A; XOR 0x80; LD A,B; XOR 0x80; (now compare unsigned)
// ... adds 4 instructions overhead

int main() {
    int fail;

    // === Verify ABS ===
    fail = 0;
    for (int i = -128; i <= 127; i++) {
        uint8_t a = (uint8_t)(int8_t)i;
        uint8_t got = branchless_abs(a);
        uint8_t exp = (uint8_t)(i < 0 ? -i : i);
        if (got != exp) { if (fail < 5) printf("ABS FAIL: %d → %d, exp %d\n", i, got, exp); fail++; }
    }
    printf("ABS(A) signed: %d/256 correct → %s\n", 256-fail, fail==0?"VERIFIED ✓":"BUG");
    printf("  Z80: LD B,A; RLCA; SBC A,A; LD C,A; XOR B; SUB C\n");
    printf("  6 instr, 24T, 6 bytes, clobbers A,B,C\n\n");

    // === Verify MIN ===
    fail = 0;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++) {
            uint8_t got = branchless_min(a, b);
            uint8_t exp = a < b ? a : b;
            if (got != exp) fail++;
        }
    printf("MIN(A,B) unsigned: %d/65536 correct → %s\n", 65536-fail, fail==0?"VERIFIED ✓":"BUG");
    printf("  Z80: LD C,A; SUB B; SBC A,A; LD D,A; LD A,C; XOR B; AND D; XOR B\n");
    printf("  8 instr, 32T, 8 bytes, clobbers A,C,D\n\n");

    // === Verify MAX ===
    fail = 0;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++) {
            uint8_t got = branchless_max(a, b);
            uint8_t exp = a > b ? a : b;
            if (got != exp) fail++;
        }
    printf("MAX(A,B) unsigned: %d/65536 correct → %s\n", 65536-fail, fail==0?"VERIFIED ✓":"BUG");
    printf("  Z80: LD C,A; SUB B; SBC A,A; LD D,A; LD A,B; XOR C; AND D; XOR C\n");
    printf("  8 instr, 32T, 8 bytes, clobbers A,C,D\n\n");

    // === Verify CLAMP ===
    fail = 0;
    for (int a = 0; a < 256; a++)
        for (int lo = 0; lo < 256; lo += 17)
            for (int hi = lo; hi < 256; hi += 19) {
                uint8_t got = branchless_clamp(a, lo, hi);
                uint8_t exp = a < lo ? lo : (a > hi ? hi : a);
                if (got != exp) fail++;
            }
    printf("CLAMP(A,lo,hi): tested → %s\n", fail==0?"VERIFIED ✓":"BUG");
    printf("  = MAX(lo, MIN(A, hi)) — two chained CMOVs\n");
    printf("  16 instr, 64T, clobbers A,B,C,D,E\n\n");

    // === Division approximations ===
    printf("=== Division by constant (multiply-and-shift) ===\n\n");

    struct { int div; int mul; int shift; const char *z80; } divs[] = {
        {3,  86, 8, "LD B,A; LD C,86; CALL mul8; LD A,H"},        // 86/256 ≈ 0.336
        {5,  52, 8, "LD B,A; LD C,52; CALL mul8; LD A,H"},        // 52/256 ≈ 0.203
        {7,  37, 8, "LD B,A; LD C,37; CALL mul8; LD A,H"},        // 37/256 ≈ 0.144
        {10, 26, 8, "LD B,A; LD C,26; CALL mul8; LD A,H"},        // 26/256 ≈ 0.102
    };

    for (int d = 0; d < 4; d++) {
        int exact = 0, maxerr = 0;
        for (int a = 0; a < 256; a++) {
            int got = (a * divs[d].mul) >> divs[d].shift;
            int exp = a / divs[d].div;
            int err = got - exp;
            if (err < 0) err = -err;
            if (err == 0) exact++;
            if (err > maxerr) maxerr = err;
        }
        printf("  A/%d ≈ A*%d>>%d: %d/256 exact, max_err=%d",
               divs[d].div, divs[d].mul, divs[d].shift, exact, maxerr);

        // Try to find better multiplier
        int bestMul = divs[d].mul, bestExact = exact, bestMaxErr = maxerr;
        for (int m = 1; m < 256; m++) {
            int ex = 0, me = 0;
            for (int a = 0; a < 256; a++) {
                int got = (a * m) >> 8;
                int exp = a / divs[d].div;
                int err = got - exp; if (err < 0) err = -err;
                if (err == 0) ex++;
                if (err > me) me = err;
            }
            if (me < bestMaxErr || (me == bestMaxErr && ex > bestExact)) {
                bestMul = m; bestExact = ex; bestMaxErr = me;
            }
        }
        if (bestMul != divs[d].mul)
            printf(" → BETTER: A*%d>>8 (%d/256 exact, max_err=%d)", bestMul, bestExact, bestMaxErr);
        printf("\n");
    }

    // Try >>9 (mul then SRL)
    printf("\n  With >>9 (extra SRL A after mul):\n");
    for (int dv = 3; dv <= 10; dv += (dv==3?2:(dv==5?2:(dv==7?3:1)))) {
        int bestMul = 0, bestExact = 0, bestMaxErr = 999;
        for (int m = 1; m < 512; m++) {
            int ex = 0, me = 0;
            for (int a = 0; a < 256; a++) {
                int got = (a * m) >> 9;
                int exp = a / dv;
                int err = got - exp; if (err < 0) err = -err;
                if (err == 0) ex++;
                if (err > me) me = err;
            }
            if (me < bestMaxErr || (me == bestMaxErr && ex > bestExact)) {
                bestMul = m; bestExact = ex; bestMaxErr = me;
            }
        }
        printf("  A/%d ≈ A*%d>>9: %d/256 exact, max_err=%d%s\n",
               dv, bestMul, bestExact, bestMaxErr, bestMaxErr==0?" [EXACT]":"");
    }

    printf("\n=== FULL SUMMARY ===\n\n");
    printf("%-20s  %-50s  %s\n", "Primitive", "Z80 sequence", "Cost");
    printf("%-20s  %-50s  %s\n", "--------", "------------", "----");
    printf("%-20s  %-50s  %s\n", "ABS(A) signed", "LD B,A; RLCA; SBC A,A; LD C,A; XOR B; SUB C", "6i 24T clob:BC");
    printf("%-20s  %-50s  %s\n", "MIN(A,B) unsigned", "LD C,A; SUB B; SBC A,A; LD D,A; LD A,C; XOR B; AND D; XOR B", "8i 32T clob:CD");
    printf("%-20s  %-50s  %s\n", "MAX(A,B) unsigned", "LD C,A; SUB B; SBC A,A; LD D,A; LD A,B; XOR C; AND D; XOR C", "8i 32T clob:CD");
    printf("%-20s  %-50s  %s\n", "CLAMP(A,lo,hi)", "MIN then MAX (chained)", "16i 64T clob:BCDE");
    printf("%-20s  %-50s  %s\n", "CY?B:0 (cmask)", "SBC A,A; AND B", "2i 8T clob:A");
    printf("%-20s  %-50s  %s\n", "CY?B:C (cmov)", "SBC A,A;LD D,A;LD A,B;XOR C;AND D;XOR C", "6i 24T clob:AD");

    return 0;
}

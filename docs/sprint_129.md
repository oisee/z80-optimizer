# Sprint 1-2-9: Idioms, 16-bit Arithmetic, Universal Chains

## Goal
Three tables that make the Z80 compiler significantly better,
plus a cross-ISA breakthrough (universal chains).

---

## Task 1: Common Idiom Table

**What:** Brute-force optimal sequences for common Z80 programming patterns.
**Output:** `data/idioms.json`, `pkg/idioms/` Go package.
**Time:** ~4 hours compute + 2 hours coding.

### Idioms to solve

| Idiom | Input | Output | Known best | Brute-force? |
|-------|-------|--------|-----------|-------------|
| Zero-extend A→HL | A | HL=(0,A) | LD L,A / LD H,0 (11T) | Verify optimal |
| Sign-extend A→HL | A | HL=sign(A) | LD L,A / RLA / SBC A,A / LD H,A (16T) | Find shorter? |
| ABS(A) | A | |A| | OR A / JP P,+2 / NEG (branchless?) | Brute-force branchless |
| SWAP A,B | A,B | B,A | LD C,A / LD A,B / LD B,C (12T) | Verify, find XOR trick? |
| MIN(A,B) | A,B | min | CP B / JR C,+1 / LD A,B (~16T) | Brute-force branchless |
| MAX(A,B) | A,B | max | CP B / JR NC,+1 / LD A,B (~16T) | Brute-force branchless |
| CLAMP(A,lo,hi) | A,lo,hi | clamped | ~24T with branches | Brute-force |
| BOOL(A) | A | 0 or 1 | OR A / JR Z,+2 / LD A,1 | Brute-force branchless |
| SIGN(A) | A | -1/0/1 | Complex (~20T) | Brute-force |
| Popcount(A) | A | bitcount | LUT or ~30T | Brute-force |
| Bit reverse(A) | A | reversed | ~30T | Brute-force |
| BCD digit(A) | A (0-9) | ASCII | ADD A,0x30 (7T) | Trivial |
| Nibble swap(A) | A | swapped | RRCA×4 (16T) or RLCA×4 | Verify optimal |

### Approach
- Branchless idioms: use existing mulopt framework (state = A,B,carry, verify all 256)
- Two-register idioms: extend state to (A,B,C,carry), verify all 256×256
- Output: JSON + Go package like mulopt

### Steps
1. Define target functions in Python (expected output for each input)
2. Adapt z80_mulopt_fast.cu: replace QuickCheck target with idiom function
3. GPU search len-1..12 per idiom
4. Package results

---

## Task 2: 16-bit Arithmetic Library

**What:** Optimal sequences for 16-bit operations the Z80 lacks natively.
**Output:** `data/arith16.json`, `pkg/arith16/` Go package.
**Time:** ~4 hours compute + 2 hours coding.

### Operations to solve

| Operation | Input | Output | Known | Notes |
|-----------|-------|--------|-------|-------|
| NEG HL | HL | -HL | 5 insts/28T (our discovery!) | Already found |
| ADD HL,imm16 | HL,nn | HL+nn | LD BC,nn / ADD HL,BC (14T) | Is there shorter? |
| SUB HL,imm16 | HL,nn | HL-nn | LD BC,nn / OR A / SBC HL,BC (19T) | Verify |
| CP HL,imm16 | HL,nn | flags | OR A / SBC HL,BC / ADD HL,BC (~26T) | Non-destructive? |
| CP HL,DE | HL,DE | flags | OR A / SBC HL,DE / ADD HL,DE (26T) | Verify |
| CP HL,0 | HL | Z flag | LD A,H / OR L (8T) | Verify optimal |
| INC HL if C | HL,carry | HL+carry | JR NC,+1 / INC HL | Branchless? |
| HL = A (zero-ext) | A | HL | LD L,A / LD H,0 (11T) | Verify |
| HL = -A (sign-ext) | A | HL=-A | Our NEG trick (28T) | Already found |
| HL >> 1 | HL | HL/2 | SRL H / RR L (16T) | Verify |
| HL << 1 | HL | HL*2 | ADD HL,HL (11T) | Native! |
| HL >> 8 | HL | HL/256 | LD L,H / LD H,0 (11T) | Verify |
| Swap HL,DE | HL,DE | DE,HL | EX DE,HL (4T) | Native! |

### Approach
- Each is a separate brute-force target
- State: (A,B,C,D,E,H,L,carry) but most ops only touch HL+temps
- Use z80_mulopt16_mini framework: initial state HL=input16, find shortest
- For imm16 ops: parameterize by the constant (like mulopt by K)

### Steps
1. Define target functions (16-bit input/output)
2. Build z80_arith16.cu kernel (or adapt mulopt16_mini)
3. GPU search per operation
4. Package results + clobber annotations

---

## Task 9: Universal Computation Chains

**What:** ISA-independent optimal chains for multiply and divide.
**Output:** `data/chains/`, `pkg/chains/` Go package, paper draft.
**Time:** ~8 hours compute + 4 hours coding + writing.

### Abstract operations (7 total)

| Op | Semantics | Cost model |
|----|-----------|-----------|
| dbl | v = v * 2 | 1 step |
| add(i) | v = v + saved[i] | 1 step |
| sub(i) | v = v - saved[i] | 1 step |
| save | push v to save stack | 1 step (max 2-3 slots) |
| shr | v = v >> 1 | 1 step |
| mask(m) | v = v & m | 1 step |
| neg | v = -v | 1 step |

### Search configurations

| Config | Ops | Save slots | Target | Search space |
|--------|-----|-----------|--------|-------------|
| Pure add chain | dbl, add, save | 1 | multiply 1-255 | tiny |
| Add-sub chain | +sub, neg | 1 | multiply 1-255 | small |
| Division chain | +shr, mask | 2 | div/mod 1-255 | medium |
| Full chain | all 7 | 2-3 | mul+div 1-65535 | large |

### Phases

**Phase A: Multiply chains 1-255** (hours)
- Abstract search with {dbl, add, sub, save, neg}, 1 save slot
- Compare chain length with Z80 mulopt results (validate)
- Output: abstract_mul8.json

**Phase B: Divide chains 1-255** (hours)
- Add {shr, mask}, 2 save slots
- Target: input/K for all 256 inputs
- Key: div3, div5, div7, div10
- Output: abstract_div8.json

**Phase C: Materialize to ISA** (code, hours)
- Z80: dbl→ADD A,A, add→ADD A,B, save→LD B,A, shr→SRL A, ...
- 6502: dbl→ASL A, add→CLC/ADC zp, save→TAX, shr→LSR A, ...
- RISC-V: dbl→SLLI rd,1, add→ADD, save→MV, shr→SRLI, ...
- Verify each materialization per-ISA (different carry/overflow)

**Phase D: 16-bit chains 1-65535** (days, optional)
- Same ops but 16-bit state
- For Z80: materializes to ADD HL,HL etc.
- For 6502: 16-bit operations via zero-page
- Output: abstract_mul16.json

### Validation
- Every abstract chain → materialize to Z80 → verify against mulopt results
- Chain length should be ≤ Z80 instruction count (chains have more freedom)
- If chain is SHORTER than Z80 mulopt → means Z80's register constraints cost extra

### Implementation
```
// Chain state
type State struct {
    Value  int        // current value
    Saved  [3]int     // save slots
    NSaved int        // how many slots used
}

// Chain search (CPU, trivial — no GPU needed for abstract chains)
func Search(target func(int)int, maxDepth int) *Chain
```

Pure CPU search — no GPU needed because abstract chains are tiny
(~10^7 at depth 12 vs 10^13 for Z80 assembly).

---

## Sprint Schedule

| Day | Task | Machine | Deliverable |
|-----|------|---------|-------------|
| Day 1 AM | Chain solver (Task 9A) | CPU any | abstract_mul8.json |
| Day 1 PM | Idiom targets (Task 1) | main GPU | idioms.json draft |
| Day 1 PM | 16-bit arith (Task 2) | i5 GPU | arith16.json draft |
| Day 2 AM | Chain div search (Task 9B) | CPU | abstract_div8.json |
| Day 2 AM | Idiom polish + package | CPU | pkg/idioms/ |
| Day 2 PM | Materialize chains (Task 9C) | CPU | chain_z80.json, chain_6502.json |
| Day 2 PM | 16-bit arith package | CPU | pkg/arith16/ |
| Day 3 | Validate all + paper section | — | docs/paper_chains.md |

## Success Criteria
- [ ] Common idioms: ≥10 patterns with proven-optimal sequences
- [ ] 16-bit arith: ≥8 operations with clobber annotations
- [ ] Abstract chains: all 254 multiply constants, ≤10 div constants
- [ ] Cross-ISA: same chains materialized to Z80 + 6502
- [ ] Go packages: pkg/idioms/, pkg/arith16/, pkg/chains/
- [ ] Paper section: "Universal Computation Chains" draft

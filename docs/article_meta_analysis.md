# What 755 Optimal Sequences Teach Us About Computation

**Meta-analysis of exhaustive Z80 arithmetic tables**

---

## Introduction

Between late 2025 and early 2026, we built a brute-force superoptimizer for the
Zilog Z80 processor. The tool exhaustively searches every possible instruction
sequence of a given length, verifying full-state equivalence across all 2^24
input combinations. The result is not a heuristic or an approximation — it is a
certificate of optimality.

The raw output is a set of tables: 254 constant multiplications (8-bit), 246
division/modulo routines, a handful of nonlinear approximations (sqrt, log2),
and 739K peephole rewrite rules. These tables are useful on their own — they
ship in the MinZ compiler and save real cycles on real hardware.

But the tables also contain *structure*. Patterns that nobody designed. Patterns
that emerge only when you look at hundreds of optimal programs simultaneously
and ask: what do they have in common? What do they avoid? What does the shape
of the solution space tell us about computation itself?

This article extracts ten such patterns. Some are engineering observations that
improve future search. Others are architectural insights about instruction set
design. A few gesture toward something more fundamental — the geometry of
optimal programs.

---

## 1. The 21-Instruction Thesis

The Z80 has approximately 700 documented instructions. Our superoptimizer
enumerates 455 opcodes expanding to 4,215 concrete instructions when you include
immediate operands. Of these, we found that only **21 instructions** appear in
any optimal arithmetic sequence across the entire multiply, divide, and
approximation tables.

This is not pool reduction as an engineering hack — we discovered it empirically,
then used it to accelerate search by 38x. But the deeper observation is
architectural: optimal computation uses a tiny, predictable fraction of any ISA.

The 21 survivors cluster into clear functional roles:

- **Shift/rotate group** (7): RLCA, RRCA, RLA, RRA, SLA A, SRA A, SRL A
- **Arithmetic group** (6): ADD A,A, ADD A,{B,C,D,E,H,L}
- **Negate/complement** (2): NEG, CPL
- **Load/save** (3): LD {B,C,D},A
- **Conditional mask** (1): SBC A,A
- **Accumulator clear** (1): XOR A
- **Subtraction** (1): SUB A,{reg}

Seven of the original 21 candidate operations *never* appear in any optimal
solution. They were plausible — they perform arithmetic — but optimality
excludes them entirely. The gap between "could be useful" and "is ever useful"
is wide.

**Is this universal?** We conjecture yes. Any sufficiently rich ISA will exhibit
a similar collapse. RISC-V's RV32I has 47 integer instructions; we predict that
exhaustive search over short arithmetic sequences would converge to fewer than
15. ARM's barrel shifter would likely produce a different 15, but the ratio
would be comparable. The reason is combinatorial: most instructions are
*dominated* — there exists a shorter sequence using other instructions that
achieves the same effect. Optimality prunes dominated instructions ruthlessly.

**Implication for ISA design**: If you are designing an ISA for a specific
computational domain, brute-force search over short sequences will tell you
which instructions actually earn their encoding space. The other 97% of the ISA
exists for programmer convenience, not computational necessity.

---

## 2. Disjoint Worlds

The 8-bit multiply table (mul8: A := A * K) and the 16-bit multiply table
(mul16: HL := A * K) share **zero instructions**. Not one.

mul8 lives entirely in the A-register world: RLCA, ADD A,A, ADD A,reg, NEG,
SBC A,A. Its saved temporaries go to B, C, D. It never touches H or L.

mul16 lives in the HL-register world: ADD HL,HL, ADD HL,BC, LD B,A, LD C,A.
It never uses RLCA or NEG.

Yet both compute the same mathematical function — constant multiplication. The
inputs differ in width (8-bit result vs 16-bit result), but the core operation
is identical: repeated conditional addition of shifted copies of the input.

This disjointness is forced by the Z80's register architecture. The A register
has rich arithmetic (rotates, negate, subtract-with-carry), but A is only 8
bits. The HL register pair can do 16-bit addition, but lacks rotates and
negate. The ISA partitions multiplication into two incompatible algorithm
families based solely on the output width.

This is not a Z80 quirk. It is a fundamental property of register-based
architectures with heterogeneous register files. On x86, 8-bit multiplication
(IMUL r/m8) uses AL/AH, while 32-bit multiplication uses EAX/EDX — different
registers, different instruction encodings, different algorithm shapes. Even
RISC-V, with its uniform register file, splits multiply across MUL (low 32
bits) and MULH (high 32 bits), forcing different code patterns for different
output widths.

**The deeper point**: the mathematical structure of multiplication is one thing;
the *algorithmic* structure is dictated by the machine. Two programs computing
the same function can be structurally incomparable because the ISA forces them
through different register topologies.

---

## 3. The Rotation Dominance

In approximate nonlinear function search — finding short instruction sequences
that compute sqrt(x) or log2(x) to within acceptable error — one instruction
dominates all others: **RRCA** (rotate right circular through accumulator).

RRCA appears at a rate of **276%** across all found approximation sequences.
That is not a typo: the average sequence uses RRCA 2.76 times. It is the only
instruction in the pool with an occurrence rate above 100%.

Why? Rotations are the mechanism by which linear arithmetic "sees" individual
bits. Addition and subtraction operate on the number as a whole. Rotate right
by one position is equivalent to dividing by 2 (with wrap-around for the low
bit), but critically, it moves the bit pattern in a way that subsequent
additions can recombine it nonlinearly.

Consider the bit-level view. After RRCA, the MSB of the result contains what
was previously the LSB of the input. An ADD A,A after RRCA produces a value
that mixes low and high bit-fields — something no sequence of pure additions
can achieve. This mixing is exactly what you need to approximate nonlinear
functions: sqrt(x) requires "spreading" the input bits across the output in a
pattern that no linear operation can produce.

The rotation-addition pair is, in a sense, the minimal nonlinear basis for
bit-level computation. It is the Z80's equivalent of the butterfly operation in
an FFT — a structured mixing step that, when repeated, builds up complex
transformations from simple ones.

**Practical consequence**: any future search for nonlinear approximations on a
shift-add architecture should weight rotations heavily in the instruction pool.
They are not just useful — they are the *mechanism* by which the search space
contains nonlinear functions at all.

---

## 4. Compression via Computation

A 256-byte lookup table for sqrt(x) occupies 256 bytes of ROM. A brute-force
search found a **12-byte instruction sequence** that generates the same table
(to within acceptable error) by computing each entry arithmetically.

This is a 20:1 compression ratio — not via entropy coding or dictionary
compression, but via *computation*. The 12 bytes are not compressed data; they
are a program. The decompressor is the CPU itself.

The error-depth tradeoff follows a consistent curve: each additional compound
operation (one instruction added to the sequence) reduces maximum error by
approximately **30%**. At depth 9, sqrt achieves 88% accuracy across all 256
input values. Extrapolating to depth 12 suggests 95%+ accuracy is reachable.

This tradeoff is not specific to sqrt. It reflects a general property: the set
of functions computable by sequences of length N grows combinatorially with N,
and the "distance" (maximum pointwise error) between the nearest computable
function and any target function shrinks as N increases. The rate of shrinkage
depends on the richness of the instruction pool — rotations help enormously (see
Section 3) because they expand the reachable function set faster than pure
arithmetic.

**Broader implication**: for embedded systems with extreme ROM constraints,
algorithmic generation of lookup tables is a viable alternative to storing them.
A superoptimizer can find the shortest program that generates a given table,
achieving compression ratios that no data compressor can match — because data
compressors cannot invent algorithms.

---

## 5. The Division-Multiplication Duality

We searched exhaustively for optimal division-by-constant sequences (A := A / K
for all K from 2 to 255). A striking structural pattern emerged: **86 out of 86**
completed division chains contain a multiplication step as a substring.

Between 64% and 92% of the instructions in a division-by-K routine are literally
multiplication code — the same ADD/shift chains that appear in the mul8 table —
followed by a small correction suffix (typically one or two right shifts and a
mask).

This is the superoptimizer rediscovering, from first principles, what
*Hacker's Delight* proved algebraically: division by a constant K is equivalent
to multiplication by the modular inverse of K, followed by a fixup. The Z80 has
no multiply instruction, so "multiplication" means shift-add chains. But the
structural relationship holds at the instruction level, not just the
mathematical level.

**Engineering consequence**: a packed arithmetic library can share code between
mul and div tables. If the mul8 table is already present (as it is in MinZ),
many division routines can be implemented as a call to the appropriate multiply
entry point followed by 2-3 fixup instructions. The mul8 library is literally
a *substrate* on which division is built.

The best known div10 routine is 27 instructions (124-135 cycles). Our GPU search
certified a lower bound of 13 instructions — no sequence of 12 or fewer
instructions computes A/10 correctly for all inputs. The gap of 14 instructions
between the lower bound and the best known solution is one of the open problems
in the project.

---

## 6. Carry-to-Mask: The Universal Conditional

`SBC A,A` — subtract A from A with carry — produces 0xFF if carry is set, 0x00
if carry is clear. It converts a 1-bit condition (the carry flag) into a
full-byte mask.

This single instruction appears in **100% of branchless conditional idioms** in
our tables and in the majority of nonlinear approximation sequences. It is the
Z80's only mechanism for converting a comparison result into a data value without
branching.

The pattern is always the same:

```
CP threshold      ; sets carry if A < threshold
SBC A,A           ; A = 0xFF if A < threshold, else 0x00
AND correction    ; mask the correction value
ADD A,base        ; apply conditional correction
```

This four-instruction template replaces a branch (JP cc, target / ... / target:)
that would cost 10-17 cycles and break pipeline locality. The branchless version
is fixed-time, shorter, and — critically — superoptimizer-friendly, because it
has no control flow for the exhaustive checker to explore.

**Cross-architecture equivalents exist but differ in character.** RISC-V has
SLT/SLTU (set-less-than), which produces 0/1 rather than 0x00/0xFF — requiring
an additional NEG to get a full mask. ARM has conditional execution on every
instruction, making the mask unnecessary but consuming encoding space. x86 has
SETcc, which like RISC-V produces 0/1 and needs a follow-up.

The Z80's SBC A,A is arguably the most elegant of these: it reuses the existing
subtract instruction with a degenerate operand (A minus A), turning a design
accident into the most important instruction in branchless arithmetic. No ISA
designer planned this. The superoptimizer found it.

---

## 7. The Prefix Sharing Structure

Consecutive constants share remarkably long common prefixes in their optimal
multiplication sequences. Specifically, the optimal program for ×K and ×(K+1)
typically share **9 to 10 instruction prefixes** before diverging.

This is exploited in the packed multi-entry library format: 254 constant
multiplications are encoded in approximately **2KB** via fall-through chains.
Each multiply entry begins partway through the previous entry's code. The
program counter falls through shared prefix instructions, then branches (or
returns) at the point of divergence.

But the engineering trick points to something deeper. The fact that "nearby"
constants have "nearby" optimal programs suggests a **metric space structure**
on the set of optimal programs.

Define the distance between two programs as the length of the shortest edit
sequence (instruction insertions, deletions, substitutions) that transforms one
into the other. Under this metric, the optimal programs for ×K cluster: numbers
that are close in value tend to have close optimal programs. Numbers that share
factors (e.g., ×6 and ×12) have even closer programs, because ×12 = ×6 followed
by a doubling step.

This is not obvious. It would be equally plausible for optimal programs to be
"scattered" — for ×41 and ×42 to use completely different strategies. But they
do not. The shift-add decomposition of integers imposes a smoothness on the
program space that mirrors the smoothness of the integers themselves.

**Open question**: can this metric space structure be formalized? Is there a
continuous mapping from the integers to the space of shift-add programs such
that the image of consecutive integers is always within bounded edit distance?
Our data says yes for Z80 constant multiplication, but the general case is
unknown.

---

## 8. Fixed-Point Scaling and Search Difficulty

When searching for sqrt approximations, we tested two output formats:
- **f4.4** (4 integer bits, 4 fractional bits — multiply by 16)
- **f3.5** (3 integer bits, 5 fractional bits — multiply by 8, more precision)

f4.4 converges faster despite having *less precision*. The reason: f4.4's output
fills more of the 0-255 byte range, giving the search more "signal" to lock
onto. With f3.5, the output values are concentrated in a narrower band, and
many candidate sequences produce outputs that are close-but-wrong in ways the
early search stages (QuickCheck, MidCheck) cannot distinguish.

This is a general principle with practical implications: **the representation of
the target function affects search difficulty independently of the function
itself.**

In our three-stage search pipeline (QuickCheck on 8 vectors, MidCheck on 32,
ExhaustiveCheck on 2^24), the early stages act as coarse filters. A target
function whose output has high variance across the test vectors produces strong
filter signal — most wrong candidates are eliminated quickly. A target function
with low output variance (because the scaling squishes values together) lets
more false positives through to the expensive ExhaustiveCheck stage.

**Practical guideline**: when setting up a superoptimizer search for an
approximate function, spend time choosing the output format. Prefer
representations that spread the output across the full register width. The
search cost difference between a good and bad representation can be 10x or more.

---

## 9. Multi-Target Search Efficiency

Our nonlinear function search tests each candidate sequence against **15 target
functions simultaneously** (sqrt, log2, exp2, reciprocal, and various scaled
variants). The verification cost scales linearly — 15 targets cost ~15x one
target. But the *generation* cost is shared entirely: enumerating and executing
each candidate sequence happens once regardless of how many targets are checked.

For searches at depth 10+, generation dominates. Producing and executing every
possible 10-instruction sequence is astronomically expensive; checking the
output against a target function is a trivial comparison. Adding more targets
to the check is essentially free.

This changes the search methodology. Instead of asking "find me a program for
sqrt," you ask "find me a program for *anything interesting*." The multi-target
framework enables **exploration** — discovering that a particular 11-instruction
sequence happens to compute a good approximation of log2, even though you were
not specifically searching for log2.

In practice, our multi-target searches have found approximations for functions
we did not originally intend to include. The "unexpected match" rate is low
(roughly 1 in 10^6 candidates matches any of 15 targets at depth 10), but when
hits are that rare, every match is valuable — and you get them for free.

**Generalization**: any brute-force search over a shared generative space should
test against multiple targets simultaneously. This applies beyond superoptimization
— to program synthesis, SAT-based optimization, even neural architecture search.
The generation-verification asymmetry is universal.

---

## 10. Abstract Chains vs. Real Instructions

We compared optimal Z80 sequences against their abstract representations in the
{dbl, add, sub, save} chain model, where each abstract operation maps to one
Z80 instruction. The overhead ratio — Z80 length divided by abstract chain
length — averages **0.99x**.

The Z80 sequences are, on average, *shorter* than their abstract representations.

This is counterintuitive. A concrete ISA should be at least as verbose as an
abstract model, because the abstract model has no register constraints, no
encoding limitations, no flags to worry about. And indeed, for most individual
sequences, the Z80 version is the same length or one instruction longer.

But the Z80 has **tricks** that the abstract model cannot express:

- **NEG** (negate A) replaces {complement, add 1} — one instruction instead of
  two abstract operations. The multiplication ×255 compiles to a single NEG
  (since 255 = -1 mod 256), taking 8 cycles. The abstract model needs at least
  two operations.

- **RLCA** (rotate left) is simultaneously a doubling and a bit-movement. In
  some multiplication chains, a single RLCA replaces what would be a dbl + save
  in the abstract model.

- **SBC A,A** has no abstract equivalent at all. It performs a conditional mask
  that the {dbl, add, sub, save} vocabulary cannot express in one step.

The average falls below 1.0 because these tricks occur often enough in the 254
multiply constants to pull the mean below parity. The ISA's irregularities —
its historical quirks and design accidents — are not overhead. They are
*computational assets* that a sufficiently thorough search will exploit.

**Lesson for ISA designers**: do not assume that a clean, orthogonal ISA
produces shorter code than a quirky, historical one. Real optimality depends on
the specific functions being computed. A single powerful special-case
instruction (like NEG or SBC A,A) can outperform a more regular instruction set
across hundreds of different programs.

---

## Conclusion

Ten patterns, one recurring theme: **structure emerges from exhaustion.**

None of these observations were predicted in advance. We did not set out to
discover that only 21 instructions matter, or that consecutive constants share
long program prefixes, or that division is literally built from multiplication.
These patterns became visible only after the search was complete — after every
sequence of length 9 had been tested, after every constant from 2 to 255 had
its optimal program, after 739K peephole rules had been extracted and verified.

The brute-force method is often dismissed as intellectually uninteresting —
"just enumerate everything." But enumeration produces data, and data at
sufficient scale reveals structure that no amount of clever reasoning would have
predicted. The 21-instruction thesis is a theorem about the Z80 ISA that no
human designer knew in 1976. The prefix sharing structure is a topological
property of the space of shift-add programs that, as far as we know, has no
prior formalization.

There is a meta-lesson here for the practice of computer science. Exhaustive
search is not just a way to find optimal programs. It is a *microscope* — a
tool for seeing the fine structure of computation. The patterns in our tables
are fossils of deep mathematical relationships between numbers, algorithms, and
machines. Reading them carefully teaches us things that no amount of theory
alone could reveal.

755 optimal sequences. 21 surviving instructions. One carry-to-mask instruction
that appears everywhere. The Z80, a processor designed in 1976 for calculators
and home computers, turns out to be a remarkably clean laboratory for studying
the nature of computation itself.

---

*Data sources: Z80 superoptimizer exhaustive tables (mul8, mul16, divmod, sqrt/log2
approximations), 2025-2026. All results verified by full-state equivalence
checking across 2^24 input combinations. Project repository:
[github.com/oisee/z80-optimizer](https://github.com/oisee/z80-optimizer)*

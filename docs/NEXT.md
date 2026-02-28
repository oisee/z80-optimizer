# What's Next: From Brute Force to GPU-Accelerated Stochastic Search

## The Problem We're Solving

The Z80 instruction set is full of redundancies that no human would find by hand. Our brute-force superoptimizer already discovered 602,008 provably correct optimizations — but it only searched **length-2 sequences**. The real treasures hide in longer sequences where instructions interact in non-obvious ways.

The challenge: the search space grows exponentially.

```
Length 2:  4,215² =        17.8M targets  → 3h on CPU     ✓ DONE
Length 3:  4,215³ =        74.8B targets  → ~months on CPU, hours on GPU
Length 4:  4,215⁴ =       315.4T targets  → not feasible, needs STOKE
Length 5+: 4,215⁵ = 1,329,000T targets    → only stochastic search
```

We need three different engines, each covering a different scale.

## Three Generations of Superoptimizers

The field has evolved from pure brute force to increasingly clever search strategies. Each approach makes a different tradeoff between **completeness** (guaranteed to find the optimum) and **scalability** (how long a sequence it can handle).

### Generation 1: Exhaustive Enumeration (1987)

> *"The Superoptimizer finds the shortest program for a given function."*
> — [Massalin, ASPLOS 1987](https://dl.acm.org/doi/10.1145/36177.36194)

The original insight: if you can execute instructions fast enough, you can simply try **every possible program** of length 1, then 2, then 3, and return the first one that matches. The first match is guaranteed to be optimal because you searched shortest-first.

**This is what we built.** Our executor runs at 2.7ns per instruction, and QuickCheck with 8 test vectors rejects 99.99% of candidates instantly. For length-2 sequences this works — 34.7 billion comparisons in 3 hours.

The wall: length 3 has 74.8 billion targets, each requiring 4,215 candidate checks. That's 315 trillion comparisons. On a single CPU, this takes months.

### Generation 2: Peephole Superoptimization (2006)

> *"We describe a system that automatically generates peephole optimizations from a description of a target machine."*
> — [Bansal & Aiken, ASPLOS 2006](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf)

Bansal and Aiken realized that brute-force results can be **compiled into rewrite rules** that a compiler applies in a single pass. Instead of running the superoptimizer at compile time, you run it once offline and produce a table of pattern → replacement entries.

This is essentially what our `rules.json` already is — 602K rewrite rules automatically discovered. The Bansal & Aiken contribution was doing this systematically and integrating it into a real compiler (GCC).

### Generation 3: Stochastic Search (2013)

> *"We use a Markov Chain Monte Carlo sampler to rapidly explore the space of all possible programs."*
> — [Schkufza, Sharma & Aiken, ASPLOS 2013 (STOKE)](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf)

STOKE changed the game. Instead of exhaustively enumerating all programs, it:

1. **Starts from the target program** (or a random program)
2. **Randomly mutates** it: replace an instruction, swap two, insert one, delete one
3. **Evaluates a cost function**: correctness (how many test cases match) + size (byte count)
4. **Accepts or rejects** the mutation using Metropolis-Hastings:
   - Better cost → always accept
   - Worse cost → accept with probability `e^(-Δcost / temperature)`
5. **Repeats** until it finds a program with cost = 0 (fully correct) and fewer bytes

The key insight: most mutations make things worse, but occasionally a random change stumbles into a fundamentally different — and shorter — program that no amount of local rewriting would find. The MCMC framework ensures the search doesn't get stuck in local optima.

STOKE successfully optimized x86-64 sequences of **10-20 instructions**, finding replacements that outperformed `gcc -O3` and sometimes even expert hand-written assembly.

### Other Notable Approaches

- **[Denali (2002)](https://www.researchgate.net/publication/314828905_Denali_a_goal-directed_superoptimizer)** — Goal-directed search using equality saturation. Applies algebraic rewrite rules to build an equivalence graph. Elegant but limited to what the rules can derive.

- **[Lens/Souper (2016)](https://mangpo.net/papers/lens-asplos16.pdf)** — Decomposes programs into smaller pieces, solves each with an SMT solver, then stitches results together. Scales better than brute force while maintaining completeness within each window.

## Our Plan: Three Engines

### Engine 1: GPU Brute Force (length 1-3, complete)

Our Z80 executor is an **ideal GPU workload**:

- **Fixed-size state**: 10 bytes per Z80 state, fits entirely in GPU registers
- **No divergence**: opcode dispatch becomes a lookup table, all threads take the same path
- **No memory access**: pure register computation, no cache misses
- **Embarrassingly parallel**: every (target, candidate) pair is independent

The architecture:

```
┌──────────────────────────────────────────────────────────┐
│                      Host (CPU)                          │
│                                                          │
│  for each target sequence:                               │
│    1. Compute target fingerprint (8 test vectors)        │
│    2. Upload target fingerprint to GPU                   │
│    3. Dispatch GPU kernel: test ALL candidates           │
│    4. Read back hit bitmap                               │
│    5. ExhaustiveCheck hits on CPU (rare, ~0.01%)         │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│                   GPU Kernel                             │
│                                                          │
│  thread[i] = candidate instruction i (0..4214)           │
│                                                          │
│  for each test vector (0..7):                            │
│    execute candidate[i] on test vector                   │
│    compare output with target fingerprint                │
│    if mismatch → results[i] = 0, return early            │
│                                                          │
│  results[i] = 1  // passed all 8 vectors                │
└──────────────────────────────────────────────────────────┘
```

For length-2 candidates, we dispatch 4,215 × 4,215 = 17.8M threads per target. Modern GPUs handle this trivially.

**Estimated performance (2× RTX 4060 Ti):**

```
CUDA cores:     2 × 4,352 = 8,704
Clock:          ~2.5 GHz boost
QuickCheck:     ~80 GPU instructions per candidate

Throughput:     8,704 cores × 2.5GHz / 80 ops ≈ 272 billion checks/sec

Length-2 total: 17.8M targets × 4,215 candidates = 75B checks
Time:           75B / 272B/sec ≈ 0.3 seconds  (was 3h16m on CPU)

Length-3 total: 74.8B targets × 4,215 candidates = 315T checks
Time:           315T / 272B/sec ≈ 19 minutes   (was ~months on CPU)
```

### Engine 2: STOKE (length 4-10+, stochastic)

For sequences beyond length 3, we switch to MCMC stochastic search:

```
┌─────────────────────────────────────────────────────────┐
│                   STOKE Engine                          │
│                                                         │
│  Input: target sequence (4-10 instructions)             │
│                                                         │
│  Initialize: candidate = copy of target                 │
│  Temperature: T = 1.0, decay = 0.999                    │
│                                                         │
│  for iteration = 1..N:                                  │
│    mutation = random choice:                            │
│      40% replace instruction with random instruction    │
│      20% swap two adjacent instructions                 │
│      20% delete a random instruction                    │
│      10% insert a random instruction                    │
│      10% replace immediate with random value            │
│                                                         │
│    new_candidate = apply(mutation, candidate)            │
│    new_cost = evaluate(new_candidate, target)            │
│    Δ = new_cost - current_cost                          │
│                                                         │
│    if Δ < 0 or random() < e^(-Δ/T):                    │
│      candidate = new_candidate                          │
│      current_cost = new_cost                            │
│                                                         │
│    T *= decay                                           │
│                                                         │
│    if current_cost == 0 and shorter(candidate, target): │
│      → ExhaustiveCheck(target, candidate)               │
│      → if verified: FOUND OPTIMIZATION                  │
└─────────────────────────────────────────────────────────┘
```

The cost function:

```
cost(candidate, target) =
    1000 × mismatches(candidate, target, test_vectors)  // correctness dominates
  + byte_size(candidate)                                 // prefer shorter
  + cycle_count(candidate) / 100                         // slight speed preference
```

When `mismatches = 0`, the candidate is functionally equivalent on the test vectors. We then verify exhaustively before declaring it an optimization.

**Parallelism**: run 16 independent MCMC chains on 16 CPU cores (or 8 cores × 2 chains each). Each chain explores a different region of the search space. This is trivially parallel — no synchronization needed.

### Engine 3: Combined Pipeline

The ultimate architecture combines both:

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Coordinator                       │
│                                                             │
│  Length 1-3: GPU brute force (complete, minutes)            │
│  Length 4+:  STOKE on CPU (stochastic, hours)               │
│  Verify:    GPU-accelerated ExhaustiveCheck                 │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  GPU Kernel  │   │ STOKE Chain  │   │ STOKE Chain     │  │
│  │  QuickCheck  │   │ (CPU core 1) │   │ (CPU core N)    │  │
│  │  all L1-L3   │   │ mutate L4+   │   │ mutate L4+      │  │
│  │  candidates  │   │ test vectors │   │ test vectors    │  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬──────────┘  │
│         │                  │                   │             │
│         ▼                  ▼                   ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Results Deduplication                    │   │
│  │         ExhaustiveCheck (GPU-accelerated)             │   │
│  │              rules.json output                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Roadmap

### Phase 1: STOKE on CPU

The fastest to implement — all the infrastructure already exists.

```
New package: pkg/stoke/
  mutator.go    — random mutations on instruction sequences
  cost.go       — cost function (correctness + size + speed)
  mcmc.go       — Metropolis-Hastings sampler
  search.go     — multi-chain parallel search

New CLI command: z80opt stoke --target "LD A, n : ADD A, B : AND 0xFF"
```

We already have: the executor (2.7ns/op), QuickCheck, ExhaustiveCheck, the full instruction catalog. STOKE just needs the mutation operators and the MCMC loop.

**Expected output**: optimizations for length 4-8 sequences that brute force can't reach.

### Phase 2: CUDA Brute Force

Port the Z80 executor to a CUDA kernel for massively parallel QuickCheck.

```
New package: pkg/gpu/
  z80_kernel.cu    — Z80 executor in CUDA C
  dispatch.go      — Go ↔ CUDA interface via cgo
  batch.go         — batch target processing

Target hardware: RTX 4060 Ti (4,352 CUDA cores, 16GB VRAM)
```

The CUDA kernel is straightforward — our executor is already a pure function with no side effects. The main engineering challenge is efficient data transfer between host and device.

**Expected output**: complete length-3 brute-force results in ~20 minutes.

### Phase 3: Combined Pipeline

Wire STOKE and GPU brute force together. STOKE runs on CPU cores while GPU handles brute-force verification. Discovered rules feed back into STOKE's mutation pool.

## Why This Matters

Every optimization found by the superoptimizer is **provably correct** — verified against all possible inputs, not just test cases. This is qualitatively different from hand-written peephole rules:

- **No false positives**: every rule preserves exact register and flag state
- **Discovers non-obvious equivalences**: `SLA A : RR A → OR A` is not something a human would think to check
- **Complete for short sequences**: brute force guarantees no optimization is missed
- **Extensible**: adding new instructions to the catalog automatically discovers new optimizations

For the Z80 specifically — still used in embedded systems, retrocomputing, and as a compiler target — these optimizations directly reduce code size and execution time in environments where every byte matters.

## References

1. **Massalin, 1987** — [Superoptimizer: A Look at the Smallest Program](https://dl.acm.org/doi/10.1145/36177.36194). The original brute-force superoptimizer. ASPLOS '87.

2. **Joshi, Nelson & Randall, 2002** — [Denali: A Goal-Directed Superoptimizer](https://www.researchgate.net/publication/314828905_Denali_a_goal-directed_superoptimizer). Equality saturation approach. PLDI '02.

3. **Bansal & Aiken, 2006** — [Automatic Generation of Peephole Superoptimizers](https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf). Brute-force → peephole rules for GCC. ASPLOS '06.

4. **Schkufza, Sharma & Aiken, 2013** — [Stochastic Superoptimization](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf). MCMC random search, outperforms `gcc -O3` on x86-64. ASPLOS '13.

5. **Phothilimthana et al., 2016** — [Scaling up Superoptimization](https://mangpo.net/papers/lens-asplos16.pdf). Decomposition + SMT solving. ASPLOS '16.

# Z80 Register Allocation Research — Glossary

## Core Concepts

- **vreg (virtual register)** — a compiler's abstract variable that needs to be assigned to a physical CPU register. A function with "6v" has 6 virtual registers.
- **nVregs** — count of virtual registers in a function
- **loc (location)** — a physical place a vreg can live. Z80 has 15: A, B, C, D, E, H, L (8-bit GPR), BC, DE, HL (16-bit pairs), IXH, IXL, IYH, IYL, mem0
- **GPR (General Purpose Register)** — the 7 basic 8-bit registers (A-L)
- **regalloc (register allocation)** — the problem of assigning vregs to physical locations, minimizing cost (move instructions, spills)

## Interference & Graphs

- **interference** — two vregs that are "live" (in use) at the same time must be in different locations. An interference pair `[i,j]` means vreg i and j can't share a register
- **interference graph** — vertices = vregs, edges = interference pairs. The "coloring" of this graph IS the register assignment
- **density** — percentage of possible edges present. 6v has 15 possible edges; 80% density = 12 edges
- **connected component** — a group of vregs all linked by interference chains. Independent components can be solved separately
- **cut vertex (articulation point)** — a vertex whose removal disconnects the graph. Splitting at a cut vertex decomposes the problem
- **2-connected** — a graph with no cut vertex. Can't be split by removing a single node

## Treewidth

- **treewidth** — a measure of how "tree-like" a graph is. tw=1 means it's a tree, tw=n-1 means it's complete (every vertex connected to every other)
- **elimination ordering** — a sequence of removing vertices from a graph, connecting their neighbors each time. The max degree seen during elimination = the width of that ordering. Treewidth = minimum width across all orderings
- **tree decomposition** — restructuring a graph into a tree of "bags" of vertices. Algorithms on graphs with low treewidth run in polynomial time O(L^tw * n) instead of exponential O(L^n)
- **tw<=3** — solvable in polynomial time by tree-DP (dynamic programming on tree decomposition)

## Search & Solving

- **brute-force** — try ALL possible assignments (L^N for N vregs and L locations). 15^15 = 437 trillion
- **backtracking** — recursive search that assigns one vreg at a time, pruning branches where interference is violated. Much faster than brute-force on sparse graphs
- **forward checking** — during backtracking, after assigning a vreg, immediately check if any unassigned neighbor has zero remaining valid locations. If so, backtrack early
- **constraint propagation** — if a vreg has exactly 1 valid location, remove that location from all interfering neighbors. Chain reaction
- **pruning factor** — ratio of brute-force space to actually explored nodes. 745,000x means we explored 745K times fewer nodes than brute force
- **search certificate** — proof that a search was exhaustive. Either "here's the optimal solution" (constructive) or "no solution exists at this length" (negative/lower bound)

## Island Decomposition

- **island** — a contiguous section of a function that can be register-allocated independently
- **liveness bottleneck** — a program point where few vregs are simultaneously alive. Good place to split into islands
- **boundary vregs** — vregs that are live across an island boundary. Must get the same location in both islands
- **shuffle cost** — T-states spent moving data between registers at island boundaries (typically 4T per register)

## GPU/CUDA

- **CUDA** — NVIDIA's GPU programming framework
- **kernel** — a function that runs on thousands of GPU threads in parallel
- **search space** — total assignments to check (L^N). Limited to 5T (5 trillion) in our GPU kernel
- **--server mode** — long-running CUDA process that reads JSON problems on stdin, outputs results on stdout. Avoids reinitializing GPU per problem
- **dual GPU** — running on both RTX 4060 Ti cards simultaneously, splitting work

## Cost Model

- **T-states (T)** — Z80 clock cycles. The fundamental cost metric. `LD A,B` = 4T, `CALL` = 17T, `RET` = 10T
- **CALL/RET overhead** — 27T per function call (17T call + 10T return)
- **shuffle overhead** — ~8T per call for moving data into the right registers before a call
- **pattern** — a specific instruction encoding with location constraints. An op might have 14 patterns, each valid for different location combinations, each with a different T-state cost

## Paper-Specific

- **feasibility** — whether ANY valid register assignment exists for a given constraint shape. 0.9% feasible at 6v means 99.1% of shapes are impossible on Z80
- **phase transition** — a sharp boundary where a problem goes from easy to hard (or feasible to infeasible). We found TWO: enumeration cliff at 6v, allocation cliff at ~16 locations
- **corpus** — collection of real compiled programs used to study which constraint patterns actually occur. Ours: 315 unique signatures from Nanz/PL/M/ABAP/SQLite/FatFS
- **signature (sig)** — SHA256 hash of a function's constraint structure (ops + patterns + interference). Two functions with the same signature have identical allocation problems
- **transfer rate** — fraction of unseen compiler output that matches existing table entries. 88.2% means training on some languages covers 88.2% of others

## Tools

- **partopt** — our call graph partition optimizer. Decides which functions to merge (inline) vs keep separate, using GPU costs
- **regalloc-enum** — exhaustive enumerator of all possible constraint shapes for a given vreg count
- **z80_regalloc** — GPU + backtracking register allocation solver
- **VIR** — Virtual Intermediate Representation, the MinZ compiler's backend
- **Z3** — Microsoft's SMT solver, used by VIR as alternative to GPU brute-force
- **ddll** — cross-session messaging tool for collaboration between Claude instances

## The 5-Level Pipeline

| Level | Method | Covers | Speed |
|---|---|---|---|
| 1 | Table lookup (17.4M entries) | <=5v, 87% of corpus | O(1) |
| 2 | Composition via cut vertices | tw<=3 dense shapes, 46% of dense | O(1) per component |
| 3 | GPU brute-force | <=12v, tw=4 | seconds |
| 4 | CPU backtracking (pruned) | <=15v, tw=4 | <1 second |
| 5 | Island decomposition + Z3 | >15v or tw>=5 | seconds-minutes |

## Key Data Points

### Exhaustive Tables

| Level | Shapes | GPU Time | Feasible |
|---|---|---|---|
| <=4v | 156,506 | 40 sec | 78.9% |
| <=5v | 17,366,874 | 20 min | 67.7% |
| 6v | ~1.9B (est.) | ~7 days | 0.9% |

### Feasibility Phase Transition

| vregs | Feasible | Infeasible |
|---|---|---|
| 2v | 95.9% | 4.1% |
| 3v | 88.5% | 11.5% |
| 4v | 78.7% | 21.3% |
| 5v | 67.7% | 32.4% |
| 6v | 0.9% | 99.1% |

### Treewidth of Dense Corpus Functions (54 functions, density >40%)

| Treewidth | Count | % | Solver |
|---|---|---|---|
| tw=2 | 1 | 1.9% | Composition |
| tw=3 | 24 | 44.4% | Composition |
| tw=4 | 19 | 35.2% | GPU/Backtrack (all <=15v) |
| tw=5 | 5 | 9.3% | Islands + Z3 |
| tw=6 | 3 | 5.6% | Islands + Z3 |
| tw=8 | 1 | 1.9% | Islands + Z3 |
| tw=13 | 1 | 1.9% | Islands + Z3 |

### Cross-Function Merging (ZSQL, 31 functions)

- 5 profitable merges, 210T saved (36%)
- ~35T saved per eliminated call boundary (CALL 17T + RET 10T + shuffle 8T)
- 3-way merge composes linearly (70T = 2 x 35T)

### Backtracking Solver Performance

- Pattern-aware location masks: 1,000-4,000x speedup over naive
- main_island0 (15v, 7 intf): 350K nodes in <1s
- _prompt_island0 (14v, 18 intf): 6.1M nodes in <1s
- Pruning techniques: pattern masks + constraint propagation + forward checking + most-constrained-first ordering

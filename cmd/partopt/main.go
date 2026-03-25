// partopt — optimal call graph partitioning via exhaustive cost oracle
//
// Given a call graph and a GPU-computed register allocation table,
// finds the partition into "islands" that minimizes total cost:
//   total = Σ(island_alloc_cost) + Σ(boundary_shuffle_cost)
//
// Algorithm: bottom-up DP on the call tree.
//   1. Topological sort (leaves first)
//   2. For each leaf: cost = table_lookup(leaf)
//   3. For each internal node: try merging with each callee subset
//      - merged cost = table_lookup(merged_desc) if combined vregs ≤ K
//      - separate cost = cost(caller) + cost(callee) + shuffle_cost
//      - pick minimum
//   4. Propagate optimal cost upward
//
// Usage: partopt --callgraph graph.json --table table.json [--max-vregs 8]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
)

// Function in the call graph
type Function struct {
	Name    string   `json:"name"`
	Sig     string   `json:"sig"`
	NVregs  int      `json:"nVregs"`
	Callees []string `json:"callees"`
	// Cost from GPU table (filled during solve)
	AllocCost int `json:"-"`
}

// Table entry: signature → optimal cost
type TableEntry struct {
	Cost       int   `json:"cost"`
	Assignment []int `json:"assignment"`
}

// Call graph
type CallGraph struct {
	Functions []Function `json:"functions"`
}

// Partition decision for one function
type Decision struct {
	Name       string   `json:"name"`
	MergedWith []string `json:"mergedWith,omitempty"` // callees inlined into this
	IslandCost int      `json:"islandCost"`
	ShuffleCost int     `json:"shuffleCost"`
	TotalCost  int      `json:"totalCost"`
}

// Result
type PartitionResult struct {
	Decisions    []Decision `json:"decisions"`
	TotalCost    int        `json:"totalCost"`
	IslandCount  int        `json:"islandCount"`
	MergeCount   int        `json:"mergeCount"`
	SeparateCount int       `json:"separateCount"`
}

const defaultShuffleCost = 4 // LD r,r' = 4T per register move
const callRetCost = 27       // CALL nn (17T) + RET (10T)

func main() {
	callgraphFile := flag.String("callgraph", "", "call graph JSON file")
	tableFile := flag.String("table", "", "GPU allocation table JSON file")
	maxVregs := flag.Int("max-vregs", 8, "max vregs for table lookup (K)")
	demo := flag.Bool("demo", false, "run built-in demo")
	flag.Parse()

	if *demo {
		runDemo(*maxVregs)
		return
	}

	if *callgraphFile == "" || *tableFile == "" {
		fmt.Fprintf(os.Stderr, "Usage: partopt --callgraph graph.json --table table.json\n")
		fmt.Fprintf(os.Stderr, "       partopt --demo\n")
		os.Exit(1)
	}

	// Load call graph
	cgData, err := os.ReadFile(*callgraphFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading call graph: %v\n", err)
		os.Exit(1)
	}
	var cg CallGraph
	if err := json.Unmarshal(cgData, &cg); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing call graph: %v\n", err)
		os.Exit(1)
	}

	// Load table
	tblData, err := os.ReadFile(*tableFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading table: %v\n", err)
		os.Exit(1)
	}
	var table map[string]TableEntry
	if err := json.Unmarshal(tblData, &table); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing table: %v\n", err)
		os.Exit(1)
	}

	result := solve(cg, table, *maxVregs)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(result)
}

func runDemo(maxVregs int) {
	// Demo call graph:
	//   main → compute → add
	//                  → multiply
	//        → print
	//
	// add: 2 vregs, cost 8T
	// multiply: 3 vregs, cost 16T
	// compute: 4 vregs (calls add + multiply)
	//   - separate: cost(compute)=20 + cost(add)=8 + cost(mul)=16 + 2×shuffle(4T) = 52T
	//   - merge compute+add: 5 vregs, cost 24T + cost(mul)=16 + shuffle = 44T
	//   - merge compute+add+mul: 7 vregs, cost 32T = 32T (no shuffles!)
	// print: 2 vregs, cost 10T
	// main: 3 vregs (calls compute + print)

	cg := CallGraph{
		Functions: []Function{
			{Name: "add", Sig: "sig_add", NVregs: 2, Callees: nil},
			{Name: "multiply", Sig: "sig_mul", NVregs: 3, Callees: nil},
			{Name: "compute", Sig: "sig_comp", NVregs: 4, Callees: []string{"add", "multiply"}},
			{Name: "print", Sig: "sig_print", NVregs: 2, Callees: nil},
			{Name: "main", Sig: "sig_main", NVregs: 3, Callees: []string{"compute", "print"}},
		},
	}

	// Mock table: separate costs
	table := map[string]TableEntry{
		"sig_add":   {Cost: 8, Assignment: []int{0, 1}},
		"sig_mul":   {Cost: 16, Assignment: []int{0, 1, 2}},
		"sig_comp":  {Cost: 20, Assignment: []int{0, 1, 2, 3}},
		"sig_print": {Cost: 10, Assignment: []int{0, 2}},
		"sig_main":  {Cost: 12, Assignment: []int{0, 1, 5}},
		// Merged signatures (would come from GPU in production)
		"sig_comp+add":         {Cost: 24, Assignment: []int{0, 1, 2, 3, 4}},     // 5 vregs
		"sig_comp+mul":         {Cost: 28, Assignment: []int{0, 1, 2, 3, 4, 5}},  // 6 vregs
		"sig_comp+add+mul":     {Cost: 32, Assignment: []int{0, 1, 2, 3, 4, 5, 6}}, // 7 vregs
		"sig_main+print":       {Cost: 18, Assignment: []int{0, 1, 2, 5, 6}},     // 5 vregs
		"sig_main+comp":        {Cost: -1}, // 7 vregs merged, feasible but expensive
	}

	fmt.Fprintf(os.Stderr, "=== Demo Call Graph ===\n")
	fmt.Fprintf(os.Stderr, "  main(3v) → compute(4v) → add(2v)\n")
	fmt.Fprintf(os.Stderr, "                         → multiply(3v)\n")
	fmt.Fprintf(os.Stderr, "           → print(2v)\n")
	fmt.Fprintf(os.Stderr, "  Max vregs for table: %d\n\n", maxVregs)

	result := solve(cg, table, maxVregs)

	fmt.Fprintf(os.Stderr, "=== Optimal Partition ===\n")
	for _, d := range result.Decisions {
		if len(d.MergedWith) > 0 {
			fmt.Fprintf(os.Stderr, "  %s + %v → island cost %dT (saved %dT shuffle + %dT call/ret)\n",
				d.Name, d.MergedWith, d.IslandCost, d.ShuffleCost, len(d.MergedWith)*callRetCost)
		} else {
			fmt.Fprintf(os.Stderr, "  %s → separate, cost %dT\n", d.Name, d.IslandCost)
		}
	}
	fmt.Fprintf(os.Stderr, "\nTotal: %dT (%d islands, %d merges, %d separate)\n",
		result.TotalCost, result.IslandCount, result.MergeCount, result.SeparateCount)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(result)
}

// solve runs the bottom-up DP partition algorithm
func solve(cg CallGraph, table map[string]TableEntry, maxVregs int) PartitionResult {
	// Build adjacency + index
	funcMap := make(map[string]*Function)
	for i := range cg.Functions {
		funcMap[cg.Functions[i].Name] = &cg.Functions[i]
	}

	// Topological sort (leaves first)
	order := topoSort(cg)

	// DP: for each function, compute optimal cost with/without merging callees
	type dpEntry struct {
		cost       int
		mergedWith []string // which callees are inlined
	}
	dp := make(map[string]dpEntry)

	for _, name := range order {
		f := funcMap[name]
		if f == nil {
			continue
		}

		// Base: cost of this function alone
		baseCost := lookupCost(f.Sig, table)
		if baseCost < 0 {
			baseCost = 9999 // infeasible — use high cost
		}

		if len(f.Callees) == 0 {
			// Leaf function: no merge options
			dp[name] = dpEntry{cost: baseCost, mergedWith: nil}
			continue
		}

		// Try all subsets of callees to merge
		nCallees := len(f.Callees)
		bestCost := 999999
		var bestMerged []string

		for mask := 0; mask < (1 << nCallees); mask++ {
			// Compute merged vreg count
			mergedVregs := f.NVregs
			var merged []string
			var separateCallees []string

			for i := 0; i < nCallees; i++ {
				callee := f.Callees[i]
				cf := funcMap[callee]
				if cf == nil {
					continue
				}
				if mask&(1<<i) != 0 {
					mergedVregs += cf.NVregs
					merged = append(merged, callee)
				} else {
					separateCallees = append(separateCallees, callee)
				}
			}

			// Skip if merged vregs exceed table capacity
			if mergedVregs > maxVregs {
				continue
			}

			// Cost of merged island
			mergedSig := buildMergedSig(f.Name, merged)
			mergedCost := lookupCost(mergedSig, table)
			if mergedCost < 0 {
				// Not in table — estimate as sum of parts (no benefit)
				continue
			}

			// Cost of separate callees (from DP)
			separateCost := 0
			for _, sc := range separateCallees {
				if entry, ok := dp[sc]; ok {
					separateCost += entry.cost
				}
				// Add shuffle cost for separate callees (2 registers shuffled avg)
				separateCost += 2 * defaultShuffleCost
				// Add CALL/RET overhead
				separateCost += callRetCost
			}

			// Merged callees: their subtree costs are absorbed into the island
			// But their own callees still need to be accounted for
			mergedSubtreeCost := 0
			for _, mc := range merged {
				cf := funcMap[mc]
				if cf == nil {
					continue
				}
				// The merged callee's own callees become separate calls from the island
				for _, subCallee := range cf.Callees {
					if entry, ok := dp[subCallee]; ok {
						mergedSubtreeCost += entry.cost
					}
					mergedSubtreeCost += 2*defaultShuffleCost + callRetCost
				}
			}

			totalCost := mergedCost + separateCost + mergedSubtreeCost
			if totalCost < bestCost {
				bestCost = totalCost
				bestMerged = merged
			}
		}

		// Also try no merging at all
		noMergeCost := baseCost
		for _, callee := range f.Callees {
			if entry, ok := dp[callee]; ok {
				noMergeCost += entry.cost
			}
			noMergeCost += 2*defaultShuffleCost + callRetCost
		}
		if noMergeCost < bestCost {
			bestCost = noMergeCost
			bestMerged = nil
		}

		dp[name] = dpEntry{cost: bestCost, mergedWith: bestMerged}
	}

	// Build result
	var decisions []Decision
	mergeCount := 0
	separateCount := 0

	for _, name := range order {
		entry, ok := dp[name]
		if !ok {
			continue
		}
		// Skip functions that were merged into a parent
		merged := false
		for _, other := range order {
			if e, ok2 := dp[other]; ok2 {
				for _, m := range e.mergedWith {
					if m == name {
						merged = true
						break
					}
				}
			}
			if merged {
				break
			}
		}
		if merged {
			continue
		}

		d := Decision{
			Name:       name,
			MergedWith: entry.mergedWith,
			TotalCost:  entry.cost,
		}

		if len(entry.mergedWith) > 0 {
			mergeCount++
			// Island cost is the merged table lookup
			mergedSig := buildMergedSig(name, entry.mergedWith)
			d.IslandCost = lookupCost(mergedSig, map[string]TableEntry{mergedSig: {Cost: entry.cost}})
		} else {
			separateCount++
			d.IslandCost = entry.cost
		}

		decisions = append(decisions, d)
	}

	// Sort by name for stable output
	sort.Slice(decisions, func(i, j int) bool {
		return decisions[i].Name < decisions[j].Name
	})

	// Find root cost
	totalCost := 0
	if len(order) > 0 {
		root := order[len(order)-1]
		if entry, ok := dp[root]; ok {
			totalCost = entry.cost
		}
	}

	return PartitionResult{
		Decisions:     decisions,
		TotalCost:     totalCost,
		IslandCount:   len(decisions),
		MergeCount:    mergeCount,
		SeparateCount: separateCount,
	}
}

func lookupCost(sig string, table map[string]TableEntry) int {
	if entry, ok := table[sig]; ok {
		return entry.Cost
	}
	return -1
}

func buildMergedSig(parent string, merged []string) string {
	if len(merged) == 0 {
		return "sig_" + parent
	}
	sig := "sig_" + parent
	sort.Strings(merged)
	for _, m := range merged {
		sig += "+" + m
	}
	return sig
}

// topoSort returns functions in dependency order (leaves first)
func topoSort(cg CallGraph) []string {
	// Build adjacency
	deps := make(map[string][]string)
	allFuncs := make(map[string]bool)
	for _, f := range cg.Functions {
		allFuncs[f.Name] = true
		deps[f.Name] = f.Callees
	}

	visited := make(map[string]bool)
	var order []string

	var visit func(string)
	visit = func(name string) {
		if visited[name] {
			return
		}
		visited[name] = true
		for _, dep := range deps[name] {
			visit(dep)
		}
		order = append(order, name)
	}

	for _, f := range cg.Functions {
		visit(f.Name)
	}
	return order
}

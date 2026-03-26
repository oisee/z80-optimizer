// regalloc-enum — enumerate realistic regalloc constraint patterns
//
// Uses actual Z80 per-vreg allowed-loc sets (not all combos):
//   8-bit:  {A}, {C}, {any GPR8}, {any GPR8 except A}
//   16-bit: {HL}, {DE}, {any pair}
//
// Enumerates all (nVregs, locSets, widths, interference) combos.
// Generates FuncDesc JSON with ops whose patterns match the loc sets.
//
// Usage: regalloc-enum [--max-vregs 5] | z80_regalloc --server > results.jsonl
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

// Realistic per-vreg allowed loc sets from actual Z80 patterns
var locSets8 = [][]int{
	{0},             // must be A (ALU dst, IN, OUT)
	{2},             // must be C (PFCCO 2nd param)
	{0, 1, 2, 3, 4, 5, 6}, // any GPR8 (LD r,n)
	{1, 2, 3, 4, 5, 6},    // any GPR8 except A (non-accumulator)
}

var locSets16 = [][]int{
	{9},       // must be HL (ADD HL,rr, CALL handle)
	{8},       // must be DE (PFCCO 2nd param ptr)
	{7, 8, 9}, // any pair (LD rr,nn)
}

type Pattern struct {
	DstLocs  []int `json:"dstLocs"`
	SrcLocs0 []int `json:"srcLocs0"`
	SrcLocs1 []int `json:"srcLocs1"`
	Cost     int   `json:"cost"`
}

type Op struct {
	Dst      int       `json:"dst"`
	Src0     int       `json:"src0"`
	Src1     int       `json:"src1"`
	Patterns []Pattern `json:"patterns"`
}

type ParamConstraint struct {
	Vreg int `json:"vreg"`
	Loc  int `json:"loc"`
}

type FuncDesc struct {
	NVregs           int               `json:"nVregs"`
	Widths           []int             `json:"widths"`
	Ops              []Op              `json:"ops"`
	Interference     [][]int           `json:"interference"`
	ParamConstraints []ParamConstraint `json:"paramConstraints"`
}

// Compute exact treewidth via minimum elimination ordering.
// For nv≤6, tries all nv! permutations (max 720). Returns exact treewidth.
func treewidth(nv int, edges [][]int) int {
	adj := make([]map[int]bool, nv)
	for i := range adj {
		adj[i] = make(map[int]bool)
	}
	for _, e := range edges {
		adj[e[0]][e[1]] = true
		adj[e[1]][e[0]] = true
	}

	perm := make([]int, nv)
	for i := range perm {
		perm[i] = i
	}

	bestTW := nv - 1 // upper bound

	// Try all permutations
	var tryAll func(int)
	tryAll = func(k int) {
		if k == nv {
			// Evaluate this elimination ordering
			localAdj := make([]map[int]bool, nv)
			for i := range localAdj {
				localAdj[i] = make(map[int]bool)
				for v := range adj[i] {
					localAdj[i][v] = true
				}
			}
			tw := 0
			for _, v := range perm {
				deg := len(localAdj[v])
				if deg > tw {
					tw = deg
				}
				if tw >= bestTW {
					return // can't improve
				}
				// Eliminate: connect all neighbors
				nbrs := make([]int, 0, deg)
				for n := range localAdj[v] {
					nbrs = append(nbrs, n)
				}
				for i := 0; i < len(nbrs); i++ {
					for j := i + 1; j < len(nbrs); j++ {
						localAdj[nbrs[i]][nbrs[j]] = true
						localAdj[nbrs[j]][nbrs[i]] = true
					}
				}
				for _, n := range nbrs {
					delete(localAdj[n], v)
				}
				localAdj[v] = nil
			}
			if tw < bestTW {
				bestTW = tw
			}
			return
		}
		for i := k; i < nv; i++ {
			perm[k], perm[i] = perm[i], perm[k]
			tryAll(k + 1)
			perm[k], perm[i] = perm[i], perm[k]
		}
	}
	tryAll(0)
	return bestTW
}

func main() {
	maxVregs := flag.Int("max-vregs", 5, "max vregs to enumerate (2-6)")
	minTW := flag.Int("min-treewidth", 0, "only emit shapes with treewidth >= this (0=all)")
	onlyNV := flag.Int("only-nv", 0, "only emit shapes with this exact nVregs (0=all)")
	denseMasksFile := flag.String("dense-masks", "", "file of interference bitmasks to enumerate (one per line, skip all others)")
	flag.Parse()

	// Load dense masks if provided (fast path: skip treewidth computation)
	denseMasks := map[int]bool{}
	if *denseMasksFile != "" {
		data, err := os.ReadFile(*denseMasksFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading dense masks: %v\n", err)
			os.Exit(1)
		}
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var m int
			fmt.Sscanf(line, "%d", &m)
			denseMasks[m] = true
		}
		fmt.Fprintf(os.Stderr, "Loaded %d dense interference masks\n", len(denseMasks))
	}

	enc := json.NewEncoder(os.Stdout)
	total := 0

	filtered := 0
	for nv := 2; nv <= *maxVregs; nv++ {
		if *onlyNV > 0 && nv != *onlyNV {
			continue
		}
		nEdges := nv * (nv - 1) / 2
		nWidthCombos := 1 << nv

		for wc := 0; wc < nWidthCombos; wc++ {
			widths := make([]int, nv)
			locSetOptions := make([][][]int, nv)
			for i := 0; i < nv; i++ {
				if wc&(1<<i) != 0 {
					widths[i] = 16
					locSetOptions[i] = locSets16
				} else {
					widths[i] = 8
					locSetOptions[i] = locSets8
				}
			}

			// Number of loc-set combos
			nLocCombos := 1
			for i := 0; i < nv; i++ {
				nLocCombos *= len(locSetOptions[i])
			}

			for lc := 0; lc < nLocCombos; lc++ {
				// Decode loc set indices
				locSetIdx := make([]int, nv)
				rem := lc
				for i := nv - 1; i >= 0; i-- {
					locSetIdx[i] = rem % len(locSetOptions[i])
					rem /= len(locSetOptions[i])
				}

				// Get actual loc sets
				vregLocs := make([][]int, nv)
				for i := 0; i < nv; i++ {
					vregLocs[i] = locSetOptions[i][locSetIdx[i]]
				}

				// Enumerate all interference graphs
				for ig := 0; ig < (1 << nEdges); ig++ {
					// Dense mask filter: skip graphs not in the mask set
					if len(denseMasks) > 0 && !denseMasks[ig] {
						total++
						filtered++
						continue
					}
					var interf [][]int
					edgeIdx := 0
					for i := 0; i < nv; i++ {
						for j := i + 1; j < nv; j++ {
							if ig&(1<<edgeIdx) != 0 {
								interf = append(interf, []int{i, j})
							}
							edgeIdx++
						}
					}

					// Build ops: one op per vreg as dst, pattern uses vreg's loc set
					var ops []Op
					for i := 0; i < nv; i++ {
						ops = append(ops, Op{
							Dst:  i,
							Src0: -1,
							Src1: -1,
							Patterns: []Pattern{{
								DstLocs:  vregLocs[i],
								SrcLocs0: []int{},
								SrcLocs1: []int{},
								Cost:     4,
							}},
						})
					}

					// Add cross-vreg ops for pairs that interfere
					// (simulates real ALU ops: dst=vi, src=vj)
					edgeIdx = 0
					for i := 0; i < nv; i++ {
						for j := i + 1; j < nv; j++ {
							if ig&(1<<edgeIdx) != 0 && len(ops) < 64 {
								ops = append(ops, Op{
									Dst:  i,
									Src0: j,
									Src1: -1,
									Patterns: []Pattern{{
										DstLocs:  vregLocs[i],
										SrcLocs0: vregLocs[j],
										SrcLocs1: []int{},
										Cost:     4,
									}},
								})
							}
							edgeIdx++
						}
					}

					// Treewidth filter: skip shapes below threshold
					if *minTW > 0 {
						tw := treewidth(nv, interf)
						if tw < *minTW {
							filtered++
							total++
							continue
						}
					}

					fd := FuncDesc{
						NVregs:           nv,
						Widths:           widths,
						Ops:              ops,
						Interference:     interf,
						ParamConstraints: nil,
					}

					enc.Encode(fd)
					total++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "  %d vregs: %d total so far\n", nv, total)
	}
	fmt.Fprintf(os.Stderr, "Total: %d patterns (%d emitted, %d filtered by treewidth)\n", total, total-filtered, filtered)
}

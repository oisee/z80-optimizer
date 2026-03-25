// regalloc-enum — enumerate all possible regalloc constraint patterns
// and pipe them through the GPU solver for exhaustive table generation.
//
// Enumerates: all (nVregs, widths, interference, paramConstraints) combos
// for 2-4 vregs. Generates minimal FuncDesc JSON (1 dummy op per vreg)
// and outputs one JSON per line for the GPU --server.
//
// Usage: regalloc-enum [--max-vregs 4] | z80_regalloc --server > results.jsonl
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

// Loc indices matching the CUDA kernel
const (
	LocA   = 0
	LocB   = 1
	LocC   = 2
	LocD   = 3
	LocE   = 4
	LocH   = 5
	LocL   = 6
	LocBC  = 7
	LocDE  = 8
	LocHL  = 9
	LocIXH = 10
	LocIXL = 11
	LocIYH = 12
	LocIYL = 13
	LocMem = 14
)

var locs8 = []int{LocA, LocB, LocC, LocD, LocE, LocH, LocL}
var locs16 = []int{LocBC, LocDE, LocHL}

// Pin options: -1 = free (unconstrained), or specific loc
func pinOptions(width int) []int {
	opts := []int{-1} // free
	if width == 8 {
		opts = append(opts, locs8...)
	} else {
		opts = append(opts, locs16...)
	}
	return opts
}

// All possible locs for a given width
func allLocs(width int) []int {
	if width == 16 {
		return locs16
	}
	return locs8
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

func main() {
	maxVregs := flag.Int("max-vregs", 4, "max vregs to enumerate (2-4)")
	flag.Parse()

	enc := json.NewEncoder(os.Stdout)
	total := 0

	for nv := 2; nv <= *maxVregs; nv++ {
		// Number of possible interference edges
		nEdges := nv * (nv - 1) / 2

		// Enumerate all width combinations
		nWidthCombos := 1 << nv // 2^nv (each vreg is 8 or 16)

		for wc := 0; wc < nWidthCombos; wc++ {
			widths := make([]int, nv)
			for i := 0; i < nv; i++ {
				if wc&(1<<i) != 0 {
					widths[i] = 16
				} else {
					widths[i] = 8
				}
			}

			// Build pin options per vreg
			pinOpts := make([][]int, nv)
			for i := 0; i < nv; i++ {
				pinOpts[i] = pinOptions(widths[i])
			}

			// Enumerate all pin combinations (cartesian product)
			pinCombos := 1
			for i := 0; i < nv; i++ {
				pinCombos *= len(pinOpts[i])
			}

			for pc := 0; pc < pinCombos; pc++ {
				pins := make([]int, nv)
				rem := pc
				for i := nv - 1; i >= 0; i-- {
					pins[i] = pinOpts[i][rem%len(pinOpts[i])]
					rem /= len(pinOpts[i])
				}

				// Enumerate all interference graphs
				for ig := 0; ig < (1 << nEdges); ig++ {
					// Build interference pairs
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

					// Build param constraints from pins
					var params []ParamConstraint
					for i, pin := range pins {
						if pin >= 0 {
							params = append(params, ParamConstraint{Vreg: i, Loc: pin})
						}
					}

					// Build a minimal op set: one op per vreg as dst
					// Each op: dst=vreg, src0=-1, src1=-1, pattern allows all valid locs
					var ops []Op
					for i := 0; i < nv; i++ {
						ops = append(ops, Op{
							Dst:  i,
							Src0: -1,
							Src1: -1,
							Patterns: []Pattern{{
								DstLocs:  allLocs(widths[i]),
								SrcLocs0: []int{},
								SrcLocs1: []int{},
								Cost:     4,
							}},
						})
					}

					fd := FuncDesc{
						NVregs:           nv,
						Widths:           widths,
						Ops:              ops,
						Interference:     interf,
						ParamConstraints: params,
					}

					enc.Encode(fd)
					total++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "  %d vregs: %d total so far\n", nv, total)
	}
	fmt.Fprintf(os.Stderr, "Total: %d function patterns\n", total)
}

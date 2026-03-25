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

func main() {
	maxVregs := flag.Int("max-vregs", 5, "max vregs to enumerate (2-6)")
	flag.Parse()

	enc := json.NewEncoder(os.Stdout)
	total := 0

	for nv := 2; nv <= *maxVregs; nv++ {
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
	fmt.Fprintf(os.Stderr, "Total: %d function patterns\n", total)
}

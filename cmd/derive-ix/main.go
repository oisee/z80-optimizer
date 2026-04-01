// derive-ix — derive IX-half alternatives from existing enriched tables.
//
// For each feasible entry in an existing 10-loc enriched table where a vreg
// is assigned to B/C/D/E (locs 1-4), derives alternative assignments using
// IXH/IXL/IYH/IYL (locs 10-13).
//
// The extra cost per IX substitution = 4T × N_uses, where N_uses is the
// number of ops involving that vreg = 1 (defining op) + degree in interference graph.
// Degree is inferred from the shape's interference bitmask (embedded in index).
//
// H/L → IX substitution is EXCLUDED (would require 16T per use, rarely optimal).
//
// Output: JSONL records compatible with build-ix-table format:
//
//	{"cost": N, "assignment": [...], "searchSpace": 0, "feasible": 1, "derived": true, ...}
//
// These records can be fed into merge-tables alongside GPU-direct results.
//
// Usage:
//
//	./derive-ix -input data/enriched_5v.enr -max-vregs 5 > data/ix_derived_5v.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/oisee/z80-optimizer/pkg/enr"
	"github.com/oisee/z80-optimizer/pkg/regalloc"
)

// Original 4-set loc options (matching regalloc-enum before IX expansion).
var origLocSets8 = [][]int{
	{0},                    // 0: must be A
	{2},                    // 1: must be C
	{0, 1, 2, 3, 4, 5, 6}, // 2: any GPR8
	{1, 2, 3, 4, 5, 6},    // 3: any GPR8 except A
}

var origLocSets16 = [][]int{
	{9},       // 0: must be HL
	{8},       // 1: must be DE
	{7, 8, 9}, // 2: any pair
}

// IX-half locations to try as substitutions.
var ixLocs = []byte{
	regalloc.LocIXH,
	regalloc.LocIXL,
	regalloc.LocIYH,
	regalloc.LocIYL,
}

// derivableGPR returns true if a loc can be substituted with an IX half.
// B(1), C(2), D(3), E(4) — yes. H(5), L(6) — no (16T per transfer).
func derivableGPR(loc byte) bool {
	return loc >= 1 && loc <= 4
}

// nOpsForVreg returns the number of ops in regalloc-enum's FuncDesc that
// involve vreg vi: 1 (defining op) + degree of vi in the interference graph.
func nOpsForVreg(vi, nVregs int, intfMask uint32) int {
	degree := 0
	edgeIdx := 0
	for a := 0; a < nVregs; a++ {
		for b := a + 1; b < nVregs; b++ {
			if intfMask&(1<<edgeIdx) != 0 {
				if a == vi || b == vi {
					degree++
				}
			}
			edgeIdx++
		}
	}
	return 1 + degree
}

type shapeInfo struct {
	nVregs    int
	widths    []int      // per-vreg width (8 or 16)
	locSetIdx []int      // per-vreg locSet index (into ls8 or ls16)
	intfMask  uint32
}

// streamShapes emits shapeInfo in the same enumeration order as regalloc-enum,
// using the given 8-bit locSets. Closes the channel when done.
func streamShapes(maxVregs int, ls8, ls16 [][]int) <-chan shapeInfo {
	ch := make(chan shapeInfo, 256)
	go func() {
		defer close(ch)
		for nv := 2; nv <= maxVregs; nv++ {
			nEdges := nv * (nv - 1) / 2
			nWidthCombos := 1 << nv

			for wc := 0; wc < nWidthCombos; wc++ {
				widths := make([]int, nv)
				locOptions := make([][][]int, nv)
				for i := 0; i < nv; i++ {
					if wc&(1<<i) != 0 {
						widths[i] = 16
						locOptions[i] = ls16
					} else {
						widths[i] = 8
						locOptions[i] = ls8
					}
				}

				nLocCombos := 1
				for i := 0; i < nv; i++ {
					nLocCombos *= len(locOptions[i])
				}

				for lc := 0; lc < nLocCombos; lc++ {
					locSetIdx := make([]int, nv)
					rem := lc
					for i := nv - 1; i >= 0; i-- {
						locSetIdx[i] = rem % len(locOptions[i])
						rem /= len(locOptions[i])
					}

					for ig := 0; ig < (1 << nEdges); ig++ {
						ch <- shapeInfo{
							nVregs:    nv,
							widths:    widths,
							locSetIdx: locSetIdx,
							intfMask:  uint32(ig),
						}
					}
				}
			}
		}
	}()
	return ch
}

type derivedRecord struct {
	Cost       int   `json:"cost"`
	Assignment []int `json:"assignment"`
	SearchSpace int  `json:"searchSpace"`
	Feasible   int   `json:"feasible"`
	Derived    bool  `json:"derived"`
	// Shape metadata for merge-tables to identify the IX-expanded shape index.
	NVregs    int   `json:"nVregs"`
	IntfMask  int   `json:"intfMask"`
	LocSetIdx []int `json:"locSetIdx"`
}

func main() {
	inputPath := flag.String("input", "", "input .enr file (uncompressed)")
	maxVregs := flag.Int("max-vregs", 5, "max vregs in the input table (2-6)")
	flag.Parse()

	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "error: -input required")
		os.Exit(1)
	}

	rdr, f, err := enr.Open(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open %s: %v\n", *inputPath, err)
		os.Exit(1)
	}
	defer f.Close()

	shapeCh := streamShapes(*maxVregs, origLocSets8, origLocSets16)

	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	enc := json.NewEncoder(out)

	var nTotal, nFeasible, nDerived uint64
	progress := uint64(500_000)

	for sh := range shapeCh {
		entry, err := rdr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "read entry %d: %v\n", nTotal, err)
			os.Exit(1)
		}
		nTotal++

		if nTotal%progress == 0 {
			fmt.Fprintf(os.Stderr, "  %d entries, %d feasible, %d derived\n", nTotal, nFeasible, nDerived)
		}

		if entry.Infeasible() {
			continue
		}
		nFeasible++

		// Try IX substitutions for each derivable vreg.
		for vi := 0; vi < sh.nVregs; vi++ {
			if sh.widths[vi] != 8 {
				continue // only 8-bit vregs can use IX halves
			}
			mainLoc := entry.Assignment[vi]
			if !derivableGPR(mainLoc) {
				continue
			}

			nOps := nOpsForVreg(vi, sh.nVregs, sh.intfMask)
			extraCost := 4 * nOps // 4T extra per use (8T vs 4T LD)

			for _, ixLoc := range ixLocs {
				// Skip if another vreg already uses this IX half.
				conflict := false
				for vj := range entry.Assignment {
					if vj != vi && entry.Assignment[vj] == ixLoc {
						conflict = true
						break
					}
				}
				if conflict {
					continue
				}

				derivedCost := entry.Cost + extraCost

				assign := make([]int, sh.nVregs)
				for k, loc := range entry.Assignment {
					assign[k] = int(loc)
				}
				assign[vi] = int(ixLoc)

				// The derived shape uses locSet 4 ({IXH,IXL,IYH,IYL}) for vreg vi.
				locSetIdxDerived := make([]int, sh.nVregs)
				copy(locSetIdxDerived, sh.locSetIdx)
				locSetIdxDerived[vi] = 4

				enc.Encode(derivedRecord{
					Cost:       derivedCost,
					Assignment: assign,
					Feasible:   1,
					Derived:    true,
					NVregs:     sh.nVregs,
					IntfMask:   int(sh.intfMask),
					LocSetIdx:  locSetIdxDerived,
				})
				nDerived++
			}
		}
	}

	out.Flush()
	fmt.Fprintf(os.Stderr, "Done: %d total, %d feasible, %d derived records emitted\n",
		nTotal, nFeasible, nDerived)
}

// gen6v-ix-feed — fast FuncDesc JSON generator for ix_expanded_6v_dense GPU run.
//
// Problem: regalloc-enum iterates ~62.9 billion (widths × locsets × masks) for nv=6,
// discarding 99.97% via treewidth filter. At 720 permutations/check that's ~hours.
//
// Solution:
//   1. Pre-compute valid 6v interference masks (treewidth≥4) by brute-force over all
//      32768 possible nv=6 graphs. Takes <100ms. Result: ~few hundred valid masks.
//   2. Iterate ONLY valid masks as OUTER loop, all 6-locSet8 widths/locsets as inner.
//   3. Total output: N_valid × 64 × 46656 FuncDesc JSON lines → feed directly to GPU.
//
// Usage:
//   ./gen6v-ix-feed | ./cuda/z80_regalloc --server > data/ix_expanded_6v_dense.jsonl
//
//   # Or pre-buffer in RAM (128GB system):
//   ./gen6v-ix-feed > /dev/shm/6v_feed.jsonl   # ~few hundred GB in /dev/shm
//   ./cuda/z80_regalloc --server < /dev/shm/6v_feed.jsonl > data/ix_expanded_6v_dense.jsonl
//
// Flags:
//   -min-tw  N  minimum treewidth (default 4)
//   -nv      N  nVregs (default 6)
//   -workers N  parallel JSON encoders (default NumCPU)
//   -buf-mb  N  stdout write buffer MB (default 64)
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/bits"
	"os"
	"runtime"
	"sync"
)

// ─── Loc sets (must match regalloc-enum exactly) ──────────────────────────────

var locSets8 = [][]int{
	{0},                              // 0: must be A
	{2},                              // 1: must be C
	{0, 1, 2, 3, 4, 5, 6},           // 2: any GPR8
	{1, 2, 3, 4, 5, 6},              // 3: any GPR8 except A
	{10, 11, 12, 13},                 // 4: must be IX/IY half
	{0, 1, 2, 3, 4, 10, 11, 12, 13}, // 5: any 8-bit except H/L
}

var locSets16 = [][]int{
	{9},       // 0: must be HL
	{8},       // 1: must be DE
	{7, 8, 9}, // 2: any pair
}

// ─── Treewidth (exact, via minimum elimination ordering) ─────────────────────

// treewidth computes the exact treewidth of an nv-vertex graph whose edges are
// encoded in adjMask (bit edgeIdx = edge (i,j) where edgeIdx = Σ_{a<i}(nv-1-a)+(j-i-1)).
// Uses exhaustive elimination ordering search (≤6! = 720 permutations for nv≤6).
func treewidth(nv int, adjMask uint32) int {
	// Build symmetric adjacency as [nv]uint8 bitmask.
	var origAdj [6]uint8
	edgeIdx := 0
	for i := 0; i < nv; i++ {
		for j := i + 1; j < nv; j++ {
			if adjMask&(1<<edgeIdx) != 0 {
				origAdj[i] |= 1 << uint(j)
				origAdj[j] |= 1 << uint(i)
			}
			edgeIdx++
		}
	}

	perm := [6]int{0, 1, 2, 3, 4, 5}
	bestTW := nv - 1 // upper bound: at most nv-1

	var tryAll func(k int)
	tryAll = func(k int) {
		if k == nv {
			// Simulate elimination ordering; track max degree.
			var adj [6]uint8 = origAdj
			tw := 0
			for p := 0; p < nv; p++ {
				v := perm[p]
				deg := bits.OnesCount8(adj[v])
				if deg > tw {
					tw = deg
				}
				if tw >= bestTW {
					return // can't beat current best
				}
				// Eliminate v: connect all pairs of its neighbors.
				nbrs := adj[v]
				for a := 0; a < nv; a++ {
					if nbrs&(1<<uint(a)) != 0 {
						adj[a] = (adj[a] | nbrs) &^ (1 << uint(a)) &^ (1 << uint(v))
					}
				}
				adj[v] = 0
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

// validMasks returns all interference masks (nv vertices) with treewidth >= minTW.
func validMasks(nv, minTW int) []uint32 {
	nEdges := nv * (nv - 1) / 2
	total := 1 << nEdges
	var masks []uint32
	for ig := 0; ig < total; ig++ {
		if treewidth(nv, uint32(ig)) >= minTW {
			masks = append(masks, uint32(ig))
		}
	}
	return masks
}

// ─── FuncDesc types ──────────────────────────────────────────────────────────

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

type FuncDesc struct {
	NVregs       int     `json:"nVregs"`
	Widths       []int   `json:"widths"`
	Ops          []Op    `json:"ops"`
	Interference [][]int `json:"interference"`
}

func edgePairs(nv int, mask uint32) [][]int {
	var edges [][]int
	idx := 0
	for i := 0; i < nv; i++ {
		for j := i + 1; j < nv; j++ {
			if mask&(1<<uint(idx)) != 0 {
				edges = append(edges, []int{i, j})
			}
			idx++
		}
	}
	return edges
}

func buildFuncDesc(nv int, widths []int, vregLocs [][]int, edges [][]int) FuncDesc {
	ops := make([]Op, 0, nv+len(edges))
	empty := []int{}
	for i := 0; i < nv; i++ {
		ops = append(ops, Op{
			Dst: i, Src0: -1, Src1: -1,
			Patterns: []Pattern{{DstLocs: vregLocs[i], SrcLocs0: empty, SrcLocs1: empty, Cost: 4}},
		})
	}
	for _, e := range edges {
		if len(ops) < 64 {
			ops = append(ops, Op{
				Dst: e[0], Src0: e[1], Src1: -1,
				Patterns: []Pattern{{DstLocs: vregLocs[e[0]], SrcLocs0: vregLocs[e[1]], SrcLocs1: empty, Cost: 4}},
			})
		}
	}
	if edges == nil {
		edges = [][]int{}
	}
	return FuncDesc{NVregs: nv, Widths: widths, Ops: ops, Interference: edges}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	minTW := flag.Int("min-tw", 4, "minimum treewidth to include")
	nv := flag.Int("nv", 6, "nVregs to enumerate")
	workers := flag.Int("workers", runtime.NumCPU(), "parallel JSON encoder goroutines")
	bufMB := flag.Int("buf-mb", 64, "stdout write buffer MB")
	maskStart := flag.Int("mask-start", 0, "first mask index to emit (for multi-GPU split)")
	maskEnd := flag.Int("mask-end", -1, "last mask index (exclusive); -1 = all")
	flag.Parse()

	// Step 1: pre-compute valid masks.
	fmt.Fprintf(os.Stderr, "gen6v-ix-feed: computing valid masks nv=%d tw>=%d ...\n", *nv, *minTW)
	masks := validMasks(*nv, *minTW)
	nEdges := *nv * (*nv - 1) / 2
	fmt.Fprintf(os.Stderr, "gen6v-ix-feed: %d valid masks out of %d total 6v graphs\n",
		len(masks), 1<<nEdges)

	nWidths := 1 << *nv

	// Count total shapes.
	var totalShapes int64
	for wc := 0; wc < nWidths; wc++ {
		nLC := 1
		for i := 0; i < *nv; i++ {
			if wc&(1<<uint(i)) != 0 {
				nLC *= len(locSets16)
			} else {
				nLC *= len(locSets8)
			}
		}
		totalShapes += int64(len(masks)) * int64(nLC)
	}
	fmt.Fprintf(os.Stderr, "gen6v-ix-feed: total shapes to emit: %.1fM\n",
		float64(totalShapes)/1e6)

	// Step 2: parallel encode.
	type workItem struct {
		mask uint32
		wc   int
	}

	workCh := make(chan workItem, *workers*8)
	outCh := make(chan []byte, *workers*64)

	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range workCh {
				edges := edgePairs(*nv, item.mask)
				widths := make([]int, *nv)
				locSetOptions := make([][][]int, *nv)
				nLC := 1
				for i := 0; i < *nv; i++ {
					if item.wc&(1<<uint(i)) != 0 {
						widths[i] = 16
						locSetOptions[i] = locSets16
					} else {
						widths[i] = 8
						locSetOptions[i] = locSets8
					}
					nLC *= len(locSetOptions[i])
				}

				for lc := 0; lc < nLC; lc++ {
					vregLocs := make([][]int, *nv)
					rem := lc
					for i := *nv - 1; i >= 0; i-- {
						idx := rem % len(locSetOptions[i])
						vregLocs[i] = locSetOptions[i][idx]
						rem /= len(locSetOptions[i])
					}
					fd := buildFuncDesc(*nv, widths, vregLocs, edges)
					b, _ := json.Marshal(fd)
					outCh <- append(b, '\n')
				}
			}
		}()
	}

	// Writer.
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		bw := bufio.NewWriterSize(os.Stdout, *bufMB<<20)
		for b := range outCh {
			bw.Write(b)
		}
		bw.Flush()
	}()

	// Apply mask range (for multi-GPU split).
	start := *maskStart
	end := *maskEnd
	if end < 0 || end > len(masks) {
		end = len(masks)
	}
	if start < 0 {
		start = 0
	}
	activeMasks := masks[start:end]
	fmt.Fprintf(os.Stderr, "gen6v-ix-feed: emitting masks [%d, %d) = %d masks\n",
		start, end, len(activeMasks))

	// Feed: masks as outer loop → eliminates 99.97% wasted iterations.
	for m, mask := range activeMasks {
		for wc := 0; wc < nWidths; wc++ {
			workCh <- workItem{mask: mask, wc: wc}
		}
		if (m+1)%100 == 0 {
			fmt.Fprintf(os.Stderr, "  mask %d/%d (%.0f%%)\r", m+1, len(activeMasks),
				float64(m+1)/float64(len(activeMasks))*100)
		}
	}
	close(workCh)
	wg.Wait()
	close(outCh)
	<-writerDone
	fmt.Fprintf(os.Stderr, "\ngen6v-ix-feed: done.\n")
}

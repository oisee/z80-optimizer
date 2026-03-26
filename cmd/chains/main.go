// chains — abstract computation chain solver
//
// Finds shortest addition-subtraction chains for constant multiplication.
// ISA-independent: results materialize to any CPU (Z80, 6502, RISC-V, ARM).
//
// Abstract ops: dbl (×2), add (+ saved), sub (- saved), save (checkpoint),
//               neg (-v), shr (÷2), mask (& const)
//
// Usage: chains [--max-depth 20] [--k 42] [--div] [--json]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"
)

// Op types
const (
	OpDbl  = iota // v = v * 2
	OpAdd         // v = v + saved[slot]
	OpSub         // v = v - saved[slot]
	OpSave        // push v to save stack
	OpNeg         // v = -v
	// For division:
	// OpShr        // v = v >> 1
	// OpMask       // v = v & constant
)

const MaxSaveSlots = 3

var opNames = []string{"dbl", "add", "sub", "save", "neg"}

type ChainResult struct {
	K     int      `json:"k"`
	Ops   []string `json:"ops"`
	Depth int      `json:"depth"`
}

type state struct {
	value int
	saved [MaxSaveSlots]int
	nSaved int
}

var (
	bestDepth int
	bestOps   []byte
	nodesExplored uint64
)

func search(s state, ops []byte, depth, maxDepth, target int) {
	nodesExplored++

	// Check if we hit target (mod 256 for 8-bit multiply)
	val := s.value & 0xFF
	if val == target && depth > 0 {
		if depth < bestDepth {
			bestDepth = depth
			bestOps = make([]byte, depth)
			copy(bestOps, ops[:depth])
		}
		return
	}

	// Prune: can't improve
	if depth >= bestDepth-1 || depth >= maxDepth {
		return
	}

	// Try each op (all arithmetic mod 256)
	// dbl
	ns := s
	ns.value = (s.value * 2) & 0xFF
	if ns.value != s.value { // skip if no effect
		ops[depth] = OpDbl
		search(ns, ops, depth+1, maxDepth, target)
	}

	// add(i) for each saved slot
	for i := 0; i < s.nSaved; i++ {
		ns = s
		ns.value = (s.value + s.saved[i]) & 0xFF
		if ns.value != s.value {
			ops[depth] = OpAdd + byte(i)*10 // encode slot in op
			search(ns, ops, depth+1, maxDepth, target)
		}
	}

	// sub(i) for each saved slot
	for i := 0; i < s.nSaved; i++ {
		ns = s
		ns.value = (s.value - s.saved[i]) & 0xFF
		if ns.value != s.value {
			ops[depth] = OpSub + byte(i)*10
			search(ns, ops, depth+1, maxDepth, target)
		}
	}

	// save (if slots available)
	if s.nSaved < MaxSaveSlots {
		ns = s
		ns.saved[ns.nSaved] = s.value
		ns.nSaved++
		ops[depth] = OpSave
		search(ns, ops, depth+1, maxDepth, target)
	}

	// neg
	ns = s
	ns.value = (-s.value) & 0xFF
	if ns.value != s.value && ns.value != 0 {
		ops[depth] = OpNeg
		search(ns, ops, depth+1, maxDepth, target)
	}
}

func decodeOps(ops []byte) []string {
	names := make([]string, len(ops))
	for i, op := range ops {
		slot := int(op) / 10
		base := op % 10
		switch base {
		case OpDbl:
			names[i] = "dbl"
		case OpAdd:
			names[i] = fmt.Sprintf("add(%d)", slot)
		case OpSub:
			names[i] = fmt.Sprintf("sub(%d)", slot)
		case OpSave:
			names[i] = "save"
		case OpNeg:
			names[i] = "neg"
		default:
			names[i] = fmt.Sprintf("?%d", op)
		}
	}
	return names
}

func solveK(k, maxDepth int) *ChainResult {
	bestDepth = maxDepth + 1
	bestOps = nil
	nodesExplored = 0

	// Initial state: value = 1 (the input multiplier)
	// For multiply by K: start with 1, reach K
	s := state{value: 1}
	ops := make([]byte, maxDepth+1)

	// Iterative deepening
	for d := 1; d <= maxDepth; d++ {
		bestDepth = d + 1
		search(s, ops, 0, d, k)
		if bestOps != nil {
			break
		}
	}

	if bestOps == nil {
		return nil
	}

	return &ChainResult{
		K:     k,
		Ops:   decodeOps(bestOps),
		Depth: len(bestOps),
	}
}

func main() {
	maxDepth := flag.Int("max-depth", 20, "maximum chain depth")
	singleK := flag.Int("k", 0, "single constant (0 = all 2..255)")
	jsonMode := flag.Bool("json", false, "JSON output")
	flag.Parse()

	startK, endK := 2, 255
	if *singleK > 0 {
		startK, endK = *singleK, *singleK
	}

	solved := 0
	var results []ChainResult

	if *jsonMode {
		fmt.Println("[")
	}

	for k := startK; k <= endK; k++ {
		start := time.Now()
		r := solveK(k, *maxDepth)
		elapsed := time.Since(start)

		if r != nil {
			solved++
			results = append(results, *r)
			if *jsonMode {
				if solved > 1 {
					fmt.Print(",")
				}
				b, _ := json.Marshal(r)
				fmt.Println(string(b))
			} else {
				fmt.Printf("x%d: %v (%d steps, %v, %d nodes)\n",
					k, r.Ops, r.Depth, elapsed, nodesExplored)
			}
		} else {
			fmt.Fprintf(os.Stderr, "x%d: not found at depth %d (%v)\n", k, *maxDepth, elapsed)
		}
	}

	if *jsonMode {
		fmt.Println("]")
	}

	fmt.Fprintf(os.Stderr, "Done: %d/%d solved\n", solved, endK-startK+1)
}

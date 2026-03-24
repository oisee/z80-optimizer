// mulopt — brute-force optimal constant multiplication sequences for Z80.
//
// For each constant K (2..255), finds the shortest instruction sequence
// where A_out = A_in * K (mod 256).
//
// Instruction pool (13 ops):
//   ADD A,A  (4T)   ADD A,B  (4T)   SUB B    (4T)   LD B,A   (4T)
//   SLA A    (8T)   SRL A    (8T)   EX AF,AF'(4T)   EXX      (4T)
//   ADC A,B  (4T)   ADC A,A  (4T)   SBC A,B  (4T)   SBC A,A  (4T)
//   OR A     (4T)
//
// Usage: mulopt [-max-len 8] [-json] [-k 42]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
)

type MulOp uint8

const (
	OpAddAA  MulOp = iota // ADD A,A (4T)
	OpAddAB               // ADD A,B (4T)
	OpSubB                // SUB B   (4T)
	OpLdBA                // LD B,A  (4T)
	OpSlaA                // SLA A   (8T)
	OpSrlA                // SRL A   (8T)
	OpExAF                // EX AF,AF' (4T)
	OpExx                 // EXX (4T)
	OpAdcAB               // ADC A,B (4T)
	OpAdcAA               // ADC A,A (4T)
	OpSbcAB               // SBC A,B (4T)
	OpSbcAA               // SBC A,A (4T)
	OpOrA                 // OR A    (4T) — clears carry
	OpNeg                 // NEG     (8T) — negate A
	NumOps
)

var opNames = [NumOps]string{
	"ADD A,A", "ADD A,B", "SUB B", "LD B,A", "SLA A", "SRL A",
	"EX AF,AF'", "EXX", "ADC A,B", "ADC A,A", "SBC A,B", "SBC A,A", "OR A", "NEG",
}

var opCost = [NumOps]int{
	4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 8,
}

// CPU state for multiply simulation.
// We track: A, F(carry), B + shadow A', F'(carry'), B'
// EXX also swaps C/D/E/H/L but we only use B, so B↔B'.
type MulState struct {
	a, b     uint8 // main registers
	carry    bool  // carry flag
	aS, bS   uint8 // shadow A', B'
	carryS   bool  // shadow carry
}

func exec(op MulOp, s MulState) MulState {
	switch op {
	case OpAddAA:
		r := uint16(s.a) + uint16(s.a)
		s.carry = r > 0xFF
		s.a = uint8(r)
	case OpAddAB:
		r := uint16(s.a) + uint16(s.b)
		s.carry = r > 0xFF
		s.a = uint8(r)
	case OpSubB:
		r := int16(s.a) - int16(s.b)
		s.carry = r < 0
		s.a = uint8(r)
	case OpLdBA:
		s.b = s.a
	case OpSlaA:
		s.carry = s.a&0x80 != 0
		s.a = s.a << 1
	case OpSrlA:
		s.carry = s.a&0x01 != 0
		s.a = s.a >> 1
	case OpExAF:
		s.a, s.aS = s.aS, s.a
		s.carry, s.carryS = s.carryS, s.carry
	case OpExx:
		s.b, s.bS = s.bS, s.b
	case OpAdcAB:
		c := uint16(0)
		if s.carry {
			c = 1
		}
		r := uint16(s.a) + uint16(s.b) + c
		s.carry = r > 0xFF
		s.a = uint8(r)
	case OpAdcAA:
		c := uint16(0)
		if s.carry {
			c = 1
		}
		r := uint16(s.a) + uint16(s.a) + c
		s.carry = r > 0xFF
		s.a = uint8(r)
	case OpSbcAB:
		c := uint16(0)
		if s.carry {
			c = 1
		}
		r := int16(s.a) - int16(s.b) - int16(c)
		s.carry = r < 0
		s.a = uint8(r)
	case OpSbcAA:
		c := uint16(0)
		if s.carry {
			c = 1
		}
		r := int16(s.a) - int16(s.a) - int16(c)
		s.carry = r < 0
		s.a = uint8(r)
	case OpOrA:
		s.carry = false
	case OpNeg:
		s.carry = s.a != 0
		s.a = -s.a
	}
	return s
}

// execSeq runs a sequence on initial A value, returns final A.
func execSeq(seq []MulOp, input uint8) uint8 {
	s := MulState{a: input}
	for _, op := range seq {
		s = exec(op, s)
	}
	return s.a
}

// Test if a sequence multiplies A by K for all inputs 0..255.
// Quick-reject with 3 discriminating inputs before full check.
func testSequence(seq []MulOp, k uint8) bool {
	// Quick reject: test 1, 2, 3 first (catches >99% of failures)
	if execSeq(seq, 1) != k {
		return false
	}
	if execSeq(seq, 2) != 2*k {
		return false
	}
	if execSeq(seq, 3) != 3*k {
		return false
	}
	// Full verification
	for input := 0; input < 256; input++ {
		if execSeq(seq, uint8(input)) != uint8(input)*k {
			return false
		}
	}
	return true
}

type MulResult struct {
	K       int      `json:"k"`
	Ops     []string `json:"ops"`
	Length  int      `json:"length"`
	TStates int      `json:"tstates"`
}

func seqCost(seq []MulOp) int {
	cost := 0
	for _, op := range seq {
		cost += opCost[op]
	}
	return cost
}

func seqNames(seq []MulOp) []string {
	names := make([]string, len(seq))
	for i, op := range seq {
		names[i] = opNames[op]
	}
	return names
}

// Find optimal sequence for constant K.
func findOptimal(k uint8, maxLen int) *MulResult {
	// Try each length, shortest first (BFS by length).
	// At each length, also track the minimum-cost sequence found.
	for length := 1; length <= maxLen; length++ {
		var bestSeq []MulOp
		bestCost := 999999

		seq := make([]MulOp, length)
		for {
			if testSequence(seq, k) {
				c := seqCost(seq)
				if c < bestCost {
					bestCost = c
					bestSeq = make([]MulOp, length)
					copy(bestSeq, seq)
				}
			}

			// Increment
			carry := true
			for i := length - 1; i >= 0 && carry; i-- {
				seq[i]++
				if seq[i] >= MulOp(NumOps) {
					seq[i] = 0
				} else {
					carry = false
				}
			}
			if carry {
				break
			}
		}

		if bestSeq != nil {
			return &MulResult{
				K:       int(k),
				Ops:     seqNames(bestSeq),
				Length:  length,
				TStates: bestCost,
			}
		}
	}
	return nil
}

func main() {
	maxLen := flag.Int("max-len", 8, "maximum sequence length to search")
	jsonOut := flag.Bool("json", false, "output as JSON array")
	singleK := flag.Int("k", 0, "search for single constant (0 = all 2..255)")
	flag.Parse()

	if *singleK > 0 {
		k := uint8(*singleK)
		fmt.Fprintf(os.Stderr, "Searching for mul×%d (max length %d, %d ops)...\n", k, *maxLen, NumOps)
		result := findOptimal(k, *maxLen)
		if result == nil {
			fmt.Fprintf(os.Stderr, "No sequence found for ×%d within length %d\n", k, *maxLen)
			os.Exit(1)
		}
		if *jsonOut {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(result)
		} else {
			fmt.Printf("×%d: %v (%d insts, %dT)\n", k, result.Ops, result.Length, result.TStates)
		}
		return
	}

	// Parallel search across all constants
	nWorkers := runtime.NumCPU()
	results := make([]*MulResult, 254)
	var mu sync.Mutex
	var wg sync.WaitGroup
	solved := 0

	ch := make(chan int, 254)
	for k := 2; k <= 255; k++ {
		ch <- k
	}
	close(ch)

	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := range ch {
				result := findOptimal(uint8(k), *maxLen)
				mu.Lock()
				results[k-2] = result
				if result != nil {
					solved++
				}
				fmt.Fprintf(os.Stderr, "\r%d/254 searched, %d solved...", k-1, solved)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	fmt.Fprintf(os.Stderr, "\rDone: %d/254 constants solved            \n", solved)

	// Collect non-nil results
	var out []*MulResult
	for _, r := range results {
		if r != nil {
			out = append(out, r)
		}
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else {
		for _, r := range out {
			fmt.Printf("×%3d: %-80s (%d insts, %dT)\n", r.K, fmt.Sprint(r.Ops), r.Length, r.TStates)
		}
		// Print unsolved
		for i, r := range results {
			if r == nil {
				fmt.Printf("×%3d: NOT FOUND (max %d)\n", i+2, *maxLen)
			}
		}
	}
}

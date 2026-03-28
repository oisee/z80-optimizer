// enrich-regalloc: Post-process regalloc tables with operation-aware cost model
//
// Reads exhaustive_Nv.bin tables, enriches each feasible assignment with
// costs for common operation patterns using the register graph cost model.
//
// Usage: enrich-regalloc -input data/exhaustive_4v.bin -output data/enriched_4v.json
//
// Output: JSON with per-shape entries including costs for different op mixes

package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

// Physical register locations (from data/README.md)
var locNames = []string{
	"A", "B", "C", "D", "E", "H", "L",
	"BC", "DE", "HL", "IXH", "IXL", "IYH", "IYL", "mem0",
}

// Move cost between 8-bit registers (T-states)
// -1 = impossible direct move
var moveCost = [15][15]int{
	//  A   B   C   D   E   H   L  BC  DE  HL IXH IXL IYH IYL mem
	{0, 4, 4, 4, 4, 4, 4, -1, -1, -1, 8, 8, 8, 8, -1},     // A
	{4, 0, 4, 4, 4, 4, 4, -1, -1, -1, 8, 8, 8, 8, -1},     // B
	{4, 4, 0, 4, 4, 4, 4, -1, -1, -1, 8, 8, 8, 8, -1},     // C
	{4, 4, 4, 0, 4, 4, 4, -1, -1, -1, 8, 8, 8, 8, -1},     // D
	{4, 4, 4, 4, 0, 4, 4, -1, -1, -1, 8, 8, 8, 8, -1},     // E
	{4, 4, 4, 4, 4, 0, 4, -1, -1, -1, 16, 16, 16, 16, -1}, // H
	{4, 4, 4, 4, 4, 4, 0, -1, -1, -1, 16, 16, 16, 16, -1}, // L
}

// ALU cost: performing binary op when dst=loc_d, src=loc_s
// All binary ALU (ADD/SUB/AND/OR/XOR/CP) require A as accumulator
func aluBinaryCost(dstLoc, srcLoc int) int {
	cost := 0
	// Move dst to A if not already there
	if dstLoc != 0 { // not A
		cost += 4 // LD A, dst
	}
	// Perform op: ADD A, src
	switch {
	case srcLoc <= 6: // A-L
		cost += 4
	case srcLoc >= 10 && srcLoc <= 13: // IXH-IYL
		cost += 8
	default:
		cost += 7 // (HL) or immediate
	}
	// Move result back if needed
	if dstLoc != 0 {
		cost += 4 // LD dst, A
	}
	return cost
}

// Unary op cost (NEG, CPL, etc) — requires A
func aluUnaryCost(loc int) int {
	cost := 0
	if loc != 0 {
		cost += 4 + 4 // LD A,r + LD r,A
	}
	cost += 8 // NEG
	return cost
}

// INC/DEC cost — works on any register
func incDecCost(loc int) int {
	if loc <= 6 {
		return 4
	}
	if loc >= 10 && loc <= 13 { // IX/IY halves
		return 8
	}
	if loc >= 7 && loc <= 9 { // 16-bit pairs
		return 6
	}
	return 11 // (HL)
}

// Operation pattern costs for a given assignment
type OpPattern struct {
	Name string `json:"name"`
	Cost int    `json:"cost"`
}

func scoreAssignment(nVregs int, assignment []int) []OpPattern {
	patterns := []OpPattern{}

	if nVregs < 2 {
		return patterns
	}

	// Pattern: ALU_HEAVY — all pairs do binary ALU
	aluTotal := 0
	aluCount := 0
	for i := 0; i < nVregs; i++ {
		for j := i + 1; j < nVregs; j++ {
			if assignment[i] < 7 && assignment[j] < 7 {
				aluTotal += aluBinaryCost(assignment[i], assignment[j])
				aluCount++
			}
		}
	}
	if aluCount > 0 {
		patterns = append(patterns, OpPattern{"alu_avg", aluTotal / aluCount})
	}

	// Pattern: A_centric — one var in A does ALU with all others
	aIdx := -1
	for i := 0; i < nVregs; i++ {
		if assignment[i] == 0 {
			aIdx = i
			break
		}
	}
	if aIdx >= 0 {
		aCost := 0
		for i := 0; i < nVregs; i++ {
			if i != aIdx && assignment[i] < 7 {
				aCost += aluBinaryCost(0, assignment[i])
			}
		}
		patterns = append(patterns, OpPattern{"a_centric", aCost})
	}

	// Pattern: best/worst pair for binary ALU
	bestPair := 9999
	worstPair := 0
	for i := 0; i < nVregs; i++ {
		for j := i + 1; j < nVregs; j++ {
			if assignment[i] < 7 && assignment[j] < 7 {
				c1 := aluBinaryCost(assignment[i], assignment[j])
				c2 := aluBinaryCost(assignment[j], assignment[i])
				c := c1
				if c2 < c1 {
					c = c2
				}
				if c < bestPair {
					bestPair = c
				}
				if c > worstPair {
					worstPair = c
				}
			}
		}
	}
	if bestPair < 9999 {
		patterns = append(patterns, OpPattern{"best_alu_pair", bestPair})
		patterns = append(patterns, OpPattern{"worst_alu_pair", worstPair})
	}

	// Pattern: total INC/DEC cost
	incTotal := 0
	for i := 0; i < nVregs; i++ {
		incTotal += incDecCost(assignment[i])
	}
	patterns = append(patterns, OpPattern{"inc_dec_total", incTotal})

	// Has A? (important for ALU feasibility)
	hasA := false
	for _, loc := range assignment {
		if loc == 0 {
			hasA = true
			break
		}
	}
	if !hasA {
		patterns = append(patterns, OpPattern{"no_accumulator", 1})
	}

	return patterns
}

func main() {
	inputPath := flag.String("input", "", "Input .bin file (uncompressed)")
	outputPath := flag.String("output", "", "Output .json file")
	limit := flag.Int("limit", 0, "Max records to process (0=all)")
	flag.Parse()

	if *inputPath == "" || *outputPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: enrich-regalloc -input FILE.bin -output FILE.json [-limit N]\n")
		os.Exit(1)
	}

	f, err := os.Open(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open %s: %v\n", *inputPath, err)
		os.Exit(1)
	}
	defer f.Close()

	// Read header
	magic := make([]byte, 4)
	f.Read(magic)
	if string(magic) != "Z80T" {
		fmt.Fprintf(os.Stderr, "bad magic: %q\n", magic)
		os.Exit(1)
	}
	var version uint32
	binary.Read(f, binary.LittleEndian, &version)

	type Entry struct {
		Index      int         `json:"index"`
		NVregs     int         `json:"nVregs"`
		OrigCost   int         `json:"orig_cost"`
		Assignment []string    `json:"assignment"`
		Patterns   []OpPattern `json:"patterns"`
	}

	out, err := os.Create(*outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create %s: %v\n", *outputPath, err)
		os.Exit(1)
	}
	defer out.Close()

	enc := json.NewEncoder(out)
	out.WriteString("[\n")

	index := 0
	feasible := 0
	first := true

	for {
		if *limit > 0 && index >= *limit {
			break
		}

		b := make([]byte, 1)
		_, err := f.Read(b)
		if err != nil {
			break
		}

		if b[0] == 0xFF {
			// infeasible — skip
			index++
			continue
		}

		nv := int(b[0])
		var cost uint16
		binary.Read(f, binary.LittleEndian, &cost)
		assign := make([]byte, nv)
		f.Read(assign)

		assignInt := make([]int, nv)
		assignStr := make([]string, nv)
		for i, a := range assign {
			assignInt[i] = int(a)
			if int(a) < len(locNames) {
				assignStr[i] = locNames[a]
			} else {
				assignStr[i] = fmt.Sprintf("loc%d", a)
			}
		}

		patterns := scoreAssignment(nv, assignInt)

		entry := Entry{
			Index:      index,
			NVregs:     nv,
			OrigCost:   int(cost),
			Assignment: assignStr,
			Patterns:   patterns,
		}

		if !first {
			out.WriteString(",\n")
		}
		enc.Encode(entry)
		first = false
		feasible++
		index++
	}

	out.WriteString("]\n")
	fmt.Fprintf(os.Stderr, "Processed %d shapes, %d feasible\n", index, feasible)
}

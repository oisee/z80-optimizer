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

// 16-bit pair indices in location table
const (
	locBC = 7
	locDE = 8
	locHL = 9
)

// Is this location a 16-bit pair?
func is16bitPair(loc int) bool { return loc >= 7 && loc <= 9 }

// Is this location an 8-bit register (A-L)?
func is8bitReg(loc int) bool { return loc <= 6 }

// Is this location an IX/IY half?
func isIXIY(loc int) bool { return loc >= 10 && loc <= 13 }

// 16-bit ALU cost: ADD HL,rr / SBC HL,rr
// dst MUST be HL (loc 9). src can be BC(7), DE(8), HL(9), SP.
func alu16AddCost(dstLoc, srcLoc int) int {
	if dstLoc == locHL && is16bitPair(srcLoc) {
		return 11 // ADD HL,rr (natural)
	}
	if dstLoc == locHL {
		// src is not a pair — need to build it
		return 11 + 8 // LD rr,src pair + ADD HL,rr
	}
	// dst is not HL — need EX DE,HL trick or move
	if dstLoc == locDE {
		// EX DE,HL; ADD HL,rr; EX DE,HL = 4+11+4 = 19T
		return 19
	}
	// dst is BC or other — very expensive
	return 30 // need multiple moves
}

func alu16SbcCost(dstLoc, srcLoc int) int {
	if dstLoc == locHL && is16bitPair(srcLoc) {
		return 15 // SBC HL,rr (ED prefix)
	}
	if dstLoc == locDE {
		return 23 // EX DE,HL; SBC HL,rr; EX DE,HL
	}
	return 35
}

// 16-bit MUL cost via our mul16 table
// mul16 requires input in A, result in HL. ~26T average.
func mul16Cost(dstLoc, srcLoc int) int {
	cost := 0
	if srcLoc != 0 { // source not in A
		cost += 4 // LD A,src_low
	}
	cost += 26 // mul16 from table (average)
	if dstLoc != locHL {
		cost += 8 // move HL result to dst pair
	}
	return cost
}

// u8 MUL cost via our mul8 table
// mul8 requires input in A, uses various regs. ~20T average.
func mul8Cost(dstLoc, srcLoc int) int {
	cost := 0
	if dstLoc != 0 {
		cost += 4 // LD A,dst
	}
	cost += 20 // mul8 average from table
	if dstLoc != 0 {
		cost += 4 // LD dst,A
	}
	return cost
}

// u16 ADD via u8 decomposition: ADD low bytes, ADC high bytes
// Works with ANY register pairs, not just HL+rr
func alu16ViaU8Cost(dstLoc, srcLoc int) int {
	// LD A,src_lo; ADD A,dst_lo; LD dst_lo,A; LD A,src_hi; ADC A,dst_hi; LD dst_hi,A
	return 24 // 6 instructions × 4T
}

func scoreAssignment(nVregs int, assignment []int) []OpPattern {
	patterns := []OpPattern{}

	if nVregs < 2 {
		return patterns
	}

	// === u8 patterns ===

	// u8 ALU average cost across all pairs
	aluTotal := 0
	aluCount := 0
	for i := 0; i < nVregs; i++ {
		for j := i + 1; j < nVregs; j++ {
			if is8bitReg(assignment[i]) && is8bitReg(assignment[j]) {
				c1 := aluBinaryCost(assignment[i], assignment[j])
				c2 := aluBinaryCost(assignment[j], assignment[i])
				if c2 < c1 {
					c1 = c2
				}
				aluTotal += c1
				aluCount++
			}
		}
	}
	if aluCount > 0 {
		patterns = append(patterns, OpPattern{"u8_alu_avg", aluTotal / aluCount})
	}

	// u8 best/worst pair
	bestPair := 9999
	worstPair := 0
	for i := 0; i < nVregs; i++ {
		for j := i + 1; j < nVregs; j++ {
			if is8bitReg(assignment[i]) && is8bitReg(assignment[j]) {
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
		patterns = append(patterns, OpPattern{"u8_best_alu", bestPair})
		patterns = append(patterns, OpPattern{"u8_worst_alu", worstPair})
	}

	// u8 MUL cost (via mul8 table)
	if aluCount > 0 {
		patterns = append(patterns, OpPattern{"u8_mul_avg", mul8Cost(assignment[0], assignment[1])})
	}

	// === u16 patterns (treating consecutive pairs of vregs as 16-bit) ===

	// Count 16-bit pair slots
	n16pairs := 0
	pairSlots := []int{} // which vregs are in 16-bit pair slots
	for i := 0; i < nVregs; i++ {
		if is16bitPair(assignment[i]) {
			n16pairs++
			pairSlots = append(pairSlots, i)
		}
	}
	patterns = append(patterns, OpPattern{"u16_pair_count", n16pairs})

	// u16 ADD natural: needs HL as dst
	hasHL := false
	for _, loc := range assignment {
		if loc == locHL {
			hasHL = true
			break
		}
	}

	if n16pairs >= 2 {
		// Best u16 ADD cost between any two pair-assigned vregs
		best16Add := 9999
		worst16Add := 0
		for i := 0; i < len(pairSlots); i++ {
			for j := i + 1; j < len(pairSlots); j++ {
				li, lj := assignment[pairSlots[i]], assignment[pairSlots[j]]
				// Try both directions
				c1 := alu16AddCost(li, lj)
				c2 := alu16AddCost(lj, li)
				c := c1
				if c2 < c1 {
					c = c2
				}
				if c < best16Add {
					best16Add = c
				}
				if c > worst16Add {
					worst16Add = c
				}
			}
		}
		if best16Add < 9999 {
			patterns = append(patterns, OpPattern{"u16_add_natural", best16Add})
			patterns = append(patterns, OpPattern{"u16_add_worst", worst16Add})
		}
	}

	// u16 ADD via u8 decomposition (always available, always 24T)
	patterns = append(patterns, OpPattern{"u16_add_via_u8", 24})

	// u16 SBC natural
	if hasHL && n16pairs >= 2 {
		patterns = append(patterns, OpPattern{"u16_sbc_natural", 15})
	} else {
		patterns = append(patterns, OpPattern{"u16_sbc_via_u8", 24})
	}

	// === Feasibility flags ===

	hasA := false
	for _, loc := range assignment {
		if loc == 0 {
			hasA = true
			break
		}
	}
	if !hasA {
		patterns = append(patterns, OpPattern{"no_accumulator", 1}) // u8 ALU needs A
	}
	if !hasHL {
		patterns = append(patterns, OpPattern{"no_hl_pair", 1}) // u16 natural ADD needs HL
	}

	// Max u16 vars feasible (without IX/IY)
	// Only 3 pairs: BC, DE, HL
	maxU16 := 0
	usedPairs := map[int]bool{}
	for _, loc := range assignment {
		if is16bitPair(loc) && !usedPairs[loc] {
			usedPairs[loc] = true
			maxU16++
		}
	}
	patterns = append(patterns, OpPattern{"u16_slots_used", maxU16})
	patterns = append(patterns, OpPattern{"u16_slots_free", 3 - maxU16})

	// INC/DEC total cost
	incTotal := 0
	for i := 0; i < nVregs; i++ {
		incTotal += incDecCost(assignment[i])
	}
	patterns = append(patterns, OpPattern{"inc_dec_total", incTotal})

	// === Width-dependent feasibility tiers ===
	// Tier 1: all u8 — any assignment works
	// Tier 2: mixed u8/u16 — u16 vars constrained to pairs
	// Tier 3: all u16 — max 3 without IX/IY
	// Tier 4: needs u32 — requires EXX (shadow)

	// How many 8-bit regs are available for u8 vars?
	usedRegs := map[int]bool{}
	for _, loc := range assignment {
		usedRegs[loc] = true
	}
	freeU8 := 0
	for i := 0; i <= 6; i++ {
		if !usedRegs[i] {
			freeU8++
		}
	}
	patterns = append(patterns, OpPattern{"u8_regs_free", freeU8})

	// Clobber pressure: how many regs are left for temporaries?
	// mul8 needs ~2 temp regs, mul16 needs ~3
	totalUsed := len(usedRegs)
	patterns = append(patterns, OpPattern{"temp_regs_avail", 15 - totalUsed})

	// === mul8/mul16 compatibility ===
	// mul8 clobbers: {C,F,H,L} (from our data: all 254 preserve A, all DE-safe)
	// Check if any live vreg sits in a mul8 clobber slot
	mul8Clobber := map[int]bool{2: true, 5: true, 6: true} // C=2, H=5, L=6 (F implicit)
	mul8Conflict := 0
	for _, loc := range assignment {
		if mul8Clobber[loc] {
			mul8Conflict++
		}
	}
	patterns = append(patterns, OpPattern{"mul8_conflicts", mul8Conflict})
	if mul8Conflict == 0 {
		patterns = append(patterns, OpPattern{"mul8_safe", 1}) // can call mul8 without save/restore
	}

	// mul16 clobbers: {A,C,F,H,L} — only DE-safe (from our enriched data)
	mul16Clobber := map[int]bool{0: true, 2: true, 5: true, 6: true} // A,C,H,L
	mul16Conflict := 0
	for _, loc := range assignment {
		if mul16Clobber[loc] {
			mul16Conflict++
		}
	}
	patterns = append(patterns, OpPattern{"mul16_conflicts", mul16Conflict})

	// === Shadow bank (EXX) enrichment ===
	// If no accumulator in assignment, estimate cost of EXX-based ALU:
	// EXX (4T) + ALU in shadow bank (4T) + EXX back (4T) = 12T per op
	// vs moving to A and back (8T) — EXX is 4T more expensive but doesn't clobber A
	if !hasA {
		// EXX cost for one ALU op: 4T + 4T(op) + 4T = 12T
		// vs via-A cost: 4T(LD A,r) + 4T(op) + 4T(LD r,A) = 12T
		// Same cost! But EXX preserves A' (useful if A used elsewhere in shadow)
		patterns = append(patterns, OpPattern{"exx_alu_cost", 12})

		// If we had an EXX-split schedule: put ALU-heavy vars in shadow bank
		// Cost: one EXX pair (8T) amortized over N ops
		// For N>=3 ops: 8T/3 = 2.7T overhead per op (cheaper than individual via-A)
		if nVregs >= 3 {
			patterns = append(patterns, OpPattern{"exx_amortized_3ops", 8 / 3})
		}
	}

	// === CALL overhead ===
	// CALL clobbers: return in A, F destroyed. Callee may clobber BC/DE/HL.
	// Save strategies (cheapest first):
	//   1. Free reg:     LD free,r + ... + LD r,free = 8T (cheapest!)
	//   2. IX/IY half:   LD IXH,r + ... + LD r,IXH = 16T (no stack)
	//   3. EX AF,AF':    EX AF,AF' + ... + EX AF,AF' = 8T (A+F only)
	//   4. PUSH/POP:     PUSH rr + POP rr = 21T (pair, classic)

	// Count regs that need saving around CALL
	regsToSave := 0
	if hasA {
		regsToSave++ // A will be clobbered by CALL return
	}
	for _, loc := range assignment {
		if loc >= 1 && loc <= 6 { // B-L
			regsToSave++
		}
	}

	// Assign cheapest save method to each reg
	callOverhead := 0
	savesRemaining := regsToSave

	// Strategy 1: use free regs (8T per save)
	freeSaves := freeU8
	if freeSaves > savesRemaining {
		freeSaves = savesRemaining
	}
	callOverhead += freeSaves * 8
	savesRemaining -= freeSaves

	// Strategy 2: EX AF,AF' for A (8T, if A needs saving and hasn't been saved)
	if hasA && savesRemaining > 0 {
		callOverhead += 8 // EX AF,AF' before + after
		savesRemaining--
	}

	// Strategy 3: IX/IY halves (16T per save, up to 4 slots)
	ixSaves := 4 // IXH, IXL, IYH, IYL
	if ixSaves > savesRemaining {
		ixSaves = savesRemaining
	}
	callOverhead += ixSaves * 16
	savesRemaining -= ixSaves

	// Strategy 4: PUSH/POP pairs for rest (21T per pair)
	callOverhead += ((savesRemaining + 1) / 2) * 21

	patterns = append(patterns, OpPattern{"call_save_cost", callOverhead})
	patterns = append(patterns, OpPattern{"call_regs_to_save", regsToSave})
	patterns = append(patterns, OpPattern{"call_free_saves", freeSaves})

	// === DJNZ compatibility ===
	// DJNZ uses B as counter. If B is occupied, DJNZ needs save/restore (8T extra).
	bOccupied := false
	for _, loc := range assignment {
		if loc == 1 { // B
			bOccupied = true
			break
		}
	}
	if bOccupied {
		patterns = append(patterns, OpPattern{"djnz_conflict", 1})
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

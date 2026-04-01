package regalloc

// EXX zone boundary cost calculator.
//
// The EXX-zone shadow bank uses the SAME location indices as the main bank:
//   B'=loc1, C'=loc2, D'=loc3, E'=loc4, H'=loc5, L'=loc6
//   BC'=loc7, DE'=loc8, HL'=loc9
//
// KEY INSIGHT: if a variable occupies the same loc in both zones,
// the EXX instruction itself performs the transfer for free.
// Zone-invariant registers (A=0, IXH=10, IXL=11, IYH=12, IYL=13) are
// unchanged by EXX — they "bridge" both zones with zero crossing cost.
//
// BoundaryCost computes the cost at the EXX entry/exit points for
// variables that are live in both the main and shadow zones.
// Variables only live in one zone contribute zero crossing cost.
//
// Shuttle options for mismatched locs:
//   A-shuttle:  LD A,r; EXX; LD r,A         =  4+4+4 = 8T per var (uses A)
//   IX-shuttle: LD IXH,r; EXX; LD r,IXH     =  8+4+8 = 20T ... but this is wrong
//   Actually the shuttle happens before or after EXX, not around it.
//   Before EXX: copy src to bridge reg.  After EXX: copy bridge reg to dst.
//   A-bridge: LD A,src (4T pre) + EXX (shared 4T) + LD dst,A (4T post) = 8T per var
//   IXH-bridge: LD IXH,src (8T pre) + EXX + LD dst,IXH (8T post) = 16T per var
//   PUSH/POP: PUSH rr (11T) + EXX + POP rr (10T) = 21T for 16-bit pair

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

// isZoneInvariant returns true for registers unchanged by EXX.
// A, IXH, IXL, IYH, IYL survive EXX unchanged.
func isZoneInvariant(loc byte) bool {
	return loc == LocA || (loc >= LocIXH && loc <= LocIYL)
}

// CrossingCost returns the extra T-states needed to make a variable available
// in its shadow-zone assignment given its main-zone assignment, at an EXX boundary.
//
// Returns 0 when:
//   - mainLoc == shadowLoc (EXX moves swapped regs, invariant regs stay)
//   - mainLoc is zone-invariant AND shadowLoc == mainLoc
//
// width is 8 or 16 (bits).
func CrossingCost(mainLoc, shadowLoc byte, width int) int {
	if mainLoc == shadowLoc {
		return 0 // EXX itself handles swapped regs; invariant regs unchanged
	}

	// Need explicit copy. Choose cheapest bridge:
	// For 16-bit pairs the cheapest route is PUSH/POP (21T round-trip).
	if width == 16 {
		return 21 // PUSH rr (11T) + POP rr (10T), split across boundary
	}

	// 8-bit mismatch. Options:
	// A-bridge (if neither src nor dst is A): 8T (LD A,src pre + LD dst,A post)
	// IXH-bridge (if src/dst not already IX): 16T (LD IXH,src + LD dst,IXH)
	// If one side is already zone-invariant (IXH etc), just need one copy (8T).

	mainInv := isZoneInvariant(mainLoc)
	shadowInv := isZoneInvariant(shadowLoc)

	if mainInv || shadowInv {
		// One side is zone-invariant — only one LD needed across boundary.
		// E.g., main=IXH, shadow=B: after EXX, LD B,IXH (8T if IX half, else 4T).
		if mainLoc == LocA || shadowLoc == LocA {
			return 4 // LD A,src or LD dst,A
		}
		if mainLoc >= LocIXH || shadowLoc >= LocIXH {
			// Transfer involves IX half: costs 8T (DD/FD prefix)
			// But H/L ↔ IX costs 16T due to DD prefix hijack.
			srcIsHL := mainLoc == LocH || mainLoc == LocL
			dstIsHL := shadowLoc == LocH || shadowLoc == LocL
			srcIsIX := mainLoc >= LocIXH && mainLoc <= LocIYL
			dstIsIX := shadowLoc >= LocIXH && shadowLoc <= LocIYL
			if (srcIsHL && dstIsIX) || (srcIsIX && dstIsHL) {
				return 16
			}
			return 8
		}
		return 4
	}

	// Both sides are non-invariant GPRs — need a bridge across EXX.
	// Best available bridge: A (if neither side uses A) = 8T.
	// If A is occupied, use IXH = 16T.
	// We can't know here whether A is free, so return the pessimistic estimate.
	// Callers that know A is free can use OverrideCrossingCost = 8.
	if mainLoc != LocA && shadowLoc != LocA {
		return 8 // A-shuttle: LD A,src pre + LD dst,A post
	}
	// One side is A but they differ — the other side is a GPR.
	// LD A,r or LD r,A = 4T on the non-A side; the A side is the invariant reg.
	return 4
}

// BoundaryCost computes the total EXX zone boundary cost.
//
//	mainAssign:   per-vreg location in the main zone
//	shadowAssign: per-vreg location in the shadow zone (same vreg indices)
//	crossing:     indices of vregs live in both zones
//	widths:       per-vreg width in bits (8 or 16)
//
// Returns the T-state overhead added at the EXX zone boundary.
// The EXX instruction itself (4T) is included once.
func BoundaryCost(mainAssign, shadowAssign []byte, crossing []int, widths []int) int {
	cost := 4 // EXX instruction

	for _, vi := range crossing {
		if vi < 0 || vi >= len(mainAssign) || vi >= len(shadowAssign) {
			continue
		}
		cost += CrossingCost(mainAssign[vi], shadowAssign[vi], widths[vi])
	}

	return cost
}

// BoundaryCostFull computes the cost when ALL vregs cross the boundary.
// Convenience wrapper for the common case where all variables span the zone.
func BoundaryCostFull(mainAssign, shadowAssign []byte, widths []int) int {
	crossing := make([]int, len(mainAssign))
	for i := range crossing {
		crossing[i] = i
	}
	return BoundaryCost(mainAssign, shadowAssign, crossing, widths)
}

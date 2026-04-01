package regalloc

import "testing"

func TestCrossingCostInvariant(t *testing.T) {
	// Same loc → always 0T regardless of type
	for _, loc := range []byte{LocA, LocB, LocC, LocD, LocE, LocH, LocL, LocBC, LocDE, LocHL, LocIXH, LocIXL, LocIYH, LocIYL} {
		w := 8
		if loc >= LocBC && loc <= LocHL {
			w = 16
		}
		if got := CrossingCost(loc, loc, w); got != 0 {
			t.Errorf("CrossingCost(%d,%d,%d) = %d, want 0", loc, loc, w, got)
		}
	}
}

func TestCrossingCostBridge(t *testing.T) {
	cases := []struct {
		main, shadow byte
		width, want  int
		desc         string
	}{
		// 16-bit: always PUSH/POP
		{LocBC, LocDE, 16, 21, "BC→DE pair"},
		{LocHL, LocBC, 16, 21, "HL→BC pair"},
		// 8-bit: A-shuttle when A not involved
		{LocB, LocC, 8, 8, "B→C via A-shuttle"},
		{LocD, LocE, 8, 8, "D→E via A-shuttle"},
		// 8-bit: one side is A → cheaper
		{LocA, LocB, 8, 4, "A→B single copy"},
		{LocB, LocA, 8, 4, "B→A single copy"},
		// IX half involved
		{LocIXH, LocB, 8, 8, "IXH→B (IX prefix LD)"},
		{LocB, LocIXH, 8, 8, "B→IXH (IX prefix LD)"},
		// H/L ↔ IX: 16T
		{LocH, LocIXH, 8, 16, "H→IXH DD hijack"},
		{LocIXH, LocH, 8, 16, "IXH→H DD hijack"},
		{LocL, LocIXL, 8, 16, "L→IXL DD hijack"},
	}

	for _, c := range cases {
		got := CrossingCost(c.main, c.shadow, c.width)
		if got != c.want {
			t.Errorf("CrossingCost(%d,%d,%d) [%s] = %d, want %d",
				c.main, c.shadow, c.width, c.desc, got, c.want)
		}
	}
}

func TestBoundaryCostEXXOnly(t *testing.T) {
	// All vregs in IX halves (zone-invariant) → only 4T for EXX
	main := []byte{LocIXH, LocIXL}
	shadow := []byte{LocIXH, LocIXL}
	widths := []int{8, 8}
	crossing := []int{0, 1}
	if got := BoundaryCost(main, shadow, crossing, widths); got != 4 {
		t.Errorf("BoundaryCost all-IX = %d, want 4", got)
	}
}

func TestBoundaryCostTypical(t *testing.T) {
	// v0=B in main, v0=B in shadow → 0T crossing (EXX swaps it)
	// v1=A in main, v1=A in shadow → 0T (invariant)
	// Total: 4T (just EXX)
	main := []byte{LocB, LocA}
	shadow := []byte{LocB, LocA}
	widths := []int{8, 8}
	crossing := []int{0, 1}
	if got := BoundaryCost(main, shadow, crossing, widths); got != 4 {
		t.Errorf("BoundaryCost matched locs = %d, want 4", got)
	}
}

func TestBoundaryCostMismatch(t *testing.T) {
	// v0=B in main, v0=C in shadow → 8T (A-shuttle) + 4T (EXX) = 12T
	main := []byte{LocB}
	shadow := []byte{LocC}
	widths := []int{8}
	crossing := []int{0}
	if got := BoundaryCost(main, shadow, crossing, widths); got != 12 {
		t.Errorf("BoundaryCost B→C mismatch = %d, want 12", got)
	}
}

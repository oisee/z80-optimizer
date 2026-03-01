package cpu

// State represents the Z80 register state relevant to the superoptimizer.
// Expanded across implementation waves:
//
//	V1:     A, F, B, C, D, E, H, L (8 bytes)
//	Wave 2: + SP uint16 (10 bytes)
//	Wave 5: + M uint8 (11 bytes) — virtual memory byte at (HL)/(BC)/(DE)
//
// Still fits a single cache line, cheap to copy by value.
type State struct {
	A, F, B, C, D, E, H, L uint8
	SP                      uint16 // Wave 2
	M                       uint8  // Wave 5: memory byte (all indirect ops share this)
}

// Equal returns true if two states are identical.
func (s State) Equal(o State) bool {
	return s == o
}

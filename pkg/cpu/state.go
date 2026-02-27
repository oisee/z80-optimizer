package cpu

// State represents the Z80 register state relevant to the superoptimizer.
// Expanded across implementation waves:
//
//	V1:     A, F, B, C, D, E, H, L (8 bytes)
//	Wave 2: + SP uint16 (10 bytes)
//
// Still fits a single cache line, cheap to copy by value.
type State struct {
	A, F, B, C, D, E, H, L uint8
	SP                      uint16 // Wave 2
}

// Equal returns true if two states are identical.
func (s State) Equal(o State) bool {
	return s == o
}

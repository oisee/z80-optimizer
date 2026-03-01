package stoke

import (
	"github.com/oisee/z80-optimizer/pkg/cpu"
	"github.com/oisee/z80-optimizer/pkg/inst"
)

// testVectors are fixed inputs for quick equivalence checking.
// Same as search.TestVectors — duplicated here to avoid import cycle.
var testVectors = []cpu.State{
	{A: 0x00, F: 0x00, B: 0x00, C: 0x00, D: 0x00, E: 0x00, H: 0x00, L: 0x00, SP: 0x0000},
	{A: 0xFF, F: 0xFF, B: 0xFF, C: 0xFF, D: 0xFF, E: 0xFF, H: 0xFF, L: 0xFF, SP: 0xFFFF},
	{A: 0x01, F: 0x00, B: 0x02, C: 0x03, D: 0x04, E: 0x05, H: 0x06, L: 0x07, SP: 0x1234},
	{A: 0x80, F: 0x01, B: 0x40, C: 0x20, D: 0x10, E: 0x08, H: 0x04, L: 0x02, SP: 0x8000},
	{A: 0x55, F: 0x00, B: 0xAA, C: 0x55, D: 0xAA, E: 0x55, H: 0xAA, L: 0x55, SP: 0x5555},
	{A: 0xAA, F: 0x01, B: 0x55, C: 0xAA, D: 0x55, E: 0xAA, H: 0x55, L: 0xAA, SP: 0xAAAA},
	{A: 0x0F, F: 0x00, B: 0xF0, C: 0x0F, D: 0xF0, E: 0x0F, H: 0xF0, L: 0x0F, SP: 0xFFFE},
	{A: 0x7F, F: 0x01, B: 0x80, C: 0x7F, D: 0x80, E: 0x7F, H: 0x80, L: 0x7F, SP: 0x7FFF},
}

// execSeq runs a sequence of instructions on a state.
func execSeq(initial cpu.State, seq []inst.Instruction) cpu.State {
	s := initial
	for i := range seq {
		cpu.Exec(&s, seq[i].Op, seq[i].Imm)
	}
	return s
}

// Cost evaluates how far a candidate is from matching the target.
// Returns: 1000 * mismatches + byteSize(candidate) + cycleCount(candidate)/100
// When Cost returns a value with mismatches==0, the candidate matches on all
// test vectors (but still needs ExhaustiveCheck to prove full equivalence).
func Cost(target, candidate []inst.Instruction) int {
	mismatches := 0
	for i := range testVectors {
		tOut := execSeq(testVectors[i], target)
		cOut := execSeq(testVectors[i], candidate)
		if tOut != cOut {
			mismatches++
		}
	}
	return 1000*mismatches + inst.SeqByteSize(candidate) + inst.SeqTStates(candidate)/100
}

// Mismatches returns only the mismatch count on test vectors.
func Mismatches(target, candidate []inst.Instruction) int {
	mismatches := 0
	for i := range testVectors {
		tOut := execSeq(testVectors[i], target)
		cOut := execSeq(testVectors[i], candidate)
		if tOut != cOut {
			mismatches++
		}
	}
	return mismatches
}

// CostMasked evaluates cost, ignoring dead flag bits in comparisons.
func CostMasked(target, candidate []inst.Instruction, deadFlags uint8) int {
	if deadFlags == 0 {
		return Cost(target, candidate)
	}
	mismatches := MismatchesMasked(target, candidate, deadFlags)
	return 1000*mismatches + inst.SeqByteSize(candidate) + inst.SeqTStates(candidate)/100
}

// MismatchesMasked returns the mismatch count, ignoring dead flag bits.
func MismatchesMasked(target, candidate []inst.Instruction, deadFlags uint8) int {
	mismatches := 0
	for i := range testVectors {
		tOut := execSeq(testVectors[i], target)
		cOut := execSeq(testVectors[i], candidate)
		if !statesEqualMasked(tOut, cOut, deadFlags) {
			mismatches++
		}
	}
	return mismatches
}

// statesEqualMasked compares two states, ignoring flag bits set in deadFlags.
func statesEqualMasked(a, b cpu.State, deadFlags uint8) bool {
	return a.A == b.A &&
		(a.F &^ deadFlags) == (b.F &^ deadFlags) &&
		a.B == b.B && a.C == b.C &&
		a.D == b.D && a.E == b.E &&
		a.H == b.H && a.L == b.L &&
		a.SP == b.SP
}

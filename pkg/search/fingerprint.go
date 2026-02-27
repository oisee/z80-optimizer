package search

import (
	"github.com/oisee/z80-optimizer/pkg/inst"
)

// FingerprintMap is a hash map from fingerprints to target sequences.
// Used for batch matching: compute fingerprints for all targets, then
// scan candidates and look up matches in O(1).
type FingerprintMap struct {
	m map[[FingerprintLen]byte][]TargetEntry
}

// TargetEntry holds a target sequence and its metadata.
type TargetEntry struct {
	Seq      []inst.Instruction
	Bytes    int
	TStates  int
}

// NewFingerprintMap creates a new map with the given capacity hint.
func NewFingerprintMap(cap int) *FingerprintMap {
	return &FingerprintMap{m: make(map[[FingerprintLen]byte][]TargetEntry, cap)}
}

// Add registers a target sequence with its fingerprint.
func (fm *FingerprintMap) Add(seq []inst.Instruction) {
	fp := Fingerprint(seq)
	// Copy the sequence since it may be reused
	seqCopy := make([]inst.Instruction, len(seq))
	copy(seqCopy, seq)
	entry := TargetEntry{
		Seq:     seqCopy,
		Bytes:   inst.SeqByteSize(seqCopy),
		TStates: inst.SeqTStates(seqCopy),
	}
	fm.m[fp] = append(fm.m[fp], entry)
}

// Lookup returns target entries matching the given fingerprint.
func (fm *FingerprintMap) Lookup(fp [FingerprintLen]byte) []TargetEntry {
	return fm.m[fp]
}

// Len returns the number of distinct fingerprints.
func (fm *FingerprintMap) Len() int {
	return len(fm.m)
}

// Entries returns the total number of target entries.
func (fm *FingerprintMap) Entries() int {
	n := 0
	for _, v := range fm.m {
		n += len(v)
	}
	return n
}

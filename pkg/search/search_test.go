package search

import (
	"testing"

	"github.com/oisee/z80-optimizer/pkg/inst"
)

// TestKnownOptimizations verifies the superoptimizer finds known Z80 optimizations.
// Note: the superoptimizer requires FULL state equivalence (including flags).
// "LD A,0 → XOR A" is NOT found because XOR A modifies flags while LD A,0 doesn't.
// We test actually-equivalent replacements here.
func TestKnownOptimizations(t *testing.T) {
	tests := []struct {
		name      string
		target    []inst.Instruction
		expectOpt bool
		bytesSave int
	}{
		{
			// AND 0xFF: A unchanged, F = FlagH | Sz53pTable[A]
			// AND A:    A unchanged, F = FlagH | Sz53pTable[A]
			// These are fully equivalent! AND 0xFF is 2 bytes, AND A is 1 byte.
			name:      "AND 0xFF -> AND A",
			target:    []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}},
			expectOpt: true,
			bytesSave: 1,
		},
		{
			// OR 0x00: A unchanged, F = Sz53pTable[A]
			// OR A:    A unchanged, F = Sz53pTable[A]
			name:      "OR 0x00 -> OR A",
			target:    []inst.Instruction{{Op: inst.OR_N, Imm: 0x00}},
			expectOpt: true,
			bytesSave: 1,
		},
		{
			// LD A, 0xFF has no 1-byte equivalent (flags differ from CPL etc.)
			name:      "LD A, 0xFF has no 1-byte equivalent",
			target:    []inst.Instruction{{Op: inst.LD_A_N, Imm: 0xFF}},
			expectOpt: false,
		},
		{
			// LD A, 0 does NOT equal XOR A because flags differ
			name:      "LD A, 0 != XOR A (flags differ)",
			target:    []inst.Instruction{{Op: inst.LD_A_N, Imm: 0x00}},
			expectOpt: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rule := SearchSingle(tc.target, 1, false)
			if !tc.expectOpt {
				if rule != nil {
					t.Errorf("expected no optimization, got: %s (-%d bytes)",
						testDisasmSeq(rule.Replacement), rule.BytesSaved)
				}
				return
			}
			if rule == nil {
				t.Fatal("expected optimization, got nil")
			}
			if rule.BytesSaved != tc.bytesSave {
				t.Errorf("bytes saved: got %d want %d", rule.BytesSaved, tc.bytesSave)
			}
			t.Logf("  %s -> %s (-%d bytes, -%d cycles)",
				testDisasmSeq(tc.target), testDisasmSeq(rule.Replacement),
				rule.BytesSaved, rule.CyclesSaved)
		})
	}
}

// TestQuickCheck verifies the quick check catches equivalences.
func TestQuickCheck(t *testing.T) {
	// XOR A is equivalent to LD A, 0 on outputs (but flags differ!)
	// Actually XOR A sets P flag, clears all others.
	// LD A, 0 doesn't change flags at all.
	// So they are NOT equivalent!

	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	cand := []inst.Instruction{{Op: inst.XOR_A}}

	// They should NOT pass QuickCheck because flags differ
	if QuickCheck(target, cand) {
		// This actually depends on whether the test vectors catch the flag difference
		// Let's verify explicitly
		t.Log("QuickCheck passed — verifying with exhaustive check")
		if ExhaustiveCheck(target, cand) {
			t.Error("LD A,0 and XOR A should NOT be equivalent (flags differ)")
		}
	}
}

// TestExhaustiveCheckEquivalent verifies truly equivalent sequences pass.
func TestExhaustiveCheckEquivalent(t *testing.T) {
	// ADD A, 1 and INC A should produce different flags (INC doesn't affect C)
	target := []inst.Instruction{{Op: inst.ADD_A_N, Imm: 1}}
	cand := []inst.Instruction{{Op: inst.INC_A}}
	if ExhaustiveCheck(target, cand) {
		t.Error("ADD A,1 and INC A should NOT be equivalent (carry flag differs)")
	}

	// Two identical sequences must be equivalent
	seq := []inst.Instruction{{Op: inst.ADD_A_B}}
	if !ExhaustiveCheck(seq, seq) {
		t.Error("identical sequences should be equivalent")
	}
}

// TestExhaustiveCheckNonEquivalent verifies different sequences are rejected.
func TestExhaustiveCheckNonEquivalent(t *testing.T) {
	target := []inst.Instruction{{Op: inst.ADD_A_B}}
	cand := []inst.Instruction{{Op: inst.SUB_B}}
	if ExhaustiveCheck(target, cand) {
		t.Error("ADD A,B and SUB B should NOT be equivalent")
	}
}

// TestPruner verifies pruning rules.
func TestPruner(t *testing.T) {
	// NOP should be pruned
	if !ShouldPrune([]inst.Instruction{{Op: inst.NOP}}) {
		t.Error("NOP should be pruned")
	}

	// Self-load should be pruned
	if !ShouldPrune([]inst.Instruction{{Op: inst.LD_A_A}}) {
		t.Error("LD A,A should be pruned")
	}

	// ADD A,B should NOT be pruned
	if ShouldPrune([]inst.Instruction{{Op: inst.ADD_A_B}}) {
		t.Error("ADD A,B should not be pruned")
	}

	// Dead write: two consecutive LD A,n
	if !ShouldPrune([]inst.Instruction{
		{Op: inst.LD_A_N, Imm: 5},
		{Op: inst.LD_A_N, Imm: 10},
	}) {
		t.Error("consecutive LD A,n should be pruned (dead write)")
	}
}

// TestFingerprint verifies fingerprint consistency.
func TestFingerprint(t *testing.T) {
	seq1 := []inst.Instruction{{Op: inst.ADD_A_B}}
	seq2 := []inst.Instruction{{Op: inst.ADD_A_B}}
	seq3 := []inst.Instruction{{Op: inst.SUB_B}}

	fp1 := Fingerprint(seq1)
	fp2 := Fingerprint(seq2)
	fp3 := Fingerprint(seq3)

	if fp1 != fp2 {
		t.Error("identical sequences should have same fingerprint")
	}
	if fp1 == fp3 {
		t.Error("different sequences should (likely) have different fingerprints")
	}
}

// TestFingerprintMap verifies the fingerprint hash map.
func TestFingerprintMap(t *testing.T) {
	fm := NewFingerprintMap(100)
	seq := []inst.Instruction{{Op: inst.ADD_A_B}}
	fm.Add(seq)

	fp := Fingerprint(seq)
	entries := fm.Lookup(fp)
	if len(entries) != 1 {
		t.Errorf("expected 1 entry, got %d", len(entries))
	}
}

// TestEnumerator verifies sequence enumeration counts.
func TestEnumerator(t *testing.T) {
	count := 0
	EnumerateSequences(1, func(seq []inst.Instruction) bool {
		count++
		return true
	})

	expected := InstructionCount()
	if count != expected {
		t.Errorf("length-1 enumeration: got %d sequences, want %d", count, expected)
	}
	t.Logf("Instructions per position: %d", expected)
}

// TestEnumeratorEarlyStop verifies enumeration stops when fn returns false.
func TestEnumeratorEarlyStop(t *testing.T) {
	count := 0
	EnumerateSequences(1, func(seq []inst.Instruction) bool {
		count++
		return count < 10
	})
	if count != 10 {
		t.Errorf("early stop: got %d iterations, want 10", count)
	}
}

// TestSearchTargeted runs a targeted search on specific length-2 sequences
// that are known to have shorter replacements.
func TestSearchTargeted(t *testing.T) {
	tests := []struct {
		name      string
		target    []inst.Instruction
		expectOpt bool
	}{
		{
			// SUB A sets A=0 and flags. Then LD A, 0 is redundant (A already 0, no flag change).
			// So SUB A : LD A, 0 → SUB A (saves 2 bytes)
			name: "SUB A : LD A, 0 -> SUB A",
			target: []inst.Instruction{
				{Op: inst.SUB_A},
				{Op: inst.LD_A_N, Imm: 0},
			},
			expectOpt: true,
		},
		{
			// AND A : AND A → AND A (idempotent)
			name: "AND A : AND A -> AND A",
			target: []inst.Instruction{
				{Op: inst.AND_A},
				{Op: inst.AND_A},
			},
			expectOpt: true,
		},
		{
			// OR A : OR A → OR A (idempotent)
			name: "OR A : OR A -> OR A",
			target: []inst.Instruction{
				{Op: inst.OR_A},
				{Op: inst.OR_A},
			},
			expectOpt: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rule := SearchSingle(tc.target, len(tc.target)-1, false)
			if tc.expectOpt {
				if rule == nil {
					t.Fatal("expected optimization, got nil")
				}
				t.Logf("  %s -> %s (-%d bytes, -%d cycles)",
					testDisasmSeq(tc.target), testDisasmSeq(rule.Replacement),
					rule.BytesSaved, rule.CyclesSaved)
			} else {
				if rule != nil {
					t.Errorf("expected no optimization, got: -%d bytes", rule.BytesSaved)
				}
			}
		})
	}
}

// TestSearchImmediateSubset searches all immediate-instruction targets
// to find single-instruction replacements. This is a fast subset of the full search.
func TestSearchImmediateSubset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	found := 0
	for _, op := range inst.ImmediateOps() {
		for imm := 0; imm < 256; imm++ {
			target := []inst.Instruction{{Op: op, Imm: uint16(imm)}}
			rule := SearchSingle(target, 1, false)
			if rule != nil {
				found++
				if found <= 20 {
					t.Logf("  %s -> %s (-%d bytes, -%d cycles)",
						testDisasmSeq(target), testDisasmSeq(rule.Replacement),
						rule.BytesSaved, rule.CyclesSaved)
				}
			}
		}
	}
	t.Logf("Found %d optimizations from immediate instructions", found)
	if found == 0 {
		t.Error("expected to find at least some optimizations")
	}
}

func testDisasmSeq(seq []inst.Instruction) string {
	s := ""
	for i, instr := range seq {
		if i > 0 {
			s += " : "
		}
		s += inst.Disassemble(instr)
	}
	return s
}

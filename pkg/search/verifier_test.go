package search

import (
	"testing"

	"github.com/oisee/z80-optimizer/pkg/cpu"
	"github.com/oisee/z80-optimizer/pkg/inst"
)

func cpu_state(a, f uint8) cpu.State {
	return cpu.State{A: a, F: f}
}

func TestQuickCheckMasked_LDA0_vs_XORA(t *testing.T) {
	// LD A, 0 vs XOR A: should fail without mask (flags differ), pass with DeadAll
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	if QuickCheck(target, candidate) {
		t.Fatal("QuickCheck should fail: LD A, 0 and XOR A have different flags")
	}

	if !QuickCheckMasked(target, candidate, DeadAll) {
		t.Fatal("QuickCheckMasked(DeadAll) should pass: registers are identical")
	}
}

func TestQuickCheckMasked_DeadNone_SameAsQuickCheck(t *testing.T) {
	target := []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}}
	candidate := []inst.Instruction{{Op: inst.AND_A}}

	full := QuickCheck(target, candidate)
	masked := QuickCheckMasked(target, candidate, DeadNone)
	if full != masked {
		t.Fatalf("DeadNone should match QuickCheck: full=%v masked=%v", full, masked)
	}
}

func TestExhaustiveCheckMasked_LDA0_vs_XORA(t *testing.T) {
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	if ExhaustiveCheck(target, candidate) {
		t.Fatal("ExhaustiveCheck should fail: flags differ")
	}

	if !ExhaustiveCheckMasked(target, candidate, DeadAll) {
		t.Fatal("ExhaustiveCheckMasked(DeadAll) should pass: registers match for all inputs")
	}
}

func TestExhaustiveCheckMasked_DeadNone_SameAsExhaustiveCheck(t *testing.T) {
	target := []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}}
	candidate := []inst.Instruction{{Op: inst.AND_A}}

	full := ExhaustiveCheck(target, candidate)
	masked := ExhaustiveCheckMasked(target, candidate, DeadNone)
	if full != masked {
		t.Fatalf("DeadNone should match ExhaustiveCheck: full=%v masked=%v", full, masked)
	}
}

func TestExhaustiveCheckMasked_UndocFlags(t *testing.T) {
	// Test that undocumented flag mask works for sequences that only differ in bits 3,5
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	// XOR A sets flags S=0,Z=1,H=0,P/V=1,N=0,C=0 — differs from LD A,0 which preserves flags
	// With DeadUndoc (0x28 = bits 3,5), they still differ on documented flags
	// So this should fail with just undoc mask
	if QuickCheckMasked(target, candidate, DeadUndoc) {
		t.Fatal("LD A,0 vs XOR A should still fail with only undoc flags masked")
	}
}

func TestFlagDiff_LDA0_vs_XORA(t *testing.T) {
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	diff := FlagDiff(target, candidate)
	if diff == 0 {
		t.Fatal("FlagDiff should be nonzero: XOR A modifies flags that LD A,0 does not")
	}

	// XOR A always produces: S=0, Z=1, H=0, P/V=1(parity of 0), N=0, C=0
	// LD A, 0 preserves all flags from input
	// So flag bits that differ depend on the input flag values
	t.Logf("FlagDiff = 0x%02X", diff)
}

func TestFlagDiff_IdenticalSequences(t *testing.T) {
	seq := []inst.Instruction{{Op: inst.AND_A}}
	diff := FlagDiff(seq, seq)
	if diff != 0 {
		t.Fatalf("FlagDiff of identical sequences should be 0, got 0x%02X", diff)
	}
}

func TestFlagDiff_RegisterDifference(t *testing.T) {
	// Sequences that differ in registers (not just flags) should return 0
	target := []inst.Instruction{{Op: inst.LD_A_B}}
	candidate := []inst.Instruction{{Op: inst.LD_A_C}}

	diff := FlagDiff(target, candidate)
	// When B != C (which happens in many test vectors), A will differ
	// FlagDiff should return 0 since registers differ
	if diff != 0 {
		t.Fatalf("FlagDiff should return 0 when registers differ, got 0x%02X", diff)
	}
}

func TestStatesEqualMasked(t *testing.T) {
	tests := []struct {
		name      string
		aF, bF    uint8
		deadFlags FlagMask
		want      bool
	}{
		{"same flags", 0xFF, 0xFF, DeadNone, true},
		{"diff flags no mask", 0xFF, 0x00, DeadNone, false},
		{"diff flags all dead", 0xFF, 0x00, DeadAll, true},
		{"diff undoc bits only", 0x28, 0x00, DeadUndoc, true},
		{"diff documented bits", 0x41, 0x00, DeadUndoc, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := cpu_state(0, tt.aF)
			b := cpu_state(0, tt.bF)
			got := statesEqualMasked(a, b, tt.deadFlags)
			if got != tt.want {
				t.Fatalf("statesEqualMasked(F=%02X, F=%02X, dead=%02X) = %v, want %v",
					tt.aF, tt.bF, tt.deadFlags, got, tt.want)
			}
		})
	}
}

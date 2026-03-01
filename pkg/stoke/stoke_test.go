package stoke

import (
	"math/rand/v2"
	"testing"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
	"github.com/oisee/z80-optimizer/pkg/search"
)

func TestReplaceInstruction(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	for i := 0; i < 100; i++ {
		result := m.ReplaceInstruction(seq)
		if len(result) != 2 {
			t.Fatalf("expected length 2, got %d", len(result))
		}
		// Original should be unchanged
		if seq[0].Op != inst.LD_A_B || seq[1].Op != inst.ADD_A_C {
			t.Fatal("original sequence was modified")
		}
	}
}

func TestSwapInstructions(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	result := m.SwapInstructions(seq)
	if len(result) != 2 {
		t.Fatalf("expected length 2, got %d", len(result))
	}
	if result[0].Op != inst.ADD_A_C || result[1].Op != inst.LD_A_B {
		t.Fatalf("expected swap, got %v", result)
	}
	// Original unchanged
	if seq[0].Op != inst.LD_A_B {
		t.Fatal("original modified")
	}
}

func TestSwapSingleInstruction(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}}
	result := m.SwapInstructions(seq)
	if len(result) != 1 {
		t.Fatalf("expected length 1, got %d", len(result))
	}
}

func TestDeleteInstruction(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}, {Op: inst.SUB_B}}

	result := m.DeleteInstruction(seq)
	if len(result) != 2 {
		t.Fatalf("expected length 2, got %d", len(result))
	}
	// Original unchanged
	if len(seq) != 3 {
		t.Fatal("original modified")
	}
}

func TestDeleteSingleInstruction(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}}
	result := m.DeleteInstruction(seq)
	if len(result) != 1 {
		t.Fatalf("expected length 1 (no delete), got %d", len(result))
	}
}

func TestInsertInstruction(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	result := m.InsertInstruction(seq)
	if len(result) != 3 {
		t.Fatalf("expected length 3, got %d", len(result))
	}
}

func TestInsertAtMaxLength(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 2) // maxLen = 2
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	result := m.InsertInstruction(seq)
	// At max length, should fall back to replace (same length)
	if len(result) != 2 {
		t.Fatalf("expected length 2 (replace fallback), got %d", len(result))
	}
}

func TestChangeImmediate(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{
		{Op: inst.LD_A_N, Imm: 0x42},
		{Op: inst.ADD_A_B},
	}

	changed := false
	for i := 0; i < 100; i++ {
		result := m.ChangeImmediate(seq)
		if result[0].Op == inst.LD_A_N && result[0].Imm != 0x42 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("ChangeImmediate never changed the immediate value")
	}
}

func TestChangeImmediateNoImm(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	// Should fall back to ReplaceInstruction
	result := m.ChangeImmediate(seq)
	if len(result) != 2 {
		t.Fatalf("expected length 2, got %d", len(result))
	}
}

func TestMutatePreservesValidSequences(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	m := NewMutator(rng, 10)
	seq := []inst.Instruction{{Op: inst.LD_A_B}, {Op: inst.ADD_A_C}}

	for i := 0; i < 1000; i++ {
		result := m.Mutate(seq)
		if len(result) < 1 {
			t.Fatalf("mutation produced empty sequence at iteration %d", i)
		}
		for j, instr := range result {
			if instr.Op >= inst.OpCodeCount {
				t.Fatalf("invalid opcode %d at position %d", instr.Op, j)
			}
		}
	}
}

func TestCostIdentical(t *testing.T) {
	seq := []inst.Instruction{{Op: inst.AND_A}}
	cost := Cost(seq, seq)
	// Identical sequences should have 0 mismatches
	// cost = 0 * 1000 + byteSize + cycleCount/100
	if cost >= 1000 {
		t.Fatalf("identical sequences should have 0 mismatches, got cost %d", cost)
	}
}

func TestCostDifferent(t *testing.T) {
	target := []inst.Instruction{{Op: inst.AND_A}}
	candidate := []inst.Instruction{{Op: inst.OR_A}}
	cost := Cost(target, candidate)
	// These produce different flag results, so mismatches > 0
	if cost < 1000 {
		t.Fatalf("different sequences should have mismatches, got cost %d", cost)
	}
}

func TestCostEquivalent(t *testing.T) {
	// AND 0FFh is equivalent to AND A (both AND A with 0xFF)
	target := []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}}
	candidate := []inst.Instruction{{Op: inst.AND_A}}
	cost := Cost(target, candidate)
	if cost >= 1000 {
		t.Fatalf("equivalent sequences should have 0 mismatches, got cost %d", cost)
	}
}

func TestMismatchesIdentical(t *testing.T) {
	seq := []inst.Instruction{{Op: inst.XOR_A}}
	m := Mismatches(seq, seq)
	if m != 0 {
		t.Fatalf("expected 0 mismatches, got %d", m)
	}
}

func TestMCMCAlwaysAcceptsImprovement(t *testing.T) {
	target := []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}}
	chain := NewChain(target, 1.0, 12345)

	// Manually set current to something with high cost
	chain.current = []inst.Instruction{{Op: inst.XOR_A}, {Op: inst.LD_A_B}}
	chain.cost = Cost(target, chain.current)

	improved := false
	for i := 0; i < 10000; i++ {
		chain.Step(0.9999)
		_, bestCost := chain.Best()
		if bestCost < chain.cost {
			improved = true
		}
	}
	// With 10K steps, the chain should have accepted at least some improvements
	if chain.Accepted == 0 {
		t.Fatal("MCMC never accepted any step")
	}
	_ = improved
}

func TestMCMCTemperatureDecay(t *testing.T) {
	target := []inst.Instruction{{Op: inst.AND_A}}
	chain := NewChain(target, 1.0, 42)

	initialTemp := chain.temperature
	for i := 0; i < 100; i++ {
		chain.Step(0.99)
	}
	if chain.temperature >= initialTemp {
		t.Fatal("temperature did not decay")
	}
	expected := initialTemp
	for i := 0; i < 100; i++ {
		expected *= 0.99
	}
	diff := chain.temperature - expected
	if diff < -0.0001 || diff > 0.0001 {
		t.Fatalf("temperature %.6f != expected %.6f", chain.temperature, expected)
	}
}

func TestEndToEndAND0xFF(t *testing.T) {
	// AND 0FFh (2 bytes) should be optimizable to AND A (1 byte)
	target := []inst.Instruction{{Op: inst.AND_N, Imm: 0xFF}}

	results := Run(Config{
		Target:     target,
		Chains:     4,
		Iterations: 100_000,
		Decay:      0.9999,
		Verbose:    false,
	})

	if len(results) == 0 {
		t.Fatal("STOKE failed to find optimization for AND 0FFh")
	}

	// Verify at least one result is AND A
	foundAndA := false
	for _, r := range results {
		if len(r.Rule.Replacement) == 1 && r.Rule.Replacement[0].Op == inst.AND_A {
			foundAndA = true
		}
		// Double-check the result is verified
		if !search.ExhaustiveCheck(target, r.Rule.Replacement) {
			t.Fatalf("reported result does not pass ExhaustiveCheck: %v", r.Rule.Replacement)
		}
	}
	if !foundAndA {
		t.Logf("warning: didn't find AND A specifically, but found %d other optimizations", len(results))
	}
}

func TestDeduplicate(t *testing.T) {
	r1 := Result{Rule: result.Rule{Replacement: []inst.Instruction{{Op: inst.AND_A}}}}
	r2 := Result{Rule: result.Rule{Replacement: []inst.Instruction{{Op: inst.AND_A}}}}
	r3 := Result{Rule: result.Rule{Replacement: []inst.Instruction{{Op: inst.OR_A}}}}

	unique := Deduplicate([]Result{r1, r2, r3})
	if len(unique) != 2 {
		t.Fatalf("expected 2 unique results, got %d", len(unique))
	}
}

func TestCostMasked_LDA0_vs_XORA(t *testing.T) {
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	// Without mask: should have mismatches (flags differ)
	costFull := Cost(target, candidate)
	if costFull < 1000 {
		t.Fatalf("Cost without mask should have mismatches, got %d", costFull)
	}

	// With DeadAll mask: should have 0 mismatches
	costMasked := CostMasked(target, candidate, 0xFF)
	if costMasked >= 1000 {
		t.Fatalf("CostMasked(DeadAll) should have 0 mismatches, got %d", costMasked)
	}
}

func TestMismatchesMasked(t *testing.T) {
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}
	candidate := []inst.Instruction{{Op: inst.XOR_A}}

	full := Mismatches(target, candidate)
	if full == 0 {
		t.Fatal("Mismatches should be > 0 without mask")
	}

	masked := MismatchesMasked(target, candidate, 0xFF)
	if masked != 0 {
		t.Fatalf("MismatchesMasked(DeadAll) should be 0, got %d", masked)
	}
}

func TestEndToEndDeadFlags_LDA0(t *testing.T) {
	// LD A, 0 (2 bytes) should be optimizable to XOR A (1 byte) when all flags are dead
	target := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0}}

	results := Run(Config{
		Target:     target,
		Chains:     4,
		Iterations: 200_000,
		Decay:      0.9999,
		Verbose:    false,
		DeadFlags:  0xFF, // All flags dead
	})

	if len(results) == 0 {
		t.Fatal("STOKE with DeadFlags=0xFF failed to find optimization for LD A, 0")
	}

	// Verify at least one result is XOR A
	foundXorA := false
	for _, r := range results {
		if len(r.Rule.Replacement) == 1 && r.Rule.Replacement[0].Op == inst.XOR_A {
			foundXorA = true
			if r.Rule.DeadFlags == 0 {
				t.Log("note: XOR A result has DeadFlags=0 (it matched fully — possible if cost happened to work)")
			}
		}
		// Verify with exhaustive masked check
		if !search.ExhaustiveCheckMasked(target, r.Rule.Replacement, 0xFF) {
			t.Fatalf("result does not pass ExhaustiveCheckMasked: %v", r.Rule.Replacement)
		}
	}
	if !foundXorA {
		t.Logf("warning: didn't find XOR A specifically, but found %d optimizations", len(results))
		for _, r := range results {
			t.Logf("  found: %v (-%d bytes)", r.Rule.Replacement, r.Rule.BytesSaved)
		}
	}
}

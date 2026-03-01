package gpu

import (
	"os/exec"
	"testing"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/search"
)

func requireCUDA(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath(CUDABinaryPath); err != nil {
		t.Skipf("CUDA binary not found at %s (run: nvcc -O2 -o cuda/z80qc cuda/z80_quickcheck.cu)", CUDABinaryPath)
	}
}

func TestCUDAProcess_BasicQuickCheck(t *testing.T) {
	requireCUDA(t)

	// Use a small set of candidates for testing
	candidates := []inst.Instruction{
		{Op: inst.XOR_A},                  // 0: XOR A — zeros A, sets Z+P
		{Op: inst.LD_A_N, Imm: 0},        // 1: LD A,0 — zeros A, no flag change
		{Op: inst.AND_A},                  // 2: AND A — H flag set
		{Op: inst.OR_A},                   // 3: OR A — no flag H
		{Op: inst.ADD_A_B},               // 4: ADD A,B
		{Op: inst.SUB_B},                 // 5: SUB B
		{Op: inst.NOP},                    // 6: NOP
		{Op: inst.LD_A_N, Imm: 0x42},     // 7: LD A,0x42
	}

	cuda, err := NewCUDAProcess(candidates, 1)
	if err != nil {
		t.Fatalf("NewCUDAProcess: %v", err)
	}
	defer cuda.Close()

	// Test 1: XOR A should match its own fingerprint
	targetSeq := []inst.Instruction{{Op: inst.XOR_A}}
	fp := search.Fingerprint(targetSeq)

	hits, err := cuda.QuickCheckGPU(fp, 0)
	if err != nil {
		t.Fatalf("QuickCheckGPU: %v", err)
	}

	// XOR A (index 0) should be the only match
	found := false
	for _, idx := range hits {
		if idx == 0 {
			found = true
		}
	}
	if !found {
		t.Errorf("XOR A (index 0) should match its own fingerprint, got hits: %v", hits)
	}

	// Test 2: NOP should match its own fingerprint
	nopSeq := []inst.Instruction{{Op: inst.NOP}}
	nopFP := search.Fingerprint(nopSeq)

	nopHits, err := cuda.QuickCheckGPU(nopFP, 0)
	if err != nil {
		t.Fatalf("QuickCheckGPU (NOP): %v", err)
	}

	foundNop := false
	for _, idx := range nopHits {
		if idx == 6 {
			foundNop = true
		}
	}
	if !foundNop {
		t.Errorf("NOP (index 6) should match its own fingerprint, got hits: %v", nopHits)
	}

	// Test 3: LD A,0x42 should match its own fingerprint
	ld42Seq := []inst.Instruction{{Op: inst.LD_A_N, Imm: 0x42}}
	ld42FP := search.Fingerprint(ld42Seq)

	ld42Hits, err := cuda.QuickCheckGPU(ld42FP, 0)
	if err != nil {
		t.Fatalf("QuickCheckGPU (LD A,0x42): %v", err)
	}

	foundLD42 := false
	for _, idx := range ld42Hits {
		if idx == 7 {
			foundLD42 = true
		}
	}
	if !foundLD42 {
		t.Errorf("LD A,0x42 (index 7) should match its own fingerprint, got hits: %v", ld42Hits)
	}
}

func TestCUDAProcess_AllOpcodes(t *testing.T) {
	requireCUDA(t)

	// Test all single opcodes — each should match its own fingerprint
	candidates := search.EnumerateFirstOp()
	t.Logf("Testing %d candidates", len(candidates))

	cuda, err := NewCUDAProcess(candidates, 1)
	if err != nil {
		t.Fatalf("NewCUDAProcess: %v", err)
	}
	defer cuda.Close()

	// Test a sample of opcodes
	testOps := []struct {
		name string
		idx  int
	}{
		{"XOR A", int(inst.XOR_A)},
		{"NOP", int(inst.NOP)},
		{"LD A,B", int(inst.LD_A_B)},
		{"INC A", int(inst.INC_A)},
		{"DAA", int(inst.DAA)},
		{"BIT 0,A", int(inst.BIT_0_A)},
		{"SET 7,L", int(inst.SET_7_L)},
		{"ADD HL,BC", int(inst.ADD_HL_BC)},
	}

	for _, tc := range testOps {
		// Find this opcode's index in the candidates list
		candIdx := -1
		for i, c := range candidates {
			if c.Op == inst.OpCode(tc.idx) && c.Imm == 0 {
				candIdx = i
				break
			}
		}
		if candIdx < 0 {
			t.Logf("Skipping %s (not in candidate list)", tc.name)
			continue
		}

		fp := search.Fingerprint([]inst.Instruction{candidates[candIdx]})
		hits, err := cuda.QuickCheckGPU(fp, 0)
		if err != nil {
			t.Fatalf("QuickCheckGPU (%s): %v", tc.name, err)
		}

		found := false
		for _, idx := range hits {
			if idx == uint32(candIdx) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%s (candidate index %d) should match its own fingerprint, got hits: %v", tc.name, candIdx, hits)
		}
	}
}

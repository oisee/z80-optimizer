// +build ignore

// verify_test.go generates test data for CUDA Z80 QuickCheck verification.
// Run: go run verify_test.go | ./z80qc (reads stdin, writes stdout)
// Or:  go run verify_test.go --gen-fingerprints > fp_data.bin
//
// This program computes fingerprints for a set of instruction sequences
// using the Go CPU executor, then feeds them to the CUDA kernel to verify
// bit-exact agreement.
package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/oisee/z80-optimizer/pkg/cpu"
	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/search"
)

// Pack an instruction as uint32: low 16 bits = opcode, high 16 bits = imm
func packInstr(op inst.OpCode, imm uint16) uint32 {
	return uint32(op) | (uint32(imm) << 16)
}

func main() {
	// Generate test cases: single-instruction sequences
	type testCase struct {
		name string
		seq  []inst.Instruction
	}

	tests := []testCase{
		// Basic loads
		{"LD A,B", []inst.Instruction{{Op: inst.LD_A_B}}},
		{"LD B,A", []inst.Instruction{{Op: inst.LD_B_A}}},
		{"LD A,0x42", []inst.Instruction{{Op: inst.LD_A_N, Imm: 0x42}}},
		{"LD H,0xFF", []inst.Instruction{{Op: inst.LD_H_N, Imm: 0xFF}}},
		// ALU
		{"ADD A,B", []inst.Instruction{{Op: inst.ADD_A_B}}},
		{"ADD A,0x01", []inst.Instruction{{Op: inst.ADD_A_N, Imm: 0x01}}},
		{"ADC A,A", []inst.Instruction{{Op: inst.ADC_A_A}}},
		{"SUB B", []inst.Instruction{{Op: inst.SUB_B}}},
		{"SBC A,C", []inst.Instruction{{Op: inst.SBC_A_C}}},
		{"AND A", []inst.Instruction{{Op: inst.AND_A}}},
		{"XOR A", []inst.Instruction{{Op: inst.XOR_A}}},
		{"OR B", []inst.Instruction{{Op: inst.OR_B}}},
		{"CP D", []inst.Instruction{{Op: inst.CP_D}}},
		// INC/DEC
		{"INC A", []inst.Instruction{{Op: inst.INC_A}}},
		{"DEC A", []inst.Instruction{{Op: inst.DEC_A}}},
		{"INC B", []inst.Instruction{{Op: inst.INC_B}}},
		{"DEC L", []inst.Instruction{{Op: inst.DEC_L}}},
		// Rotates
		{"RLCA", []inst.Instruction{{Op: inst.RLCA}}},
		{"RRCA", []inst.Instruction{{Op: inst.RRCA}}},
		{"RLA", []inst.Instruction{{Op: inst.RLA}}},
		{"RRA", []inst.Instruction{{Op: inst.RRA}}},
		// CB rotates
		{"RLC A", []inst.Instruction{{Op: inst.RLC_A}}},
		{"RRC B", []inst.Instruction{{Op: inst.RRC_B}}},
		{"RL C", []inst.Instruction{{Op: inst.RL_C}}},
		{"RR D", []inst.Instruction{{Op: inst.RR_D}}},
		{"SLA E", []inst.Instruction{{Op: inst.SLA_E}}},
		{"SRA H", []inst.Instruction{{Op: inst.SRA_H}}},
		{"SRL L", []inst.Instruction{{Op: inst.SRL_L}}},
		{"SLL A", []inst.Instruction{{Op: inst.SLL_A}}},
		// Special
		{"DAA", []inst.Instruction{{Op: inst.DAA}}},
		{"CPL", []inst.Instruction{{Op: inst.CPL}}},
		{"SCF", []inst.Instruction{{Op: inst.SCF}}},
		{"CCF", []inst.Instruction{{Op: inst.CCF}}},
		{"NEG", []inst.Instruction{{Op: inst.NEG}}},
		{"NOP", []inst.Instruction{{Op: inst.NOP}}},
		// BIT/RES/SET
		{"BIT 0,A", []inst.Instruction{{Op: inst.BIT_0_A}}},
		{"BIT 7,B", []inst.Instruction{{Op: inst.BIT_7_B}}},
		{"RES 3,C", []inst.Instruction{{Op: inst.RES_3_C}}},
		{"SET 5,D", []inst.Instruction{{Op: inst.SET_5_D}}},
		// 16-bit
		{"INC BC", []inst.Instruction{{Op: inst.INC_BC}}},
		{"DEC DE", []inst.Instruction{{Op: inst.DEC_DE}}},
		{"ADD HL,BC", []inst.Instruction{{Op: inst.ADD_HL_BC}}},
		{"EX DE,HL", []inst.Instruction{{Op: inst.EX_DE_HL}}},
		{"LD SP,HL", []inst.Instruction{{Op: inst.LD_SP_HL}}},
		// 16-bit immediate
		{"LD BC,0x1234", []inst.Instruction{{Op: inst.LD_BC_NN, Imm: 0x1234}}},
		{"LD HL,0xABCD", []inst.Instruction{{Op: inst.LD_HL_NN, Imm: 0xABCD}}},
		// ADC/SBC HL
		{"ADC HL,BC", []inst.Instruction{{Op: inst.ADC_HL_BC}}},
		{"SBC HL,DE", []inst.Instruction{{Op: inst.SBC_HL_DE}}},
	}

	// Compute Go fingerprints and print
	fmt.Fprintf(os.Stderr, "Computing %d fingerprints (Go CPU)...\n", len(tests))
	for i, tc := range tests {
		fp := search.Fingerprint(tc.seq)
		fmt.Fprintf(os.Stderr, "%3d. %-15s FP[0:4]=%02x %02x %02x %02x\n",
			i, tc.name, fp[0], fp[1], fp[2], fp[3])
	}

	// Use XOR A as target — known to only match itself
	targetSeq := []inst.Instruction{{Op: inst.XOR_A}}
	targetFP := search.Fingerprint(targetSeq)

	candidateCount := uint32(len(tests))
	seqLen := uint32(1)
	deadFlags := uint32(0)

	// Write binary protocol to stdout
	binary.Write(os.Stdout, binary.LittleEndian, candidateCount)
	binary.Write(os.Stdout, binary.LittleEndian, seqLen)
	binary.Write(os.Stdout, binary.LittleEndian, deadFlags)
	os.Stdout.Write(targetFP[:])

	for _, tc := range tests {
		packed := packInstr(tc.seq[0].Op, tc.seq[0].Imm)
		binary.Write(os.Stdout, binary.LittleEndian, packed)
	}

	fmt.Fprintf(os.Stderr, "Wrote %d candidates, target=XOR A\n", len(tests))

	// Now: for comprehensive verification, also dump all fingerprints so we can
	// compare each candidate's GPU fingerprint against Go's.
	// Write a second file with expected fingerprints.
	fpFile, err := os.Create("cuda/expected_fps.bin")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: couldn't create expected_fps.bin: %v\n", err)
		return
	}
	defer fpFile.Close()
	binary.Write(fpFile, binary.LittleEndian, candidateCount)
	for _, tc := range tests {
		fp := search.Fingerprint(tc.seq)
		fpFile.Write(fp[:])
	}
	fmt.Fprintf(os.Stderr, "Wrote expected fingerprints to cuda/expected_fps.bin\n")

	// Also print which candidates should match XOR A
	fmt.Fprintf(os.Stderr, "\nExpected matches for target XOR A:\n")
	for i, tc := range tests {
		if search.QuickCheck(targetSeq, tc.seq) {
			fmt.Fprintf(os.Stderr, "  MATCH: %d. %s\n", i, tc.name)
		}
	}
}

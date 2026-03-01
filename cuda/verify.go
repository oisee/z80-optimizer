//go:build ignore

// verify.go — Cross-verify CUDA Z80 executor against Go CPU executor.
// Run: go run cuda/verify.go
// Requires: cuda/dump_fps binary (built from dump_fps.cu)
package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/search"
)

func main() {
	// Build list of all opcodes with representative immediates
	type testInstr struct {
		op  inst.OpCode
		imm uint16
	}

	var instrs []testInstr
	for op := inst.OpCode(0); op < inst.OpCodeCount; op++ {
		if inst.HasImmediate(op) {
			// Test with a few representative immediate values
			if inst.HasImm16(op) {
				instrs = append(instrs, testInstr{op, 0x0000})
				instrs = append(instrs, testInstr{op, 0x1234})
				instrs = append(instrs, testInstr{op, 0xFFFF})
			} else {
				instrs = append(instrs, testInstr{op, 0x00})
				instrs = append(instrs, testInstr{op, 0x42})
				instrs = append(instrs, testInstr{op, 0xFF})
			}
		} else {
			instrs = append(instrs, testInstr{op, 0})
		}
	}

	fmt.Fprintf(os.Stderr, "Testing %d instruction variants (%d opcodes)...\n",
		len(instrs), inst.OpCodeCount)

	// Compute Go fingerprints
	goFPs := make(map[string]string) // "op=X imm=Y" → hex fingerprint
	for _, ti := range instrs {
		seq := []inst.Instruction{{Op: ti.op, Imm: ti.imm}}
		fp := search.Fingerprint(seq)
		key := fmt.Sprintf("op=%d imm=%d", ti.op, ti.imm)
		hex := ""
		for _, b := range fp {
			hex += fmt.Sprintf("%02x", b)
		}
		goFPs[key] = hex
	}

	// Generate input for CUDA dump_fps
	var input strings.Builder
	for _, ti := range instrs {
		fmt.Fprintf(&input, "%d %d\n", ti.op, ti.imm)
	}

	// Run CUDA dump_fps
	cmd := exec.Command("cuda/dump_fps")
	cmd.Stdin = strings.NewReader(input.String())
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to run cuda/dump_fps: %v\n", err)
		os.Exit(1)
	}

	// Parse CUDA output and compare
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	mismatches := 0
	matches := 0
	lineNum := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineNum++
		// Format: "op=X imm=Y fp=HEXHEX..."
		parts := strings.SplitN(line, " fp=", 2)
		if len(parts) != 2 {
			fmt.Fprintf(os.Stderr, "Bad line %d: %s\n", lineNum, line)
			continue
		}
		key := parts[0]
		cudaFP := parts[1]

		goFP, ok := goFPs[key]
		if !ok {
			fmt.Fprintf(os.Stderr, "Unknown key: %s\n", key)
			continue
		}

		if cudaFP == goFP {
			matches++
		} else {
			mismatches++
			if mismatches <= 20 {
				fmt.Fprintf(os.Stderr, "MISMATCH %s:\n  Go:   %s\n  CUDA: %s\n", key, goFP, cudaFP)
				// Show first differing byte position
				for i := 0; i < len(goFP) && i < len(cudaFP); i += 2 {
					if goFP[i:i+2] != cudaFP[i:i+2] {
						byteIdx := i / 2
						vecIdx := byteIdx / 10
						regIdx := byteIdx % 10
						regNames := []string{"A", "F", "B", "C", "D", "E", "H", "L", "SPh", "SPl"}
						fmt.Fprintf(os.Stderr, "  First diff at byte %d (vector %d, %s): Go=%s CUDA=%s\n",
							byteIdx, vecIdx, regNames[regIdx], goFP[i:i+2], cudaFP[i:i+2])
						break
					}
				}
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\nResults: %d matches, %d mismatches out of %d tests\n",
		matches, mismatches, matches+mismatches)
	if mismatches > 0 {
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "ALL TESTS PASSED\n")
}

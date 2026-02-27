package search

import "github.com/oisee/z80-optimizer/pkg/inst"

// EnumConfig configures sequence enumeration.
type EnumConfig struct {
	MaxLen int // Maximum sequence length to enumerate
}

// EnumerateSequences generates all instruction sequences of exactly length n.
// It calls fn for each sequence. The slice passed to fn is reused between calls.
// fn should return false to stop enumeration early.
// Includes 16-bit immediate ops (LD rr,nn) with all 65536 values.
func EnumerateSequences(n int, fn func(seq []inst.Instruction) bool) {
	nonImm := inst.NonImmediateOps()
	imm8Ops := inst.ImmediateOps()
	imm16Ops := inst.Imm16Ops()

	seq := make([]inst.Instruction, n)
	enumerateRec(seq, 0, nonImm, imm8Ops, imm16Ops, fn)
}

// enumerateRec recursively builds sequences.
func enumerateRec(seq []inst.Instruction, pos int, nonImm []inst.OpCode, imm8Ops []inst.OpCode, imm16Ops []inst.OpCode, fn func([]inst.Instruction) bool) bool {
	if pos == len(seq) {
		return fn(seq)
	}

	// Non-immediate instructions
	for _, op := range nonImm {
		seq[pos] = inst.Instruction{Op: op, Imm: 0}
		if !enumerateRec(seq, pos+1, nonImm, imm8Ops, imm16Ops, fn) {
			return false
		}
	}

	// 8-bit immediate instructions with all 256 values
	for _, op := range imm8Ops {
		for imm := 0; imm < 256; imm++ {
			seq[pos] = inst.Instruction{Op: op, Imm: uint16(imm)}
			if !enumerateRec(seq, pos+1, nonImm, imm8Ops, imm16Ops, fn) {
				return false
			}
		}
	}

	// 16-bit immediate instructions with all 65536 values
	for _, op := range imm16Ops {
		for imm := 0; imm < 65536; imm++ {
			seq[pos] = inst.Instruction{Op: op, Imm: uint16(imm)}
			if !enumerateRec(seq, pos+1, nonImm, imm8Ops, imm16Ops, fn) {
				return false
			}
		}
	}

	return true
}

// InstructionCount returns the number of distinct instructions.
func InstructionCount() int {
	return len(inst.NonImmediateOps()) + len(inst.ImmediateOps())*256 + len(inst.Imm16Ops())*65536
}

// EnumerateFirstOp returns all possible first instructions (for partitioning).
func EnumerateFirstOp() []inst.Instruction {
	result := make([]inst.Instruction, 0, InstructionCount())
	for _, op := range inst.NonImmediateOps() {
		result = append(result, inst.Instruction{Op: op, Imm: 0})
	}
	for _, op := range inst.ImmediateOps() {
		for imm := 0; imm < 256; imm++ {
			result = append(result, inst.Instruction{Op: op, Imm: uint16(imm)})
		}
	}
	for _, op := range inst.Imm16Ops() {
		for imm := 0; imm < 65536; imm++ {
			result = append(result, inst.Instruction{Op: op, Imm: uint16(imm)})
		}
	}
	return result
}

package gpu

import (
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"sync"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/search"
)

// CUDAProcess manages a long-running z80qc --server child process.
// Candidates are uploaded once at startup; each QuickCheck call sends
// a target fingerprint and reads back matching indices.
type CUDAProcess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	mu     sync.Mutex // serialize queries
	count  uint32     // candidate count
}

// CUDABinaryPath is the path to the z80qc CUDA binary.
// Override this before calling NewCUDAProcess if the binary is elsewhere.
var CUDABinaryPath = "cuda/z80qc"

// NewCUDAProcess starts the CUDA server process and uploads candidates.
// The candidates slice should be the full set of instruction variants to test.
// seqLen is the number of instructions per candidate sequence.
func NewCUDAProcess(candidates []inst.Instruction, seqLen int) (*CUDAProcess, error) {
	cmd := exec.Command(CUDABinaryPath, "--server")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("cuda: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("cuda: stdout pipe: %w", err)
	}
	// Stderr goes to parent's stderr for diagnostics.
	cmd.Stderr = nil // inherit

	if err := cmd.Start(); err != nil {
		stdin.Close()
		return nil, fmt.Errorf("cuda: start %s: %w", CUDABinaryPath, err)
	}

	cp := &CUDAProcess{
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
		count:  uint32(len(candidates)),
	}

	// Write header: candidate_count, seq_len
	header := [2]uint32{uint32(len(candidates)), uint32(seqLen)}
	if err := binary.Write(stdin, binary.LittleEndian, header); err != nil {
		cp.Close()
		return nil, fmt.Errorf("cuda: write header: %w", err)
	}

	// Write packed candidate data: each instruction is uint32(op | imm<<16)
	for _, c := range candidates {
		packed := uint32(c.Op) | (uint32(c.Imm) << 16)
		if err := binary.Write(stdin, binary.LittleEndian, packed); err != nil {
			cp.Close()
			return nil, fmt.Errorf("cuda: write candidate: %w", err)
		}
	}

	return cp, nil
}

// QuickCheckGPU sends a target fingerprint to the GPU and returns indices
// of candidates whose fingerprints match (subject to dead flag masking).
func (cp *CUDAProcess) QuickCheckGPU(fp [search.FingerprintLen]byte, deadFlags search.FlagMask) ([]uint32, error) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Write target fingerprint (80 bytes) + dead_flags (4 bytes)
	if _, err := cp.stdin.Write(fp[:]); err != nil {
		return nil, fmt.Errorf("cuda: write fingerprint: %w", err)
	}
	if err := binary.Write(cp.stdin, binary.LittleEndian, uint32(deadFlags)); err != nil {
		return nil, fmt.Errorf("cuda: write dead_flags: %w", err)
	}

	// Read response: match_count + match_indices
	var matchCount uint32
	if err := binary.Read(cp.stdout, binary.LittleEndian, &matchCount); err != nil {
		return nil, fmt.Errorf("cuda: read match_count: %w", err)
	}

	if matchCount == 0 {
		return nil, nil
	}

	matches := make([]uint32, matchCount)
	if err := binary.Read(cp.stdout, binary.LittleEndian, matches); err != nil {
		return nil, fmt.Errorf("cuda: read matches: %w", err)
	}

	return matches, nil
}

// Close shuts down the CUDA process.
func (cp *CUDAProcess) Close() error {
	cp.stdin.Close()
	return cp.cmd.Wait()
}

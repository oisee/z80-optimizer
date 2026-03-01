package gpu

import (
	"fmt"
	"time"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
	"github.com/oisee/z80-optimizer/pkg/search"
)

// SearchConfig holds GPU search configuration.
type SearchConfig struct {
	MaxTargetLen int
	MaxCandLen   int
	Verbose      bool
	DeadFlags    search.FlagMask
}

// SearchGPU runs the superoptimizer search using CUDA GPU-accelerated QuickCheck.
// For each target, the GPU tests all candidates in parallel, then the CPU
// does ExhaustiveCheck only on the ~0.01% of candidates that pass QuickCheck.
func SearchGPU(cfg SearchConfig) (*result.Table, error) {
	if cfg.MaxCandLen <= 0 {
		cfg.MaxCandLen = cfg.MaxTargetLen - 1
	}

	startTime := time.Now()

	// Build candidate list.
	if cfg.Verbose {
		fmt.Println("Building candidate list...")
	}
	candidates := search.EnumerateFirstOp()

	if cfg.Verbose {
		fmt.Printf("  %d candidate instructions\n", len(candidates))
	}

	// Start CUDA process.
	if cfg.Verbose {
		fmt.Println("Starting CUDA GPU process...")
	}
	cuda, err := NewCUDAProcess(candidates, 1)
	if err != nil {
		return nil, fmt.Errorf("gpu: %w", err)
	}
	defer cuda.Close()

	if cfg.Verbose {
		fmt.Printf("GPU initialized (%.1fs)\n", time.Since(startTime).Seconds())
	}

	results := result.NewTable()

	for targetLen := 2; targetLen <= cfg.MaxTargetLen; targetLen++ {
		candLen := targetLen - 1
		if candLen > cfg.MaxCandLen {
			candLen = cfg.MaxCandLen
		}

		if cfg.Verbose {
			fmt.Printf("=== Searching target length %d (GPU) ===\n", targetLen)
		}

		if err := searchLengthGPU(cuda, candidates, targetLen, candLen, cfg, results); err != nil {
			return results, fmt.Errorf("gpu: length %d: %w", targetLen, err)
		}

		if cfg.Verbose {
			elapsed := time.Since(startTime)
			fmt.Printf("  Elapsed: %s, Found: %d\n", elapsed.Round(time.Millisecond), results.Len())
		}
	}

	return results, nil
}

// searchLengthGPU searches for optimizations for targets of a specific length.
func searchLengthGPU(cuda *CUDAProcess, candidates []inst.Instruction, targetLen, candLen int, cfg SearchConfig, results *result.Table) error {
	// Collect targets.
	var targets [][]inst.Instruction
	search.EnumerateSequences8(targetLen, func(seq []inst.Instruction) bool {
		if search.ShouldPrune(seq) {
			return true
		}
		seqCopy := make([]inst.Instruction, len(seq))
		copy(seqCopy, seq)
		targets = append(targets, seqCopy)
		return true
	})

	if cfg.Verbose {
		fmt.Printf("  %d target sequences\n", len(targets))
	}

	// Process each target: GPU QuickCheck → CPU ExhaustiveCheck.
	gpuChecks := 0
	cpuVerifies := 0
	found := 0
	reportTime := time.Now()

	for i, target := range targets {
		// Progress reporting.
		if cfg.Verbose && time.Since(reportTime) > 10*time.Second {
			reportTime = time.Now()
			pct := float64(i) / float64(len(targets)) * 100
			fmt.Printf("  [%.1f%%] %d/%d targets | %d GPU hits → %d verified | %d found\n",
				pct, i, len(targets), gpuChecks, cpuVerifies, found)
		}

		// Compute target fingerprint.
		fp := search.Fingerprint(target)

		// GPU QuickCheck: tests all candidates in one dispatch.
		hits, err := cuda.QuickCheckGPU(fp, cfg.DeadFlags)
		if err != nil {
			return fmt.Errorf("target %d: %w", i, err)
		}

		gpuChecks += len(hits)

		// CPU ExhaustiveCheck on hits only.
		targetBytes := inst.SeqByteSize(target)
		targetTStates := inst.SeqTStates(target)

		for _, hitIdx := range hits {
			if int(hitIdx) >= len(candidates) {
				continue
			}

			cand := []inst.Instruction{candidates[hitIdx]}
			candBytes := inst.SeqByteSize(cand)
			if candBytes >= targetBytes {
				continue
			}

			if search.ShouldPrune(cand) {
				continue
			}

			// MidCheck: 32-vector filter to catch false positives
			if cfg.DeadFlags == search.DeadNone {
				if !search.MidCheck(target, cand) {
					continue
				}
			} else {
				if !search.MidCheckMasked(target, cand, cfg.DeadFlags) {
					continue
				}
			}

			cpuVerifies++

			if cfg.DeadFlags == search.DeadNone {
				if !search.ExhaustiveCheck(target, cand) {
					continue
				}
			} else {
				if !search.ExhaustiveCheckMasked(target, cand, cfg.DeadFlags) {
					continue
				}
			}

			// Found a valid replacement.
			found++
			candCopy := make([]inst.Instruction, len(cand))
			copy(candCopy, cand)
			candTStates := inst.SeqTStates(candCopy)

			rule := result.Rule{
				Source:      copySeq(target),
				Replacement: candCopy,
				BytesSaved:  targetBytes - candBytes,
				CyclesSaved: targetTStates - candTStates,
			}

			if cfg.DeadFlags != search.DeadNone {
				flagDiff := search.FlagDiff(target, cand)
				rule.DeadFlags = flagDiff
			}

			results.Add(rule)

			if cfg.Verbose {
				fmt.Printf("  FOUND: %s -> %s (-%d bytes, -%d cycles)\n",
					disasmSeq(target), disasmSeq(candCopy),
					rule.BytesSaved, rule.CyclesSaved)
			}
		}
	}

	if cfg.Verbose {
		fmt.Printf("  GPU hits: %d, CPU verifies: %d, Found: %d\n",
			gpuChecks, cpuVerifies, found)
	}

	return nil
}

func copySeq(seq []inst.Instruction) []inst.Instruction {
	c := make([]inst.Instruction, len(seq))
	copy(c, seq)
	return c
}

func disasmSeq(seq []inst.Instruction) string {
	s := ""
	for i, instr := range seq {
		if i > 0 {
			s += " : "
		}
		s += inst.Disassemble(instr)
	}
	return s
}

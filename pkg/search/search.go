package search

import (
	"fmt"
	"runtime"
	"time"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
)

// Config holds search configuration.
type Config struct {
	MaxTargetLen int  // Maximum target sequence length (2-5)
	MaxCandLen   int  // Maximum candidate length (defaults to MaxTargetLen-1)
	NumWorkers   int  // Number of parallel workers (defaults to NumCPU)
	Verbose      bool // Print progress
}

// Run executes the superoptimizer search.
// For each target length L, it tries to find shorter replacements.
func Run(cfg Config) *result.Table {
	if cfg.NumWorkers <= 0 {
		cfg.NumWorkers = runtime.NumCPU()
	}
	if cfg.MaxCandLen <= 0 {
		cfg.MaxCandLen = cfg.MaxTargetLen - 1
	}

	pool := NewWorkerPool(cfg.NumWorkers)
	startTime := time.Now()

	for targetLen := 2; targetLen <= cfg.MaxTargetLen; targetLen++ {
		if cfg.Verbose {
			fmt.Printf("=== Searching target length %d ===\n", targetLen)
		}

		tasks := collectTasks(targetLen, cfg.MaxCandLen)
		if cfg.Verbose {
			fmt.Printf("  Generated %d target sequences (after pruning)\n", len(tasks))
		}

		pool.RunTasks(tasks, cfg.Verbose)

		checked, found := pool.Stats()
		if cfg.Verbose {
			elapsed := time.Since(startTime)
			fmt.Printf("  Checked: %d, Found: %d, Elapsed: %s\n", checked, found, elapsed.Round(time.Millisecond))
		}
	}

	return pool.Results
}

// collectTasks generates all non-prunable target sequences of the given length.
// Uses 8-bit-only enumeration for targets to keep the search space feasible.
// 16-bit immediate ops are still considered as candidate replacements.
func collectTasks(targetLen, maxCandLen int) []SearchTask {
	var tasks []SearchTask

	EnumerateSequences8(targetLen, func(seq []inst.Instruction) bool {
		if ShouldPrune(seq) {
			return true
		}

		// Copy the sequence for the task
		seqCopy := make([]inst.Instruction, len(seq))
		copy(seqCopy, seq)
		tasks = append(tasks, SearchTask{
			Target:     seqCopy,
			MaxCandLen: maxCandLen,
		})
		return true
	})

	return tasks
}

// SearchSingle finds the shortest replacement for a specific target sequence.
func SearchSingle(target []inst.Instruction, maxCandLen int, verbose bool) *result.Rule {
	pool := NewWorkerPool(1)
	pool.RunTasks([]SearchTask{{
		Target:     target,
		MaxCandLen: maxCandLen,
	}}, verbose)

	rules := pool.Results.Rules()
	if len(rules) == 0 {
		return nil
	}
	return &rules[0]
}

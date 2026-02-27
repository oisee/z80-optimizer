package search

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
)

// WorkerPool manages parallel search workers.
type WorkerPool struct {
	NumWorkers int
	Results    *result.Table
	mu         sync.Mutex
	checked    atomic.Int64
	found      atomic.Int64
}

// NewWorkerPool creates a pool with the given number of workers.
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	return &WorkerPool{
		NumWorkers: numWorkers,
		Results:    result.NewTable(),
	}
}

// SearchTask represents a unit of work: find a shorter replacement for target.
type SearchTask struct {
	Target    []inst.Instruction
	MaxCandLen int
}

// Stats returns search statistics.
func (wp *WorkerPool) Stats() (checked, found int64) {
	return wp.checked.Load(), wp.found.Load()
}

// RunTasks distributes search tasks across workers.
func (wp *WorkerPool) RunTasks(tasks []SearchTask, verbose bool) {
	ch := make(chan SearchTask, len(tasks))
	for _, t := range tasks {
		ch <- t
	}
	close(ch)

	var wg sync.WaitGroup
	for i := 0; i < wp.NumWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range ch {
				wp.processTask(task, verbose)
			}
		}()
	}
	wg.Wait()
}

// processTask finds the shortest replacement for a target sequence.
func (wp *WorkerPool) processTask(task SearchTask, verbose bool) {
	targetBytes := inst.SeqByteSize(task.Target)
	targetTStates := inst.SeqTStates(task.Target)

	// Try candidate lengths from 1 up to maxCandLen
	for candLen := 1; candLen <= task.MaxCandLen; candLen++ {
		found := false
		EnumerateSequences(candLen, func(cand []inst.Instruction) bool {
			wp.checked.Add(1)

			// Skip if candidate is not shorter
			candBytes := inst.SeqByteSize(cand)
			if candBytes >= targetBytes {
				return true // continue
			}

			// Pruning
			if ShouldPrune(cand) {
				return true
			}

			// QuickCheck first (fast rejection)
			if !QuickCheck(task.Target, cand) {
				return true
			}

			// Exhaustive verification
			if !ExhaustiveCheck(task.Target, cand) {
				return true
			}

			// Found a valid replacement!
			wp.found.Add(1)
			candCopy := make([]inst.Instruction, len(cand))
			copy(candCopy, cand)
			candTStates := inst.SeqTStates(candCopy)

			rule := result.Rule{
				Source:      copySeq(task.Target),
				Replacement: candCopy,
				BytesSaved:  targetBytes - candBytes,
				CyclesSaved: targetTStates - candTStates,
			}

			wp.mu.Lock()
			wp.Results.Add(rule)
			wp.mu.Unlock()

			if verbose {
				fmt.Printf("  FOUND: %s -> %s (-%d bytes, -%d cycles)\n",
					disasmSeq(task.Target), disasmSeq(candCopy),
					rule.BytesSaved, rule.CyclesSaved)
			}

			found = true
			return false // stop enumeration for this candidate length
		})
		if found {
			break // found optimal (shortest) replacement
		}
	}
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

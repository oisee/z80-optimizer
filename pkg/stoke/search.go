package stoke

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
	"github.com/oisee/z80-optimizer/pkg/search"
)

// Config holds STOKE search configuration.
type Config struct {
	Target     []inst.Instruction
	Chains     int     // Number of independent MCMC chains (goroutines)
	Iterations int     // Iterations per chain
	Decay      float64 // Temperature decay factor per step
	Verbose    bool
	DeadFlags  uint8 // If nonzero, ignore these flag bits during equivalence checks
}

// Result holds a verified optimization found by STOKE.
type Result struct {
	Rule    result.Rule
	ChainID int
	Iter    int
}

// Run launches N independent MCMC chains in parallel and collects verified results.
func Run(cfg Config) []Result {
	if cfg.Chains <= 0 {
		cfg.Chains = 1
	}
	if cfg.Iterations <= 0 {
		cfg.Iterations = 1_000_000
	}
	if cfg.Decay <= 0 || cfg.Decay >= 1 {
		cfg.Decay = 0.9999
	}

	targetBytes := inst.SeqByteSize(cfg.Target)
	targetCycles := inst.SeqTStates(cfg.Target)

	if cfg.Verbose {
		fmt.Printf("STOKE search: %d chains × %d iterations (decay=%.6f)\n",
			cfg.Chains, cfg.Iterations, cfg.Decay)
		fmt.Printf("Target: ")
		for i, instr := range cfg.Target {
			if i > 0 {
				fmt.Print(" : ")
			}
			fmt.Print(inst.Disassemble(instr))
		}
		fmt.Printf(" (%d bytes, %d T-states)\n\n", targetBytes, targetCycles)
	}

	var mu sync.Mutex
	var results []Result
	var wg sync.WaitGroup

	// Seed from random source
	baseSeed := rand.Uint64()

	// Progress tracking
	startTime := time.Now()
	done := make(chan struct{})

	if cfg.Verbose {
		go func() {
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					elapsed := time.Since(startTime)
					mu.Lock()
					found := len(results)
					mu.Unlock()
					fmt.Printf("  [%s] %d verified results found\n",
						elapsed.Round(time.Second), found)
				}
			}
		}()
	}

	for i := 0; i < cfg.Chains; i++ {
		wg.Add(1)
		go func(chainID int) {
			defer wg.Done()

			seed := baseSeed + uint64(chainID)*0x9E3779B97F4A7C15
			chain := NewChain(cfg.Target, 1.0, seed)
			if cfg.DeadFlags != 0 {
				chain.deadFlags = cfg.DeadFlags
			}

			for iter := 0; iter < cfg.Iterations; iter++ {
				chain.Step(cfg.Decay)

				// Check if best has zero mismatches and is shorter
				best, bestCost := chain.Best()
				// Zero mismatches means cost < 1000 (since 1000*0 + size + cycles/100 < 1000 for any reasonable seq)
				if bestCost < 1000 && chain.IsShorter() {
					// Verify with ExhaustiveCheck (masked or full)
					verified := false
					var deadFlags uint8
					if cfg.DeadFlags != 0 {
						verified = search.ExhaustiveCheckMasked(cfg.Target, best, cfg.DeadFlags)
						if verified {
							deadFlags = search.FlagDiff(cfg.Target, best)
						}
					} else {
						verified = search.ExhaustiveCheck(cfg.Target, best)
					}

					if verified {
						candBytes := inst.SeqByteSize(best)
						candCycles := inst.SeqTStates(best)
						r := Result{
							Rule: result.Rule{
								Source:      copySeq(cfg.Target),
								Replacement: copySeq(best),
								BytesSaved:  targetBytes - candBytes,
								CyclesSaved: targetCycles - candCycles,
								DeadFlags:   deadFlags,
							},
							ChainID: chainID,
							Iter:    iter,
						}

						mu.Lock()
						results = append(results, r)
						mu.Unlock()

						if cfg.Verbose {
							fmt.Printf("  Chain %d @ iter %d: ", chainID, iter)
							for j, instr := range best {
								if j > 0 {
									fmt.Print(" : ")
								}
								fmt.Print(inst.Disassemble(instr))
							}
							if deadFlags != 0 {
								fmt.Printf(" (-%d bytes, -%d cycles, dead flags 0x%02X) VERIFIED\n",
									r.Rule.BytesSaved, r.Rule.CyclesSaved, deadFlags)
							} else {
								fmt.Printf(" (-%d bytes, -%d cycles) VERIFIED\n",
									r.Rule.BytesSaved, r.Rule.CyclesSaved)
							}
						}

						// Reset chain to explore more
						chain = NewChain(cfg.Target, 1.0, seed+uint64(iter))
						if cfg.DeadFlags != 0 {
							chain.deadFlags = cfg.DeadFlags
						}
					}
				}
			}

			if cfg.Verbose {
				fmt.Printf("  Chain %d done: %d accepted, %d rejected\n",
					chainID, chain.Accepted, chain.Rejected)
			}
		}(i)
	}

	wg.Wait()
	close(done)

	if cfg.Verbose {
		elapsed := time.Since(startTime)
		fmt.Printf("\nSTOKE complete: %d verified results in %s\n",
			len(results), elapsed.Round(time.Millisecond))
	}

	return results
}

// Deduplicate removes duplicate results (same replacement sequence).
func Deduplicate(results []Result) []Result {
	seen := make(map[string]bool)
	var unique []Result
	for _, r := range results {
		key := seqKey(r.Rule.Replacement)
		if !seen[key] {
			seen[key] = true
			unique = append(unique, r)
		}
	}
	return unique
}

func seqKey(seq []inst.Instruction) string {
	key := make([]byte, 0, len(seq)*4)
	for _, instr := range seq {
		key = append(key, byte(instr.Op>>8), byte(instr.Op), byte(instr.Imm>>8), byte(instr.Imm))
	}
	return string(key)
}

package stoke

import (
	"math"
	"math/rand/v2"

	"github.com/oisee/z80-optimizer/pkg/inst"
)

// Chain is a single Metropolis-Hastings MCMC chain with simulated annealing.
type Chain struct {
	current     []inst.Instruction
	best        []inst.Instruction
	cost        int
	bestCost    int
	temperature float64
	rng         *rand.Rand
	mutator     *Mutator
	target      []inst.Instruction
	targetBytes int
	deadFlags   uint8 // If nonzero, ignore these flag bits in cost evaluation

	// Stats
	Accepted int64
	Rejected int64
}

// NewChain creates a new MCMC chain initialized from the target sequence.
func NewChain(target []inst.Instruction, temperature float64, seed uint64) *Chain {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))
	maxLen := len(target) + 2 // allow some growth
	if maxLen < 10 {
		maxLen = 10
	}
	current := copySeq(target)
	cost := Cost(target, current)

	return &Chain{
		current:     current,
		best:        copySeq(current),
		cost:        cost,
		bestCost:    cost,
		temperature: temperature,
		rng:         rng,
		mutator:     NewMutator(rng, maxLen),
		target:      target,
		targetBytes: inst.SeqByteSize(target),
	}
}

// Step performs one MCMC iteration: mutate, evaluate, accept/reject.
// Returns true if the step was accepted.
func (c *Chain) Step(decay float64) bool {
	candidate := c.mutator.Mutate(c.current)
	newCost := CostMasked(c.target, candidate, c.deadFlags)
	delta := newCost - c.cost

	accepted := false
	if delta <= 0 {
		// Always accept improvements (or equal)
		accepted = true
	} else if c.temperature > 0 {
		// Accept worse solutions with probability e^(-delta/T)
		prob := math.Exp(-float64(delta) / c.temperature)
		if c.rng.Float64() < prob {
			accepted = true
		}
	}

	if accepted {
		c.current = candidate
		c.cost = newCost
		c.Accepted++

		// Track best: prefer lower cost, and among zero-mismatch candidates prefer shorter
		if newCost < c.bestCost {
			c.best = copySeq(candidate)
			c.bestCost = newCost
		}
	} else {
		c.Rejected++
	}

	// Anneal
	c.temperature *= decay

	return accepted
}

// Best returns the best candidate found and its cost.
func (c *Chain) Best() ([]inst.Instruction, int) {
	return c.best, c.bestCost
}

// Current returns the current candidate and its cost.
func (c *Chain) Current() ([]inst.Instruction, int) {
	return c.current, c.cost
}

// IsShorter returns true if the best candidate is shorter (in bytes) than the target.
func (c *Chain) IsShorter() bool {
	return inst.SeqByteSize(c.best) < c.targetBytes
}

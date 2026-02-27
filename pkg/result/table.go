package result

import (
	"sort"
	"sync"

	"github.com/oisee/z80-optimizer/pkg/inst"
)

// Rule represents a single optimization: replacing Source with Replacement.
type Rule struct {
	Source      []inst.Instruction
	Replacement []inst.Instruction
	BytesSaved  int
	CyclesSaved int
}

// Table stores discovered optimization rules.
type Table struct {
	mu    sync.Mutex
	rules []Rule
}

// NewTable creates an empty table.
func NewTable() *Table {
	return &Table{}
}

// Add inserts a rule into the table.
func (t *Table) Add(r Rule) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rules = append(t.rules, r)
}

// Rules returns a copy of all rules, sorted by bytes saved (descending).
func (t *Table) Rules() []Rule {
	t.mu.Lock()
	defer t.mu.Unlock()
	result := make([]Rule, len(t.rules))
	copy(result, t.rules)
	sort.Slice(result, func(i, j int) bool {
		if result[i].BytesSaved != result[j].BytesSaved {
			return result[i].BytesSaved > result[j].BytesSaved
		}
		return result[i].CyclesSaved > result[j].CyclesSaved
	})
	return result
}

// Len returns the number of rules.
func (t *Table) Len() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.rules)
}

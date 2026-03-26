// Package regalloc provides a reader for the Z80T binary register allocation tables.
//
// Tables are indexed by enumeration order — shape N in the file corresponds to
// shape N from regalloc-enum. Callers must compute the enumeration index from
// their constraint shape to perform a lookup.
//
// Binary format (Z80T v1):
//
//	Header:  4 bytes magic "Z80T" + 4 bytes version (uint32 LE)
//	Records: variable-length, one per shape:
//	  Infeasible: 0xFF (1 byte)
//	  Feasible:   nVregs (uint8) + cost (uint16 LE) + assignment (nVregs bytes)
package regalloc

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Entry is a single register allocation result.
type Entry struct {
	Cost       int    // T-states cost, -1 if infeasible
	Assignment []byte // physical location per vreg (nil if infeasible)
}

// Infeasible returns true if no valid assignment exists for this shape.
func (e *Entry) Infeasible() bool { return e.Cost < 0 }

// Table holds all entries from a Z80T binary file, indexed by enumeration order.
type Table struct {
	Entries []Entry
}

// LoadBinary reads a Z80T binary table (optionally zstd-compressed).
// Returns a Table with entries indexed by enumeration order.
func LoadBinary(path string) (*Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read header
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "Z80T" {
		return nil, fmt.Errorf("bad magic: %q (expected Z80T)", magic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	// Read all records
	var entries []Entry
	buf := make([]byte, 1)

	for {
		_, err := io.ReadFull(f, buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read record %d: %w", len(entries), err)
		}

		marker := buf[0]
		if marker == 0xFF {
			// Infeasible
			entries = append(entries, Entry{Cost: -1})
		} else {
			// Feasible: marker is nVregs
			nv := int(marker)
			var cost uint16
			if err := binary.Read(f, binary.LittleEndian, &cost); err != nil {
				return nil, fmt.Errorf("read cost at record %d: %w", len(entries), err)
			}
			assign := make([]byte, nv)
			if _, err := io.ReadFull(f, assign); err != nil {
				return nil, fmt.Errorf("read assignment at record %d: %w", len(entries), err)
			}
			entries = append(entries, Entry{Cost: int(cost), Assignment: assign})
		}
	}

	return &Table{Entries: entries}, nil
}

// Lookup returns the entry at the given enumeration index, or nil if out of range.
func (t *Table) Lookup(index int) *Entry {
	if index < 0 || index >= len(t.Entries) {
		return nil
	}
	return &t.Entries[index]
}

// Stats returns table statistics.
func (t *Table) Stats() (total, feasible, infeasible int) {
	total = len(t.Entries)
	for i := range t.Entries {
		if t.Entries[i].Infeasible() {
			infeasible++
		} else {
			feasible++
		}
	}
	return
}

// Location name constants matching the binary format.
var LocNames = [15]string{
	"A", "B", "C", "D", "E", "H", "L",
	"BC", "DE", "HL",
	"IXH", "IXL", "IYH", "IYL",
	"mem0",
}

// FormatAssignment returns a human-readable assignment string like "A=HL, B=DE".
func FormatAssignment(assignment []byte) string {
	s := ""
	for i, loc := range assignment {
		if i > 0 {
			s += ", "
		}
		name := "?"
		if int(loc) < len(LocNames) {
			name = LocNames[loc]
		}
		s += fmt.Sprintf("v%d=%s", i, name)
	}
	return s
}

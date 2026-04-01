// Package enr reads and writes Z80 enriched register allocation tables (.enr files).
//
// File format (ENRT v1):
//
//	Header (16 bytes):
//	  magic:       "ENRT"     (4 bytes)
//	  version:     1          (uint32 LE)
//	  n_entries:   N          (uint32 LE) — total record count (feasible + infeasible)
//	  max_vregs:   V          (uint8)
//	  num_patterns: P         (uint8)  — always 12 for v1
//	  reserved:    0          (uint16)
//
//	Records (variable length):
//	  Infeasible: 0xFF
//	  Feasible:   nVregs(u8) + cost(u16le) + assignment[nVregs](u8) + flags(u16le) + patterns[P×u16le]
//
// Flags bitfield (v1):
//
//	bit 0: no_accumulator  (A not in assignment)
//	bit 1: no_hl_pair      (HL not in assignment)
//	bit 2: mul8_safe       (no clobber conflicts)
//	bit 3: djnz_conflict   (B occupied)
//	bit 4: u16_capable     (≥2 pair slots)
//
// Pattern cost order (v1, 12 uint16 values):
//
//	0:u8_alu_avg, 1:u8_best_alu, 2:u8_worst_alu, 3:u16_add_natural,
//	4:u8_mul_avg, 5:mul8_conflicts, 6:mul16_conflicts, 7:call_save_cost,
//	8:call_free_saves, 9:temp_regs_avail, 10:u16_slots_free, 11:u8_regs_free
package enr

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	Magic       = "ENRT"
	Version     = 1
	NumPatterns = 12

	// Pattern indices
	PatU8AluAvg      = 0
	PatU8BestAlu     = 1
	PatU8WorstAlu    = 2
	PatU16AddNatural = 3
	PatU8MulAvg      = 4
	PatMul8Conflicts = 5
	PatMul16Conflicts = 6
	PatCallSaveCost  = 7
	PatCallFreeSaves = 8
	PatTempRegsAvail = 9
	PatU16SlotsFree  = 10
	PatU8RegsFree    = 11

	// Flags
	FlagNoAccumulator = 1 << 0
	FlagNoHLPair      = 1 << 1
	FlagMul8Safe      = 1 << 2
	FlagDJNZConflict  = 1 << 3
	FlagU16Capable    = 1 << 4
)

// Entry is a single enriched register allocation result.
type Entry struct {
	NVregs     int
	Cost       int    // -1 if infeasible
	Assignment []byte // nil if infeasible
	Flags      uint16
	Patterns   [NumPatterns]uint16
}

// Infeasible returns true if no valid assignment exists for this shape.
func (e *Entry) Infeasible() bool { return e.Cost < 0 }

// Reader streams entries from an .enr file.
type Reader struct {
	r        *bufio.Reader
	NEntries uint32
	MaxVregs uint8
	NumPat   uint8
	pos      int
}

// Open opens an .enr file for sequential reading.
// The file must be uncompressed (caller decompresses zstd if needed).
func Open(path string) (*Reader, *os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	r, err := NewReader(f)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return r, f, nil
}

// NewReader creates a Reader from an io.Reader already positioned at byte 0.
func NewReader(r io.Reader) (*Reader, error) {
	br := bufio.NewReaderSize(r, 1<<20)

	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != Magic {
		return nil, fmt.Errorf("bad magic %q (expected ENRT)", magic)
	}

	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != Version {
		return nil, fmt.Errorf("unsupported version %d", version)
	}

	var nEntries uint32
	if err := binary.Read(br, binary.LittleEndian, &nEntries); err != nil {
		return nil, fmt.Errorf("read n_entries: %w", err)
	}

	header12 := make([]byte, 4)
	if _, err := io.ReadFull(br, header12); err != nil {
		return nil, fmt.Errorf("read header tail: %w", err)
	}

	return &Reader{
		r:        br,
		NEntries: nEntries,
		MaxVregs: header12[0],
		NumPat:   header12[1],
	}, nil
}

// Next reads and returns the next entry, or io.EOF when done.
func (rd *Reader) Next() (*Entry, error) {
	b := make([]byte, 1)
	if _, err := io.ReadFull(rd.r, b); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil, io.EOF
		}
		return nil, err
	}
	rd.pos++

	if b[0] == 0xFF {
		return &Entry{Cost: -1}, nil
	}

	nv := int(b[0])
	var cost16 uint16
	if err := binary.Read(rd.r, binary.LittleEndian, &cost16); err != nil {
		return nil, fmt.Errorf("entry %d: read cost: %w", rd.pos, err)
	}

	assign := make([]byte, nv)
	if _, err := io.ReadFull(rd.r, assign); err != nil {
		return nil, fmt.Errorf("entry %d: read assignment: %w", rd.pos, err)
	}

	var flags uint16
	if err := binary.Read(rd.r, binary.LittleEndian, &flags); err != nil {
		return nil, fmt.Errorf("entry %d: read flags: %w", rd.pos, err)
	}

	np := int(rd.NumPat)
	if np == 0 {
		np = NumPatterns
	}
	var pats [NumPatterns]uint16
	for i := 0; i < np && i < NumPatterns; i++ {
		if err := binary.Read(rd.r, binary.LittleEndian, &pats[i]); err != nil {
			return nil, fmt.Errorf("entry %d: read pattern %d: %w", rd.pos, i, err)
		}
	}

	return &Entry{
		NVregs:     nv,
		Cost:       int(cost16),
		Assignment: assign,
		Flags:      flags,
		Patterns:   pats,
	}, nil
}

// Position returns the number of records read so far (1-based).
func (rd *Reader) Position() int { return rd.pos }

// Writer writes entries to an .enr file.
type Writer struct {
	w        *bufio.Writer
	f        *os.File
	nWritten uint32
}

// Create creates a new .enr file for writing.
// maxVregs should be the maximum nVregs value across all entries.
func Create(path string, maxVregs uint8) (*Writer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriterSize(f, 1<<20)

	// Write header with placeholder count.
	w.WriteString(Magic)
	binary.Write(w, binary.LittleEndian, uint32(Version))
	binary.Write(w, binary.LittleEndian, uint32(0)) // placeholder
	w.WriteByte(maxVregs)
	w.WriteByte(NumPatterns)
	binary.Write(w, binary.LittleEndian, uint16(0)) // reserved

	if err := w.Flush(); err != nil {
		f.Close()
		return nil, err
	}

	return &Writer{w: w, f: f}, nil
}

// WriteInfeasible writes an infeasible record.
func (wr *Writer) WriteInfeasible() error {
	wr.nWritten++
	return wr.w.WriteByte(0xFF)
}

// WriteFeasible writes a feasible entry.
func (wr *Writer) WriteFeasible(e *Entry) error {
	wr.nWritten++
	wr.w.WriteByte(byte(e.NVregs))
	binary.Write(wr.w, binary.LittleEndian, uint16(e.Cost))
	wr.w.Write(e.Assignment)
	binary.Write(wr.w, binary.LittleEndian, e.Flags)
	for i := 0; i < NumPatterns; i++ {
		binary.Write(wr.w, binary.LittleEndian, e.Patterns[i])
	}
	return nil
}

// Close flushes buffered data, seeks back to write the actual record count,
// and closes the file.
func (wr *Writer) Close() error {
	if err := wr.w.Flush(); err != nil {
		wr.f.Close()
		return err
	}
	// Seek to n_entries offset (4+4=8) and write actual count.
	if _, err := wr.f.Seek(8, io.SeekStart); err != nil {
		wr.f.Close()
		return err
	}
	if err := binary.Write(wr.f, binary.LittleEndian, wr.nWritten); err != nil {
		wr.f.Close()
		return err
	}
	return wr.f.Close()
}

// NWritten returns the number of records written so far.
func (wr *Writer) NWritten() uint32 { return wr.nWritten }

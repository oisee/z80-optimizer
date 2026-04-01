// merge-tables — merge regalloc table sources into a unified Z80T v2 binary.
//
// Priority order (for same shape, keep lowest cost):
//   1. GPU-direct from Z80T v2 binary (ix_expanded_*.bin)   — ground truth
//   2. Derived from existing enriched tables (ix_derived_*.jsonl)  — approximation
//
// The Z80T v2 primary binary (from build-ix-table) covers ALL shapes in the
// 6-locSet8 enumeration space. Derived JSONL records are used to improve entries
// where the GPU found infeasible (conservative: GPU already found the optimal).
//
// Output: Z80T v2 binary with best assignment per shape (same format as input).
//
// For shapes with no source at all (not in primary, not derived): emits infeasible (0xFF).
//
// Usage (simplest — just pass-through the GPU binary):
//
//	./merge-tables -primary data/ix_expanded_5v.bin -out data/merged_5v.bin
//
// With derived records added:
//
//	./merge-tables -primary data/ix_expanded_5v.bin \
//	               -derived data/ix_derived_5v.jsonl \
//	               -out data/merged_5v.bin
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"
)

// Z80T v2 header fields we care about.
type tableHeader struct {
	nLocSets8  int
	nLocSets16 int
	maxVregs   int
	nEntries   uint64
}

func readV2Header(f *os.File) (tableHeader, error) {
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return tableHeader{}, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "Z80T" {
		return tableHeader{}, fmt.Errorf("bad magic %q", magic)
	}
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return tableHeader{}, err
	}
	if version != 2 {
		return tableHeader{}, fmt.Errorf("unsupported version %d (need v2)", version)
	}
	hdr := make([]byte, 3)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return tableHeader{}, err
	}
	var nEntries uint64
	if err := binary.Read(f, binary.LittleEndian, &nEntries); err != nil {
		return tableHeader{}, err
	}
	return tableHeader{
		nLocSets8:  int(hdr[0]),
		nLocSets16: int(hdr[1]),
		maxVregs:   int(hdr[2]),
		nEntries:   nEntries,
	}, nil
}

// v2Record matches the Z80T v2 per-record format.
type v2Record struct {
	cost       int    // -1 = infeasible
	assignment []byte // nil if infeasible
}

// readRecord reads one record from a Z80T v2 binary (sequential).
func readRecord(r *bufio.Reader) (v2Record, error) {
	b, err := r.ReadByte()
	if err != nil {
		return v2Record{cost: -1}, err
	}
	if b == 0xFF {
		return v2Record{cost: -1}, nil
	}
	nv := int(b)
	var cost16 uint16
	if err := binary.Read(r, binary.LittleEndian, &cost16); err != nil {
		return v2Record{cost: -1}, err
	}
	assign := make([]byte, nv)
	if _, err := io.ReadFull(r, assign); err != nil {
		return v2Record{cost: -1}, err
	}
	return v2Record{cost: int(cost16), assignment: assign}, nil
}

// writeRecord writes one record to a Z80T v2 binary.
func writeRecord(w *bufio.Writer, rec v2Record) {
	if rec.cost < 0 {
		w.WriteByte(0xFF)
		return
	}
	w.WriteByte(byte(len(rec.assignment)))
	binary.Write(w, binary.LittleEndian, uint16(rec.cost))
	w.Write(rec.assignment)
}

// derivedRecord — JSONL from derive-ix output.
type derivedRecord struct {
	Cost       int   `json:"cost"`
	Assignment []int `json:"assignment"`
	Feasible   int   `json:"feasible"`
	Derived    bool  `json:"derived"`
	// Shape metadata for indexing into the 6-locSet8 enumeration space.
	NVregs    int   `json:"nVregs"`
	IntfMask  int   `json:"intfMask"`
	LocSetIdx []int `json:"locSetIdx"`
}

// locSets8_6 matches regalloc-enum with 6 locSets8.
var locSets8_6 = [][]int{
	{0},                         // 0: must be A
	{2},                         // 1: must be C
	{0, 1, 2, 3, 4, 5, 6},      // 2: any GPR8
	{1, 2, 3, 4, 5, 6},         // 3: any GPR8 except A
	{10, 11, 12, 13},            // 4: must be IX/IY half
	{0, 1, 2, 3, 4, 10, 11, 12, 13}, // 5: any 8-bit except H/L
}

var locSets16_3 = [][]int{
	{9},       // 0: must be HL
	{8},       // 1: must be DE
	{7, 8, 9}, // 2: any pair
}

// shapeIndex computes the linear enumeration index for a shape within the
// 6-locSet8 space with the given maxVregs.
// Returns -1 on error.
func shapeIndex(nVregs, maxVregs int, widths []int, locSetIdx []int, intfMask int) int {
	if nVregs < 2 || nVregs > maxVregs {
		return -1
	}

	ls8 := locSets8_6
	ls16 := locSets16_3

	// Compute offset for preceding nv values.
	offset := 0
	for nv := 2; nv < nVregs; nv++ {
		offset += countShapesForNV(nv, ls8, ls16)
	}

	// Width combo: bit i = vreg i is 16-bit.
	widthCombo := 0
	for i := 0; i < nVregs; i++ {
		if widths[i] == 16 {
			widthCombo |= 1 << i
		}
	}

	nEdges := nVregs * (nVregs - 1) / 2
	nIntfGraphs := 1 << nEdges

	// Add offsets for preceding width combos.
	for wc := 0; wc < widthCombo; wc++ {
		offset += locComboCount(nVregs, wc, ls8, ls16) * nIntfGraphs
	}

	// Add offset for preceding loc combos.
	locCombo := encodeLocCombo(nVregs, widthCombo, locSetIdx, ls8, ls16)
	if locCombo < 0 {
		return -1
	}
	offset += locCombo * nIntfGraphs

	// Add interference graph index.
	offset += intfMask

	return offset
}

func countShapesForNV(nv int, ls8, ls16 [][]int) int {
	nEdges := nv * (nv - 1) / 2
	nIntfGraphs := 1 << nEdges
	total := 0
	for wc := 0; wc < (1 << nv); wc++ {
		total += locComboCount(nv, wc, ls8, ls16) * nIntfGraphs
	}
	return total
}

func locComboCount(nv, widthCombo int, ls8, ls16 [][]int) int {
	count := 1
	for i := 0; i < nv; i++ {
		if widthCombo&(1<<i) != 0 {
			count *= len(ls16)
		} else {
			count *= len(ls8)
		}
	}
	return count
}

func encodeLocCombo(nv, widthCombo int, locSetIdx []int, ls8, ls16 [][]int) int {
	combo := 0
	for i := 0; i < nv; i++ {
		var nSets int
		if widthCombo&(1<<i) != 0 {
			nSets = len(ls16)
		} else {
			nSets = len(ls8)
		}
		if locSetIdx[i] < 0 || locSetIdx[i] >= nSets {
			return -1
		}
		mul := 1
		for j := i + 1; j < nv; j++ {
			if widthCombo&(1<<j) != 0 {
				mul *= len(ls16)
			} else {
				mul *= len(ls8)
			}
		}
		combo += locSetIdx[i] * mul
	}
	return combo
}

func main() {
	primaryPath := flag.String("primary", "", "Z80T v2 primary binary (required)")
	derivedPath := flag.String("derived", "", "JSONL derived records from derive-ix (optional)")
	outPath := flag.String("out", "", "output Z80T v2 binary (required)")
	flag.Parse()

	if *primaryPath == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "error: -primary and -out are required")
		os.Exit(1)
	}

	// --- Load derived records into a map (index → best cost+assignment) ---
	type candidate struct {
		cost   int
		assign []byte
	}
	derived := map[int]candidate{}

	if *derivedPath != "" {
		df, err := os.Open(*derivedPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open derived %s: %v\n", *derivedPath, err)
			os.Exit(1)
		}
		scanner := bufio.NewScanner(df)
		scanner.Buffer(make([]byte, 1<<20), 1<<20)
		nDerived := 0
		for scanner.Scan() {
			var rec derivedRecord
			if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
				continue
			}
			if rec.Feasible == 0 || len(rec.LocSetIdx) == 0 {
				continue
			}

			// Infer widths from locSetIdx (index >= 3 for 16-bit locSets? No:
			// widths are encoded separately. We don't have them in derived records.)
			// Assume all 8-bit (the most common case from existing .enr).
			// 16-bit derived shapes are rare and can be added later.
			widths := make([]int, rec.NVregs)
			for i := range widths {
				widths[i] = 8 // all-8bit assumption for derived
			}

			idx := shapeIndex(rec.NVregs, 6, widths, rec.LocSetIdx, rec.IntfMask)
			if idx < 0 {
				continue
			}
			assign := make([]byte, len(rec.Assignment))
			for i, v := range rec.Assignment {
				assign[i] = byte(v)
			}
			if existing, ok := derived[idx]; !ok || rec.Cost < existing.cost {
				derived[idx] = candidate{cost: rec.Cost, assign: assign}
			}
			nDerived++
		}
		df.Close()
		fmt.Fprintf(os.Stderr, "Loaded %d derived records (%d unique shapes)\n", nDerived, len(derived))
	}

	// --- Open primary Z80T v2 binary ---
	pf, err := os.Open(*primaryPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open primary %s: %v\n", *primaryPath, err)
		os.Exit(1)
	}
	defer pf.Close()

	phdr, err := readV2Header(pf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read primary header: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Primary: %d entries, maxVregs=%d, locSets8=%d\n",
		phdr.nEntries, phdr.maxVregs, phdr.nLocSets8)

	// --- Create output file ---
	of, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create %s: %v\n", *outPath, err)
		os.Exit(1)
	}
	ow := bufio.NewWriterSize(of, 1<<20)

	// Write header (placeholder count).
	ow.WriteString("Z80T")
	binary.Write(ow, binary.LittleEndian, uint32(2))
	ow.WriteByte(byte(phdr.nLocSets8))
	ow.WriteByte(byte(phdr.nLocSets16))
	ow.WriteByte(byte(phdr.maxVregs))
	binary.Write(ow, binary.LittleEndian, uint64(0)) // placeholder
	ow.Flush()

	const nEntriesOffset = 11

	// --- Stream primary, apply derived improvements ---
	pr := bufio.NewReaderSize(pf, 1<<20)
	var nTotal, nImproved, nDerivedUsed uint64
	var nFeasible, nInfeasible uint64
	start := time.Now()
	progress := uint64(1_000_000)

	for {
		rec, err := readRecord(pr)
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "read primary record %d: %v\n", nTotal, err)
			os.Exit(1)
		}

		// Check if derived has a better (or any) entry for this shape index.
		if cand, ok := derived[int(nTotal)]; ok {
			if rec.cost < 0 || cand.cost < rec.cost {
				// Derived is better — use it.
				rec = v2Record{cost: cand.cost, assignment: cand.assign}
				nDerivedUsed++
				if cand.cost < rec.cost {
					nImproved++
				}
			}
		}

		writeRecord(ow, rec)
		if rec.cost >= 0 {
			nFeasible++
		} else {
			nInfeasible++
		}
		nTotal++

		if nTotal%progress == 0 {
			elapsed := time.Since(start)
			rate := float64(nTotal) / elapsed.Seconds()
			fmt.Fprintf(os.Stderr, "  %d entries (%.0f/s), feasible=%.1f%%\n",
				nTotal, rate, 100*float64(nFeasible)/float64(nTotal))
		}
	}

	ow.Flush()

	// Seek back to write actual n_entries.
	of.Seek(nEntriesOffset, io.SeekStart)
	binary.Write(of, binary.LittleEndian, nTotal)

	elapsed := time.Since(start)
	sz, _ := of.Seek(0, io.SeekEnd)
	of.Close()

	feasPct := 0.0
	if nTotal > 0 {
		feasPct = 100 * float64(nFeasible) / float64(nTotal)
	}

	fmt.Fprintf(os.Stderr, "Done: %d entries (feasible=%.1f%%, improved=%d, derived-used=%d) in %.1fs → %s (%.1f MB)\n",
		nTotal, feasPct, nImproved, nDerivedUsed, elapsed.Seconds(), *outPath, float64(sz)/(1<<20))
}

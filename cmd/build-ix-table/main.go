// build-ix-table — convert regalloc GPU server JSONL output to Z80T v2 binary.
//
// Reads JSONL records from stdin (one JSON object per line) produced by:
//
//	./regalloc-enum | ./cuda/z80_regalloc --server > data/foo.jsonl
//
// Writes a Z80T v2 binary file to -out.
//
// Z80T v2 header format:
//
//	magic:       "Z80T"    (4 bytes)
//	version:     2         (uint32 LE)
//	nLocSets8:   N         (uint8) — number of 8-bit loc-set options (4=original, 6=IX-expanded)
//	nLocSets16:  M         (uint8) — number of 16-bit loc-set options (always 3)
//	maxVregs:    V         (uint8) — max vregs this table covers
//	n_entries:   E         (uint64 LE) — total number of records
//
// Records (same as v1):
//
//	Infeasible: 0xFF
//	Feasible:   nVregs(uint8) + cost(uint16 LE) + assignment[nVregs](uint8 each)
//
// Usage:
//
//	./build-ix-table -out data/ix_expanded_5v.bin -n-locsets8 6 -max-vregs 5 < data/ix_expanded_5v.jsonl
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

type jsonRecord struct {
	Cost        int   `json:"cost"`
	Assignment  []int `json:"assignment"`
	SearchSpace int   `json:"searchSpace"`
	Feasible    int   `json:"feasible"`
}

func main() {
	outPath := flag.String("out", "", "output binary file path (required)")
	nLocSets8 := flag.Int("n-locsets8", 6, "number of 8-bit loc sets (4=original, 6=IX-expanded)")
	nLocSets16 := flag.Int("n-locsets16", 3, "number of 16-bit loc sets (default 3)")
	maxVregs := flag.Int("max-vregs", 5, "max vregs this table covers")
	flag.Parse()

	if *outPath == "" {
		fmt.Fprintln(os.Stderr, "error: -out is required")
		os.Exit(1)
	}
	if *nLocSets8 < 1 || *nLocSets8 > 255 {
		fmt.Fprintln(os.Stderr, "error: -n-locsets8 must be 1-255")
		os.Exit(1)
	}

	// Open output file for writing (will seek back to fill n_entries).
	out, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create %s: %v\n", *outPath, err)
		os.Exit(1)
	}
	defer out.Close()

	w := bufio.NewWriterSize(out, 1<<20) // 1MB write buffer

	// Write header — n_entries placeholder = 0, will seek back to fill.
	w.WriteString("Z80T")
	binary.Write(w, binary.LittleEndian, uint32(2))      // version
	w.WriteByte(byte(*nLocSets8))                         // nLocSets8
	w.WriteByte(byte(*nLocSets16))                        // nLocSets16
	w.WriteByte(byte(*maxVregs))                          // maxVregs
	binary.Write(w, binary.LittleEndian, uint64(0))       // n_entries placeholder
	if err := w.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "flush header: %v\n", err)
		os.Exit(1)
	}

	// Header layout offsets: "Z80T"(4) + version(4) + nLS8(1) + nLS16(1) + maxVregs(1) = 11
	// n_entries starts at byte 11.
	const nEntriesOffset = 11

	// Process JSONL from stdin.
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)

	w.Reset(out)

	var nEntries, nFeasible, nInfeasible uint64
	start := time.Now()
	progressInterval := 500_000

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var rec jsonRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			fmt.Fprintf(os.Stderr, "parse line %d: %v\n", nEntries+1, err)
			os.Exit(1)
		}

		if rec.Cost < 0 {
			// Infeasible
			w.WriteByte(0xFF)
			nInfeasible++
		} else {
			// Feasible
			nVregs := len(rec.Assignment)
			if nVregs > 127 {
				fmt.Fprintf(os.Stderr, "line %d: nVregs=%d too large\n", nEntries+1, nVregs)
				os.Exit(1)
			}
			if rec.Cost > 65534 {
				fmt.Fprintf(os.Stderr, "line %d: cost=%d overflows uint16\n", nEntries+1, rec.Cost)
				os.Exit(1)
			}
			w.WriteByte(byte(nVregs))
			binary.Write(w, binary.LittleEndian, uint16(rec.Cost))
			for _, loc := range rec.Assignment {
				w.WriteByte(byte(loc))
			}
			nFeasible++
		}
		nEntries++

		if int(nEntries)%progressInterval == 0 {
			elapsed := time.Since(start)
			rate := float64(nEntries) / elapsed.Seconds()
			fmt.Fprintf(os.Stderr, "  %d entries (%.0f/s), feasible=%.1f%%\n",
				nEntries, rate, 100*float64(nFeasible)/float64(nEntries))
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read stdin: %v\n", err)
		os.Exit(1)
	}

	// Flush remaining data before seeking back.
	if err := w.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "flush: %v\n", err)
		os.Exit(1)
	}

	// Seek back and write actual n_entries.
	if _, err := out.Seek(nEntriesOffset, io.SeekStart); err != nil {
		fmt.Fprintf(os.Stderr, "seek: %v\n", err)
		os.Exit(1)
	}
	if err := binary.Write(out, binary.LittleEndian, nEntries); err != nil {
		fmt.Fprintf(os.Stderr, "write n_entries: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	feasPct := 0.0
	if nEntries > 0 {
		feasPct = 100 * float64(nFeasible) / float64(nEntries)
	}
	sz, _ := out.Seek(0, io.SeekEnd)

	fmt.Fprintf(os.Stderr, "Done: %d entries (feasible=%.1f%%, infeasible=%d) in %.1fs → %s (%.1f MB)\n",
		nEntries, feasPct, nInfeasible, elapsed.Seconds(), *outPath, float64(sz)/(1<<20))
}

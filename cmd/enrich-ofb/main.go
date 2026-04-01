// enrich-ofb: generate OFB (Op Feasibility Bag) sidecar for regalloc tables.
//
// Supports two input formats (auto-detected by magic bytes):
//   - ENRT (enriched .enr files): enriched_4v.enr, enriched_5v.enr, enriched_6v_dense.enr
//   - Z80T v2 (.bin files):       merged_ix_5v.bin, ix_expanded_5v.bin, etc.
//
// Output: .ofb sidecar binary aligned to the source file:
//   magic "OFB1" (4 bytes) + n_entries (uint64 LE) + [n_entries × uint32 LE OFB]
//   Infeasible entries get OFB = 0.
//
// Usage:
//   enrich-ofb -input data/enriched_5v.enr        -output data/enriched_5v.ofb
//   enrich-ofb -input data/enriched_6v_dense.enr   -output data/enriched_6v_dense.ofb
//   enrich-ofb -input data/merged_ix_5v.bin        -output data/merged_ix_5v.ofb
//
// OFB bit definitions — see pkg/regalloc/ofb.go (regalloc.OFB* constants).
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/oisee/z80-optimizer/pkg/enr"
	"github.com/oisee/z80-optimizer/pkg/regalloc"
)

func main() {
	input := flag.String("input", "", "source file (.enr or Z80T .bin) — required")
	output := flag.String("output", "", "output .ofb sidecar file — required")
	flag.Parse()

	if *input == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "Usage: enrich-ofb -input FILE -output FILE.ofb")
		fmt.Fprintln(os.Stderr, "  Accepts ENRT (.enr) and Z80T v2 (.bin) formats (auto-detected).")
		os.Exit(1)
	}

	if err := run(*input, *output); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(inputPath, outputPath string) error {
	f, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return fmt.Errorf("read magic: %w", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek: %w", err)
	}

	switch string(magic[:]) {
	case "ENRT":
		return runENRT(f, inputPath, outputPath)
	case "Z80T":
		return runZ80T(f, inputPath, outputPath)
	default:
		return fmt.Errorf("unknown format magic %q (expected ENRT or Z80T)", magic)
	}
}

// writeOFBHeader writes the OFB1 file header with a placeholder n_entries.
// Returns the file offset where n_entries should be written on close.
func writeOFBHeader(bw *bufio.Writer) {
	bw.WriteString("OFB1")
	binary.Write(bw, binary.LittleEndian, uint64(0)) // placeholder, filled in after
}

// closeOFB flushes bw, seeks back to offset 4 in f, and writes the final entry count.
func closeOFB(bw *bufio.Writer, f *os.File, n uint64, outputPath string) error {
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("flush %s: %w", outputPath, err)
	}
	if _, err := f.Seek(4, io.SeekStart); err != nil {
		return fmt.Errorf("seek to n_entries: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, n); err != nil {
		return fmt.Errorf("write n_entries: %w", err)
	}
	return nil
}

// runENRT processes an ENRT (.enr) input file.
func runENRT(f *os.File, inputPath, outputPath string) error {
	r, err := enr.NewReader(f)
	if err != nil {
		return fmt.Errorf("open ENRT: %w", err)
	}
	nEntries := uint64(r.NEntries)
	fmt.Fprintf(os.Stderr, "enrich-ofb ENRT: %s → %s  (%d entries)\n",
		inputPath, outputPath, nEntries)

	out, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer out.Close()
	bw := bufio.NewWriterSize(out, 1<<20)
	writeOFBHeader(bw)

	var nFeasible, nProcessed uint64
	for {
		entry, err := r.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read entry %d: %w", nProcessed, err)
		}
		nProcessed++
		var ofb uint32
		if !entry.Infeasible() {
			nFeasible++
			ofb = regalloc.ComputeOFB(entry.Assignment)
		}
		binary.Write(bw, binary.LittleEndian, ofb)
		if nProcessed%2_000_000 == 0 {
			fmt.Fprintf(os.Stderr, "  %dM / %dM  (%.0f%%)\n",
				nProcessed/1_000_000, nEntries/1_000_000,
				float64(nProcessed)/float64(nEntries)*100)
		}
	}

	if err := closeOFB(bw, out, nProcessed, outputPath); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "enrich-ofb ENRT: done. %d feasible / %d total.\n",
		nFeasible, nProcessed)
	return nil
}

// runZ80T processes a Z80T v2 (.bin) input file.
//
// Z80T v2 header: "Z80T"(4) + version(uint32le) + nLocSets8(u8) + nLocSets16(u8) +
//
//	maxVregs(u8) + n_entries(uint64le)
//
// Records: 0xFF = infeasible | nVregs(u8) + cost(uint16le) + assignment[nVregs](u8...)
func runZ80T(f *os.File, inputPath, outputPath string) error {
	br := bufio.NewReaderSize(f, 1<<20)

	var hdr struct {
		Magic      [4]byte
		Version    uint32
		NLocSets8  uint8
		NLocSets16 uint8
		MaxVregs   uint8
		NEntries   uint64
	}
	// Read fields individually (struct has alignment issues with binary.Read)
	io.ReadFull(br, hdr.Magic[:])
	binary.Read(br, binary.LittleEndian, &hdr.Version)
	binary.Read(br, binary.LittleEndian, &hdr.NLocSets8)
	binary.Read(br, binary.LittleEndian, &hdr.NLocSets16)
	binary.Read(br, binary.LittleEndian, &hdr.MaxVregs)
	binary.Read(br, binary.LittleEndian, &hdr.NEntries)

	if hdr.Version != 2 {
		return fmt.Errorf("Z80T version %d not supported (expected 2)", hdr.Version)
	}

	fmt.Fprintf(os.Stderr, "enrich-ofb Z80T v2: %s → %s  (%d entries, nLS8=%d nLS16=%d maxV=%d)\n",
		inputPath, outputPath, hdr.NEntries, hdr.NLocSets8, hdr.NLocSets16, hdr.MaxVregs)

	out, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer out.Close()
	bw := bufio.NewWriterSize(out, 1<<20)
	writeOFBHeader(bw)

	var nFeasible, nProcessed uint64
	oneByte := make([]byte, 1)
	for nProcessed < hdr.NEntries {
		if _, err := io.ReadFull(br, oneByte); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return fmt.Errorf("read record %d: %w", nProcessed, err)
		}
		nProcessed++

		var ofb uint32
		if oneByte[0] != 0xFF {
			nv := int(oneByte[0])
			var cost uint16
			binary.Read(br, binary.LittleEndian, &cost)
			assign := make([]byte, nv)
			if _, err := io.ReadFull(br, assign); err != nil {
				return fmt.Errorf("read assignment at record %d: %w", nProcessed, err)
			}
			nFeasible++
			ofb = regalloc.ComputeOFB(assign)
		}
		binary.Write(bw, binary.LittleEndian, ofb)

		if nProcessed%5_000_000 == 0 {
			fmt.Fprintf(os.Stderr, "  %.0fM / %.0fM  (%.0f%%)\n",
				float64(nProcessed)/1e6, float64(hdr.NEntries)/1e6,
				float64(nProcessed)/float64(hdr.NEntries)*100)
		}
	}

	if err := closeOFB(bw, out, nProcessed, outputPath); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "enrich-ofb Z80T v2: done. %d feasible / %d total.\n",
		nFeasible, nProcessed)
	return nil
}

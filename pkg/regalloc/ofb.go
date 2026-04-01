// OFB (Op Feasibility Bag) — per-assignment operation capability bitmask.
//
// Each bit indicates whether a Z80 operation class is natively supported
// by a given register assignment without extra save/restore moves.
// Computed in O(1) purely from the assignment bytes.
//
// # OFB Bit Reference
//
//	Bit  Constant          Meaning
//	 0   OFBALuGPR         A assigned + another vreg in B/C/D/E/H/L → native "op A,r" (4T)
//	 1   OFBALuIX          A assigned + vreg in IXH/IXL/IYH/IYL    → "op A,IXh" (8T, DD prefix)
//	 2   OFBHLArith        HL pair assigned                         → ADD/ADC/SBC HL,rr native
//	 3   OFBHLPtr          HL pair assigned                         → LD r,(HL) / LD (HL),r native
//	 4   OFBDEPtr          DE pair assigned                         → LD A,(DE) / LD (DE),A native
//	 5   OFBMul8Safe       H, L, C all free                         → mul8 clobber zone clear
//	 6   OFBDJNZFree       B not assigned                           → DJNZ loop, no save needed
//	 7   OFBIXBridge       ≥1 vreg in IXH/IXL/IYH/IYL              → EXX zone crossing free
//	 8   OFBAFree          A not assigned                           → A available as scratch
//	 9   OFBEXXSplit       ≥2 vregs in B/C/D/E/H/L                 → EXX 2-bank split worthwhile
//	10   OFBBCFree         BC pair not assigned                     → BC free for ADC HL,BC / loop
//	11   OFBDEFree         DE pair not assigned                     → DE free as pointer/scratch
//	12   OFBADCSBCSrc      No IX halves assigned                    → ADC/SBC HL,rr src valid
//	                                                                   (IX halves can't be ADC/SBC src: DD+ED conflict)
//	13   OFBSPFree         SP not occupied (always true in cur model) → PUSH/POP tier safe
//	14   OFBHLHLu32        HL assigned + BC or DE free               → HLH'L' u32 pattern via EXX viable
//
// # Usage — backend (code generator)
//
//	entry, _ := table.Lookup(idx)
//	ofb := regalloc.ComputeOFB(entry.Assignment)
//
//	// Before emitting a multiply:
//	if ofb&regalloc.OFBMul8Safe != 0 {
//	    ops, T, _, _, ok := mulopt.Emit16c(k) // H,L,C all clear: safe to clobber
//	} else {
//	    // emit save/restore for H, L, C around mul8 sequence
//	}
//
//	// Before emitting DJNZ loop:
//	if ofb&regalloc.OFBDJNZFree != 0 {
//	    // emit: LD B,n; loop: ...; DJNZ loop  (free, no B save needed)
//	} else {
//	    // emit: PUSH BC; LD B,n; loop: ...; DJNZ loop; POP BC
//	}
//
//	// EXX zone split decision:
//	if ofb&regalloc.OFBEXXSplit != 0 {
//	    // ≥2 GPR8 vregs — worth inserting EXX boundary for shadow bank
//	}
//
//	// ADC/SBC HL,rr safety check:
//	if ofb&regalloc.OFBADCSBCSrc != 0 {
//	    // safe: no vreg is in IXH/IXL/IYH/IYL acting as ADC/SBC source
//	    // (DD+ED prefix conflict makes IXH/IXL invalid ADC/SBC src)
//	}
//
// # OFB sidecar files
//
// For the enriched .enr tables, precomputed OFB sidecars are available:
//
//	data/enriched_4v.ofb        — 156K entries × 4 bytes  = ~625KB
//	data/enriched_5v.ofb        — 17.4M entries × 4 bytes = ~67MB
//	data/enriched_6v_dense.ofb  — 66.1M entries × 4 bytes = ~253MB
//
// Sidecar format: magic "OFB1" (4B) + n_entries (uint32 LE) + [n_entries × uint32 LE]
// Entry i in the sidecar corresponds to entry i in the .enr file (same enumeration order).
// Infeasible entries have OFB = 0.
//
// For merged_ix_5v.bin (IX-expanded 5v, 60.9M entries), no sidecar file exists;
// call ComputeOFB(entry.Assignment) directly — it is O(1).
package regalloc

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// OFB bit constants.
const (
	OFBALuGPR    uint32 = 1 << 0  // A assigned + another vreg in B/C/D/E/H/L → native "op A,r" (4T)
	OFBALuIX     uint32 = 1 << 1  // A assigned + vreg in IXH/IXL/IYH/IYL     → "op A,IXh" (8T)
	OFBHLArith   uint32 = 1 << 2  // HL pair assigned → ADD/ADC/SBC HL,rr native
	OFBHLPtr     uint32 = 1 << 3  // HL pair assigned → LD r,(HL) / LD (HL),r native
	OFBDEPtr     uint32 = 1 << 4  // DE pair assigned → LD A,(DE) / LD (DE),A native
	OFBMul8Safe  uint32 = 1 << 5  // H, L, C all free → mul8 clobber zone clear
	OFBDJNZFree  uint32 = 1 << 6  // B not assigned   → DJNZ loop without save
	OFBIXBridge  uint32 = 1 << 7  // ≥1 vreg in IXH/IXL/IYH/IYL → EXX zone crossing free
	OFBAFree     uint32 = 1 << 8  // A not assigned → A available as scratch accumulator
	OFBEXXSplit  uint32 = 1 << 9  // ≥2 vregs in B/C/D/E/H/L → EXX 2-bank split worthwhile
	OFBBCFree    uint32 = 1 << 10 // BC pair not assigned → ADC HL,BC / free loop counter
	OFBDEFree    uint32 = 1 << 11 // DE pair not assigned → DE free as pointer/scratch
	OFBADCSBCSrc uint32 = 1 << 12 // No IX halves → ADC/SBC HL,rr src valid (no DD+ED conflict)
	OFBSPFree    uint32 = 1 << 13 // SP not occupied → PUSH/POP tier safe
	OFBHLHLu32   uint32 = 1 << 14 // HL assigned + BC or DE free → HLH'L' u32 pattern via EXX
)

// ComputeOFB derives the OFB bitmask from a physical register assignment.
// assignment[i] is the physical location of vreg i (use Loc* constants).
// O(1) — no table lookup needed.
func ComputeOFB(assignment []byte) uint32 {
	var present [15]bool
	for _, loc := range assignment {
		if int(loc) < len(present) {
			present[loc] = true
		}
	}

	hasA   := present[LocA]
	hasB   := present[LocB]
	hasC   := present[LocC]
	hasH   := present[LocH]
	hasL   := present[LocL]
	hasBC  := present[LocBC]
	hasDE  := present[LocDE]
	hasHL  := present[LocHL]
	hasIXH := present[LocIXH]
	hasIXL := present[LocIXL]
	hasIYH := present[LocIYH]
	hasIYL := present[LocIYL]

	hasAnyIX := hasIXH || hasIXL || hasIYH || hasIYL

	gpr8Count := 0
	for _, loc := range []byte{LocB, LocC, LocD, LocE, LocH, LocL} {
		if present[loc] {
			gpr8Count++
		}
	}

	var ofb uint32

	if hasA && gpr8Count > 0 {
		ofb |= OFBALuGPR
	}
	if hasA && hasAnyIX {
		ofb |= OFBALuIX
	}
	if hasHL {
		ofb |= OFBHLArith | OFBHLPtr
	}
	if hasDE {
		ofb |= OFBDEPtr
	}
	if !hasC && !hasH && !hasL {
		ofb |= OFBMul8Safe
	}
	if !hasB {
		ofb |= OFBDJNZFree
	}
	if hasAnyIX {
		ofb |= OFBIXBridge
	}
	if !hasA {
		ofb |= OFBAFree
	}
	if gpr8Count >= 2 {
		ofb |= OFBEXXSplit
	}
	if !hasBC {
		ofb |= OFBBCFree
	}
	if !hasDE {
		ofb |= OFBDEFree
	}
	if !hasAnyIX {
		ofb |= OFBADCSBCSrc
	}
	ofb |= OFBSPFree // SP is never a vreg location in the current model
	if hasHL && (!hasBC || !hasDE) {
		ofb |= OFBHLHLu32
	}

	return ofb
}

// OFBNames returns a human-readable list of set OFB flag names.
func OFBNames(ofb uint32) []string {
	all := []struct {
		bit  uint32
		name string
	}{
		{OFBALuGPR, "ALU_GPR"},
		{OFBALuIX, "ALU_IX"},
		{OFBHLArith, "HL_ARITH"},
		{OFBHLPtr, "HL_PTR"},
		{OFBDEPtr, "DE_PTR"},
		{OFBMul8Safe, "MUL8_SAFE"},
		{OFBDJNZFree, "DJNZ_FREE"},
		{OFBIXBridge, "IX_BRIDGE"},
		{OFBAFree, "A_FREE"},
		{OFBEXXSplit, "EXX_SPLIT"},
		{OFBBCFree, "BC_FREE"},
		{OFBDEFree, "DE_FREE"},
		{OFBADCSBCSrc, "ADCSBC_SRC"},
		{OFBSPFree, "SP_FREE"},
		{OFBHLHLu32, "HLHL_U32"},
	}
	var names []string
	for _, f := range all {
		if ofb&f.bit != 0 {
			names = append(names, f.name)
		}
	}
	return names
}

// OFBTable holds precomputed OFB values indexed parallel to an .enr file.
type OFBTable struct {
	Values []uint32 // OFB per entry; 0 for infeasible entries
}

// LoadOFB reads a .ofb sidecar binary file.
// The sidecar must correspond to the same .enr file used for table lookup.
func LoadOFB(path string) (*OFBTable, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "OFB1" {
		return nil, fmt.Errorf("bad magic: %q (expected OFB1)", magic)
	}

	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("read n_entries: %w", err)
	}

	values := make([]uint32, n)
	if err := binary.Read(f, binary.LittleEndian, values); err != nil {
		return nil, fmt.Errorf("read OFB values: %w", err)
	}

	return &OFBTable{Values: values}, nil
}

// Get returns the OFB value for the given entry index.
// Returns 0 if index is out of range or entry is infeasible.
func (t *OFBTable) Get(index int) uint32 {
	if index < 0 || index >= len(t.Values) {
		return 0
	}
	return t.Values[index]
}

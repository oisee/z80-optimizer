// Materialize virtual mul16 ops into real Z80 instructions.
package mulopt

// Materialize16 expands virtual ops into real Z80 assembly lines.
// Virtual ops: "ADD HL,HL", "ADD HL,BC", "LD C,A" pass through as-is.
// "SWAP_HL" → LD H,L / LD L,0
// "SUB HL,BC" → OR A / SBC HL,BC
func Materialize16(virtualOps []string) []string {
	var real []string
	for _, op := range virtualOps {
		switch op {
		case "SWAP_HL":
			real = append(real, "LD H, L", "LD L, 0")
		case "SUB HL,BC":
			real = append(real, "OR A", "SBC HL, BC")
		default:
			real = append(real, op)
		}
	}
	return real
}

// Emit8 returns Z80 assembly lines for multiplying A by constant k.
// Returns nil if k is not in the table.
// If bPreserve is true, only returns B-preserving sequences.
func Emit8(k int, bPreserve bool) []string {
	var e *Mul8Entry
	if bPreserve {
		e = Lookup8Safe(k)
	} else {
		e = Lookup8(k)
	}
	if e == nil {
		return nil
	}
	return e.Ops
}

// Emit16 returns real Z80 assembly for 16-bit multiply A*k→HL.
// Includes the implicit preamble: LD L, A / LD H, 0
func Emit16(k int, includePreamble bool) []string {
	e := Lookup16(k)
	if e == nil {
		return nil
	}
	var lines []string
	if includePreamble {
		lines = append(lines, "LD L, A", "LD H, 0")
	}
	lines = append(lines, Materialize16(e.Ops)...)
	return lines
}

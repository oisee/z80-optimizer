package cpu

// Z80 flag bit positions in the F register.
const (
	FlagC uint8 = 0x01 // Carry
	FlagN uint8 = 0x02 // Subtract
	FlagP uint8 = 0x04 // Parity/Overflow
	FlagV       = FlagP // Overflow (same bit as Parity)
	Flag3 uint8 = 0x08 // Undocumented bit 3
	FlagH uint8 = 0x10 // Half-carry
	Flag5 uint8 = 0x20 // Undocumented bit 5
	FlagZ uint8 = 0x40 // Zero
	FlagS uint8 = 0x80 // Sign
)

// Precomputed flag tables, ported from remogatto/z80.
var (
	// sz53Table: S, Z, 5, 3 flags for each byte value
	Sz53Table [256]uint8
	// sz53pTable: sz53 with parity flag included
	Sz53pTable [256]uint8
	// parityTable: parity flag for each byte value
	ParityTable [256]uint8

	// Half-carry and overflow lookup tables (from remogatto/z80).
	// For 8-bit ops: index from bits 3 of {result, arg1, arg2}.
	// For 16-bit ops (ADC/SBC HL): index from bits 11 and 15, same tables.
	HalfcarryAddTable = [8]uint8{0, FlagH, FlagH, FlagH, 0, 0, 0, FlagH}
	HalfcarrySubTable = [8]uint8{0, 0, FlagH, 0, FlagH, 0, FlagH, FlagH}
	OverflowAddTable  = [8]uint8{0, 0, 0, FlagV, FlagV, 0, 0, 0}
	OverflowSubTable  = [8]uint8{0, FlagV, 0, 0, 0, 0, FlagV, 0}
)

func init() {
	for i := 0; i < 256; i++ {
		Sz53Table[i] = uint8(i) & (Flag3 | Flag5 | FlagS)

		// Count parity (number of 1 bits)
		j := uint8(i)
		parity := uint8(0)
		for k := 0; k < 8; k++ {
			parity ^= j & 1
			j >>= 1
		}
		if parity == 0 {
			ParityTable[i] = FlagP
		}
		Sz53pTable[i] = Sz53Table[i] | ParityTable[i]
	}
	// Zero flag for value 0
	Sz53Table[0] |= FlagZ
	Sz53pTable[0] |= FlagZ
}

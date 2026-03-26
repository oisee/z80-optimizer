// Package gpugen generates GPU compute kernels for brute-force search
// across multiple backends (CUDA, Metal, OpenCL) from a single ISA definition.
//
// Usage:
//
//	kernel := gpugen.Emit(gpugen.Z80Mul, gpugen.Metal)
//	// → returns Metal shader source as string
package gpugen

// Reg describes a register in the simulated CPU state.
type Reg struct {
	Name string // C identifier (e.g. "a", "carry")
	Type Type   // U8, U16, Bool
}

// Type is a register type in the simulated state.
type Type int

const (
	U8   Type = iota
	U16
	Bool
)

// Op describes one instruction in the search pool.
type Op struct {
	Name string // Human-readable (e.g. "ADD A,A")
	Cost int    // T-states / cycles
	Body string // C fragment executed in switch case. Uses register names directly.
}

// Var is a temporary variable used in Op.Body fragments.
// Declared at the top of the exec_op function in generated code.
type Var struct {
	Name string
	Type Type
}

// ISA defines an instruction set for brute-force search.
type ISA struct {
	Name       string   // Identifier (e.g. "z80_mulopt")
	State      []Reg    // Simulated registers
	Ops        []Op     // Instruction pool
	Locals     []Var    // Temp variables used in Op Bodies
	QuickCheck []uint8  // Test inputs for quick rejection (e.g. {1, 2, 127, 255})
	InputReg   string   // Register loaded with input value (e.g. "a")
	InputRegs  []string // Additional registers loaded with input (e.g. ["a","l"] for arith16)
	OutputReg  string   // Register checked for result (e.g. "a")
	OutputExpr string   // C expression for result if OutputReg is not a register (e.g. "((UINT16)h << 8) | l")
	OutputType Type     // Return type (default U8; use U16 for 16-bit results)
}

// Masked returns a new ISA with only the ops at the given indices enabled.
func (isa ISA) Masked(indices []int) ISA {
	out := isa
	out.Ops = make([]Op, 0, len(indices))
	for _, idx := range indices {
		if idx >= 0 && idx < len(isa.Ops) {
			out.Ops = append(out.Ops, isa.Ops[idx])
		}
	}
	return out
}

// MaskedByBits returns a new ISA with ops enabled where mask[i]=='1'.
func (isa ISA) MaskedByBits(mask string) ISA {
	var indices []int
	for i := 0; i < len(mask) && i < len(isa.Ops); i++ {
		if mask[i] == '1' {
			indices = append(indices, i)
		}
	}
	return isa.Masked(indices)
}

// MaskedDisable returns a new ISA with the named ops removed.
func (isa ISA) MaskedDisable(names []string) ISA {
	nameSet := make(map[string]bool, len(names))
	for _, n := range names {
		nameSet[n] = true
	}
	var indices []int
	for i, op := range isa.Ops {
		if !nameSet[op.Name] {
			indices = append(indices, i)
		}
	}
	return isa.Masked(indices)
}

// Backend selects the GPU target language.
type Backend int

const (
	CUDA   Backend = iota
	Metal
	OpenCL
	Vulkan
)

func (b Backend) String() string {
	switch b {
	case CUDA:
		return "CUDA"
	case Metal:
		return "Metal"
	case OpenCL:
		return "OpenCL"
	case Vulkan:
		return "Vulkan"
	default:
		return "?"
	}
}

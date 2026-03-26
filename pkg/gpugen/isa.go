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

// ISA defines an instruction set for brute-force search.
type ISA struct {
	Name       string   // Identifier (e.g. "z80_mulopt")
	State      []Reg    // Simulated registers
	Ops        []Op     // Instruction pool
	QuickCheck []uint8  // Test inputs for quick rejection (e.g. {1, 2, 127, 255})
	InputReg   string   // Register loaded with input value (e.g. "a")
	OutputReg  string   // Register checked for result (e.g. "a")
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

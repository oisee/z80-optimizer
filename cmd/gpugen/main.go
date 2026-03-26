// gpugen — generate GPU kernels for brute-force search from ISA definitions.
//
// Usage:
//
//	gpugen -isa z80 -backend metal > z80_mulopt.metal
//	gpugen -isa 6502 -backend cuda > 6502_mulopt.cu
//	gpugen -isa z80 -backend all   # emit all backends to files
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/oisee/z80-optimizer/pkg/gpugen"
)

var isas = map[string]gpugen.ISA{
	"z80":         gpugen.Z80Mul,
	"z80_arith16": gpugen.Z80Arith16,
	"6502":        gpugen.MOS6502Mul,
}

var backends = map[string]gpugen.Backend{
	"cuda":   gpugen.CUDA,
	"metal":  gpugen.Metal,
	"opencl": gpugen.OpenCL,
	"vulkan": gpugen.Vulkan,
}

var extensions = map[gpugen.Backend]string{
	gpugen.CUDA:   ".cu",
	gpugen.Metal:  ".metal",
	gpugen.OpenCL: ".cl",
	gpugen.Vulkan: ".comp",
}

func main() {
	isaName := flag.String("isa", "z80", "ISA: z80, 6502")
	backendName := flag.String("backend", "metal", "Backend: cuda, metal, opencl, vulkan, all, host-header")
	outDir := flag.String("out", "", "Output directory (for -backend all)")
	flag.Parse()

	isa, ok := isas[*isaName]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown ISA: %s (available: z80, z80_arith16, 6502)\n", *isaName)
		os.Exit(1)
	}

	if *backendName == "host-header" {
		fmt.Print(gpugen.EmitHostHeader(isa))
		return
	}

	if *backendName == "all" {
		dir := *outDir
		if dir == "" {
			dir = "."
		}
		for name, b := range backends {
			src := gpugen.Emit(isa, b)
			path := fmt.Sprintf("%s/%s%s", dir, isa.Name, extensions[b])
			os.WriteFile(path, []byte(src), 0644)
			fmt.Fprintf(os.Stderr, "Wrote %s (%d bytes)\n", path, len(src))
			_ = name
		}
		return
	}

	b, ok := backends[*backendName]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown backend: %s (available: cuda, metal, opencl, all)\n", *backendName)
		os.Exit(1)
	}

	fmt.Print(gpugen.Emit(isa, b))
}

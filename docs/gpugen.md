# gpugen — Multi-Backend GPU Kernel Generator

Generate GPU compute kernels for brute-force search from a single ISA definition.
One definition → CUDA, Metal, OpenCL, Vulkan.

## Quick Start

```bash
# Generate kernels
go run cmd/gpugen/main.go -isa z80 -backend metal  > z80_mulopt.metal
go run cmd/gpugen/main.go -isa z80 -backend cuda   > z80_mulopt.cu
go run cmd/gpugen/main.go -isa z80 -backend opencl > z80_mulopt.cl
go run cmd/gpugen/main.go -isa z80 -backend vulkan > z80_mulopt.comp

# Generate all backends at once
go run cmd/gpugen/main.go -isa z80 -backend all -out metal/

# Available ISAs: z80, 6502
go run cmd/gpugen/main.go -isa 6502 -backend metal > mos6502_mulopt.metal
```

## Compile & Run

### Metal (macOS)
```bash
xcrun -sdk macosx metal -O2 -c z80_mulopt.metal -o z80.air
xcrun -sdk macosx metallib z80.air -o /tmp/mulopt.metallib
clang -O2 -framework Metal -framework Foundation -o metal_mulopt metal/mulopt_host.m
./metal_mulopt --k 27 --max-len 8
```

### CUDA (NVIDIA)
```bash
nvcc -O3 -c z80_mulopt.cu -o z80.o  # kernel only, needs host
```

### Vulkan (Linux, AMD/NVIDIA)
```bash
glslc --target-env=vulkan1.3 -fshader-stage=compute z80_mulopt.comp -o z80.spv
gcc -O2 -o vk_mulopt vulkan/mulopt_host.c -lvulkan -lm
./vk_mulopt --k 27 --max-len 8
```

### OpenCL (macOS, Linux)
```bash
# macOS
gcc -O2 -o ocl_mulopt opencl/z80_mulopt.c -framework OpenCL
# Linux
gcc -O2 -o ocl_mulopt opencl/z80_mulopt.c -lOpenCL -lm
```

## Architecture

```
pkg/gpugen/
  isa.go       — ISA, Op, Reg, Backend types
  z80.go       — Z80 14-op multiply pool
  mos6502.go   — 6502 14-op multiply pool
  emit.go      — multi-backend code generator
  emit_test.go — tests (all ISA × backend combos)

cmd/gpugen/    — CLI tool

metal/         — Metal host + hand-written shader (reference)
vulkan/        — Vulkan compute host
opencl/        — OpenCL host (works on macOS + Linux AMD)
```

## Defining a New ISA

Add a new file in `pkg/gpugen/`:

```go
package gpugen

var MyISA = ISA{
    Name:       "myisa_mulopt",     // must be valid C identifier
    InputReg:   "a",                // register loaded with test input
    OutputReg:  "a",                // register checked for result
    QuickCheck: []uint8{1, 2, 127, 255},  // fast rejection inputs
    State: []Reg{
        {Name: "a", Type: U8},
        {Name: "x", Type: U8},
        {Name: "carry", Type: Bool},
    },
    Ops: []Op{
        {Name: "SHL", Cost: 2, Body: `carry = (a & 0x80) != 0; a = a << 1;`},
        {Name: "ADD", Cost: 3, Body: `r = (UINT16)a + x; carry = r > 0xFF; a = (UINT8)r;`},
        // ...
    },
}
```

Then register it in `cmd/gpugen/main.go` and generate for any backend.

### Body DSL

Op `Body` is a C fragment. Use these placeholders for cross-backend types:
- `UINT8` → `uint8_t` (CUDA), `uchar` (Metal/OpenCL), `uint` (Vulkan)
- `UINT16` → `uint16_t` (CUDA), `ushort` (Metal/OpenCL), `uint` (Vulkan)
- `INT16` → `int16_t` (CUDA), `short` (Metal/OpenCL), `int` (Vulkan)

Available temp vars (pre-declared): `r` (u16), `c` (u16), `bit` (u8).

Register names from `State` are used directly in the Body.

## Benchmark Results (2026-03-26)

### M2 MacBook Air — len-7, 254 constants, 63 solved

| Backend | Time | Notes |
|---------|------|-------|
| Generated Metal | 2:00.5 | gpugen output |
| Hand-written Metal | 1:58.7 | reference implementation |
| OpenCL | 1:59.5 | hand-written |

**Generated code within 1% of hand-written. 100% correctness (63/63 same optimal length).**

### AMD RX 580 (i3) — x27, len-8

| Backend | Time | Notes |
|---------|------|-------|
| Vulkan (generated) | 4.2s | RADV driver, stable |
| OpenCL (hand-written) | 79s + crash | rusticl, unstable |

**Vulkan 19x faster than OpenCL on AMD.** RADV >> rusticl for compute.

### Cross-platform verification

All backends produce identical optimal lengths. Different sequences at same length
are expected (multiple optimal solutions exist).

## Backend Differences

| Feature | CUDA | Metal | OpenCL | Vulkan |
|---------|------|-------|--------|--------|
| 8-bit types | uint8_t | uchar | uchar | uint (masked) |
| Params | pointers | thread & | pointers | inout |
| Thread ID | blockIdx×blockDim+threadIdx | [[thread_position_in_grid]] | get_global_id(0) | gl_GlobalInvocationID.x |
| Atomics | atomicMin() | atomic_fetch_min_explicit() | atomic_min() | atomicMin() |
| Shader format | .cu source | .metallib binary | .cl source | .spv (SPIR-V) |

## Future Work

- Add `Reads`/`Writes` annotations to Ops for automatic dead-write pruning
- Parameterize search pattern (mulopt, divmod, peephole) beyond just ISA
- Vulkan host auto-detection of best device
- Meet-in-the-middle search pattern

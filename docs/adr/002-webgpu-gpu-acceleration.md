# ADR 002: WebGPU GPU Acceleration

## Status

Accepted (infrastructure complete, runtime integration in progress)

## Context

The brute-force superoptimizer discovered 602K optimizations for length-2 sequences in 3h16m on CPU. Length-3 has 74.8B targets × 4,215 candidates = 315T comparisons — months on CPU. The Z80 executor is an ideal GPU workload: 10-byte fixed state, no memory access, no branching, embarrassingly parallel.

The original plan (docs/GenPlan.md) called for CUDA. We chose WebGPU instead for portability and simpler build requirements.

## Decision

We port the QuickCheck pipeline to a WebGPU compute shader (WGSL), using the `go-webgpu/webgpu` zero-CGo Go bindings to wgpu-native (Vulkan backend on Linux).

### Why WebGPU over CUDA

| Criterion | CUDA | WebGPU/WGSL |
|-----------|------|-------------|
| GPU vendor lock-in | NVIDIA only | Any Vulkan/Metal/DX12 GPU |
| Build complexity | CUDA toolkit + CGo | Zero-CGo (goffi FFI) |
| Language | CUDA C (separate toolchain) | WGSL (embedded as Go string) |
| Portability | Linux/Windows + NVIDIA | Linux/macOS/Windows, any GPU |
| Ecosystem | Mature, well-documented | Newer, less tooling |
| Performance | Slightly higher (hardware-specific) | ~95% of CUDA for compute |

The primary factor was **build simplicity**: WebGPU requires only a single shared library (`libwgpu_native.so`) with no CGo, no CUDA toolkit, no nvcc compiler. The WGSL shader is embedded in the Go binary via `//go:embed`.

### Architecture

```
Host (CPU)                              GPU (Vulkan via wgpu-native)
─────────────                           ──────────────────────────────
For each target:                        Compute Shader (@workgroup_size(256)):
  1. Compute target fingerprint           thread[i] = candidate i
  2. Convert 80B → 96B GPU format         decode candidate from u32 encoding
  3. Upload fingerprint to GPU            execute on 8 test vectors
  4. Dispatch compute shader              compare with target fingerprint
  5. Read back hit bitmap                 atomicOr 1 bit per candidate
  6. ExhaustiveCheck hits on CPU
```

### Buffer layout (5 bindings)

| Binding | Type | Size | Purpose |
|---------|------|------|---------|
| 0 | storage, read | 800 B | Packed lookup tables (Sz53, Sz53p, Parity, halfcarry, overflow) |
| 1 | storage, read | N × 4 B | Candidate instructions (u32: opcode<<16 \| imm) |
| 2 | storage, read | 96 B | Target fingerprint (3 × u32 per test vector × 8) |
| 3 | storage, read_write | ceil(N/32) × 4 B | Results bitmap (1 bit per candidate, atomic) |
| 4 | uniform | 16 B | Params (candidate_count, seq_len, num_candidates_per_pos, dead_flags) |

### Fingerprint format conversion

Go fingerprint: 80 bytes (10 bytes per test vector × 8 vectors: A, F, B, C, D, E, H, L, SP_hi, SP_lo).

GPU fingerprint: 96 bytes (3 × u32 per test vector × 8 vectors):
- w0: A<<24 | F<<16 | B<<8 | C
- w1: D<<24 | E<<16 | H<<8 | L
- w2: SP_hi<<24 | SP_lo<<16 | 0 | 0

The GPU format packs state into u32 words for efficient comparison with WGSL bitwise ops.

### Shader design

The WGSL compute shader (`pkg/gpu/shader/z80_quickcheck.wgsl`, 1171 lines) is a complete Z80 executor:

- **394-case opcode switch**: direct port of Go `cpu.Exec()`. WGSL `switch` on `u32` produces a GPU jump table.
- **20 ALU helper functions**: `exec_add`, `exec_sub`, `exec_and`, `exec_xor`, `exec_cp`, `exec_inc`, `exec_dec`, `exec_daa`, 8 CB-prefix rotate/shift ops, `exec_add_hl`, `exec_adc_hl`, `exec_sbc_hl`, `exec_bit`.
- **Packed lookup tables**: 256-entry tables stored as `array<u32, 64>` with byte extraction: `(table[idx >> 2] >> ((idx & 3) * 8)) & 0xFF`.
- **8 hardcoded test vectors**: same as Go `search.TestVectors`.
- **Dead-flags masking**: configurable via `params.dead_flags`, masks F register bits during comparison.

### Go host code

| File | Purpose |
|------|---------|
| `pkg/gpu/device.go` | wgpu device/adapter/queue lifecycle, `Available()` check |
| `pkg/gpu/pipeline.go` | Compute pipeline from embedded WGSL (auto-layout) |
| `pkg/gpu/dispatch.go` | Buffer management, fingerprint conversion, GPU dispatch + readback |
| `pkg/gpu/search.go` | GPU search loop: enumerate targets → GPU QuickCheck → CPU ExhaustiveCheck |
| `pkg/gpu/gpu_test.go` | Unit tests (packing, encoding) + GPU integration tests |

### Multi-length candidate support

For length-1: each GPU thread tests `candidates[gid.x]`.

For length-2: each GPU thread decodes two instructions from a flat index:
```
i0 = idx / num_candidates_per_pos
i1 = idx % num_candidates_per_pos
```
Total threads = N². For 4,215 candidates: ~17.8M threads, dispatched as ~69,500 workgroups of 256.

## Consequences

### Positive

- Zero-CGo build: no CUDA toolkit, no C compiler, single `libwgpu_native.so` dependency
- Portable: works on any GPU with Vulkan/Metal/DX12 support
- WGSL shader embedded in Go binary via `//go:embed` — single binary deployment
- Exact same ALU semantics as CPU executor (394 opcodes, same flag tables)
- Auto-layout pipeline simplifies bind group setup
- Dead-flags support built into shader from day one

### Negative

- wgpu-native is still maturing; encountered SIGSEGV in `RequestAdapter` (ABI compatibility issue between go-webgpu v0.4.0 and wgpu-native v27)
- Slightly less GPU performance than hand-tuned CUDA (shared memory, warp-level ops not available in WGSL)
- 1171-line shader is hard to debug — no GPU printf, limited tooling
- go-webgpu/goffi requires `CGO_ENABLED=0` (counterintuitively)

### Neutral

- The ~800 bytes of lookup tables fit easily in GPU cache
- 394-opcode switch compiles to efficient jump table on modern GPU compilers
- Atomic bitmap output minimizes GPU→CPU bandwidth (1 bit per candidate)

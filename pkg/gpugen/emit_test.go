package gpugen

import (
	"strings"
	"testing"
)

func TestEmitZ80AllBackends(t *testing.T) {
	for _, backend := range []Backend{CUDA, Metal, OpenCL, Vulkan} {
		src := Emit(Z80Mul, backend)
		if !strings.Contains(src, "NUM_OPS") {
			t.Errorf("%s: missing NUM_OPS", backend)
		}
		if !strings.Contains(src, "exec_op") {
			t.Errorf("%s: missing exec_op", backend)
		}
		if !strings.Contains(src, "run_seq") {
			t.Errorf("%s: missing run_seq", backend)
		}
		// Vulkan uses void main(), others use named kernel
		if backend == Vulkan {
			if !strings.Contains(src, "void main()") {
				t.Errorf("%s: missing main", backend)
			}
		} else if !strings.Contains(src, "z80_mulopt_kernel") {
			t.Errorf("%s: missing kernel", backend)
		}
		// Check op count
		if !strings.Contains(src, "#define NUM_OPS 14") {
			t.Errorf("%s: wrong NUM_OPS", backend)
		}
	}
}

func TestEmit6502AllBackends(t *testing.T) {
	for _, backend := range []Backend{CUDA, Metal, OpenCL, Vulkan} {
		src := Emit(MOS6502Mul, backend)
		if backend == Vulkan {
			if !strings.Contains(src, "void main()") {
				t.Errorf("%s: missing main", backend)
			}
		} else if !strings.Contains(src, "mos6502_mulopt_kernel") {
			t.Errorf("%s: missing 6502 kernel", backend)
		}
		if !strings.Contains(src, "#define NUM_OPS 14") {
			t.Errorf("%s: wrong NUM_OPS for 6502", backend)
		}
	}
}

func TestMetalCompiles(t *testing.T) {
	src := Emit(Z80Mul, Metal)
	// Verify Metal-specific constructs
	if !strings.Contains(src, "using namespace metal") {
		t.Error("missing Metal header")
	}
	if !strings.Contains(src, "thread uchar &a") {
		t.Error("missing Metal thread address space")
	}
	if !strings.Contains(src, "atomic_fetch_min_explicit") {
		t.Error("missing Metal atomic")
	}
}

func TestCUDACompiles(t *testing.T) {
	src := Emit(Z80Mul, CUDA)
	if !strings.Contains(src, "__global__") {
		t.Error("missing CUDA __global__")
	}
	if !strings.Contains(src, "__device__") {
		t.Error("missing CUDA __device__")
	}
	if !strings.Contains(src, "atomicMin") {
		t.Error("missing CUDA atomicMin")
	}
}

func TestOpenCLCompiles(t *testing.T) {
	src := Emit(Z80Mul, OpenCL)
	if !strings.Contains(src, "__kernel") {
		t.Error("missing OpenCL __kernel")
	}
	if !strings.Contains(src, "get_global_id") {
		t.Error("missing OpenCL get_global_id")
	}
}

func TestVulkanCompiles(t *testing.T) {
	src := Emit(Z80Mul, Vulkan)
	if !strings.Contains(src, "#version 450") {
		t.Error("missing GLSL version")
	}
	if !strings.Contains(src, "GL_EXT_shader_explicit_arithmetic_types_int64") {
		t.Error("missing int64 extension")
	}
	if !strings.Contains(src, "gl_GlobalInvocationID") {
		t.Error("missing Vulkan thread ID")
	}
	if !strings.Contains(src, "layout(binding = 0)") {
		t.Error("missing Vulkan SSBO binding")
	}
	if !strings.Contains(src, "atomicMin") {
		t.Error("missing Vulkan atomicMin")
	}
	if !strings.Contains(src, "void main()") {
		t.Error("missing Vulkan main")
	}
	// GLSL uses inout, not pointers
	if strings.Contains(src, "uint *a") {
		t.Error("Vulkan should use inout, not pointers")
	}
	// Must mask to 8 bits in verification
	if !strings.Contains(src, "& 0xFF") {
		t.Error("missing 8-bit masking for Vulkan")
	}
}

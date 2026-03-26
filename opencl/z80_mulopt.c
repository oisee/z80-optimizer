// z80_mulopt.c — OpenCL brute-force optimal constant multiplication for Z80
//
// Port of cuda/z80_mulopt_fast.cu for AMD GPUs via Mesa rusticl/OpenCL.
// Same 14-op reduced pool, same QuickCheck + full verification.
//
// Build: gcc -O2 -o z80_mulopt_ocl opencl/z80_mulopt.c -lOpenCL -lm
// Usage: z80_mulopt_ocl [--max-len 8] [--k 42] [--json]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <CL/cl.h>

#define NUM_OPS 14
#define MAX_LEN 12

// Op names for output
static const char *opNames[] = {
    "ADD A,A", "ADD A,B", "SUB B", "LD B,A",
    "ADC A,B", "ADC A,A", "SBC A,B", "SBC A,A",
    "SRL A", "RLA", "RRA", "RLCA", "RRCA", "NEG"
};

static const uint8_t opCosts[] = {
    4,4,4,4,4,4,4,4, 8, 4,4,4,4, 8
};

// OpenCL kernel source — the Z80 executor runs on GPU
static const char *kernel_source =
"#define NUM_OPS 14\n"
"#define OP_ADD_AA 0\n"
"#define OP_ADD_AB 1\n"
"#define OP_SUB_B  2\n"
"#define OP_LD_BA  3\n"
"#define OP_ADC_AB 4\n"
"#define OP_ADC_AA 5\n"
"#define OP_SBC_AB 6\n"
"#define OP_SBC_AA 7\n"
"#define OP_SRL_A  8\n"
"#define OP_RLA    9\n"
"#define OP_RRA   10\n"
"#define OP_RLCA  11\n"
"#define OP_RRCA  12\n"
"#define OP_NEG   13\n"
"\n"
"void exec_op(uchar op, uchar *a, uchar *b, int *carry) {\n"
"    uint r; int c; uchar bit;\n"
"    switch (op) {\n"
"    case OP_ADD_AA: r = (uint)*a + *a; *carry = r > 0xFF; *a = (uchar)r; break;\n"
"    case OP_ADD_AB: r = (uint)*a + *b; *carry = r > 0xFF; *a = (uchar)r; break;\n"
"    case OP_SUB_B:  *carry = (*a < *b); *a = *a - *b; break;\n"
"    case OP_LD_BA:  *b = *a; break;\n"
"    case OP_ADC_AB: c = *carry ? 1 : 0; r = (uint)*a + *b + c; *carry = r > 0xFF; *a = (uchar)r; break;\n"
"    case OP_ADC_AA: c = *carry ? 1 : 0; r = (uint)*a + *a + c; *carry = r > 0xFF; *a = (uchar)r; break;\n"
"    case OP_SBC_AB: c = *carry ? 1 : 0; *carry = ((int)*a - (int)*b - c) < 0; *a = *a - *b - (uchar)c; break;\n"
"    case OP_SBC_AA: c = *carry ? 1 : 0; *carry = c > 0; *a = -(uchar)c; break;\n"
"    case OP_SRL_A:  *carry = (*a & 1) != 0; *a = *a >> 1; break;\n"
"    case OP_RLA:    bit = *carry ? 1 : 0; *carry = (*a & 0x80) != 0; *a = (*a << 1) | bit; break;\n"
"    case OP_RRA:    bit = *carry ? 0x80 : 0; *carry = (*a & 1) != 0; *a = (*a >> 1) | bit; break;\n"
"    case OP_RLCA:   *carry = (*a & 0x80) != 0; *a = (*a << 1) | (*a >> 7); break;\n"
"    case OP_RRCA:   *carry = (*a & 1) != 0; *a = (*a >> 1) | (*a << 7); break;\n"
"    case OP_NEG:    *carry = (*a != 0); *a = (uchar)(0 - *a); break;\n"
"    }\n"
"}\n"
"\n"
"uchar run_seq(uchar *ops, int len, uchar input) {\n"
"    uchar a = input, b = 0;\n"
"    int carry = 0;\n"
"    for (int i = 0; i < len; i++) exec_op(ops[i], &a, &b, &carry);\n"
"    return a;\n"
"}\n"
"\n"
"__kernel void mulopt_kernel(uchar k, int seqLen, ulong offset, ulong count,\n"
"                            __global uint *bestScore, __global ulong *bestIdx) {\n"
"    ulong tid = get_global_id(0);\n"
"    if (tid >= count) return;\n"
"\n"
"    ulong seqIdx = offset + tid;\n"
"    uchar ops[12];\n"
"    ulong tmp = seqIdx;\n"
"    for (int i = seqLen - 1; i >= 0; i--) {\n"
"        ops[i] = (uchar)(tmp % NUM_OPS);\n"
"        tmp /= NUM_OPS;\n"
"    }\n"
"\n"
"    // QuickCheck: 4 test values\n"
"    if (run_seq(ops, seqLen, 1) != (uchar)(1 * k)) return;\n"
"    if (run_seq(ops, seqLen, 2) != (uchar)(2 * k)) return;\n"
"    if (run_seq(ops, seqLen, 127) != (uchar)(127 * k)) return;\n"
"    if (run_seq(ops, seqLen, 255) != (uchar)(255 * k)) return;\n"
"\n"
"    // Full verification\n"
"    for (int input = 0; input < 256; input++) {\n"
"        if (run_seq(ops, seqLen, (uchar)input) != (uchar)(input * k)) return;\n"
"    }\n"
"\n"
"    // Compute cost\n"
"    __constant uchar costs[] = {4,4,4,4,4,4,4,4,8,4,4,4,4,8};\n"
"    uint cost = 0;\n"
"    for (int i = 0; i < seqLen; i++) cost += costs[ops[i]];\n"
"    uint score = ((uint)seqLen << 16) | cost;\n"
"\n"
"    atomic_min(bestScore, score);\n"
"    if (score <= *bestScore) {\n"
"        // Store index (race possible, but we verify on CPU)\n"
"        bestIdx[0] = seqIdx;\n"
"    }\n"
"}\n";

// Host helpers
static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; i++) r *= base;
    return r;
}

static void decode_seq(uint64_t idx, int len, uint8_t *ops) {
    for (int i = len - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(idx % NUM_OPS);
        idx /= NUM_OPS;
    }
}

// CPU verification (same logic as GPU)
static uint8_t cpu_run_seq(uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    int carry = 0;
    for (int i = 0; i < len; i++) {
        uint16_t r; int c; uint8_t bit;
        switch (ops[i]) {
        case 0:  r=a+a; carry=r>0xFF; a=(uint8_t)r; break;
        case 1:  r=a+b; carry=r>0xFF; a=(uint8_t)r; break;
        case 2:  carry=(a<b); a=a-b; break;
        case 3:  b=a; break;
        case 4:  c=carry?1:0; r=a+b+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 5:  c=carry?1:0; r=a+a+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 6:  c=carry?1:0; carry=((int)a-(int)b-c)<0; a=a-b-(uint8_t)c; break;
        case 7:  c=carry?1:0; carry=c>0; a=-(uint8_t)c; break;
        case 8:  carry=a&1; a=a>>1; break;
        case 9:  bit=carry?1:0; carry=(a&0x80)!=0; a=(a<<1)|bit; break;
        case 10: bit=carry?0x80:0; carry=a&1; a=(a>>1)|bit; break;
        case 11: carry=(a&0x80)!=0; a=(a<<1)|(a>>7); break;
        case 12: carry=a&1; a=(a>>1)|(a<<7); break;
        case 13: carry=(a!=0); a=(uint8_t)(0-a); break;
        }
    }
    return a;
}

typedef struct {
    int k, length, tstates;
    uint8_t ops[MAX_LEN];
    int found;
} MulResult;

static MulResult solve_k(cl_context ctx, cl_command_queue queue, cl_kernel kernel,
                         int k, int maxLen) {
    MulResult result = {.k = k, .found = 0};

    cl_mem d_bestScore = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, NULL);
    cl_mem d_bestIdx   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint64_t), NULL, NULL);

    uint32_t initScore = 0xFFFFFFFF;
    uint64_t initIdx = 0;

    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);

        clEnqueueWriteBuffer(queue, d_bestScore, CL_TRUE, 0, 4, &initScore, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_bestIdx,   CL_TRUE, 0, 8, &initIdx,   0, NULL, NULL);

        uint8_t kk = (uint8_t)k;
        size_t batchSize = 256 * 65535;
        uint64_t offset = 0;

        while (offset < total) {
            uint64_t count = total - offset;
            if (count > batchSize) count = batchSize;

            clSetKernelArg(kernel, 0, sizeof(uint8_t),  &kk);
            clSetKernelArg(kernel, 1, sizeof(int),      &len);
            clSetKernelArg(kernel, 2, sizeof(uint64_t), &offset);
            clSetKernelArg(kernel, 3, sizeof(uint64_t), &count);
            clSetKernelArg(kernel, 4, sizeof(cl_mem),   &d_bestScore);
            clSetKernelArg(kernel, 5, sizeof(cl_mem),   &d_bestIdx);

            size_t globalSize = ((count + 255) / 256) * 256;
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            clFinish(queue);

            offset += count;
        }

        // Read result
        uint32_t bestScore;
        uint64_t bestIdx;
        clEnqueueReadBuffer(queue, d_bestScore, CL_TRUE, 0, 4, &bestScore, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_bestIdx,   CL_TRUE, 0, 8, &bestIdx,   0, NULL, NULL);

        if (bestScore != 0xFFFFFFFF) {
            result.found = 1;
            result.length = bestScore >> 16;
            result.tstates = bestScore & 0xFFFF;
            decode_seq(bestIdx, len, result.ops);

            // CPU verify
            int ok = 1;
            for (int inp = 0; inp < 256 && ok; inp++) {
                if (cpu_run_seq(result.ops, len, (uint8_t)inp) != (uint8_t)(inp * k))
                    ok = 0;
            }
            if (!ok) {
                fprintf(stderr, "WARNING: GPU result for x%d failed CPU verify!\n", k);
                result.found = 0;
            }
            break;
        }
    }

    clReleaseMemObject(d_bestScore);
    clReleaseMemObject(d_bestIdx);
    return result;
}

int main(int argc, char *argv[]) {
    int maxLen = 8, singleK = 0, jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) singleK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) jsonMode = 1;
        else { fprintf(stderr, "Usage: z80_mulopt_ocl [--max-len 8] [--k 42] [--json]\n"); return 1; }
    }

    // OpenCL init
    cl_platform_id plat;
    cl_device_id dev;
    clGetPlatformIDs(1, &plat, NULL);
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) {
        fprintf(stderr, "No GPU found, trying CPU...\n");
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }

    char devName[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
    fprintf(stderr, "OpenCL device: %s\n", devName);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &err);

    // Build kernel
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(prog, "mulopt_kernel", &err);

    // Run
    int startK = singleK > 0 ? singleK : 2;
    int endK   = singleK > 0 ? singleK : 255;
    int solved = 0;

    if (jsonMode) printf("[\n");

    for (int k = startK; k <= endK; k++) {
        MulResult r = solve_k(ctx, queue, kernel, k, maxLen);
        if (r.found) {
            solved++;
            if (jsonMode) {
                printf("  {\"k\": %d, \"ops\": [", k);
                for (int i = 0; i < r.length; i++) {
                    printf("%s\"%s\"", i ? "," : "", opNames[r.ops[i]]);
                }
                printf("], \"length\": %d, \"tstates\": %d}%s\n",
                       r.length, r.tstates, (k < endK) ? "," : "");
            } else {
                printf("x%d:", k);
                for (int i = 0; i < r.length; i++) printf(" %s", opNames[r.ops[i]]);
                printf(" (%d insts, %dT)\n", r.length, r.tstates);
            }
        }
        if (!singleK)
            fprintf(stderr, "x%d/%d (%d solved)...", k, endK, solved);
    }
    fprintf(stderr, "\nDone: %d/%d constants solved\n", solved, endK - startK + 1);

    if (jsonMode) printf("]\n");

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}

// mulopt_host.m — Metal host for Z80 constant multiply search
//
// Build:
//   xcrun -sdk macosx metal -O2 -c metal/mulopt.metal -o /tmp/mulopt.air
//   xcrun -sdk macosx metallib /tmp/mulopt.air -o /tmp/mulopt.metallib
//   clang -O2 -framework Metal -framework Foundation -o metal_mulopt metal/mulopt_host.m
//
// Usage: metal_mulopt [--max-len 8] [--k 42] [--json]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int NUM_OPS = 14;
#define MAX_LEN 12
#define MAX_OPS 128

// Default op names (z80_mul 14-op pool). Overridden by --ops-file.
static const char *defaultOpNames[] = {
    "ADD A,A", "ADD A,B", "SUB B", "LD B,A",
    "ADC A,B", "ADC A,A", "SBC A,B", "SBC A,A",
    "SRL A", "RLA", "RRA", "RLCA", "RRCA", "NEG"
};

static const char *dynOpNames[MAX_OPS];
static const char **opNames = defaultOpNames;
static int opNamesCount = 14;

// Load op names from text file (one name per line)
static void loadOpNames(const char *path) {
    static char buf[8192];
    static char *ptrs[MAX_OPS];
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open ops file: %s\n", path); return; }
    int n = 0;
    char *p = buf;
    while (n < MAX_OPS && fgets(p, (int)(buf + sizeof(buf) - p), f)) {
        size_t len = strlen(p);
        if (len > 0 && p[len-1] == '\n') p[len-1] = 0;
        if (p[0] == 0) continue;
        ptrs[n] = p;
        dynOpNames[n] = p;
        p += strlen(p) + 1;
        n++;
    }
    fclose(f);
    opNames = dynOpNames;
    opNamesCount = n;
    NUM_OPS = n;
    fprintf(stderr, "Loaded %d op names from %s\n", n, path);
}

typedef struct __attribute__((packed)) {
    uint32_t k;
    int32_t seqLen;
    uint64_t offset;
    uint64_t count;
} MuloptArgs;

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

// CPU verification
static uint8_t cpu_run_seq(uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    int carry = 0;
    for (int i = 0; i < len; i++) {
        uint16_t r; uint8_t bit, c;
        switch (ops[i]) {
        case 0:  r=a+a; carry=r>0xFF; a=(uint8_t)r; break;
        case 1:  r=a+b; carry=r>0xFF; a=(uint8_t)r; break;
        case 2:  carry=(a<b); a=a-b; break;
        case 3:  b=a; break;
        case 4:  c=carry?1:0; r=a+b+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 5:  c=carry?1:0; r=a+a+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 6:  c=carry?1:0; carry=((int)a-(int)b-c)<0; a=a-b-c; break;
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        int maxLen = 8, minLen = 1, singleK = 0, jsonMode = 0, skipCpuVerify = 0;
        const char *skipFile = NULL;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--min-len") && i+1 < argc) minLen = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--k") && i+1 < argc) singleK = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--json")) jsonMode = 1;
            else if (!strcmp(argv[i], "--skip") && i+1 < argc) skipFile = argv[++i];
            else if (!strcmp(argv[i], "--num-ops") && i+1 < argc) NUM_OPS = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--no-verify")) skipCpuVerify = 1;
            else if (!strcmp(argv[i], "--ops-file") && i+1 < argc) loadOpNames(argv[++i]);
        }

        // Load skip set (already-solved constants from previous JSON)
        int skipSet[256] = {0};
        int skipCount = 0;
        if (skipFile) {
            FILE *sf = fopen(skipFile, "r");
            if (sf) {
                char buf[1024*1024];
                size_t n = fread(buf, 1, sizeof(buf)-1, sf);
                buf[n] = 0; fclose(sf);
                // Quick parse: find "k": N patterns
                char *p = buf;
                while ((p = strstr(p, "\"k\"")) != NULL) {
                    p += 3;
                    while (*p && (*p < '0' || *p > '9')) p++;
                    int kk = atoi(p);
                    if (kk >= 2 && kk <= 255) { skipSet[kk] = 1; skipCount++; }
                }
                fprintf(stderr, "Skipping %d already-solved constants\n", skipCount);
            }
        }

        // Metal setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }
        fprintf(stderr, "Metal device: %s\n", [[device name] UTF8String]);

        // Load compiled metallib
        NSError *error = nil;
        NSString *libPath = @"/tmp/mulopt.metallib";
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        if (!library) {
            fprintf(stderr, "Failed to load metallib: %s\n", [[error description] UTF8String]);
            return 1;
        }

        // Try kernel names: generated arith16, generated mulopt, hand-written
        id<MTLFunction> function = [library newFunctionWithName:@"z80_arith16_kernel"];
        if (!function) function = [library newFunctionWithName:@"z80_mulopt_kernel"];
        if (!function) function = [library newFunctionWithName:@"mulopt_kernel"];
        if (!function) { fprintf(stderr, "Kernel not found\n"); return 1; }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { fprintf(stderr, "Pipeline error: %s\n", [[error description] UTF8String]); return 1; }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSUInteger maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadGroupSize = maxThreadsPerGroup > 256 ? 256 : maxThreadsPerGroup;

        fprintf(stderr, "Thread group size: %lu, max threads/group: %lu\n",
                (unsigned long)threadGroupSize, (unsigned long)maxThreadsPerGroup);

        // Buffers
        id<MTLBuffer> argsBuf = [device newBufferWithLength:sizeof(MuloptArgs) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scoreBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> idxLoBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> idxHiBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        int startK = singleK > 0 ? singleK : 2;
        int endK   = singleK > 0 ? singleK : 255;
        int solved = 0;

        if (jsonMode) printf("[\n");

        int toSearch = 0;
        for (int k = startK; k <= endK; k++) {
            if (!skipSet[k]) toSearch++;
        }
        fprintf(stderr, "Searching %d constants, len %d..%d\n", toSearch, minLen, maxLen);
        int searched = 0;

        for (int k = startK; k <= endK; k++) {
            if (skipSet[k]) continue;
            searched++;
            MulResult result = {.k = k, .found = 0};

            for (int len = minLen; len <= maxLen; len++) {
                uint64_t total = ipow(NUM_OPS, len);

                // Reset best
                *(uint32_t *)[scoreBuf contents] = 0xFFFFFFFF;
                *(uint32_t *)[idxLoBuf contents] = 0;
                *(uint32_t *)[idxHiBuf contents] = 0;

                uint64_t batchSize = (uint64_t)threadGroupSize * 65535;
                uint64_t offset = 0;

                while (offset < total) {
                    uint64_t count = total - offset;
                    if (count > batchSize) count = batchSize;

                    MuloptArgs *args = (MuloptArgs *)[argsBuf contents];
                    args->k = k;
                    args->seqLen = len;
                    args->offset = offset;
                    args->count = count;

                    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:argsBuf offset:0 atIndex:0];
                    [encoder setBuffer:scoreBuf offset:0 atIndex:1];
                    [encoder setBuffer:idxLoBuf offset:0 atIndex:2];
                    [encoder setBuffer:idxHiBuf offset:0 atIndex:3];

                    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
                    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
                    [encoder endEncoding];
                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];

                    offset += count;
                }

                uint32_t bestScore = *(uint32_t *)[scoreBuf contents];
                if (bestScore != 0xFFFFFFFF) {
                    uint32_t lo = *(uint32_t *)[idxLoBuf contents];
                    uint32_t hi = *(uint32_t *)[idxHiBuf contents];
                    uint64_t bestIdx = ((uint64_t)hi << 32) | lo;

                    result.found = 1;
                    result.length = len;
                    result.tstates = bestScore & 0xFFFF;
                    decode_seq(bestIdx, len, result.ops);

                    // CPU verify (only for default 14-op z80_mul pool)
                    if (!skipCpuVerify) {
                        int ok = 1;
                        for (int inp = 0; inp < 256 && ok; inp++) {
                            if (cpu_run_seq(result.ops, len, (uint8_t)inp) != (uint8_t)(inp * k))
                                ok = 0;
                        }
                        if (!ok) {
                            fprintf(stderr, "WARNING: GPU result for x%d failed CPU verify!\n", k);
                            result.found = 0;
                            continue;
                        }
                    }
                    break;
                }
            }

            if (result.found) {
                solved++;
                if (jsonMode) {
                    printf("  {\"k\": %d, \"ops\": [", k);
                    for (int i = 0; i < result.length; i++)
                        printf("%s\"%s\"", i ? "," : "", (result.ops[i] < opNamesCount ? opNames[result.ops[i]] : "?"));
                    printf("], \"length\": %d, \"tstates\": %d}%s\n",
                           result.length, result.tstates, (k < endK) ? "," : "");
                } else {
                    printf("x%d:", k);
                    for (int i = 0; i < result.length; i++) printf(" %s", (result.ops[i] < opNamesCount ? opNames[result.ops[i]] : "?"));
                    printf(" (%d insts, %dT)\n", result.length, result.tstates);
                }
            }
            if (!singleK)
                fprintf(stderr, "\r[%d/%d] x%d (%d solved)...", searched, toSearch, k, solved);
        }
        fprintf(stderr, "\nDone: %d/%d constants solved\n", solved, endK - startK + 1);
        if (jsonMode) printf("]\n");
    }
    return 0;
}

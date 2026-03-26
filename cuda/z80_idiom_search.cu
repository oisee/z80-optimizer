// z80_idiom_search.cu — GPU brute-force search for Z80 idiom patterns
// Reuses mulopt's 14-op pool but with arbitrary A→A target functions.
// Build: nvcc -O3 -o z80_idiom_search z80_idiom_search.cu
// Usage: z80_idiom_search --idiom abs --max-len 10

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_OPS 14

// Same executor as mulopt_fast
__device__ void exec_op(uint8_t op, uint8_t &a, uint8_t &b, bool &carry) {
    uint16_t r; int c; uint8_t bit;
    switch (op) {
    case 0: r=a+a; carry=r>0xFF; a=(uint8_t)r; break;
    case 1: r=a+b; carry=r>0xFF; a=(uint8_t)r; break;
    case 2: carry=(a<b); a=a-b; break;
    case 3: b=a; break;
    case 4: c=carry?1:0; r=a+b+c; carry=r>0xFF; a=(uint8_t)r; break;
    case 5: c=carry?1:0; r=a+a+c; carry=r>0xFF; a=(uint8_t)r; break;
    case 6: c=carry?1:0; carry=((int)a-(int)b-c)<0; a=a-b-(uint8_t)c; break;
    case 7: c=carry?1:0; carry=c>0; a=-(uint8_t)c; break;
    case 8: carry=a&1; a=a>>1; break;
    case 9: bit=carry?1:0; carry=(a&0x80)!=0; a=(a<<1)|bit; break;
    case 10: bit=carry?0x80:0; carry=a&1; a=(a>>1)|bit; break;
    case 11: carry=(a&0x80)!=0; a=(a<<1)|(a>>7); break;
    case 12: carry=a&1; a=(a>>1)|(a<<7); break;
    case 13: carry=(a!=0); a=(uint8_t)(0-a); break;
    }
}

__device__ uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    bool carry = false;
    for (int i = 0; i < len; i++) exec_op(ops[i], a, b, carry);
    return a;
}

// Target function table (256 entries: target[input] = expected output)
__constant__ uint8_t d_target[256];

__constant__ uint8_t opCost[NUM_OPS] = {4,4,4,4,4,4,4,4,8,4,4,4,4,8};

__global__ void idiom_kernel(int seqLen, uint64_t offset, uint64_t count,
                              uint32_t *bestScore, uint64_t *bestIdx) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint64_t seqIdx = offset + tid;
    uint8_t ops[12];
    uint64_t tmp = seqIdx;
    for (int i = seqLen - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(tmp % NUM_OPS);
        tmp /= NUM_OPS;
    }
    
    // QuickCheck: 4 test inputs
    if (run_seq(ops, seqLen, 0) != d_target[0]) return;
    if (run_seq(ops, seqLen, 1) != d_target[1]) return;
    if (run_seq(ops, seqLen, 127) != d_target[127]) return;
    if (run_seq(ops, seqLen, 255) != d_target[255]) return;
    
    // Full verify
    for (int i = 0; i < 256; i++) {
        if (run_seq(ops, seqLen, (uint8_t)i) != d_target[i]) return;
    }
    
    uint16_t cost = 0;
    for (int i = 0; i < seqLen; i++) cost += opCost[ops[i]];
    uint32_t score = ((uint32_t)seqLen << 16) | cost;
    
    uint32_t old = atomicMin(bestScore, score);
    if (score <= old) atomicExch((unsigned long long*)bestIdx, (unsigned long long)seqIdx);
}

static const char *opNames[] = {
    "ADD A,A","ADD A,B","SUB B","LD B,A","ADC A,B","ADC A,A",
    "SBC A,B","SBC A,A","SRL A","RLA","RRA","RLCA","RRCA","NEG"
};

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

// Generate target tables
static void gen_target(const char *name, uint8_t *tgt) {
    for (int i = 0; i < 256; i++) {
        uint8_t a = (uint8_t)i;
        if (!strcmp(name, "abs"))       tgt[i] = (a < 128) ? a : (256 - a);
        else if (!strcmp(name, "sign")) tgt[i] = (a == 0) ? 0 : (a < 128) ? 1 : 255;
        else if (!strcmp(name, "bool")) tgt[i] = (a != 0) ? 1 : 0;
        else if (!strcmp(name, "not"))  tgt[i] = (a == 0) ? 1 : 0;
        else if (!strcmp(name, "nibswap")) tgt[i] = ((a & 0xF) << 4) | ((a >> 4) & 0xF);
        else if (!strcmp(name, "bitrev")) {
            uint8_t r = 0;
            for (int b = 0; b < 8; b++) if (a & (1<<b)) r |= (1<<(7-b));
            tgt[i] = r;
        }
        else if (!strcmp(name, "clz")) {
            uint8_t c = 8;
            for (int b = 7; b >= 0; b--) { if (a & (1<<b)) { c = 7-b; break; } }
            tgt[i] = c;
        }
        else if (!strcmp(name, "popcnt")) {
            uint8_t c = 0;
            for (int b = 0; b < 8; b++) if (a & (1<<b)) c++;
            tgt[i] = c;
        }
        else if (!strcmp(name, "toupper")) tgt[i] = (a >= 0x61 && a <= 0x7A) ? a - 32 : a;
        else if (!strcmp(name, "tolower")) tgt[i] = (a >= 0x41 && a <= 0x5A) ? a + 32 : a;
        else tgt[i] = a; // identity
    }
}

int main(int argc, char *argv[]) {
    int maxLen = 10;
    const char *idiom = "abs";
    int runAll = 0;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--idiom") && i+1 < argc) idiom = argv[++i];
        else if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--all")) runAll = 1;
    }
    
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best, 4); cudaMalloc(&d_idx, 8);
    uint32_t dummy = 0;
    cudaMemcpy(d_best, &dummy, 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    const char *all_idioms[] = {"abs","sign","bool","not","nibswap","bitrev","clz","popcnt","toupper","tolower",NULL};
    
    for (int ii = 0; all_idioms[ii]; ii++) {
        if (!runAll && strcmp(idiom, all_idioms[ii]) != 0) continue;
        
        const char *name = all_idioms[ii];
        uint8_t tgt[256];
        gen_target(name, tgt);
        cudaMemcpyToSymbol(d_target, tgt, 256);
        
        fprintf(stderr, "Searching: %s (max-len %d, %d ops)\n", name, maxLen, NUM_OPS);
        
        uint32_t initScore = 0xFFFFFFFF;
        uint64_t initIdx = 0;
        int found = 0;
        
        for (int len = 1; len <= maxLen; len++) {
            uint64_t total = ipow(NUM_OPS, len);
            if (total > 5000000000000ULL) break;
            
            cudaMemcpy(d_best, &initScore, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, &initIdx, 8, cudaMemcpyHostToDevice);
            
            int bs = 256;
            uint64_t batch = (uint64_t)bs * 65535;
            for (uint64_t off = 0; off < total; off += batch) {
                uint64_t cnt = total - off;
                if (cnt > batch) cnt = batch;
                idiom_kernel<<<(unsigned int)((cnt+bs-1)/bs), bs>>>(len, off, cnt, d_best, d_idx);
                cudaDeviceSynchronize();
            }
            
            uint32_t bestScore;
            uint64_t bestIdx;
            cudaMemcpy(&bestScore, d_best, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&bestIdx, d_idx, 8, cudaMemcpyDeviceToHost);
            
            if (bestScore != 0xFFFFFFFF) {
                found = 1;
                int rlen = bestScore >> 16;
                int rcost = bestScore & 0xFFFF;
                uint8_t ops[12];
                uint64_t tmp = bestIdx;
                for (int i = rlen-1; i >= 0; i--) { ops[i] = tmp % NUM_OPS; tmp /= NUM_OPS; }
                
                printf("%s:", name);
                for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
                printf(" (%d insts, %dT)\n", rlen, rcost);
                fflush(stdout);
                break;
            }
        }
        if (!found) printf("%s: NOT FOUND at len %d\n", name, maxLen);
    }
    
    cudaFree(d_best); cudaFree(d_idx);
    return 0;
}

// Guided division search: 16-bit pool, check A = floor(input/K)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_OPS 6

__device__ uint8_t run_seq(const uint8_t *ops, int len, uint8_t input) {
    uint8_t a=input, b=0, c=0, h=0, l=input;
    for (int i=0; i<len; i++) {
        uint16_t hl, bc, r;
        switch (ops[i]) {
        case 0: // ADD HL,HL (11T)
            hl=((uint16_t)h<<8)|l; r=hl+hl; h=(uint8_t)(r>>8); l=(uint8_t)r; break;
        case 1: // ADD HL,BC (11T) — B=0, C=input initially
            hl=((uint16_t)h<<8)|l; bc=((uint16_t)b<<8)|c; r=hl+bc; h=(uint8_t)(r>>8); l=(uint8_t)r; break;
        case 2: // LD C,A (4T)
            c=a; break;
        case 3: // SHR_HL (16T) — 16-bit shift right
            { uint8_t hbit=h&1; h=h>>1; l=(l>>1)|(hbit<<7); } break;
        case 4: // LD A,H (4T) — extract high byte to A
            a=h; break;
        case 5: // SRL A (8T) — shift A right
            a=a>>1; break;
        }
    }
    return a;  // Result in A!
}

__constant__ uint8_t d_target[256];
__constant__ uint8_t opCost[NUM_OPS] = {11,11,4,16,4,8};

static const char *opNames[] = {"ADD HL,HL","ADD HL,BC","LD C,A","SHR_HL","LD A,H","SRL A"};

__global__ void kernel(int seqLen, uint64_t offset, uint64_t count,
                       uint32_t *bestScore, uint64_t *bestIdx) {
    uint64_t tid=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x;
    if(tid>=count) return;
    uint64_t seqIdx=offset+tid;
    uint8_t ops[20]; uint64_t tmp=seqIdx;
    for(int i=seqLen-1;i>=0;i--){ops[i]=tmp%NUM_OPS;tmp/=NUM_OPS;}
    if(run_seq(ops,seqLen,0)!=d_target[0]) return;
    if(run_seq(ops,seqLen,1)!=d_target[1]) return;
    if(run_seq(ops,seqLen,127)!=d_target[127]) return;
    if(run_seq(ops,seqLen,255)!=d_target[255]) return;
    for(int i=0;i<256;i++) if(run_seq(ops,seqLen,(uint8_t)i)!=d_target[i]) return;
    uint16_t cost=0;
    for(int i=0;i<seqLen;i++) cost+=opCost[ops[i]];
    uint32_t score=((uint32_t)seqLen<<16)|cost;
    atomicMin(bestScore,score);
    if(score<=*bestScore) atomicExch((unsigned long long*)bestIdx,(unsigned long long)seqIdx);
}

static uint64_t ipow(uint64_t b,int e){uint64_t r=1;for(int i=0;i<e;i++)r*=b;return r;}

int main(int argc, char *argv[]) {
    int maxLen=15,divK=3;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--max-len")&&i+1<argc) maxLen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--k")&&i+1<argc) divK=atoi(argv[++i]);
    }
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    uint32_t *d_best; uint64_t *d_idx;
    cudaMalloc(&d_best,4); cudaMalloc(&d_idx,8);
    uint32_t dummy=0; cudaMemcpy(d_best,&dummy,4,cudaMemcpyHostToDevice); cudaDeviceSynchronize();
    
    uint8_t tgt[256];
    for(int i=0;i<256;i++) tgt[i]=(uint8_t)(i/divK);
    cudaMemcpyToSymbol(d_target,tgt,256);
    
    fprintf(stderr,"GUIDED div%d: 6-op 16-bit pool, max-len %d\n",divK,maxLen);
    uint32_t initScore=0xFFFFFFFF; uint64_t initIdx=0;
    
    for(int len=1;len<=maxLen;len++){
        uint64_t total=ipow(NUM_OPS,len);
        if(total>5000000000000ULL) break;
        cudaMemcpy(d_best,&initScore,4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx,&initIdx,8,cudaMemcpyHostToDevice);
        int bs=256; uint64_t batch=(uint64_t)bs*65535;
        for(uint64_t off=0;off<total;off+=batch){
            uint64_t cnt=total-off; if(cnt>batch) cnt=batch;
            kernel<<<(unsigned int)((cnt+bs-1)/bs),bs>>>(len,off,cnt,d_best,d_idx);
            cudaDeviceSynchronize();
        }
        uint32_t bestScore; uint64_t bestIdx;
        cudaMemcpy(&bestScore,d_best,4,cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestIdx,d_idx,8,cudaMemcpyDeviceToHost);
        if(bestScore!=0xFFFFFFFF){
            int rlen=bestScore>>16,rcost=bestScore&0xFFFF;
            uint8_t ops[20]; uint64_t tmp=bestIdx;
            for(int i=rlen-1;i>=0;i--){ops[i]=tmp%NUM_OPS;tmp/=NUM_OPS;}
            printf("div%d:",divK);
            for(int i=0;i<rlen;i++) printf(" %s",opNames[ops[i]]);
            printf(" (%d insts, %dT)\n",rlen,rcost);
            return 0;
        }
        fprintf(stderr,"  len-%d: %llu tested\n",len,(unsigned long long)total);
    }
    printf("div%d: NOT FOUND at len %d\n",divK,maxLen);
}

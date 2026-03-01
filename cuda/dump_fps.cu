// dump_fps.cu — Compute fingerprints on GPU and dump them for verification.
// Build: nvcc -O2 -o cuda/dump_fps cuda/dump_fps.cu
// Usage: echo "OP IMM" | ./cuda/dump_fps
//        Reads opcode+imm pairs from stdin (text), one per line.
//        Outputs each instruction's 80-byte fingerprint as hex.
//        This is for verifying GPU matches Go CPU executor.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// Include all the Z80 definitions from the main file
// (In a real build we'd share a header, but for this standalone test
// we inline the needed parts.)

#define FLAG_C  0x01u
#define FLAG_N  0x02u
#define FLAG_P  0x04u
#define FLAG_V  0x04u
#define FLAG_3  0x08u
#define FLAG_H  0x10u
#define FLAG_5  0x20u
#define FLAG_Z  0x40u
#define FLAG_S  0x80u

__constant__ uint8_t d_sz53[256];
__constant__ uint8_t d_sz53p[256];
__constant__ uint8_t d_parity[256];
__constant__ uint8_t d_halfcarry_add[8];
__constant__ uint8_t d_halfcarry_sub[8];
__constant__ uint8_t d_overflow_add[8];
__constant__ uint8_t d_overflow_sub[8];

static uint8_t h_sz53[256];
static uint8_t h_sz53p[256];
static uint8_t h_parity[256];
static const uint8_t h_halfcarry_add[8] = {0, FLAG_H, FLAG_H, FLAG_H, 0, 0, 0, FLAG_H};
static const uint8_t h_halfcarry_sub[8] = {0, 0, FLAG_H, 0, FLAG_H, 0, FLAG_H, FLAG_H};
static const uint8_t h_overflow_add[8]  = {0, 0, 0, FLAG_V, FLAG_V, 0, 0, 0};
static const uint8_t h_overflow_sub[8]  = {0, FLAG_V, 0, 0, 0, 0, FLAG_V, 0};

static void init_tables() {
    for (int i = 0; i < 256; i++) {
        h_sz53[i] = (uint8_t)(i) & (FLAG_3 | FLAG_5 | FLAG_S);
        uint8_t j = (uint8_t)i, p = 0;
        for (int k = 0; k < 8; k++) { p ^= j & 1; j >>= 1; }
        h_parity[i] = (p == 0) ? FLAG_P : 0;
        h_sz53p[i] = h_sz53[i] | h_parity[i];
    }
    h_sz53[0] |= FLAG_Z;
    h_sz53p[0] |= FLAG_Z;
}

static void upload_tables() {
    cudaMemcpyToSymbol(d_sz53, h_sz53, 256);
    cudaMemcpyToSymbol(d_sz53p, h_sz53p, 256);
    cudaMemcpyToSymbol(d_parity, h_parity, 256);
    cudaMemcpyToSymbol(d_halfcarry_add, h_halfcarry_add, 8);
    cudaMemcpyToSymbol(d_halfcarry_sub, h_halfcarry_sub, 8);
    cudaMemcpyToSymbol(d_overflow_add, h_overflow_add, 8);
    cudaMemcpyToSymbol(d_overflow_sub, h_overflow_sub, 8);
}

struct Z80State {
    uint8_t r[8];
    uint16_t sp;
};

#define REG_A 0
#define REG_F 1
#define REG_B 2
#define REG_C 3
#define REG_D 4
#define REG_E 5
#define REG_H 6
#define REG_L 7

__constant__ Z80State d_test_vectors[8];
static const Z80State h_test_vectors[8] = {
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 0x0000},
    {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 0xFFFF},
    {{0x01, 0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}, 0x1234},
    {{0x80, 0x01, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02}, 0x8000},
    {{0x55, 0x00, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55}, 0x5555},
    {{0xAA, 0x01, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA}, 0xAAAA},
    {{0x0F, 0x00, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F}, 0xFFFE},
    {{0x7F, 0x01, 0x80, 0x7F, 0x80, 0x7F, 0x80, 0x7F}, 0x7FFF},
};

__device__ __constant__ uint8_t LD_DST[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t LD_FULL_SRC[49] = {
    REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
    REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L,
};
__device__ __constant__ uint8_t ALU_SRC[7] = {REG_B, REG_C, REG_D, REG_E, REG_H, REG_L, REG_A};
__device__ __constant__ uint8_t CB_REG[7]  = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t IMM_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};
__device__ __constant__ uint8_t INCDEC_REG[7] = {REG_A, REG_B, REG_C, REG_D, REG_E, REG_H, REG_L};

// Copy exec_instruction and all helpers from z80_quickcheck.cu
// (In a real project, this would be a shared header)
__device__ inline uint8_t bsel(bool cond, uint8_t a, uint8_t b) { return cond ? a : b; }
__device__ void alu_add(Z80State &s, uint8_t val) { uint16_t r=(uint16_t)s.r[0]+val; uint8_t lookup=((s.r[0]&0x88)>>3)|((val&0x88)>>2)|(uint8_t)((r&0x88)>>1); s.r[0]=(uint8_t)r; s.r[1]=bsel(r&0x100,FLAG_C,(uint8_t)0)|d_halfcarry_add[lookup&7]|d_overflow_add[lookup>>4]|d_sz53[s.r[0]]; }
__device__ void alu_adc(Z80State &s, uint8_t val) { uint16_t r=(uint16_t)s.r[0]+val+(s.r[1]&FLAG_C); uint8_t lookup=(uint8_t)(((uint16_t)(s.r[0])&0x88)>>3|((uint16_t)(val)&0x88)>>2|(r&0x88)>>1); s.r[0]=(uint8_t)r; s.r[1]=bsel(r&0x100,FLAG_C,(uint8_t)0)|d_halfcarry_add[lookup&7]|d_overflow_add[lookup>>4]|d_sz53[s.r[0]]; }
__device__ void alu_sub(Z80State &s, uint8_t val) { uint16_t r=(uint16_t)s.r[0]-val; uint8_t lookup=((s.r[0]&0x88)>>3)|((val&0x88)>>2)|(uint8_t)((r&0x88)>>1); s.r[0]=(uint8_t)r; s.r[1]=bsel(r&0x100,FLAG_C,(uint8_t)0)|FLAG_N|d_halfcarry_sub[lookup&7]|d_overflow_sub[lookup>>4]|d_sz53[s.r[0]]; }
__device__ void alu_sbc(Z80State &s, uint8_t val) { uint16_t r=(uint16_t)s.r[0]-val-(s.r[1]&FLAG_C); uint8_t lookup=((s.r[0]&0x88)>>3)|((val&0x88)>>2)|(uint8_t)((r&0x88)>>1); s.r[0]=(uint8_t)r; s.r[1]=bsel(r&0x100,FLAG_C,(uint8_t)0)|FLAG_N|d_halfcarry_sub[lookup&7]|d_overflow_sub[lookup>>4]|d_sz53[s.r[0]]; }
__device__ void alu_and(Z80State &s, uint8_t val) { s.r[0]&=val; s.r[1]=FLAG_H|d_sz53p[s.r[0]]; }
__device__ void alu_xor(Z80State &s, uint8_t val) { s.r[0]^=val; s.r[1]=d_sz53p[s.r[0]]; }
__device__ void alu_or(Z80State &s, uint8_t val) { s.r[0]|=val; s.r[1]=d_sz53p[s.r[0]]; }
__device__ void alu_cp(Z80State &s, uint8_t val) { uint16_t r=(uint16_t)s.r[0]-val; uint8_t lookup=((s.r[0]&0x88)>>3)|((val&0x88)>>2)|(uint8_t)((r&0x88)>>1); s.r[1]=bsel(r&0x100,FLAG_C,bsel(r!=0,(uint8_t)0,FLAG_Z))|FLAG_N|d_halfcarry_sub[lookup&7]|d_overflow_sub[lookup>>4]|(val&(FLAG_3|FLAG_5))|(uint8_t)(r&FLAG_S); }
__device__ void alu_inc(Z80State &s, int reg) { s.r[reg]++; s.r[1]=(s.r[1]&FLAG_C)|bsel(s.r[reg]==0x80,FLAG_V,(uint8_t)0)|bsel((s.r[reg]&0x0F)!=0,(uint8_t)0,FLAG_H)|d_sz53[s.r[reg]]; }
__device__ void alu_dec(Z80State &s, int reg) { s.r[1]=(s.r[1]&FLAG_C)|bsel((s.r[reg]&0x0F)!=0,(uint8_t)0,FLAG_H)|FLAG_N; s.r[reg]--; s.r[1]|=bsel(s.r[reg]==0x7F,FLAG_V,(uint8_t)0)|d_sz53[s.r[reg]]; }
__device__ uint8_t cb_rlc(Z80State &s, uint8_t v) { v=(v<<1)|(v>>7); s.r[1]=(v&FLAG_C)|d_sz53p[v]; return v; }
__device__ uint8_t cb_rrc(Z80State &s, uint8_t v) { s.r[1]=v&FLAG_C; v=(v>>1)|(v<<7); s.r[1]|=d_sz53p[v]; return v; }
__device__ uint8_t cb_rl(Z80State &s, uint8_t v) { uint8_t old=v; v=(v<<1)|(s.r[1]&FLAG_C); s.r[1]=(old>>7)|d_sz53p[v]; return v; }
__device__ uint8_t cb_rr(Z80State &s, uint8_t v) { uint8_t old=v; v=(v>>1)|(s.r[1]<<7); s.r[1]=(old&FLAG_C)|d_sz53p[v]; return v; }
__device__ uint8_t cb_sla(Z80State &s, uint8_t v) { s.r[1]=v>>7; v<<=1; s.r[1]|=d_sz53p[v]; return v; }
__device__ uint8_t cb_sra(Z80State &s, uint8_t v) { s.r[1]=v&FLAG_C; v=(v&0x80)|(v>>1); s.r[1]|=d_sz53p[v]; return v; }
__device__ uint8_t cb_srl(Z80State &s, uint8_t v) { s.r[1]=v&FLAG_C; v>>=1; s.r[1]|=d_sz53p[v]; return v; }
__device__ uint8_t cb_sll(Z80State &s, uint8_t v) { s.r[1]=v>>7; v=(v<<1)|0x01; s.r[1]|=d_sz53p[v]; return v; }
__device__ void exec_bit(Z80State &s, uint8_t val, int bit) { s.r[1]=(s.r[1]&FLAG_C)|FLAG_H|(val&(FLAG_3|FLAG_5)); if((val&(1<<bit))==0) s.r[1]|=FLAG_P|FLAG_Z; if(bit==7&&(val&0x80)) s.r[1]|=FLAG_S; }
__device__ void exec_daa(Z80State &s) { uint8_t add=0,carry=s.r[1]&FLAG_C; if((s.r[1]&FLAG_H)||(s.r[0]&0x0F)>9) add=6; if(carry||s.r[0]>0x99) add|=0x60; if(s.r[0]>0x99) carry=FLAG_C; if(s.r[1]&FLAG_N) alu_sub(s,add); else alu_add(s,add); s.r[1]=(s.r[1]&~(FLAG_C|FLAG_P))|carry|d_parity[s.r[0]]; }
__device__ void exec_add_hl(Z80State &s, uint16_t val) { uint16_t hl=((uint16_t)s.r[6]<<8)|s.r[7]; uint32_t result=(uint32_t)hl+val; uint16_t hc=(hl&0x0FFF)+(val&0x0FFF); s.r[1]=(s.r[1]&(FLAG_S|FLAG_Z|FLAG_P))|bsel(hc&0x1000,FLAG_H,(uint8_t)0)|bsel(result&0x10000,FLAG_C,(uint8_t)0)|((uint8_t)(result>>8)&(FLAG_3|FLAG_5)); s.r[6]=(uint8_t)(result>>8); s.r[7]=(uint8_t)result; }
__device__ void exec_adc_hl(Z80State &s, uint16_t val) { uint16_t hl=((uint16_t)s.r[6]<<8)|s.r[7]; uint32_t carry=s.r[1]&FLAG_C; uint32_t result=(uint32_t)hl+val+carry; uint8_t lookup=(uint8_t)(((uint32_t)(hl&0x8800)>>11)|((uint32_t)(val&0x8800)>>10)|((result&0x8800)>>9)); s.r[6]=(uint8_t)(result>>8); s.r[7]=(uint8_t)result; s.r[1]=bsel(result&0x10000,FLAG_C,(uint8_t)0)|d_overflow_add[lookup>>4]|(s.r[6]&(FLAG_3|FLAG_5|FLAG_S))|d_halfcarry_add[lookup&7]|bsel((s.r[6]|s.r[7])!=0,(uint8_t)0,FLAG_Z); }
__device__ void exec_sbc_hl(Z80State &s, uint16_t val) { uint16_t hl=((uint16_t)s.r[6]<<8)|s.r[7]; uint32_t carry=s.r[1]&FLAG_C; uint32_t result=(uint32_t)hl-val-carry; uint8_t lookup=(uint8_t)(((uint32_t)(hl&0x8800)>>11)|((uint32_t)(val&0x8800)>>10)|((result&0x8800)>>9)); s.r[6]=(uint8_t)(result>>8); s.r[7]=(uint8_t)result; s.r[1]=bsel(result&0x10000,FLAG_C,(uint8_t)0)|FLAG_N|d_overflow_sub[lookup>>4]|(s.r[6]&(FLAG_3|FLAG_5|FLAG_S))|d_halfcarry_sub[lookup&7]|bsel((s.r[6]|s.r[7])!=0,(uint8_t)0,FLAG_Z); }
__device__ uint16_t get_pair(const Z80State &s, int pair) { switch(pair){ case 0:return((uint16_t)s.r[2]<<8)|s.r[3]; case 1:return((uint16_t)s.r[4]<<8)|s.r[5]; case 2:return((uint16_t)s.r[6]<<8)|s.r[7]; case 3:return s.sp; } return 0; }
__device__ void set_pair(Z80State &s, int pair, uint16_t val) { switch(pair){ case 0:s.r[2]=(uint8_t)(val>>8);s.r[3]=(uint8_t)val;break; case 1:s.r[4]=(uint8_t)(val>>8);s.r[5]=(uint8_t)val;break; case 2:s.r[6]=(uint8_t)(val>>8);s.r[7]=(uint8_t)val;break; case 3:s.sp=val;break; } }

__device__ void exec_instruction(Z80State &s, uint16_t op, uint16_t imm) {
    if(op<49){s.r[LD_DST[op/7]]=s.r[LD_FULL_SRC[op]];return;}
    if(op<56){s.r[IMM_REG[op-49]]=(uint8_t)imm;return;}
    if(op<120){int ao=(op-56)/8,si=(op-56)%8;uint8_t val=(si<7)?s.r[ALU_SRC[si]]:(uint8_t)imm;switch(ao){case 0:alu_add(s,val);break;case 1:alu_adc(s,val);break;case 2:alu_sub(s,val);break;case 3:alu_sbc(s,val);break;case 4:alu_and(s,val);break;case 5:alu_xor(s,val);break;case 6:alu_or(s,val);break;case 7:alu_cp(s,val);break;}return;}
    if(op<127){alu_inc(s,INCDEC_REG[op-120]);return;}
    if(op<134){alu_dec(s,INCDEC_REG[op-127]);return;}
    if(op==134){s.r[0]=(s.r[0]<<1)|(s.r[0]>>7);s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_C|FLAG_3|FLAG_5));return;}
    if(op==135){s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&FLAG_C);s.r[0]=(s.r[0]>>1)|(s.r[0]<<7);s.r[1]|=s.r[0]&(FLAG_3|FLAG_5);return;}
    if(op==136){uint8_t old=s.r[0];s.r[0]=(s.r[0]<<1)|(s.r[1]&FLAG_C);s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_3|FLAG_5))|(old>>7);return;}
    if(op==137){uint8_t old=s.r[0];s.r[0]=(s.r[0]>>1)|(s.r[1]<<7);s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_3|FLAG_5))|(old&FLAG_C);return;}
    if(op==138){exec_daa(s);return;}
    if(op==139){s.r[0]^=0xFF;s.r[1]=(s.r[1]&(FLAG_C|FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_3|FLAG_5))|FLAG_N|FLAG_H;return;}
    if(op==140){s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_3|FLAG_5))|FLAG_C;return;}
    if(op==141){uint8_t oldC=s.r[1]&FLAG_C;s.r[1]=(s.r[1]&(FLAG_P|FLAG_Z|FLAG_S))|(s.r[0]&(FLAG_3|FLAG_5));if(oldC)s.r[1]|=FLAG_H;else s.r[1]|=FLAG_C;return;}
    if(op==142){uint8_t old=s.r[0];s.r[0]=0;alu_sub(s,old);return;}
    if(op==143)return;
    if(op>=144&&op<=192){int co=(op-144)/7,reg=CB_REG[(op-144)%7];switch(co){case 0:s.r[reg]=cb_rlc(s,s.r[reg]);break;case 1:s.r[reg]=cb_rrc(s,s.r[reg]);break;case 2:s.r[reg]=cb_rl(s,s.r[reg]);break;case 3:s.r[reg]=cb_rr(s,s.r[reg]);break;case 4:s.r[reg]=cb_sla(s,s.r[reg]);break;case 5:s.r[reg]=cb_sra(s,s.r[reg]);break;case 6:s.r[reg]=cb_srl(s,s.r[reg]);break;}return;}
    if(op==193){s.r[0]=cb_sll(s,s.r[0]);return;}
    if(op>=194&&op<200){s.r[CB_REG[(op-194)+1]]=cb_sll(s,s.r[CB_REG[(op-194)+1]]);return;}
    if(op>=200&&op<256){int idx=op-200;exec_bit(s,s.r[CB_REG[idx%7]],idx/7);return;}
    if(op>=256&&op<312){int idx=op-256;s.r[CB_REG[idx%7]]&=~(1u<<(idx/7));return;}
    if(op>=312&&op<368){int idx=op-312;s.r[CB_REG[idx%7]]|=(1u<<(idx/7));return;}
    if(op>=368&&op<376){int idx=op-368,pair=idx%4;bool dec=idx>=4;uint16_t v=get_pair(s,pair);set_pair(s,pair,dec?v-1:v+1);return;}
    if(op>=376&&op<380){exec_add_hl(s,get_pair(s,op-376));return;}
    if(op==380){uint8_t td=s.r[4],te=s.r[5];s.r[4]=s.r[6];s.r[5]=s.r[7];s.r[6]=td;s.r[7]=te;return;}
    if(op==381){s.sp=((uint16_t)s.r[6]<<8)|s.r[7];return;}
    if(op>=382&&op<386){set_pair(s,op-382,imm);return;}
    if(op>=386&&op<390){exec_adc_hl(s,get_pair(s,op-386));return;}
    if(op>=390&&op<394){exec_sbc_hl(s,get_pair(s,op-390));return;}
}

// Kernel: compute fingerprint for each candidate, write to output
#define FP_SIZE 10
#define NUM_VECTORS 8
#define FP_LEN (FP_SIZE * NUM_VECTORS)

__global__ void fingerprint_kernel(
    const uint32_t* __restrict__ candidates,
    uint8_t* __restrict__ fps_out,  // FP_LEN bytes per candidate
    uint32_t count,
    uint32_t seq_len
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const uint32_t* my_seq = candidates + (uint64_t)tid * seq_len;
    uint8_t* my_fp = fps_out + (uint64_t)tid * FP_LEN;

    for (int v = 0; v < NUM_VECTORS; v++) {
        Z80State s = d_test_vectors[v];
        for (uint32_t i = 0; i < seq_len; i++) {
            uint32_t packed = my_seq[i];
            exec_instruction(s, (uint16_t)(packed & 0xFFFF), (uint16_t)(packed >> 16));
        }
        int off = v * FP_SIZE;
        my_fp[off+0]=s.r[0]; my_fp[off+1]=s.r[1]; my_fp[off+2]=s.r[2]; my_fp[off+3]=s.r[3];
        my_fp[off+4]=s.r[4]; my_fp[off+5]=s.r[5]; my_fp[off+6]=s.r[6]; my_fp[off+7]=s.r[7];
        my_fp[off+8]=(uint8_t)(s.sp>>8); my_fp[off+9]=(uint8_t)s.sp;
    }
}

int main() {
    init_tables();
    upload_tables();
    cudaMemcpyToSymbol(d_test_vectors, h_test_vectors, sizeof(h_test_vectors));

    // Read instructions from stdin: "OP IMM\n" per line
    std::vector<uint32_t> instrs;
    uint32_t op, imm;
    while (scanf("%u %u", &op, &imm) == 2) {
        instrs.push_back(op | (imm << 16));
    }

    uint32_t count = (uint32_t)instrs.size();
    uint32_t seq_len = 1;
    fprintf(stderr, "Computing fingerprints for %u instructions on GPU...\n", count);

    uint32_t *d_candidates;
    uint8_t  *d_fps;
    cudaMalloc(&d_candidates, count * sizeof(uint32_t));
    cudaMalloc(&d_fps, count * FP_LEN);
    cudaMemcpy(d_candidates, instrs.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    fingerprint_kernel<<<gridSize, blockSize>>>(d_candidates, d_fps, count, seq_len);
    cudaDeviceSynchronize();

    uint8_t* h_fps = (uint8_t*)malloc(count * FP_LEN);
    cudaMemcpy(h_fps, d_fps, count * FP_LEN, cudaMemcpyDeviceToHost);

    // Output fingerprints as hex
    for (uint32_t i = 0; i < count; i++) {
        uint32_t packed = instrs[i];
        printf("op=%u imm=%u fp=", packed & 0xFFFF, packed >> 16);
        for (int j = 0; j < FP_LEN; j++)
            printf("%02x", h_fps[i * FP_LEN + j]);
        printf("\n");
    }

    free(h_fps);
    cudaFree(d_candidates);
    cudaFree(d_fps);
    return 0;
}

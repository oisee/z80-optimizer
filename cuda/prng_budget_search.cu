/*
 * prng_budget_search.cu — LFSR-16 AND-cascade: budget-constrained + area-shrinking search
 *
 * Two modes:
 *
 *   KEYFRAME (--keyframe):  first frame, budget=256 (configurable)
 *     Phases blk=4→2→1→1→1; each phase SHRINKS the allowed position area (default 90%/phase).
 *     Artistic effect: coarse seeds scatter everywhere, fine seeds concentrate toward center.
 *
 *   DELTA    (--delta):     subsequent frames, budget=128 (configurable)
 *     Start canvas from --init-canvas (previous frame result).
 *     All phases search FULL area — need to fix changes anywhere in the canvas.
 *     Start from blk=4 (broader stroke on first delta pass).
 *
 * All params configurable:
 *   --budget N         total seeds (default: 256 keyframe, 128 delta)
 *   --shrink F         area shrink factor per phase, keyframe only (default 0.90)
 *   --center-x X       focal point x (default 64, canvas center)
 *   --center-y Y       focal point y (default 48)
 *   --phase-budgets a,b,c,d,e   per-phase seed counts (overrides auto split)
 *
 * Build:
 *   nvcc -O3 -o cuda/prng_budget_search cuda/prng_budget_search.cu -lm
 *
 * Usage:
 *   # Keyframe
 *   ./cuda/prng_budget_search --keyframe --target frame_001.pgm --out kf_001.json
 *
 *   # Delta from previous
 *   ./cuda/prng_budget_search --delta --target frame_002.pgm \
 *       --init-canvas result_001.pgm --out delta_002.json
 *
 *   # Custom budget / shrink
 *   ./cuda/prng_budget_search --keyframe --budget 128 --shrink 0.80 --center-x 56 --center-y 28 \
 *       --target frame_001.pgm --out kf_001.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <algorithm>

#define W       128
#define H        96
#define PS      (W * H / 8)   /* packed bytes */
#define GRID      8            /* position grid snap */
#define BUF_N   768            /* 32×24 blocks */
#define NSEEDS  65535

/* ====================================================================
 * LFSR-16, poly 0xB400
 * ==================================================================== */
__device__ __host__ __forceinline__ uint16_t lfsr16(uint16_t s) {
    uint16_t bit = s & 1u;
    s >>= 1;
    if (bit) s ^= 0xB400u;
    return s;
}

/* ====================================================================
 * makeBuf: 768-block buffer, AND of andN consecutive bits
 * ==================================================================== */
__device__ void makeBuf(uint16_t seed, int warmup, int andN, uint8_t buf[BUF_N]) {
    uint16_t s = seed ? seed : 1u;
    for (int i = 0; i < warmup; i++) s = lfsr16(s);
    for (int i = 0; i < BUF_N; i++) {
        uint16_t acc = 1u;
        for (int k = 0; k < andN; k++) { s = lfsr16(s); acc &= (s & 1u); }
        buf[i] = (uint8_t)acc;
    }
}

/* ====================================================================
 * buildPositionsConstrained:
 *   Generate (ox, oy) grid positions within area_frac of canvas,
 *   centered at (cx, cy). area_frac=1.0 = full canvas.
 *
 *   For blk ≥ 4: patch = 128×96+ → always single position (0,0).
 *   For blk < 4: constrain patch CENTER to [cx±W*area_frac/2, cy±H*area_frac/2].
 * ==================================================================== */
int buildPositionsConstrained(int blk, float area_frac, int cx, int cy,
                               int ox_out[], int oy_out[]) {
    int pw = 32 * blk;
    int ph = 24 * blk;

    /* If patch covers full canvas, only (0,0) is valid */
    if (pw >= W && ph >= H) {
        ox_out[0] = 0; oy_out[0] = 0;
        return 1;
    }

    /* Valid ox range (patch must fit): [0, W-pw], [0, H-ph] */
    int ox_lo = 0, ox_hi = W - pw;
    int oy_lo = 0, oy_hi = H - ph;

    /* Constrain by area_frac, centered at (cx, cy):
       patch center = ox + pw/2; constrain to [cx - W*area_frac/2, cx + W*area_frac/2] */
    if (area_frac < 1.0f) {
        float hw = (W * area_frac) * 0.5f;
        float hh = (H * area_frac) * 0.5f;
        int ox_lo_c = (int)floorf((cx - hw) - pw * 0.5f);
        int ox_hi_c = (int)floorf((cx + hw) - pw * 0.5f);
        int oy_lo_c = (int)floorf((cy - hh) - ph * 0.5f);
        int oy_hi_c = (int)floorf((cy + hh) - ph * 0.5f);
        ox_lo = (ox_lo_c > ox_lo) ? ox_lo_c : ox_lo;
        ox_hi = (ox_hi_c < ox_hi) ? ox_hi_c : ox_hi;
        oy_lo = (oy_lo_c > oy_lo) ? oy_lo_c : oy_lo;
        oy_hi = (oy_hi_c < oy_hi) ? oy_hi_c : oy_hi;
        /* Clamp to valid range */
        if (ox_lo < 0) ox_lo = 0;
        if (oy_lo < 0) oy_lo = 0;
        if (ox_hi > W - pw) ox_hi = W - pw;
        if (oy_hi > H - ph) oy_hi = H - ph;
    }

    /* Snap to grid */
    ox_lo = (ox_lo / GRID) * GRID;
    oy_lo = (oy_lo / GRID) * GRID;

    int n = 0;
    for (int oy = oy_lo; oy <= oy_hi; oy += GRID)
        for (int ox = ox_lo; ox <= ox_hi; ox += GRID) {
            ox_out[n] = ox;
            oy_out[n] = oy;
            n++;
            if (n >= 512) goto done;
        }
done:
    return (n > 0) ? n : 0;
}

/* ====================================================================
 * buildPositionsMultiZone:
 *   Union of constrained position sets from N focal zones.
 *   Deduplicates using a visited grid. Returns total unique positions.
 * ==================================================================== */
int buildPositionsMultiZone(int blk, float area_frac,
                             int zones_x[], int zones_y[], int n_zones,
                             int ox_out[], int oy_out[]) {
    /* visited[gx][gy]: each cell = one GRID-snapped (ox,oy) */
    uint8_t visited[(W/GRID)+2][(H/GRID)+2];
    memset(visited, 0, sizeof(visited));
    int total = 0;
    for (int z = 0; z < n_zones; z++) {
        int tmp_ox[512], tmp_oy[512];
        int n = buildPositionsConstrained(blk, area_frac, zones_x[z], zones_y[z], tmp_ox, tmp_oy);
        for (int i = 0; i < n && total < 512; i++) {
            int gx = tmp_ox[i] / GRID, gy = tmp_oy[i] / GRID;
            if (!visited[gx][gy]) {
                visited[gx][gy] = 1;
                ox_out[total] = tmp_ox[i];
                oy_out[total] = tmp_oy[i];
                total++;
            }
        }
    }
    return total;
}

/* ====================================================================
 * computeBaseErrors (host)
 * ==================================================================== */
void computeBaseErrors(const uint8_t *canvas, const uint8_t *target,
                       int npos, const int *ox_arr, const int *oy_arr,
                       int blk, int *baseErr) {
    int pw = 32 * blk, ph = 24 * blk;
    for (int pi = 0; pi < npos; pi++) {
        int ox = ox_arr[pi], oy = oy_arr[pi], err = 0;
        for (int y = oy; y < oy + ph && y < H; y++)
            for (int x = ox; x < ox + pw && x < W; x++) {
                uint8_t cb = (canvas[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
                uint8_t tb = (target[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
                err += (cb != tb);
            }
        baseErr[pi] = err;
    }
}

/* ====================================================================
 * CUDA kernel: one thread per seed, searches all positions in list.
 * Minimizes delta (improvement), not absolute error — foveal property.
 * ==================================================================== */
__global__ void searchKernel(
    const uint8_t * __restrict__ canvas_d,
    const uint8_t * __restrict__ target_d,
    const int     * __restrict__ baseErr_d,
    const int     * __restrict__ ox_d,
    const int     * __restrict__ oy_d,
    int npos, int blk, int andN, int warmup,
    uint16_t * __restrict__ out_seed,
    int      * __restrict__ out_pos,
    int      * __restrict__ out_err,
    const uint8_t * __restrict__ weight_d   /* per-pixel importance, NULL = flat */
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = tid + 1;
    if (seed > NSEEDS) return;

    uint8_t buf[BUF_N];
    makeBuf((uint16_t)seed, warmup, andN, buf);

    int bestPos = 0, bestDelta = 0x7fffffff;
    for (int pi = 0; pi < npos; pi++) {
        int ox = ox_d[pi], oy = oy_d[pi];
        int delta = 0;
        for (int by = 0; by < 24; by++)
            for (int bx = 0; bx < 32; bx++) {
                if (!buf[by*32+bx]) continue;
                for (int dy = 0; dy < blk; dy++)
                    for (int dx = 0; dx < blk; dx++) {
                        int x = ox + bx*blk + dx, y = oy + by*blk + dy;
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int bidx = y*(W/8) + x/8, bbit = 7-(x%8);
                            uint8_t cb = (canvas_d[bidx] >> bbit) & 1u;
                            uint8_t tb = (target_d[bidx]  >> bbit) & 1u;
                            int w = weight_d ? (int)weight_d[y*W + x] : 1;
                            delta += (cb == tb) ? w : -w;
                        }
                    }
            }
        if (delta < bestDelta) { bestDelta = delta; bestPos = pi; }
    }
    out_seed[tid] = (uint16_t)seed;
    out_pos [tid] = bestPos;
    out_err [tid] = bestDelta;
}

/* ====================================================================
 * applyBuf (host)
 * ==================================================================== */
void applyBuf(uint8_t *canvas, const uint8_t buf[BUF_N], int ox, int oy, int blk) {
    for (int by = 0; by < 24; by++)
        for (int bx = 0; bx < 32; bx++) {
            if (!buf[by*32+bx]) continue;
            for (int dy = 0; dy < blk; dy++)
                for (int dx = 0; dx < blk; dx++) {
                    int x = ox+bx*blk+dx, y = oy+by*blk+dy;
                    if (x>=0&&x<W&&y>=0&&y<H)
                        canvas[y*(W/8)+x/8] ^= (1u<<(7-(x%8)));
                }
        }
}

double lBin(const uint8_t *a, const uint8_t *b) {
    int err=0;
    for(int i=0;i<PS;i++) err+=__builtin_popcount(a[i]^b[i]);
    return (double)err/(W*H);
}

/* ====================================================================
 * PGM I/O
 * ==================================================================== */
int loadPGM(const char *path, uint8_t *out, int *ow, int *oh) {
    FILE *f = fopen(path,"rb"); if(!f) return -1;
    int w,h,maxval; char magic[8];
    fscanf(f,"%7s %d %d %d",magic,&w,&h,&maxval); fgetc(f);
    if(strcmp(magic,"P5")!=0){fclose(f);return -1;}
    *ow=w; *oh=h;
    uint8_t *raw=(uint8_t*)malloc(w*h);
    if((int)fread(raw,1,w*h,f)!=w*h){free(raw);fclose(f);return -1;}
    fclose(f);
    memset(out,0,PS);
    for(int i=0;i<w*h;i++) if(raw[i]>maxval/2) out[i/8]|=(1u<<(7-(i%8)));
    free(raw); return 0;
}

void savePGM(const char *path, const uint8_t *canvas) {
    FILE *f=fopen(path,"wb"); if(!f) return;
    fprintf(f,"P5\n%d %d\n255\n",W,H);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) {
        uint8_t b=(canvas[y*(W/8)+x/8]>>(7-(x%8)))&1u;
        fputc(b?255:0,f);
    }
    fclose(f);
}

/* ====================================================================
 * Phase config
 * ==================================================================== */
typedef struct {
    int    blk;
    int    andN;
    int    budget;      /* seeds for this phase */
    float  area_frac;   /* allowed position area fraction (1.0 = full) */
    char   label[32];
} PhaseConfig;

/* Default keyframe schedule (sum=256) */
static PhaseConfig kf_phases[] = {
    {4, 3,   1, 1.00f, "KF-L0-AND3"},
    {2, 4,   8, 0.90f, "KF-L1-AND4"},
    {1, 5,  16, 0.81f, "KF-L2-AND5"},
    {1, 6,  64, 0.73f, "KF-L3-AND6"},
    {1, 7, 167, 0.66f, "KF-L4-AND7"},
};
static int nkf_phases = 5;

/* Fine keyframe: 8 phases, shrink=0.75 per phase (sum=256) */
static PhaseConfig kf_fine_phases[] = {
    {4, 3,   1, 1.00f, "KF-L0-AND3"},
    {2, 4,   4, 0.75f, "KF-L1-AND4"},
    {1, 4,   8, 0.56f, "KF-L2-AND4"},
    {1, 5,  12, 0.42f, "KF-L3-AND5"},
    {1, 5,  20, 0.32f, "KF-L4-AND5"},
    {1, 6,  32, 0.24f, "KF-L5-AND6"},
    {1, 6,  64, 0.18f, "KF-L6-AND6"},
    {1, 7, 115, 0.13f, "KF-L7-AND7"},
};
static int nkf_fine_phases = 8;

/* Default delta schedule — proportions derived from canonical phase analysis:
 *   AND-3: 0.2%, AND-4: 2%, AND-5: 16%, AND-6: 39%, AND-7: 43%
 * Sum=128. AND-7 is critical — canonical spends 43% here for fine detail.
 * Budget is extended via last phase when --budget > sum. */
static PhaseConfig dt_phases[] = {
    {2, 3,   1, 1.00f, "DT-L0-AND3"},   /* blk=2: 63 positions, coarse bounce */
    {1, 4,   3, 1.00f, "DT-L1-AND4"},
    {1, 5,  20, 1.00f, "DT-L2-AND5"},
    {1, 6,  50, 1.00f, "DT-L3-AND6"},
    {1, 7,  54, 1.00f, "DT-L4-AND7"},   /* canonical: 43% of budget here */
};
static int ndt_phases = 5;

/* ====================================================================
 * CP MODE: Carrier-Payload hierarchical search (CPU-only, u8 budget)
 *
 * Carrier:  blk=8 at (0,0), seeds 1-255 × andN 3-8  → identifies hot zones
 * Payload4: blk=4 at all valid positions, seeds 1-255 × andN 3-6
 * Payload2: blk=2 at all valid positions, seeds 1-255 × andN 4-6
 * Payload1: blk=1 at all valid positions, seeds 1-255 × andN 5-7
 *
 * Payloads scored and applied masked to carrier-active 8×8 blocks.
 *
 * Usage:
 *   ./cuda/prng_budget_search --cp --target frame.pgm [--init-canvas prev.pgm]
 *                             --out result.json --out-pgm result.pgm
 * ==================================================================== */

/* CPU LFSR-16 (device version is __device__ __host__, use it directly) */

static void makeBuf_h(uint16_t seed, int warmup, int andN, uint8_t buf[BUF_N]) {
    uint16_t s = seed ? seed : 1u;
    for (int i = 0; i < warmup; i++) s = lfsr16(s);
    for (int i = 0; i < BUF_N; i++) {
        uint8_t acc = 1;
        for (int k = 0; k < andN; k++) { s = lfsr16(s); acc &= (uint8_t)(s & 1u); }
        buf[i] = acc;
    }
}

static void unpackPS(const uint8_t packed[PS], uint8_t out[W*H]) {
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        out[y*W+x] = (packed[y*(W/8)+x/8] >> (7-(x%8))) & 1u;
}

static void packPS(const uint8_t in[W*H], uint8_t packed[PS]) {
    memset(packed, 0, PS);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        if (in[y*W+x]) packed[y*(W/8)+x/8] |= (1u << (7-(x%8)));
}

/* Apply buf to unpacked canvas. If cmask!=NULL, restrict to carrier-active 8×8 blocks. */
static void applyBuf_h(uint8_t canvas[W*H], const uint8_t buf[BUF_N],
                        int ox, int oy, int blk, const uint8_t cmask[BUF_N]) {
    for (int by = 0; by < 24; by++) for (int bx = 0; bx < 32; bx++) {
        if (!buf[by*32+bx]) continue;
        for (int dy = 0; dy < blk; dy++) for (int dx = 0; dx < blk; dx++) {
            int x = ox+bx*blk+dx, y = oy+by*blk+dy;
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            if (cmask) {
                int cbx = x/8, cby = y/8;
                if (cbx >= 16 || cby >= 12 || !cmask[cby*32+cbx]) continue;
            }
            canvas[y*W+x] ^= 1;
        }
    }
}

/* Score buf against target (returns delta: negative = improvement).
   If cmask!=NULL, count only pixels within carrier-active 8×8 blocks. */
static int scoreBuf_h(const uint8_t canvas[W*H], const uint8_t target[W*H],
                       const uint8_t buf[BUF_N], int ox, int oy, int blk,
                       const uint8_t cmask[BUF_N]) {
    int d = 0;
    for (int by = 0; by < 24; by++) for (int bx = 0; bx < 32; bx++) {
        if (!buf[by*32+bx]) continue;
        for (int dy = 0; dy < blk; dy++) for (int dx = 0; dx < blk; dx++) {
            int x = ox+bx*blk+dx, y = oy+by*blk+dy;
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            if (cmask) {
                int cbx = x/8, cby = y/8;
                if (cbx >= 16 || cby >= 12 || !cmask[cby*32+cbx]) continue;
            }
            d += (canvas[y*W+x] != target[y*W+x]) ? -1 : +1;
        }
    }
    return d;
}

typedef struct { uint16_t seed; int ox, oy, blk, andN; } CPPayload;

int cp_search(const char *target_path, const char *init_canvas_path,
              const char *out_json, const char *out_pgm) {

    uint8_t tgt_pack[PS], cnv_pack[PS];
    int tw, th;
    memset(cnv_pack, 0, PS);

    if (loadPGM(target_path, tgt_pack, &tw, &th) < 0) return 1;
    if (tw != W || th != H) { fprintf(stderr, "Target size mismatch: %dx%d\n", tw, th); return 1; }
    if (init_canvas_path && loadPGM(init_canvas_path, cnv_pack, &tw, &th) < 0) {
        fprintf(stderr, "Warning: init canvas not found, using black\n");
        memset(cnv_pack, 0, PS);
    }

    uint8_t canvas[W*H], target_px[W*H];
    unpackPS(cnv_pack, canvas);
    unpackPS(tgt_pack, target_px);

    int base_err = 0;
    for (int i = 0; i < W*H; i++) base_err += (canvas[i] != target_px[i]);
    printf("CP mode | target=%s | base err=%d (%.2f%%)\n",
           target_path, base_err, 100.0f*base_err/(W*H));

    /* ── Phase 1: Carrier (blk=8, andN 3-8, seeds 1-255) ───────────────── */
    uint8_t carrier_buf[BUF_N], tmp_buf[BUF_N];
    uint16_t cs = 1; int can = 5, c_score = INT_MAX;

    for (int andN = 3; andN <= 8; andN++) {
        for (int s = 1; s <= 255; s++) {
            makeBuf_h((uint16_t)s, 0, andN, tmp_buf);
            /* Score carrier: only 16×12 valid blocks (blk=8 at ox=0,oy=0) */
            int d = 0;
            for (int by = 0; by < 12; by++) for (int bx = 0; bx < 16; bx++) {
                if (!tmp_buf[by*32+bx]) continue;
                for (int dy = 0; dy < 8; dy++) for (int dx = 0; dx < 8; dx++) {
                    int x = bx*8+dx, y = by*8+dy;
                    d += (canvas[y*W+x] != target_px[y*W+x]) ? -1 : +1;
                }
            }
            if (d < c_score) {
                c_score = d; cs = (uint16_t)s; can = andN;
                memcpy(carrier_buf, tmp_buf, BUF_N);
            }
        }
    }

    int n_active = 0;
    for (int i = 0; i < BUF_N; i++) n_active += carrier_buf[i];
    printf("[carrier] seed=%u andN=%d score=%+d active=%d/192 (%.1f%%)\n",
           cs, can, c_score, n_active, 100.0f*n_active/192);

    applyBuf_h(canvas, carrier_buf, 0, 0, 8, NULL);

    int err_c = 0;
    for (int i = 0; i < W*H; i++) err_c += (canvas[i] != target_px[i]);
    printf("[carrier] after: %d px (%.2f%%)\n", err_c, 100.0f*err_c/(W*H));

    /* ── Phases 2-4: Payload (blk=4,2,1, scored within carrier zone) ───── */
    CPPayload payloads[32]; int npl = 0;

    struct { int blk, anL, anH; } pl_cfg[] = {{4,3,6},{2,4,6},{1,5,7}};
    int n_pl_cfg = (int)(sizeof(pl_cfg)/sizeof(pl_cfg[0]));

    for (int ph = 0; ph < n_pl_cfg && npl < 16; ph++) {
        int blk = pl_cfg[ph].blk, anL = pl_cfg[ph].anL, anH = pl_cfg[ph].anH;
        int ox_list[512], oy_list[512];
        int npos = buildPositionsConstrained(blk, 1.0f, W/2, H/2, ox_list, oy_list);

        int best_d = 0; uint16_t bseed = 0; int box = 0, boy = 0, ban = anL;

        for (int s = 1; s <= 255 && npl < 16; s++) {
            for (int pi = 0; pi < npos; pi++) {
                int ox = ox_list[pi], oy = oy_list[pi];
                for (int andN = anL; andN <= anH; andN++) {
                    makeBuf_h((uint16_t)s, 0, andN, tmp_buf);
                    int d = scoreBuf_h(canvas, target_px, tmp_buf, ox, oy, blk, carrier_buf);
                    if (d < best_d) { best_d=d; bseed=(uint16_t)s; box=ox; boy=oy; ban=andN; }
                }
            }
        }

        if (bseed && best_d < 0) {
            makeBuf_h(bseed, 0, ban, tmp_buf);
            applyBuf_h(canvas, tmp_buf, box, boy, blk, carrier_buf);
            payloads[npl].seed=bseed; payloads[npl].ox=box;
            payloads[npl].oy=boy; payloads[npl].blk=blk; payloads[npl].andN=ban;
            npl++;
            int err_pl = 0;
            for (int i = 0; i < W*H; i++) err_pl += (canvas[i] != target_px[i]);
            printf("[payload blk=%d] seed=%u andN=%d (%d,%d) score=%+d | err=%d (%.2f%%)\n",
                   blk, bseed, ban, box, boy, best_d, err_pl, 100.0f*err_pl/(W*H));
        }
    }

    int final_err = 0;
    for (int i = 0; i < W*H; i++) final_err += (canvas[i] != target_px[i]);
    printf("Final: %d seeds, %.2f%% error  (CP: 1 carrier + %d payload)\n",
           1+npl, 100.0f*final_err/(W*H), npl);

    /* Write CP JSON */
    FILE *f = fopen(out_json, "w"); if (!f) { perror(out_json); return 1; }
    fprintf(f, "{\n  \"lfsr16_poly\": \"0xB400\",\n");
    fprintf(f, "  \"canvas_w\": %d,\n  \"canvas_h\": %d,\n", W, H);
    fprintf(f, "  \"mode\": \"delta\",\n  \"budget\": 255,\n");
    fprintf(f, "  \"seeds\": [\n");
    fprintf(f, "    {\"type\":\"cp\",\"cs\":%u,\"cx\":0,\"cy\":0,\"can\":%d,\"ps\":[", cs, can);
    for (int i = 0; i < npl; i++) {
        fprintf(f, "[%u,%d,%d,%d,%d]", payloads[i].seed, payloads[i].ox,
                payloads[i].oy, payloads[i].blk, payloads[i].andN);
        if (i < npl-1) fprintf(f, ",");
    }
    fprintf(f, "]}\n  ]\n}\n");
    fclose(f);
    printf("JSON: %s\n", out_json);

    uint8_t result_pack[PS];
    packPS(canvas, result_pack);
    savePGM(out_pgm, result_pack);
    printf("PGM:  %s\n", out_pgm);

    return 0;
}

/* ====================================================================
 * Seed record + JSON output
 * ==================================================================== */
typedef struct {
    int step; uint16_t seed;
    int ox, oy, blk, andN, warmup;
    float area_frac;
    char label[32];
} SeedRecord;

void writeJSON(const char *path, SeedRecord *recs, int nrecs,
               int budget, int mode_keyframe, float shrink, int cx, int cy) {
    FILE *f=fopen(path,"w"); if(!f) return;
    fprintf(f,"{\n  \"lfsr16_poly\": \"0xB400\",\n");
    fprintf(f,"  \"canvas_w\": %d,\n  \"canvas_h\": %d,\n",W,H);
    fprintf(f,"  \"position_grid\": %d,\n",GRID);
    fprintf(f,"  \"mode\": \"%s\",\n", mode_keyframe?"keyframe":"delta");
    fprintf(f,"  \"budget\": %d,\n  \"shrink\": %.3f,\n",budget,shrink);
    fprintf(f,"  \"center_x\": %d,\n  \"center_y\": %d,\n",cx,cy);
    fprintf(f,"  \"seeds\": [\n");
    for(int i=0;i<nrecs;i++){
        SeedRecord *r=&recs[i];
        fprintf(f,"    {\"step\":%d,\"seed\":%u,\"ox\":%d,\"oy\":%d,"
                "\"blk\":%d,\"and_n\":%d,\"warmup\":%d,\"area_frac\":%.3f,\"label\":\"%s\"}%s\n",
                r->step,r->seed,r->ox,r->oy,r->blk,r->andN,r->warmup,r->area_frac,r->label,
                i<nrecs-1?",":"");
    }
    fprintf(f,"  ]\n}\n");
    fclose(f);
}

/* ====================================================================
 * main
 * ==================================================================== */
int main(int argc, char **argv) {
    const char *target_path = NULL;
    const char *init_canvas = NULL;
    const char *out_json    = "/tmp/cuda_budget_result.json";
    const char *out_pgm     = "/tmp/cuda_budget_result.pgm";
    int   gpu_id      = 0;
    int   mode_kf     = -1;   /* -1=auto, 1=keyframe, 0=delta */
    int   budget      = -1;   /* -1=auto */
    float shrink      = -1.f; /* -1=preset default */
    int   center_x    = W/2;
    int   center_y    = H/2;
    int   verbose     = 0;
    int   preset_fine = 0;    /* --preset fine */
    int   delta_blk   = -1;   /* --delta-blk N: override first delta phase blk */
    int   auto_bounce = 0;    /* --auto-bounce: probe blk=1,2,4 and pick best for L0 */
    const char *weight_map  = NULL; /* --weight-map file.wmap: per-pixel uint8 importance */
    int   auto_weight = 0;          /* --auto-weight: derive weight from canvas/target diff each step */
    int   mode_cp     = 0;          /* --cp: carrier-payload hierarchical search (CPU, u8 budget) */

    /* Multi-zone: up to 8 zones (x,y pairs) */
    int zones_x[8], zones_y[8], n_zones = 0;

    /* Per-phase budget overrides (comma-separated) */
    int   phase_budgets[16]; int n_phase_budgets = 0;

    for(int i=1;i<argc;i++){
        if     (!strcmp(argv[i],"--target")       && i+1<argc) target_path   = argv[++i];
        else if(!strcmp(argv[i],"--init-canvas")  && i+1<argc) init_canvas   = argv[++i];
        else if(!strcmp(argv[i],"--out")          && i+1<argc) out_json      = argv[++i];
        else if(!strcmp(argv[i],"--out-pgm")      && i+1<argc) out_pgm       = argv[++i];
        else if(!strcmp(argv[i],"--gpu")          && i+1<argc) gpu_id        = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--keyframe"))                  mode_kf       = 1;
        else if(!strcmp(argv[i],"--delta"))                     mode_kf       = 0;
        else if(!strcmp(argv[i],"--budget")       && i+1<argc) budget        = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--shrink")       && i+1<argc) shrink        = atof(argv[++i]);
        else if(!strcmp(argv[i],"--center-x")     && i+1<argc) center_x      = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--center-y")     && i+1<argc) center_y      = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--delta-blk")    && i+1<argc) delta_blk     = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--auto-bounce"))               auto_bounce   = 1;
        else if(!strcmp(argv[i],"--verbose"))                   verbose       = 1;
        else if(!strcmp(argv[i],"--weight-map")   && i+1<argc) weight_map    = argv[++i];
        else if(!strcmp(argv[i],"--auto-weight"))               auto_weight   = 1;
        else if(!strcmp(argv[i],"--cp"))                        mode_cp       = 1;
        else if(!strcmp(argv[i],"--preset")       && i+1<argc) {
            if(!strcmp(argv[++i],"fine")) preset_fine = 1;
        }
        /* --zones x1,y1,x2,y2,x3,y3 — comma-separated zone centers */
        else if(!strcmp(argv[i],"--zones")        && i+1<argc) {
            char *tok = strtok(argv[++i], ",");
            while(tok && n_zones < 8) {
                zones_x[n_zones] = atoi(tok);
                tok = strtok(NULL, ",");
                if(!tok) break;
                zones_y[n_zones] = atoi(tok);
                tok = strtok(NULL, ",");
                n_zones++;
            }
        }
        else if(!strcmp(argv[i],"--phase-budgets")&& i+1<argc) {
            char *tok = strtok(argv[++i], ",");
            while(tok && n_phase_budgets < 16) {
                phase_budgets[n_phase_budgets++] = atoi(tok);
                tok = strtok(NULL, ",");
            }
        }
    }

    /* Auto-detect mode from --init-canvas */
    if(mode_kf < 0) mode_kf = (init_canvas == NULL) ? 1 : 0;

    /* Select phase schedule */
    PhaseConfig *phases;
    int nphases;
    if(mode_kf){
        if(preset_fine){ phases=kf_fine_phases; nphases=nkf_fine_phases; }
        else            { phases=kf_phases;      nphases=nkf_phases; }
    } else {
        phases=dt_phases; nphases=ndt_phases;
    }

    /* Override first delta phase blk if requested */
    if(!mode_kf && delta_blk > 0) {
        dt_phases[0].blk = delta_blk;
        /* single position for blk>=4 is handled by buildPositionsConstrained */
    }

    /* Apply shrink to keyframe phases (rebuild area_frac from shrink) */
    if(mode_kf && shrink > 0.f) {
        float af = 1.0f;
        for(int p=0;p<nphases;p++){ phases[p].area_frac = af; af *= shrink; }
    }

    /* Override per-phase budgets if specified */
    if(n_phase_budgets > 0) {
        for(int p=0;p<nphases&&p<n_phase_budgets;p++)
            phases[p].budget = phase_budgets[p];
    }

    /* Total budget cap */
    int total_phase_budget = 0;
    for(int p=0;p<nphases;p++) total_phase_budget += phases[p].budget;
    int budget_cap = (budget < 0) ? total_phase_budget : budget;
    /* If budget > phase sum, extend last phase to fill the gap */
    if(budget_cap > total_phase_budget)
        phases[nphases-1].budget += (budget_cap - total_phase_budget);

    if(!target_path) { fprintf(stderr,"--target required\n"); return 1; }

    /* CP mode: CPU-only, no GPU needed */
    if(mode_cp) return cp_search(target_path, init_canvas, out_json, out_pgm);

    cudaSetDevice(gpu_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,gpu_id);
    printf("GPU %d: %s\n", gpu_id, prop.name);
    float shrink_display = (shrink > 0.f) ? shrink : (preset_fine ? 0.75f : 0.90f);
    if(n_zones > 0) {
        printf("Mode: %s | budget=%d | shrink=%.2f | %d zones:",
               mode_kf?"keyframe":"delta", budget_cap, shrink_display, n_zones);
        for(int z=0;z<n_zones;z++) printf(" (%d,%d)", zones_x[z], zones_y[z]);
        printf("\n");
    } else {
        printf("Mode: %s | budget=%d | shrink=%.2f | center=(%d,%d)%s\n",
               mode_kf?"keyframe":"delta", budget_cap, shrink_display, center_x, center_y,
               preset_fine?" | preset=fine":"");
    }

    /* Load target */
    uint8_t target_bin[PS]; int tw,th;
    if(loadPGM(target_path,target_bin,&tw,&th)<0) return 1;
    if(tw!=W||th!=H){ fprintf(stderr,"Size mismatch: %dx%d\n",tw,th); return 1; }
    printf("Target: %s\n", target_path);

    /* Print phase table */
    printf("\n%-4s  %-4s  %-5s  %-6s  %-7s  %-8s  %s\n",
           "ph","blk","AND-N","budget","area%%","n_pos","label");
    for(int p=0;p<nphases;p++){
        int ox_tmp[512],oy_tmp[512];
        int np;
        if(n_zones > 0)
            np = buildPositionsMultiZone(phases[p].blk, phases[p].area_frac,
                                         zones_x, zones_y, n_zones, ox_tmp, oy_tmp);
        else
            np = buildPositionsConstrained(phases[p].blk, phases[p].area_frac,
                                           center_x, center_y, ox_tmp, oy_tmp);
        printf("  %d   blk=%d  AND-%d  %4d    %5.1f%%   %4d      %s\n",
               p, phases[p].blk, phases[p].andN, phases[p].budget,
               phases[p].area_frac*100.0f, np, phases[p].label);
    }
    printf("\n");

    /* Load weight map (optional) */
    uint8_t weight_h[W*H]; uint8_t *weight_d_ptr = NULL;
    if(weight_map){
        FILE *wf = fopen(weight_map,"rb");
        if(!wf){ fprintf(stderr,"Cannot open --weight-map %s\n",weight_map); return 1; }
        size_t nr = fread(weight_h,1,W*H,wf); fclose(wf);
        if((int)nr != W*H){
            fprintf(stderr,"Weight map size mismatch: got %zu, expected %d\n",nr,W*H);
            return 1;
        }
        cudaMalloc(&weight_d_ptr, W*H);
        cudaMemcpy(weight_d_ptr, weight_h, W*H, cudaMemcpyHostToDevice);
        printf("Weight map: %s  (min=%d max=%d mean=%.1f)\n", weight_map,
               *std::min_element(weight_h,weight_h+W*H),
               *std::max_element(weight_h,weight_h+W*H),
               [&](){ double s=0; for(int i=0;i<W*H;i++) s+=weight_h[i]; return s/(W*H); }());
    }

    /* Allocate GPU */
    uint8_t *canvas_d, *target_d;
    cudaMalloc(&canvas_d,PS); cudaMalloc(&target_d,PS);
    cudaMemset(canvas_d,0,PS);
    cudaMemcpy(target_d,target_bin,PS,cudaMemcpyHostToDevice);

    uint16_t *out_seed_d; int *out_pos_d, *out_err_d;
    cudaMalloc(&out_seed_d, NSEEDS*sizeof(uint16_t));
    cudaMalloc(&out_pos_d,  NSEEDS*sizeof(int));
    cudaMalloc(&out_err_d,  NSEEDS*sizeof(int));
    uint16_t out_seed_h[NSEEDS];
    int      out_pos_h[NSEEDS], out_err_h[NSEEDS];

    int *pos_ox_d, *pos_oy_d, *baseErr_d;
    cudaMalloc(&pos_ox_d, 512*sizeof(int));
    cudaMalloc(&pos_oy_d, 512*sizeof(int));
    cudaMalloc(&baseErr_d,512*sizeof(int));

    /* Init canvas */
    uint8_t canvas_h[PS]; memset(canvas_h,0,PS);
    if(init_canvas){
        int iw,ih;
        if(loadPGM(init_canvas,canvas_h,&iw,&ih)==0 && iw==W && ih==H)
            printf("Init canvas: %s  err=%.2f%%\n",
                   init_canvas, lBin(canvas_h,target_bin)*100.0);
        else{ fprintf(stderr,"Warning: cannot load --init-canvas, starting blank\n"); memset(canvas_h,0,PS); }
    }
    cudaMemcpy(canvas_d, canvas_h, PS, cudaMemcpyHostToDevice);

    const int BLOCK=128, GRID_DIM=(NSEEDS+BLOCK-1)/BLOCK;

    /* --auto-bounce: probe blk=1,2,4 with warmup=0, pick best delta for L0 */
    if(!mode_kf && auto_bounce && delta_blk < 0) {
        int probe_blks[] = {1, 2, 4};
        int best_blk = 2, best_probe_delta = 0x7fffffff;
        printf("Auto-bounce probe: ");
        for(int pi=0; pi<3; pi++){
            int pb = probe_blks[pi];
            int pox[512], poy[512];
            int pnpos = buildPositionsConstrained(pb, 1.0f, center_x, center_y, pox, poy);
            int pbe[512];
            computeBaseErrors(canvas_h, target_bin, pnpos, pox, poy, pb, pbe);
            cudaMemcpy(pos_ox_d, pox, pnpos*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(pos_oy_d, poy, pnpos*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(baseErr_d, pbe, pnpos*sizeof(int), cudaMemcpyHostToDevice);
            searchKernel<<<GRID_DIM,BLOCK>>>(
                canvas_d, target_d, baseErr_d, pos_ox_d, pos_oy_d,
                pnpos, pb, 3/*andN*/, 0/*warmup*/,
                out_seed_d, out_pos_d, out_err_d, weight_d_ptr);
            cudaDeviceSynchronize();
            cudaMemcpy(out_err_h, out_err_d, NSEEDS*sizeof(int), cudaMemcpyDeviceToHost);
            int best_d = 0x7fffffff;
            for(int s=0;s<NSEEDS;s++) if(out_err_h[s]<best_d) best_d=out_err_h[s];
            printf("blk=%d:delta=%d  ", pb, best_d);
            if(best_d < best_probe_delta){ best_probe_delta=best_d; best_blk=pb; }
        }
        printf("→ winner=blk=%d\n", best_blk);
        dt_phases[0].blk = best_blk;
    }

    SeedRecord *seedLog=(SeedRecord*)malloc(budget_cap*sizeof(SeedRecord));
    int nlog=0;

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    int global_step = 0;

    printf("%-6s  %-8s  %-6s  %-5s  %-5s  %-8s  %s\n",
           "step","L_bin","elapsed","ox","oy","seed","label");
    printf("------  --------  ------  -----  -----  --------  ------\n");

    for(int p=0; p<nphases && nlog<budget_cap; p++){
        PhaseConfig *ph = &phases[p];
        int pos_ox_h[512], pos_oy_h[512];
        int npos;
        if(n_zones > 0)
            npos = buildPositionsMultiZone(ph->blk, ph->area_frac,
                                           zones_x, zones_y, n_zones, pos_ox_h, pos_oy_h);
        else
            npos = buildPositionsConstrained(ph->blk, ph->area_frac, center_x, center_y,
                                             pos_ox_h, pos_oy_h);
        if(npos == 0){
            printf("Phase %d (%s): no valid positions, skipping\n", p, ph->label);
            continue;
        }
        cudaMemcpy(pos_ox_d, pos_ox_h, npos*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(pos_oy_d, pos_oy_h, npos*sizeof(int), cudaMemcpyHostToDevice);

        int phase_budget = ph->budget;
        if(nlog + phase_budget > budget_cap) phase_budget = budget_cap - nlog;

        for(int si=0; si<phase_budget; si++){
            int warmup = global_step;

            /* --auto-weight: build weight from current canvas diff.
             * Use 4:1 ratio (wrong:correct).  Much higher ratios cause thrashing:
             * a seed fixing 1 wrong pixel at cost of breaking 200 correct ones
             * would look attractive with 255:1 but makes the canvas far worse.
             * At 4:1, a seed must fix > 25% of the pixels it breaks to be preferred. */
            if(auto_weight){
                if(!weight_d_ptr) cudaMalloc(&weight_d_ptr, W*H);
                for(int y=0;y<H;y++) for(int x=0;x<W;x++){
                    int bidx=y*(W/8)+x/8, bbit=7-(x%8);
                    uint8_t cb=(canvas_h[bidx]>>bbit)&1u;
                    uint8_t tb=(target_bin[bidx]>>bbit)&1u;
                    weight_h[y*W+x] = (cb != tb) ? 4u : 1u;
                }
                cudaMemcpy(weight_d_ptr, weight_h, W*H, cudaMemcpyHostToDevice);
            }

            int baseErr_h[512];
            computeBaseErrors(canvas_h,target_bin,npos,pos_ox_h,pos_oy_h,ph->blk,baseErr_h);
            cudaMemcpy(baseErr_d,baseErr_h,npos*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(canvas_d,canvas_h,PS,cudaMemcpyHostToDevice);

            searchKernel<<<GRID_DIM,BLOCK>>>(
                canvas_d,target_d,baseErr_d,pos_ox_d,pos_oy_d,
                npos,ph->blk,ph->andN,warmup,
                out_seed_d,out_pos_d,out_err_d,weight_d_ptr);
            cudaDeviceSynchronize();

            cudaMemcpy(out_seed_h,out_seed_d,NSEEDS*sizeof(uint16_t),cudaMemcpyDeviceToHost);
            cudaMemcpy(out_pos_h, out_pos_d, NSEEDS*sizeof(int),     cudaMemcpyDeviceToHost);
            cudaMemcpy(out_err_h, out_err_d, NSEEDS*sizeof(int),     cudaMemcpyDeviceToHost);

            int bestSeed=-1, bestPos=0, bestDelta=0x7fffffff;
            for(int s=0;s<NSEEDS;s++){
                if(out_err_h[s]<bestDelta){
                    bestDelta=out_err_h[s]; bestSeed=s; bestPos=out_pos_h[s];
                }
            }

            if(bestSeed<0||bestDelta>=0){ /* no improvement possible */
                printf("  No improvement at step %d (delta=%d), stopping phase.\n",
                       nlog+1, bestDelta);
                break;
            }

            int ox=pos_ox_h[bestPos], oy=pos_oy_h[bestPos];
            uint16_t seed_val=(uint16_t)(bestSeed+1);
            uint8_t buf[BUF_N];
            uint16_t s=seed_val; for(int w=0;w<warmup;w++) s=lfsr16(s);
            { uint16_t ss=s;
              for(int i=0;i<BUF_N;i++){
                uint16_t acc=1u;
                for(int k=0;k<ph->andN;k++){ss=lfsr16(ss);acc&=(ss&1u);}
                buf[i]=(uint8_t)acc;
              }
            }
            applyBuf(canvas_h,buf,ox,oy,ph->blk);
            global_step++;
            nlog++;

            SeedRecord *r=&seedLog[nlog-1];
            r->step=nlog; r->seed=seed_val;
            r->ox=ox; r->oy=oy; r->blk=ph->blk; r->andN=ph->andN;
            r->warmup=warmup; r->area_frac=ph->area_frac;
            strncpy(r->label,ph->label,31);

            /* Log milestone or first of phase */
            if(verbose || si==0 || nlog==budget_cap || nlog%16==0){
                struct timespec tn; clock_gettime(CLOCK_MONOTONIC,&tn);
                double el=tn.tv_sec-t0.tv_sec+(tn.tv_nsec-t0.tv_nsec)*1e-9;
                double err=lBin(canvas_h,target_bin)*100.0;
                printf("%6d  %6.2f%%  %5.1fs  %5d  %5d  %6u    %s\n",
                       nlog,err,el,ox,oy,(unsigned)seed_val,ph->label);
            }
        }
    }

    /* Final stats */
    {
        struct timespec tn; clock_gettime(CLOCK_MONOTONIC,&tn);
        double el=tn.tv_sec-t0.tv_sec+(tn.tv_nsec-t0.tv_nsec)*1e-9;
        double err=lBin(canvas_h,target_bin)*100.0;
        printf("\nFinal: %d seeds, %.3f%% error, %.1fs\n", nlog, err, el);
    }

    /* Save outputs */
    writeJSON(out_json, seedLog, nlog, budget_cap, mode_kf, shrink, center_x, center_y);
    savePGM(out_pgm, canvas_h);
    printf("PGM:  %s\n", out_pgm);

    free(seedLog);
    cudaFree(canvas_d); cudaFree(target_d);
    cudaFree(out_seed_d); cudaFree(out_pos_d); cudaFree(out_err_d);
    cudaFree(pos_ox_d); cudaFree(pos_oy_d); cudaFree(baseErr_d);
    return 0;
}

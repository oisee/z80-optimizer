/*
 * prng_layered_search.cu — Layered LFSR brute-force image search
 *
 * Approach from Mona (Atari 256b) and BB (Introspec, ZX Spectrum 256b):
 * Image = XOR of 64 LFSR-seeded layers, each with its own 16-bit seed.
 * Each layer is brute-forced independently (65536 candidates) against
 * the residual error from all previous layers.
 *
 * Multi-resolution: early layers draw large blocks (coarse shape),
 * later layers draw smaller blocks (fine detail).
 *
 * Build: nvcc -O3 -o cuda/prng_layered_search cuda/prng_layered_search.cu -lm
 * Usage: ./cuda/prng_layered_search --target file.pgm [--layers 64] [--gpu 0]
 *
 * Data budget: 64 layers × 2 bytes = 128 bytes + ~128 bytes Z80 code = 256 byte intro
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define W 128
#define H 96
#define PIXELS (W * H)  /* 12288 */

/* ====== LFSR: 32-bit Galois, same polynomial as Mona ====== */
__device__ __host__ uint32_t lfsr_step(uint32_t state) {
    uint32_t bit = state & 1;
    state >>= 1;
    if (bit) state ^= 0xB4BCD35C;  /* maximal-length polynomial */
    return state;
}

/* ====== Draw one layer: N points at LFSR-determined positions ====== */
/* block_size: 1=pixel, 2=2x2, 4=4x4, 8=8x8 */
/* seed: 16-bit, combined with layer_idx for 32-bit LFSR init */
__device__ void draw_layer(
    uint8_t* canvas,   /* W*H bits packed as bytes (W/8 * H) */
    uint16_t seed,
    int layer_idx,
    int num_points,
    int block_size
) {
    /* Initialize LFSR from seed + layer index */
    uint32_t state = ((uint32_t)seed << 16) | ((uint32_t)(layer_idx * 7 + 0x1337));
    /* Warm up LFSR */
    for (int i = 0; i < 8; i++) state = lfsr_step(state);

    for (int p = 0; p < num_points; p++) {
        state = lfsr_step(state);
        int px = (state >> 0) & 0x7F;   /* 0-127 */
        int py = ((state >> 7) & 0x7F) % H;  /* 0-95 */

        /* Align to block grid */
        px = (px / block_size) * block_size;
        py = (py / block_size) * block_size;

        /* XOR a block_size × block_size block */
        for (int dy = 0; dy < block_size && (py + dy) < H; dy++) {
            for (int dx = 0; dx < block_size && (px + dx) < W; dx++) {
                int x = px + dx;
                int y = py + dy;
                int byte_idx = y * (W / 8) + (x / 8);
                int bit_idx = 7 - (x % 8);
                canvas[byte_idx] ^= (1 << bit_idx);
            }
        }
    }
}

/* ====== Kernel: test all 65536 seeds for one layer ====== */
/* Each thread tests one seed value */
__global__ void search_layer_kernel(
    const uint8_t* __restrict__ current,    /* current canvas (packed bits) */
    const uint8_t* __restrict__ target,     /* target image (packed bits) */
    uint32_t* __restrict__ errors,          /* output: error count per seed */
    int layer_idx,
    int num_points,
    int block_size
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= 65536) return;

    /* Copy current canvas to local buffer */
    uint8_t canvas[PIXELS / 8];  /* 1536 bytes — fits in registers/local mem */
    for (int i = 0; i < PIXELS / 8; i++)
        canvas[i] = current[i];

    /* Draw this layer's contribution */
    draw_layer(canvas, (uint16_t)seed, layer_idx, num_points, block_size);

    /* Count errors vs target */
    uint32_t err = 0;
    for (int i = 0; i < PIXELS / 8; i++) {
        uint8_t diff = canvas[i] ^ target[i];
        err += __popc(diff);  /* popcount of XOR = number of different bits */
    }

    errors[seed] = err;
}

/* ====== Host: layer schedule ====== */
struct LayerSpec {
    int block_size;
    int num_points;
};

void make_schedule(LayerSpec* spec, int num_layers) {
    /* BB-style: early layers = many large points, later = fewer small points
     * Key: num_points DECREASES with layer (opposite of naive!)
     * Early layers set coarse shape, later layers do fine correction */
    for (int i = 0; i < num_layers; i++) {
        int remaining = num_layers - i;
        if (i < num_layers / 8) {
            spec[i].block_size = 8;
            spec[i].num_points = remaining * 2;  /* ~128 down to ~112 */
        } else if (i < num_layers / 4) {
            spec[i].block_size = 4;
            spec[i].num_points = remaining * 2;  /* ~96 down to ~80 */
        } else if (i < num_layers / 2) {
            spec[i].block_size = 2;
            spec[i].num_points = remaining;       /* ~48 down to ~32 */
        } else {
            spec[i].block_size = 1;
            spec[i].num_points = remaining / 2 + 8; /* ~24 down to ~8 */
        }
        if (spec[i].num_points < 4) spec[i].num_points = 4;
    }
}

/* ====== Host: apply a layer to canvas ====== */
void host_draw_layer(uint8_t* canvas, uint16_t seed, int layer_idx, int num_points, int block_size) {
    uint32_t state = ((uint32_t)seed << 16) | ((uint32_t)(layer_idx * 7 + 0x1337));
    for (int i = 0; i < 8; i++) {
        uint32_t bit = state & 1;
        state >>= 1;
        if (bit) state ^= 0xB4BCD35C;
    }

    for (int p = 0; p < num_points; p++) {
        uint32_t bit = state & 1;
        state >>= 1;
        if (bit) state ^= 0xB4BCD35C;

        int px = (state >> 0) & 0x7F;
        int py = ((state >> 7) & 0x7F) % H;
        px = (px / block_size) * block_size;
        py = (py / block_size) * block_size;

        for (int dy = 0; dy < block_size && (py + dy) < H; dy++) {
            for (int dx = 0; dx < block_size && (px + dx) < W; dx++) {
                int x = px + dx;
                int y = py + dy;
                int byte_idx = y * (W / 8) + (x / 8);
                int bit_idx = 7 - (x % 8);
                canvas[byte_idx] ^= (1 << bit_idx);
            }
        }
    }
}

/* ====== I/O ====== */
int load_pgm_binary(const char* path, uint8_t* packed) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4] = {0};
    int w = 0, h = 0, maxval = 0;
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return -2; }
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); }
        else if (c > ' ') { ungetc(c, f); break; }
    }
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) { fclose(f); return -3; }
    fgetc(f);
    if (w != W || h != H) { fclose(f); return -4; }

    /* Read pixels and pack to bits */
    memset(packed, 0, PIXELS / 8);
    uint8_t* raw = (uint8_t*)malloc(w * h);
    fread(raw, 1, w * h, f);
    fclose(f);

    for (int i = 0; i < w * h; i++) {
        if (raw[i] > maxval / 2) {
            int byte_idx = i / 8;
            int bit_idx = 7 - (i % 8);
            packed[byte_idx] |= (1 << bit_idx);
        }
    }
    free(raw);
    return 0;
}

void save_pgm_from_packed(const char* path, const uint8_t* packed) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int byte_idx = y * (W / 8) + (x / 8);
            int bit_idx = 7 - (x % 8);
            uint8_t v = (packed[byte_idx] >> bit_idx) & 1 ? 255 : 0;
            fwrite(&v, 1, 1, f);
        }
    }
    fclose(f);
}

void save_comparison(const char* path, const uint8_t* target, const uint8_t* generated) {
    int cw = W * 2 + 4;
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", cw, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < cw; x++) {
            uint8_t v;
            int px, byte_idx, bit_idx;
            if (x < W) {
                px = x; byte_idx = y*(W/8)+(px/8); bit_idx = 7-(px%8);
                v = (target[byte_idx] >> bit_idx) & 1 ? 255 : 0;
            } else if (x < W + 4) {
                v = 128;
            } else {
                px = x - W - 4; byte_idx = y*(W/8)+(px/8); bit_idx = 7-(px%8);
                v = (generated[byte_idx] >> bit_idx) & 1 ? 255 : 0;
            }
            fwrite(&v, 1, 1, f);
        }
    }
    fclose(f);
}

int count_errors(const uint8_t* a, const uint8_t* b) {
    int err = 0;
    for (int i = 0; i < PIXELS / 8; i++) {
        uint8_t diff = a[i] ^ b[i];
        err += __builtin_popcount(diff);
    }
    return err;
}

/* ====== Main ====== */
int main(int argc, char** argv) {
    int num_layers = 64;
    int device_id = 0;
    const char* target_path = NULL;
    const char* output_dir = "media/prng_images/layered";
    int save_every = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target") && i+1<argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1<argc) num_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i+1<argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1<argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--save-every") && i+1<argc) save_every = atoi(argv[++i]);
    }

    if (!target_path) {
        fprintf(stderr, "Usage: %s --target file.pgm [--layers 64] [--gpu 0] [--output dir]\n", argv[0]);
        return 1;
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU: %s\n", prop.name);

    /* Load target */
    uint8_t h_target[PIXELS / 8];
    if (load_pgm_binary(target_path, h_target) != 0) {
        fprintf(stderr, "Failed to load %s\n", target_path);
        return 1;
    }
    printf("Target: %s (%d white pixels)\n", target_path,
           count_errors(h_target, (uint8_t*)calloc(PIXELS/8, 1)));  /* count white = XOR with black */

    /* Output dir */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);

    /* Save target */
    char path[512];
    snprintf(path, sizeof(path), "%s/target.pgm", output_dir);
    save_pgm_from_packed(path, h_target);

    /* Layer schedule */
    LayerSpec* schedule = (LayerSpec*)malloc(num_layers * sizeof(LayerSpec));
    make_schedule(schedule, num_layers);

    printf("Layers: %d\n", num_layers);
    printf("Schedule: L0: blk=%d pts=%d ... L%d: blk=%d pts=%d\n",
           schedule[0].block_size, schedule[0].num_points,
           num_layers-1, schedule[num_layers-1].block_size, schedule[num_layers-1].num_points);

    /* GPU allocations */
    uint8_t *d_current, *d_target;
    uint32_t *d_errors;
    cudaMalloc(&d_current, PIXELS / 8);
    cudaMalloc(&d_target, PIXELS / 8);
    cudaMalloc(&d_errors, 65536 * sizeof(uint32_t));

    cudaMemcpy(d_target, h_target, PIXELS / 8, cudaMemcpyHostToDevice);

    /* Host canvas (starts black) */
    uint8_t h_canvas[PIXELS / 8];
    memset(h_canvas, 0, sizeof(h_canvas));

    /* Results */
    uint16_t* best_seeds = (uint16_t*)malloc(num_layers * sizeof(uint16_t));
    uint32_t h_errors[65536];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int initial_err = count_errors(h_canvas, h_target);
    printf("Initial error: %d / %d pixels (%.1f%%)\n", initial_err, PIXELS,
           100.0 * initial_err / PIXELS);

    for (int layer = 0; layer < num_layers; layer++) {
        int blk = schedule[layer].block_size;
        int pts = schedule[layer].num_points;

        /* Upload current canvas */
        cudaMemcpy(d_current, h_canvas, PIXELS / 8, cudaMemcpyHostToDevice);

        /* Search all 65536 seeds */
        int threads = 256;
        int blocks = (65536 + threads - 1) / threads;
        search_layer_kernel<<<blocks, threads>>>(
            d_current, d_target, d_errors, layer, pts, blk);
        cudaDeviceSynchronize();

        /* Download errors */
        cudaMemcpy(h_errors, d_errors, 65536 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        /* Find best seed */
        uint32_t best_err = 0xFFFFFFFF;
        uint16_t best_seed = 0;
        for (int s = 0; s < 65536; s++) {
            if (h_errors[s] < best_err) {
                best_err = h_errors[s];
                best_seed = (uint16_t)s;
            }
        }

        /* Apply best seed to host canvas */
        int prev_err = count_errors(h_canvas, h_target);
        host_draw_layer(h_canvas, best_seed, layer, pts, blk);
        int new_err = count_errors(h_canvas, h_target);
        best_seeds[layer] = best_seed;

        printf("Layer %2d: blk=%d pts=%3d seed=0x%04X err=%d→%d (-%d, %.1f%%)\n",
               layer, blk, pts, best_seed, prev_err, new_err,
               prev_err - new_err, 100.0 * new_err / PIXELS);

        /* Save checkpoint */
        if (layer % save_every == save_every - 1 || layer == num_layers - 1) {
            snprintf(path, sizeof(path), "%s/layer%02d_err%d.pgm", output_dir, layer, new_err);
            save_pgm_from_packed(path, h_canvas);
            snprintf(path, sizeof(path), "%s/layer%02d_compare.pgm", output_dir, layer);
            save_comparison(path, h_target, h_canvas);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    int final_err = count_errors(h_canvas, h_target);
    printf("\n=== RESULT ===\n");
    printf("Final error: %d / %d pixels (%.1f%%)\n", final_err, PIXELS,
           100.0 * final_err / PIXELS);
    printf("Layers: %d, Data: %d bytes (seeds only)\n", num_layers, num_layers * 2);
    printf("Time: %.1fs\n", elapsed);

    /* Save seed table */
    snprintf(path, sizeof(path), "%s/seeds.bin", output_dir);
    FILE* f = fopen(path, "wb");
    fwrite(best_seeds, 2, num_layers, f);
    fclose(f);

    /* Save seed table as text */
    snprintf(path, sizeof(path), "%s/seeds.txt", output_dir);
    f = fopen(path, "w");
    fprintf(f, "# Layer seeds for %s (%d layers, %d bytes)\n", target_path, num_layers, num_layers * 2);
    for (int i = 0; i < num_layers; i++)
        fprintf(f, "layer %2d: 0x%04X  blk=%d pts=%d\n",
                i, best_seeds[i], schedule[i].block_size, schedule[i].num_points);
    fclose(f);

    printf("Seeds: %s\n", path);
    printf("Output: %s/\n", output_dir);

    /* Cleanup */
    free(schedule);
    free(best_seeds);
    cudaFree(d_current);
    cudaFree(d_target);
    cudaFree(d_errors);

    return 0;
}

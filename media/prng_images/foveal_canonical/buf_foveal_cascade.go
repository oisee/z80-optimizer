package main

// Foveal AND-cascade: same LFSR-16 buffer, but position is searched greedily.
// Each step finds best (seed, ox, oy) — naturally focuses on high-error regions.
// ox, oy snapped to 8px (character-cell) grid.
//
// Layer schedule:
//   L0:   1 seed,  blk=4, AND-3  — full screen broad stroke
//   L1:   8 seeds, blk=2, AND-3  — greedy position search (64×48 patches)
//   L2:  16 seeds, blk=1, AND-4  — greedy position search (32×24 patches)
//   L3: 128 seeds, blk=1, AND-5
//   L4: 256 seeds, blk=1, AND-6
//   L5: 800 seeds, blk=1, AND-7

import (
	"encoding/json"
	"fmt"
	"math/bits"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

const (
	W       = 128
	H       = 96
	PS      = W * H / 8
	WORKERS = 12
	GRID    = 8 // position snap (pixels)
)

func lfsr16(s uint16) uint16 {
	bit := s & 1
	s >>= 1
	if bit != 0 {
		s ^= 0xB400
	}
	return s
}

func makeBuf(seed uint16, warmup, andN int) [768]byte {
	s := seed
	if s == 0 {
		s = 1
	}
	for i := 0; i < warmup; i++ {
		s = lfsr16(s)
	}
	var buf [768]byte
	for i := 0; i < 768; i++ {
		acc := uint16(1)
		for k := 0; k < andN; k++ {
			s = lfsr16(s)
			acc &= s & 1
		}
		buf[i] = byte(acc)
	}
	return buf
}

func applyBuf(pix *[PS]byte, buf *[768]byte, ox, oy, blk int) {
	for by := 0; by < 24; by++ {
		for bx := 0; bx < 32; bx++ {
			if buf[by*32+bx] == 0 {
				continue
			}
			for dy := 0; dy < blk; dy++ {
				for dx := 0; dx < blk; dx++ {
					x := ox + bx*blk + dx
					y := oy + by*blk + dy
					if x >= 0 && x < W && y >= 0 && y < H {
						pix[y*(W/8)+x/8] ^= 1 << (7 - (x % 8))
					}
				}
			}
		}
	}
}

func lBin(canvas, target *[PS]byte) float64 {
	err := 0
	for i := 0; i < PS; i++ {
		err += bits.OnesCount8(canvas[i] ^ target[i])
	}
	return float64(err) / float64(W*H)
}

// positions returns all valid (ox, oy) for given blk, snapped to GRID
func positions(blk int) [][2]int {
	pw := 32 * blk // patch pixel width
	ph := 24 * blk // patch pixel height
	var ps [][2]int
	for oy := 0; oy+ph <= H; oy += GRID {
		for ox := 0; ox+pw <= W; ox += GRID {
			ps = append(ps, [2]int{ox, oy})
		}
	}
	return ps
}

// findBest searches all seeds × all positions for (seed, ox, oy) with min region error.
// Uses delta approach: precompute baseErr per position, then delta per (buf, position).
func findBest(canvas, target *[PS]byte, andN, warmup, blk int) (uint16, int, int, int) {
	pos := positions(blk)
	pw := 32 * blk
	ph := 24 * blk

	// Precompute base region error for each position (once per step)
	baseErr := make([]int, len(pos))
	for pi, p := range pos {
		ox, oy := p[0], p[1]
		for y := oy; y < oy+ph && y < H; y++ {
			for x := ox; x < ox+pw && x < W; x++ {
				cb := (canvas[y*(W/8)+x/8] >> (7 - (x % 8))) & 1
				tb := (target[y*(W/8)+x/8] >> (7 - (x % 8))) & 1
				if cb != tb {
					baseErr[pi]++
				}
			}
		}
	}

	type res struct {
		seed          uint16
		ox, oy, delta int
	}

	chunk := 65536 / WORKERS
	results := make(chan res, WORKERS)
	var wg sync.WaitGroup

	for w := 0; w < WORKERS; w++ {
		wg.Add(1)
		start, end := w*chunk, (w+1)*chunk
		go func() {
			defer wg.Done()
			bestSeed := uint16(1)
			bestOx, bestOy, bestDelta := 0, 0, 1<<30

			for s := start; s < end; s++ {
				if s == 0 {
					continue
				}
				buf := makeBuf(uint16(s), warmup, andN)

				for pi, p := range pos {
					ox, oy := p[0], p[1]
					// compute delta: for each set bit in buf, check if flip helps or hurts
					delta := 0
					for by := 0; by < 24; by++ {
						for bx := 0; bx < 32; bx++ {
							if buf[by*32+bx] == 0 {
								continue
							}
							for dy := 0; dy < blk; dy++ {
								for dx := 0; dx < blk; dx++ {
									x := ox + bx*blk + dx
									y := oy + by*blk + dy
									if x >= 0 && x < W && y >= 0 && y < H {
										cb := (canvas[y*(W/8)+x/8] >> (7 - (x % 8))) & 1
										tb := (target[y*(W/8)+x/8] >> (7 - (x % 8))) & 1
										if cb == tb {
											delta++ // flip hurts
										} else {
											delta-- // flip helps
										}
									}
								}
							}
						}
					}
					// Foveal: minimize delta (maximize improvement) — not absolute newErr.
					// This finds the most error-dense region to fix, not the already-good one.
					_ = baseErr[pi] // baseErr still precomputed (available for callers)
					if delta < bestDelta {
						bestDelta = delta
						bestSeed = uint16(s)
						bestOx = ox
						bestOy = oy
					}
				}
			}
			results <- res{bestSeed, bestOx, bestOy, bestDelta}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	bestSeed := uint16(1)
	bestOx, bestOy, bestDelta := 0, 0, 1<<30
	for r := range results {
		if r.delta < bestDelta {
			bestDelta = r.delta
			bestSeed = r.seed
			bestOx = r.ox
			bestOy = r.oy
		}
	}
	return bestSeed, bestOx, bestOy, bestDelta
}

func saveCanvas(pix *[PS]byte, path string) {
	f, _ := os.Create(path)
	fmt.Fprintf(f, "P5\n%d %d\n255\n", W, H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			b := (pix[y*(W/8)+x/8] >> (7 - (x % 8))) & 1
			if b != 0 {
				f.Write([]byte{255})
			} else {
				f.Write([]byte{0})
			}
		}
	}
	f.Close()
}

type SeedRecord struct {
	Step   int    `json:"step"`
	Seed   uint16 `json:"seed"`
	Ox     int    `json:"ox"`
	Oy     int    `json:"oy"`
	Blk    int    `json:"blk"`
	AndN   int    `json:"and_n"`
	Warmup int    `json:"warmup"`
	Label  string `json:"label"`
}

type Phase struct {
	andN, blk, count int
	label             string
}

func loadPGM(path string) ([]byte, int, int, int, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, 0, 0, 0, err }
	var w, h, maxval int
	off := 3
	for off < len(data) && data[off] == '#' {
		for off < len(data) && data[off] != '\n' { off++ }
		off++
	}
	fmt.Sscanf(string(data[off:]), "%d %d", &w, &h)
	for data[off] != '\n' { off++ }
	off++
	for data[off] == '#' {
		for data[off] != '\n' { off++ }
		off++
	}
	fmt.Sscanf(string(data[off:]), "%d", &maxval)
	for data[off] != '\n' { off++ }
	off++
	return data[off:], w, h, maxval, nil
}

func main() {
	runtime.GOMAXPROCS(WORKERS)

	raw, tw, th, maxval, err := loadPGM("/home/alice/dev/z80-optimizer/media/prng_images/targets/che.pgm")
	if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
	if tw != W || th != H { fmt.Fprintln(os.Stderr, "size mismatch"); os.Exit(1) }

	var targetBin [PS]byte
	for i := 0; i < W*H; i++ {
		if int(raw[i]) > maxval/2 {
			targetBin[i/8] |= 1 << (7 - (i % 8))
		}
	}

	// Phase schedule
	phases := []Phase{
		{3, 4, 1,   "L0-AND3"},  // 1 seed: full screen blk=4
		{3, 2, 8,   "L1-AND3"},  // 8 seeds: blk=2 position search
		{4, 1, 16,  "L2-AND4"},  // 16 seeds: blk=1 position search
		{5, 1, 128, "L3-AND5"},
		{6, 1, 256, "L4-AND6"},
		{7, 1, 800, "L5-AND7"},
	}

	total := 0
	for _, p := range phases { total += p.count }

	fmt.Printf("Foveal cascade: %d total steps\n", total)
	fmt.Printf("  pos grid: %dpx  blk=4→1→1→1\n", GRID)
	for _, p := range phases {
		npos := len(positions(p.blk))
		fmt.Printf("  %s: %d steps × %d positions × 65535 seeds\n", p.label, p.count, npos)
	}
	fmt.Println()

	milestones := map[int]bool{1: true, 5: true, 9: true, 25: true, 153: true}
	for _, n := range []int{25, 50, 100, 153, 213, 409, 597, total} { milestones[n] = true }
	for i := 50; i <= total; i += 50 { milestones[i] = true }
	ml := make([]int, 0, len(milestones))
	for k := range milestones { ml = append(ml, k) }
	sort.Ints(ml)

	var canvas [PS]byte
	var seedLog []SeedRecord
	t0 := time.Now()
	step := 0
	snapshots := map[int]string{}

	fmt.Printf("%-6s  %-8s  %-8s  %-14s  %s\n", "step", "L_bin", "elapsed", "pos", "label")
	fmt.Println("------  --------  --------  --------------  ------")

	for _, phase := range phases {
		for pi := 0; pi < phase.count; pi++ {
			warmup := step
			seed, ox, oy, _ := findBest(&canvas, &targetBin, phase.andN, warmup, phase.blk)

			// apply if it improves global error
			buf := makeBuf(seed, warmup, phase.andN)
			var test [PS]byte
			copy(test[:], canvas[:])
			applyBuf(&test, &buf, ox, oy, phase.blk)
			if lBin(&test, &targetBin) < lBin(&canvas, &targetBin) {
				applyBuf(&canvas, &buf, ox, oy, phase.blk)
				seedLog = append(seedLog, SeedRecord{
					Step: len(seedLog) + 1, Seed: seed,
					Ox: ox, Oy: oy, Blk: phase.blk,
					AndN: phase.andN, Warmup: warmup, Label: phase.label,
				})
			}

			step++
			if milestones[step] || step == total {
				lb := lBin(&canvas, &targetBin)
				snap := fmt.Sprintf("/tmp/foveal_s%04d.pgm", step)
				saveCanvas(&canvas, snap)
				snapshots[step] = snap
				fmt.Printf("%-6d  %6.2f%%  %8v  (%3d,%2d) blk=%d  %s\n",
					step, lb*100, time.Since(t0).Round(time.Second),
					ox, oy, phase.blk, phase.label)
			}
		}
	}

	fmt.Printf("\nApplied %d / %d seeds\n", len(seedLog), total)

	jdata, _ := json.MarshalIndent(map[string]any{
		"lfsr16_poly":  "0xB400",
		"canvas_w":     W,
		"canvas_h":     H,
		"position_grid": GRID,
		"seeds":        seedLog,
	}, "", "  ")
	os.WriteFile("/tmp/foveal_cascade_seeds.json", jdata, 0644)
	fmt.Printf("Seeds: /tmp/foveal_cascade_seeds.json (%d entries)\n", len(seedLog))
	saveCanvas(&canvas, "/tmp/foveal_cascade_result.pgm")
}

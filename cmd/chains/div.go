package main

// Division chain search: find shortest abstract chain where
// f(n) = floor(n / K) for all n = 0..255.
//
// Abstract ops: dbl, add(i), sub(i), save, neg, shr, shr4
// State: current value (integer, unbounded during computation)
//        saved slots (up to 3)
// Verification: run chain on all 256 inputs, check each gives floor(n/K)

// divState tracks the chain computation for ALL 256 inputs simultaneously.
// This is the key insight: instead of tracking one value, track a vector of 256 values.
type divState struct {
	values [256]int32 // current value for each input 0..255
	saved  [3][256]int32
	nSaved int
}

var (
	divBestDepth int
	divBestOps   []byte
	divNodes     uint64
)

func divSearch(s divState, ops []byte, depth, maxDepth int, target [256]uint8) {
	divNodes++

	// Check if current values match target for all 256 inputs
	if depth > 0 {
		match := true
		for i := 0; i < 256; i++ {
			if uint8(s.values[i]&0xFF) != target[i] {
				match = false
				break
			}
		}
		if match {
			if depth < divBestDepth {
				divBestDepth = depth
				divBestOps = make([]byte, depth)
				copy(divBestOps, ops[:depth])
			}
			return
		}
	}

	if depth >= divBestDepth-1 || depth >= maxDepth {
		return
	}

	// Try each op (applied to ALL 256 values simultaneously)

	// dbl: v = v * 2
	{
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = s.values[i] * 2
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 0 // dbl
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}

	// shr: v = v >> 1 (arithmetic for signed, logical for unsigned)
	{
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = s.values[i] >> 1
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 5 // shr
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}

	// shr4: v = v >> 4
	{
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = s.values[i] >> 4
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 6 // shr4
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}

	// add(i) for each saved slot
	for si := 0; si < s.nSaved; si++ {
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = s.values[i] + s.saved[si][i]
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 1 + byte(si)*10 // add(si)
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}

	// sub(i) for each saved slot
	for si := 0; si < s.nSaved; si++ {
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = s.values[i] - s.saved[si][i]
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 2 + byte(si)*10 // sub(si)
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}

	// save
	if s.nSaved < 3 {
		ns := s
		ns.saved[ns.nSaved] = s.values
		ns.nSaved++
		ops[depth] = 3 // save
		divSearch(ns, ops, depth+1, maxDepth, target)
	}

	// neg: v = -v
	{
		ns := s
		diff := false
		for i := 0; i < 256; i++ {
			ns.values[i] = -s.values[i]
			if ns.values[i] != s.values[i] {
				diff = true
			}
		}
		if diff {
			ops[depth] = 4 // neg
			divSearch(ns, ops, depth+1, maxDepth, target)
		}
	}
}

var divOpNames = map[byte]string{
	0: "dbl", 1: "add(0)", 2: "sub(0)", 3: "save", 4: "neg", 5: "shr", 6: "shr4",
	11: "add(1)", 12: "sub(1)", 21: "add(2)", 22: "sub(2)",
}

func decodeDivOps(ops []byte) []string {
	names := make([]string, len(ops))
	for i, op := range ops {
		if n, ok := divOpNames[op]; ok {
			names[i] = n
		} else {
			names[i] = "?"
		}
	}
	return names
}

func solveDivK(k, maxDepth int) *ChainResult {
	divBestDepth = maxDepth + 1
	divBestOps = nil
	divNodes = 0

	// Target: floor(n / k) for all n = 0..255
	var target [256]uint8
	for i := 0; i < 256; i++ {
		target[i] = uint8(i / k)
	}

	// Initial state: values[i] = i (the input)
	var s divState
	for i := 0; i < 256; i++ {
		s.values[i] = int32(i)
	}

	ops := make([]byte, maxDepth+1)

	// Iterative deepening
	for d := 1; d <= maxDepth; d++ {
		divBestDepth = d + 1
		divSearch(s, ops, 0, d, target)
		if divBestOps != nil {
			break
		}
	}

	if divBestOps == nil {
		return nil
	}

	return &ChainResult{
		K:     k,
		Ops:   decodeDivOps(divBestOps),
		Depth: len(divBestOps),
	}
}

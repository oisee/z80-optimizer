package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/oisee/z80-optimizer/pkg/gpu"
	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
	"github.com/oisee/z80-optimizer/pkg/search"
	"github.com/oisee/z80-optimizer/pkg/stoke"
	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "z80opt",
		Short: "Z80 superoptimizer — find optimal instruction sequences",
	}

	// enumerate command
	var maxTarget int
	var output string
	var checkpoint string
	var verbose bool
	var numWorkers int
	var deadFlagsStr string
	var useGPU bool

	enumCmd := &cobra.Command{
		Use:   "enumerate",
		Short: "Enumerate all target sequences and find shorter replacements",
		RunE: func(cmd *cobra.Command, args []string) error {
			deadFlags, err := parseDeadFlags(deadFlagsStr)
			if err != nil {
				return err
			}

			fmt.Printf("Z80 Superoptimizer\n")
			fmt.Printf("  Max target length: %d\n", maxTarget)
			fmt.Printf("  Target instructions: %d per position (8-bit only)\n", search.InstructionCount8())
			fmt.Printf("  Candidate instructions: %d per position (incl. 16-bit)\n", search.InstructionCount())
			if useGPU {
				fmt.Printf("  Mode: GPU (WebGPU/Vulkan)\n")
			} else {
				fmt.Printf("  Workers: %d\n", numWorkers)
			}
			if deadFlags != 0 {
				fmt.Printf("  Dead flags: 0x%02X (%s)\n", deadFlags, result.DeadFlagDesc(deadFlags))
			}
			fmt.Println()

			var rules []result.Rule

			if useGPU {
				gpuCfg := gpu.SearchConfig{
					MaxTargetLen: maxTarget,
					Verbose:      verbose,
					DeadFlags:    deadFlags,
				}
				table, err := gpu.SearchGPU(gpuCfg)
				if err != nil {
					return fmt.Errorf("GPU search failed: %w", err)
				}
				rules = table.Rules()
			} else {
				cfg := search.Config{
					MaxTargetLen: maxTarget,
					NumWorkers:   numWorkers,
					Verbose:      verbose,
					DeadFlags:    deadFlags,
				}
				table := search.Run(cfg)
				rules = table.Rules()
			}

			fmt.Printf("\nFound %d optimizations\n", len(rules))

			if output != "" {
				f, err := os.Create(output)
				if err != nil {
					return err
				}
				defer f.Close()
				if err := result.WriteJSON(f, rules); err != nil {
					return err
				}
				fmt.Printf("Written to %s\n", output)
			}

			_ = checkpoint // TODO: implement checkpoint resume
			return nil
		},
	}
	enumCmd.Flags().IntVar(&maxTarget, "max-target", 2, "Maximum target sequence length")
	enumCmd.Flags().StringVar(&output, "output", "", "Output JSON file path")
	enumCmd.Flags().StringVar(&checkpoint, "checkpoint", "", "Checkpoint file for resume")
	enumCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
	enumCmd.Flags().IntVar(&numWorkers, "workers", 0, "Number of workers (0 = NumCPU)")
	enumCmd.Flags().StringVar(&deadFlagsStr, "dead-flags", "none", "Dead flags mask: none, undoc, all, or hex (e.g. 0x13)")
	enumCmd.Flags().BoolVar(&useGPU, "gpu", false, "Use GPU acceleration (WebGPU/Vulkan)")

	// target command
	var maxCand int

	targetCmd := &cobra.Command{
		Use:   "target [instructions]",
		Short: "Find optimal replacement for a specific instruction sequence",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			// Parse the target sequence from assembly
			input := strings.Join(args, " ")
			seq, err := parseAssembly(input)
			if err != nil {
				return fmt.Errorf("failed to parse: %w", err)
			}

			fmt.Printf("Target: %s (%d bytes, %d T-states)\n",
				input, inst.SeqByteSize(seq), inst.SeqTStates(seq))

			rule := search.SearchSingle(seq, maxCand, verbose)
			if rule == nil {
				fmt.Println("No shorter replacement found.")
				return nil
			}

			fmt.Printf("Replacement: ")
			for i, instr := range rule.Replacement {
				if i > 0 {
					fmt.Print(" : ")
				}
				fmt.Print(inst.Disassemble(instr))
			}
			fmt.Printf(" (-%d bytes, -%d cycles)\n", rule.BytesSaved, rule.CyclesSaved)
			return nil
		},
	}
	targetCmd.Flags().IntVar(&maxCand, "max-candidate", 4, "Maximum candidate length")
	targetCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")

	// verify command
	verifyCmd := &cobra.Command{
		Use:   "verify [rules.json]",
		Short: "Re-verify all rules in a JSON file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()

			rules, err := result.ReadJSON(f)
			if err != nil {
				return err
			}

			fmt.Printf("Verifying %d rules...\n", len(rules))
			for i, r := range rules {
				fmt.Printf("  [%d] %s -> %s ... ", i+1, r.SourceASM, r.ReplacementASM)
				// TODO: parse assembly back to instructions and ExhaustiveCheck
				fmt.Println("(parse-back not yet implemented)")
			}
			return nil
		},
	}

	// export command
	var format string

	exportCmd := &cobra.Command{
		Use:   "export [rules.json]",
		Short: "Export rules in various formats",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()

			_, err = result.ReadJSON(f)
			if err != nil {
				return err
			}

			switch format {
			case "go":
				fmt.Println("// Go export not yet implemented — use verify + manual integration")
			default:
				return fmt.Errorf("unknown format: %s", format)
			}
			return nil
		},
	}
	exportCmd.Flags().StringVarP(&format, "format", "f", "go", "Output format (go)")

	// stoke command
	var stokeChains int
	var stokeIter int
	var stokeDecay float64
	var stokeOutput string
	var stokeVerbose bool
	var stokeDeadFlagsStr string

	stokeCmd := &cobra.Command{
		Use:   "stoke",
		Short: "Run STOKE stochastic superoptimizer on a target sequence",
		RunE: func(cmd *cobra.Command, args []string) error {
			targetStr, _ := cmd.Flags().GetString("target")
			if targetStr == "" {
				return fmt.Errorf("--target is required")
			}
			seq, err := parseAssembly(targetStr)
			if err != nil {
				return fmt.Errorf("failed to parse target: %w", err)
			}

			deadFlags, err := parseDeadFlags(stokeDeadFlagsStr)
			if err != nil {
				return err
			}

			cfg := stoke.Config{
				Target:     seq,
				Chains:     stokeChains,
				Iterations: stokeIter,
				Decay:      stokeDecay,
				Verbose:    stokeVerbose,
				DeadFlags:  deadFlags,
			}

			results := stoke.Run(cfg)
			results = stoke.Deduplicate(results)

			fmt.Printf("\n%d unique optimizations found\n", len(results))
			for i, r := range results {
				fmt.Printf("  %d. ", i+1)
				for j, instr := range r.Rule.Replacement {
					if j > 0 {
						fmt.Print(" : ")
					}
					fmt.Print(inst.Disassemble(instr))
				}
				if r.Rule.DeadFlags != 0 {
					fmt.Printf(" (-%d bytes, -%d cycles, dead flags: %s)\n",
						r.Rule.BytesSaved, r.Rule.CyclesSaved, result.DeadFlagDesc(r.Rule.DeadFlags))
				} else {
					fmt.Printf(" (-%d bytes, -%d cycles)\n",
						r.Rule.BytesSaved, r.Rule.CyclesSaved)
				}
			}

			if stokeOutput != "" && len(results) > 0 {
				rules := make([]result.Rule, len(results))
				for i, r := range results {
					rules[i] = r.Rule
				}
				f, err := os.Create(stokeOutput)
				if err != nil {
					return err
				}
				defer f.Close()
				if err := result.WriteJSON(f, rules); err != nil {
					return err
				}
				fmt.Printf("Written to %s\n", stokeOutput)
			}
			return nil
		},
	}
	stokeCmd.Flags().String("target", "", "Target assembly sequence (colon-separated)")
	stokeCmd.Flags().IntVar(&stokeChains, "chains", runtime.NumCPU(), "Number of MCMC chains")
	stokeCmd.Flags().IntVar(&stokeIter, "iterations", 10_000_000, "Iterations per chain")
	stokeCmd.Flags().Float64Var(&stokeDecay, "decay", 0.9999, "Temperature decay factor")
	stokeCmd.Flags().StringVar(&stokeOutput, "output", "", "Output JSON file path")
	stokeCmd.Flags().BoolVarP(&stokeVerbose, "verbose", "v", false, "Verbose output")
	stokeCmd.Flags().StringVar(&stokeDeadFlagsStr, "dead-flags", "none", "Dead flags mask: none, undoc, all, or hex (e.g. 0xFF)")

	// verify-jsonl command: verify CUDA JSONL output against CPU ExhaustiveCheck
	var verifyDeadFlagsStr string
	verifyJSONLCmd := &cobra.Command{
		Use:   "verify-jsonl [file.jsonl]",
		Short: "Verify JSONL rules from CUDA search using CPU ExhaustiveCheck",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return verifyJSONL(args[0], verifyDeadFlagsStr, verbose)
		},
	}
	verifyJSONLCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
	verifyJSONLCmd.Flags().StringVar(&verifyDeadFlagsStr, "dead-flags", "none", "Dead flags mask for verification")

	rootCmd.AddCommand(enumCmd, targetCmd, verifyCmd, exportCmd, stokeCmd, verifyJSONLCmd)
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// parseDeadFlags parses the --dead-flags flag value.
func parseDeadFlags(s string) (search.FlagMask, error) {
	switch strings.ToLower(s) {
	case "none", "":
		return search.DeadNone, nil
	case "undoc":
		return search.DeadUndoc, nil
	case "all":
		return search.DeadAll, nil
	default:
		// Try parsing as hex (0xFF or FF)
		s = strings.TrimPrefix(strings.ToLower(s), "0x")
		v, err := strconv.ParseUint(s, 16, 8)
		if err != nil {
			return 0, fmt.Errorf("invalid --dead-flags value %q: use none, undoc, all, or hex (e.g. 0xFF)", s)
		}
		return search.FlagMask(v), nil
	}
}

// parseAssembly converts assembly text like "LD A, 0" into instructions.
func parseAssembly(text string) ([]inst.Instruction, error) {
	// Split on : for multi-instruction sequences
	parts := strings.Split(text, ":")
	var seq []inst.Instruction

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		instr, err := parseSingleInstruction(part)
		if err != nil {
			return nil, fmt.Errorf("cannot parse %q: %w", part, err)
		}
		seq = append(seq, instr)
	}

	if len(seq) == 0 {
		return nil, fmt.Errorf("no instructions parsed from %q", text)
	}
	return seq, nil
}

func parseSingleInstruction(text string) (inst.Instruction, error) {
	text = strings.TrimSpace(text)
	upper := strings.ToUpper(text)

	// Try to match against all catalog mnemonics
	for op := inst.OpCode(0); op < inst.OpCodeCount; op++ {
		info := &inst.Catalog[op]
		if info.Mnemonic == "" {
			continue
		}

		if !inst.HasImmediate(op) {
			if strings.EqualFold(text, info.Mnemonic) {
				return inst.Instruction{Op: op}, nil
			}
			continue
		}

		// For immediate instructions, the mnemonic has "n" as placeholder
		// Match the pattern with any hex/decimal value
		pattern := strings.ToUpper(info.Mnemonic)
		nIdx := strings.LastIndex(pattern, "N")
		if nIdx < 0 {
			continue
		}
		prefix := pattern[:nIdx]
		suffix := pattern[nIdx+1:]

		if !strings.HasPrefix(upper, prefix) {
			continue
		}
		if suffix != "" && !strings.HasSuffix(upper, suffix) {
			continue
		}

		valStr := upper[len(prefix):]
		if suffix != "" {
			valStr = valStr[:len(valStr)-len(suffix)]
		}
		valStr = strings.TrimSpace(valStr)

		val, err := parseImmediate(valStr)
		if err != nil {
			continue
		}
		return inst.Instruction{Op: op, Imm: uint16(val)}, nil
	}

	return inst.Instruction{}, fmt.Errorf("unknown instruction: %s", text)
}

func verifyJSONL(path string, deadFlagsStr string, verbose bool) error {
	deadFlags, err := parseDeadFlags(deadFlagsStr)
	if err != nil {
		return err
	}

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	total, passed, failed, skipped := 0, 0, 0, 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		total++

		var rule struct {
			SourceASM      string `json:"source_asm"`
			ReplacementASM string `json:"replacement_asm"`
			BytesSaved     int    `json:"bytes_saved"`
			CyclesSaved    int    `json:"cycles_saved"`
		}
		if err := json.Unmarshal([]byte(line), &rule); err != nil {
			fmt.Fprintf(os.Stderr, "  [%d] JSON parse error: %v\n", total, err)
			skipped++
			continue
		}

		source, err := parseAssembly(rule.SourceASM)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%d] Cannot parse source %q: %v\n", total, rule.SourceASM, err)
			skipped++
			continue
		}
		replacement, err := parseAssembly(rule.ReplacementASM)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%d] Cannot parse replacement %q: %v\n", total, rule.ReplacementASM, err)
			skipped++
			continue
		}

		var ok bool
		if deadFlags == search.DeadNone {
			ok = search.ExhaustiveCheck(source, replacement)
		} else {
			ok = search.ExhaustiveCheckMasked(source, replacement, deadFlags)
		}

		if ok {
			passed++
			if verbose {
				fmt.Printf("  [%d] PASS: %s -> %s\n", total, rule.SourceASM, rule.ReplacementASM)
			}
		} else {
			failed++
			fmt.Printf("  [%d] FAIL: %s -> %s\n", total, rule.SourceASM, rule.ReplacementASM)
		}

		if total%10000 == 0 {
			fmt.Fprintf(os.Stderr, "  Progress: %d verified (%d pass, %d fail, %d skip)\n",
				total, passed, failed, skipped)
		}
	}

	fmt.Printf("\nVerification complete: %d total, %d passed, %d failed, %d skipped\n",
		total, passed, failed, skipped)
	if failed > 0 {
		return fmt.Errorf("%d rules failed verification", failed)
	}
	return nil
}

func parseImmediate(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty")
	}

	// Handle hex: 0xFF, FFh, 0x00, etc.
	if strings.HasPrefix(s, "0X") || strings.HasPrefix(s, "0x") {
		var v int
		_, err := fmt.Sscanf(s, "0x%x", &v)
		if err != nil {
			_, err = fmt.Sscanf(s, "0X%x", &v)
		}
		return v, err
	}
	if strings.HasSuffix(strings.ToUpper(s), "H") {
		s = s[:len(s)-1]
		var v int
		_, err := fmt.Sscanf(s, "%x", &v)
		return v, err
	}

	// Decimal
	var v int
	_, err := fmt.Sscanf(s, "%d", &v)
	return v, err
}

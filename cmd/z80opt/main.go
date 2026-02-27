package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/oisee/z80-optimizer/pkg/inst"
	"github.com/oisee/z80-optimizer/pkg/result"
	"github.com/oisee/z80-optimizer/pkg/search"
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

	enumCmd := &cobra.Command{
		Use:   "enumerate",
		Short: "Enumerate all target sequences and find shorter replacements",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg := search.Config{
				MaxTargetLen: maxTarget,
				NumWorkers:   numWorkers,
				Verbose:      verbose,
			}

			fmt.Printf("Z80 Superoptimizer\n")
			fmt.Printf("  Max target length: %d\n", cfg.MaxTargetLen)
			fmt.Printf("  Instruction count: %d per position\n", search.InstructionCount())
			fmt.Printf("  Workers: %d\n", cfg.NumWorkers)
			fmt.Println()

			table := search.Run(cfg)
			rules := table.Rules()

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

	rootCmd.AddCommand(enumCmd, targetCmd, verifyCmd, exportCmd)
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
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

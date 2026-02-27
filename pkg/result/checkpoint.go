package result

import (
	"encoding/gob"
	"os"

	"github.com/oisee/z80-optimizer/pkg/inst"
)

// Checkpoint holds state for resuming a search.
type Checkpoint struct {
	Rules           []Rule
	CompletedTarget int // Number of target sequences fully searched
	TargetLen       int // Current target length being searched
}

func init() {
	// Register types for gob encoding
	gob.Register(inst.Instruction{})
	gob.Register(inst.OpCode(0))
}

// SaveCheckpoint writes search state to a file.
func SaveCheckpoint(path string, ckpt *Checkpoint) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(ckpt)
}

// LoadCheckpoint loads search state from a file.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ckpt Checkpoint
	if err := gob.NewDecoder(f).Decode(&ckpt); err != nil {
		return nil, err
	}
	return &ckpt, nil
}

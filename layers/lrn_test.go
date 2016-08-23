package layers

import (
	"testing"

	"github.com/Rompei/mat"
)

func TestLRN(t *testing.T) {
	lrn := LRN{
		N:     5,
		K:     2,
		Alpha: 0.0005,
		Beta:  0.75,
	}

	input := make([][][]float32, 96)
	for i := range input {
		m := mat.Random(64, 64)
		input[i] = m.M
	}

	res := lrn.Forward(input)
	t.Log(res)
}

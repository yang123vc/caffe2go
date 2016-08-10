package layers

import (
	"testing"

	"github.com/Rompei/mat"
)

func TestSoftmax(t *testing.T) {
	input := make([][][]float32, 1000)
	for i := 0; i < 1000; i++ {
		input[i] = mat.Random(1, 1).M
	}
	t.Log(input)

	softMax := SoftmaxLossLayer{}
	output, err := softMax.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(output)
}

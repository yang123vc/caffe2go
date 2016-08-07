package layers

import "testing"

func TestDropout(t *testing.T) {
	input := [][][]float32{
		{
			{0, 1, -1},
			{-100, 23, 43},
			{1.234, -0.22, 0.5},
		},
	}

	dropout := DropoutLayer{
		Ratio: 0.5,
	}

	output, err := dropout.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	t.Log(output)
}

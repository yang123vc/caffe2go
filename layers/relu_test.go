package layers

import "testing"

func TestReLU(t *testing.T) {
	input := [][][]float32{
		{
			{0, 1, -1},
			{-100, 23, 43},
			{1.234, -0.22, 0.5},
		},
	}
	relu := ReLULayer{}
	output, err := relu.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	ans := [][][]float32{
		{
			{0, 1, 0},
			{0, 23, 43},
			{1.234, 0, 0.5},
		},
	}

	for i := range output {
		for j := range output[i] {
			for k := range output[i][j] {
				if output[i][j][k] != ans[i][j][k] {
					t.Log(output)
					t.Fatal("not same")
				}
			}
		}
	}
}

package layers

import (
	"testing"
)

func TestFullconnect(t *testing.T) {
	input := [][][]float32{
		{
			{1},
		},
		{
			{2},
		},
		{
			{3},
		},
	}

	weights := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
		{13, 14, 15},
	}

	bias := []float32{
		1, 2, 1, 0, 0,
	}

	fc := FullconnectLayer{
		Width:    3,
		Height:   5,
		BiasTerm: true,
		Weights:  weights,
		Bias:     bias,
	}

	ans := [][][]float32{
		{
			{15},
		},
		{
			{34},
		},
		{
			{51},
		},
		{
			{68},
		},
		{
			{86},
		},
	}

	output, err := fc.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	for i := range output {
		for j := range output[i] {
			for k := range output[i][j] {
				if ans[i][j][k] != output[i][j][k] {
					t.Error("not same")
					t.Log(output)
				}
			}
		}
	}

}

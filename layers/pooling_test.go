package layers

import "testing"

func TestPooling(t *testing.T) {
	input := [][][]float32{
		{
			{1, 1, 2, 4},
			{5, 6, 7, 8},
			{3, 2, 1, 0},
			{1, 2, 3, 4},
		},
	}

	ans := [][][]float32{
		{
			{6, 8},
			{3, 4},
		},
	}

	pool := PoolingLayer{
		KernelSize: 2,
		Stride:     2,
		Padding:    0,
	}

	output, err := pool.Forward(input)
	if err != nil {
		t.Fatal(err)
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

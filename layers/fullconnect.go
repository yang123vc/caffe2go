package layers

import (
	"github.com/gonum/matrix/mat64"
)

// FullconnectLayer is a layer.
type FullconnectLayer struct {
	*BaseLayer
	Width    int
	Height   int
	Weights  [][]float32
	BiasTerm bool
	Bias     []float32
}

// NewFullconnectLayer is constructor.
func NewFullconnectLayer(name, t string, width, height int, biasTerm bool) *FullconnectLayer {
	w := make([][]float32, height)

	return &FullconnectLayer{
		BaseLayer: NewBaseLayer(name, t),
		Width:     width,
		Height:    height,
		BiasTerm:  biasTerm,
		Weights:   w,
	}
}

// Forward forawards a step.
func (f *FullconnectLayer) Forward(input [][][]float32) ([][][]float32, error) {
	in := make([]float64, f.Width)

	idx := 0
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				in[idx] = float64(input[i][j][k])
				idx++
			}
		}
	}

	inMat := mat64.NewDense(f.Width, 1, in)
	weights := ConvertMatrix(f.Weights)
	var out mat64.Dense
	out.Mul(weights, inMat)

	output := make([][][]float32, f.Height)
	for i := range output {
		output[i] = make([][]float32, 1)
		for j := range output[i] {
			output[i][j] = make([]float32, 1)
			for k := range output[i][j] {
				output[i][j][k] = float32(out.At(i, 0)) + f.Bias[i]
			}
		}
	}

	return output, nil
}

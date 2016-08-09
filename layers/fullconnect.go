package layers

import (
	"github.com/Rompei/mat"
)

// FullconnectLayer is a layer.
type FullconnectLayer struct {
	*BaseLayer
	Width    int32
	Height   int32
	Weights  [][]float32
	BiasTerm bool
	Bias     []float32
}

// NewFullconnectLayer is constructor.
func NewFullconnectLayer(name, t string, width, height int32, biasTerm bool) *FullconnectLayer {
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
	output := make([][][]float32, f.Height)
	for i := range output {
		output[i] = make([][]float32, 1)
		for j := range output[i] {
			output[i][j] = make([]float32, 1)
		}
	}
	output[0] = make([][]float32, 1)
	output[0][0] = make([]float32, 1)
	vec := make([][]float32, 1)
	vec[0] = make([]float32, f.Width)
	idx := 0
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				vec[0][idx] = input[i][j][k]
				idx++
			}
		}
	}
	for i := range f.Weights {
		w := make([][]float32, 1)
		w[0] = f.Weights[i]
		wMatrix := mat.NewMatrix(w).T()
		res, err := mat.Mul(mat.NewMatrix(vec), wMatrix)
		if err != nil {
			return nil, err
		}
		output[i][0][0] = res.M[0][0]
	}

	return output, nil
}

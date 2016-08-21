package layers

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// ReLULayer is layer of ReLU.
type ReLULayer struct {
	*BaseLayer
	Slope float32
}

// NewReLULayer is constructor.
func NewReLULayer(name, t string, slope float32) *ReLULayer {
	return &ReLULayer{
		BaseLayer: NewBaseLayer(name, t),
		Slope:     slope,
	}
}

// Forward forwards a step.
func (r *ReLULayer) Forward(input [][][]float32) ([][][]float32, error) {
	for i := range input {
		t := ConvertMatrix(input[i])
		var out mat64.Dense
		out.Apply(func(i, j int, v float64) float64 {
			return math.Max(0, v)
		}, t)
		input[i] = ConvertMat64(&out)
	}
	return input, nil
}

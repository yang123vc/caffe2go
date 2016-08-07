package layers

import (
	"github.com/Rompei/mat"
	"math"
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
		target := mat.NewMatrix(input[i])
		target.BroadcastFunc(func(v float32, a ...interface{}) float32 {
			return float32(math.Max(0, float64(v)))
		})
		input[i] = target.M
	}
	return input, nil
}

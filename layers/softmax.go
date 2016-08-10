package layers

import (
	"math"
)

// SoftmaxLossLayer is layer of Softmax loss.
type SoftmaxLossLayer struct {
	*BaseLayer
}

// NewSoftmaxLossLayer is constructor.
func NewSoftmaxLossLayer(name, t string) *SoftmaxLossLayer {
	return &SoftmaxLossLayer{
		BaseLayer: NewBaseLayer(name, t),
	}
}

// Forward forwards a step.
func (s *SoftmaxLossLayer) Forward(input [][][]float32) ([][][]float32, error) {
	total := float32(0.0)
	max := float32(0.0)

	// Calculate maximum.
	for i := range input {
		if max < input[i][0][0] {
			max = input[i][0][0]
		}
	}

	// Calculate total.
	for i := range input {
		input[i][0][0] = float32(math.Exp(float64(input[i][0][0] - max)))
		total += input[i][0][0]
	}

	for i := range input {
		input[i][0][0] = input[i][0][0] / total
	}

	return input, nil
}

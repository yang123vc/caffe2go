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
		for j := range input[i] {
			for k := range input[i][j] {
				if max < input[i][j][k] {
					max = input[i][j][k]
				}
			}
		}
	}

	// Calculate total.
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				input[i][j][k] = float32(math.Exp(float64(input[i][j][k] - max)))
				total += input[i][j][k]
			}
		}
	}

	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				input[i][j][k] = input[i][j][k] / total
			}
		}
	}

	return input, nil
}

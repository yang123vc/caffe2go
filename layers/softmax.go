package layers

import "fmt"
import "math"

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
	total := 0.0
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				total += math.Exp(float64(input[i][j][k]))
			}
		}
	}
	fmt.Println(total)

	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				input[i][j][k] = float32(math.Exp(float64(input[i][j][k])) / total)
			}
		}
	}
	return input, nil
}

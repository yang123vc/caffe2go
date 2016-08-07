package layers

import (
	"math/rand"
	"time"
)

// DropoutLayer is layer of Dropout.
type DropoutLayer struct {
	*BaseLayer
	Ratio float32
}

// NewDropoutLayer is constructor.
func NewDropoutLayer(name, t string, ratio float32) *DropoutLayer {
	return &DropoutLayer{
		BaseLayer: NewBaseLayer(name, t),
		Ratio:     ratio,
	}
}

// Forward fowards a step.
func (d *DropoutLayer) Forward(input [][][]float32) ([][][]float32, error) {
	rand.Seed(time.Now().UnixNano())
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				if rand.Float32() < d.Ratio {
					input[i][j][k] = 0
				}
			}
		}
	}
	return input, nil
}

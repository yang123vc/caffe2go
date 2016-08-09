package layers

import "github.com/Rompei/mat"

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
	for i := range input {
		t := mat.NewMatrix(input[i])
		res := t.BroadcastMul(d.Ratio)
		input[i] = res.M
	}
	return input, nil
}

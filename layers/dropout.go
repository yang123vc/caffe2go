package layers

import "github.com/gonum/matrix/mat64"

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
	r := float64(d.Ratio)
	doneCh := make(chan bool, len(input))
	for i := range input {
		go func(idx int) {
			t := ConvertMatrix(input[idx])
			var out mat64.Dense
			out.Apply(func(i, j int, v float64) float64 {
				return v * r
			}, t)
			input[idx] = ConvertMat64(&out)
			doneCh <- true
		}(i)
	}
	for range input {
		<-doneCh
	}
	return input, nil
}

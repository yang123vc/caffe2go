package layers

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

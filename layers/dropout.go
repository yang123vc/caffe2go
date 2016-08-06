package layers

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

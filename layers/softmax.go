package layers

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

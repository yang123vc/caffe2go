package layers

// FullConnectLayer is a layer.
type FullConnectLayer struct {
	*BaseLayer
	Width    int32
	Height   int32
	Weights  []float32
	BiasTerm bool
	Bias     []float32
}

// NewFullConnetLayer is constructor.
func NewFullConnetLayer(name, t string, width, height int32, biasTerm bool) *FullConnectLayer {
	return &FullConnectLayer{
		BaseLayer: NewBaseLayer(name, t),
		Width:     width,
		Height:    height,
		BiasTerm:  biasTerm,
	}
}

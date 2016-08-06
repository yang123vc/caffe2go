package layers

// PoolingLayer is layer of Pooling.
type PoolingLayer struct {
	*BaseLayer
	KernelSize uint32
	Stride     uint32
	Padding    uint32
}

// NewPoolingLayer is constructor.
func NewPoolingLayer(name, t string, kernelSize, stride, padding uint32) *PoolingLayer {
	return &PoolingLayer{
		BaseLayer:  NewBaseLayer(name, t),
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    padding,
	}
}

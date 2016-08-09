package layers

import (
	"github.com/Rompei/mat"
)

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

// Forward fowards a step.
func (pool *PoolingLayer) Forward(input [][][]float32) ([][][]float32, error) {
	output := make([][][]float32, len(input))
	for i := range input {
		output[i] = mat.NewMatrix(input[i]).Pooling(uint(pool.KernelSize), uint(pool.Stride), mat.Max).M
	}
	return output, nil
}

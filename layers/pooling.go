package layers

import (
	"github.com/Rompei/mat"
)

// PoolingLayer is layer of Pooling.
type PoolingLayer struct {
	*BaseLayer
	KernelSize int
	Stride     int
	Padding    int
}

// NewPoolingLayer is constructor.
func NewPoolingLayer(name, t string, kernelSize, stride, padding int) *PoolingLayer {
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
	doneCh := make(chan bool, len(input))
	for i := range input {
		go func(i int, doneCh chan bool) {
			output[i] = mat.NewMatrix(input[i]).Pooling(uint(pool.KernelSize), uint(pool.Stride), mat.Max).M
			doneCh <- true
		}(i, doneCh)
	}
	for range input {
		<-doneCh
	}
	return output, nil
}

package layers

import "github.com/Rompei/exmat"

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
			rows := len(input[i])
			cols := len(input[i][0])
			t := make([]float64, rows*cols)
			for j := range input[i] {
				for k := range input[i][j] {
					t[cols*j+k] = float64(input[i][j][k])
				}
			}
			in := exmat.NewExMat(rows, cols, t)
			var out exmat.ExMat
			out.Pooling(pool.KernelSize, pool.Stride, exmat.Max, in)
			output[i] = ConvertMat64(out)
			doneCh <- true
		}(i, doneCh)
	}
	for range input {
		<-doneCh
	}
	close(doneCh)
	return output, nil
}

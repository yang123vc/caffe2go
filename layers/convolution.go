package layers

import (
	"github.com/Rompei/exmat"
	"github.com/Rompei/mat"
	"github.com/gonum/matrix/mat64"
)

// ConvolutionLayer is layer of Convolution.
type ConvolutionLayer struct {
	*BaseLayer
	NInput     uint32
	NOutput    uint32
	KernelSize int
	Stride     int
	Padding    int
	Weights    [][][][]float32
	Bias       []float32
	BiasTerm   bool
}

// NewConvolutionLayer is constructor.
func NewConvolutionLayer(name, t string, nInput, nOutput uint32, kernelSize, stride, padding int, biasTerm bool) *ConvolutionLayer {
	w := make([][][][]float32, nOutput)
	for i := 0; i < int(nOutput); i++ {
		w[i] = make([][][]float32, nInput)
		for j := 0; j < int(nInput); j++ {
			w[i][j] = make([][]float32, kernelSize)
			for k := 0; k < int(kernelSize); k++ {
				w[i][j][k] = make([]float32, kernelSize)
			}
		}
	}
	return &ConvolutionLayer{
		BaseLayer:  NewBaseLayer(name, t),
		NInput:     nInput,
		NOutput:    nOutput,
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    padding,
		Weights:    w,
		BiasTerm:   biasTerm,
	}
}

// Forward forwards a step.
func (conv *ConvolutionLayer) Forward(input [][][]float32) ([][][]float32, error) {
	if conv.Padding > 0 {
		doneCh := make(chan bool, len(input))
		for i := range input {
			go func(i int, doneCh chan bool) {
				in := ConvertMatrix(input[i])
				inExMat := exmat.NewExMatFromDense(in)
				input[i] = ConvertMat64(inExMat.ZeroPadding(conv.Padding))
				doneCh <- true
			}(i, doneCh)
		}
		for i := 0; i < len(input); i++ {
			<-doneCh
		}
		close(doneCh)
	}
	in := ConvertMatrix(Im2Col(input, conv.KernelSize, conv.Stride))
	kernels := make([][]float32, conv.NOutput)
	doneCh := make(chan bool, conv.NOutput)
	for i := 0; i < int(conv.NOutput); i++ {
		go func(i int, doneCh chan bool) {
			kernels[i] = Im2Col(conv.Weights[i], conv.KernelSize, conv.Stride)[0]
			doneCh <- true
		}(i, doneCh)
	}
	for i := 0; i < int(conv.NOutput); i++ {
		<-doneCh
	}
	close(doneCh)
	kernelMatrix := ConvertMatrix(kernels)
	var out mat64.Dense
	out.Mul(in, kernelMatrix.T())
	output := make([][][]float32, conv.NOutput)
	rows := (len(input[0])-conv.KernelSize)/conv.Stride + 1
	cols := (len(input[0][0])-conv.KernelSize)/conv.Stride + 1
	outTransposed := out.T()
	r, c := outTransposed.Dims()
	errCh := make(chan error, r)
	for i := 0; i < r; i++ {
		go func(i int, errCh chan error) {
			part := make([][]float32, 1)
			part[0] = make([]float32, c)
			for j := 0; j < c; j++ {
				part[0][j] = float32(outTransposed.At(i, j))
			}
			res, err := mat.NewMatrix(part).Reshape(uint(rows), uint(cols))
			if err != nil {
				errCh <- err
				return
			}
			output[i] = res.M
			errCh <- nil
		}(i, errCh)
	}
	for i := 0; i < r; i++ {
		if err := <-errCh; err != nil {
			return nil, err
		}
	}
	close(errCh)

	if conv.BiasTerm {
		doneCh := make(chan bool, len(output))
		for i := range output {
			go func(idx int) {
				m := ConvertMatrix(output[idx])
				var res mat64.Dense
				res.Apply(func(i, j int, v float64) float64 {
					return v + float64(conv.Bias[idx])
				}, m)
				output[idx] = ConvertMat64(&res)
				doneCh <- true
			}(i)
		}
		for range output {
			<-doneCh
		}
		close(doneCh)
	}

	return output, nil
}

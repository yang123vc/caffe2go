package layers

import (
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

func (conv *ConvolutionLayer) addMatrixes(ms []*mat.Matrix) (res float32, err error) {
	sumurise := mat.Zeros(ms[0].Rows, ms[0].Cols)
	for i := range ms {
		if sumurise, err = mat.Add(sumurise, ms[i]); err != nil {
			return
		}
	}
	for y := range sumurise.M {
		for x := range sumurise.M[y] {
			res += sumurise.M[y][x]
		}
	}
	return
}

// Forward forwards a step.
func (conv *ConvolutionLayer) Forward(input [][][]float32) ([][][]float32, error) {
	if conv.Padding > 0 {
		doneCh := make(chan bool, len(input))
		for i := range input {
			go func(i int, doneCh chan bool) {
				input[i] = mat.NewMatrix(input[i]).Pad(uint(conv.Padding), mat.Max).M
				doneCh <- true
			}(i, doneCh)
		}
		for i := 0; i < len(input); i++ {
			<-doneCh
		}
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

	if conv.BiasTerm {
		for i := range output {
			output[i] = mat.NewMatrix(output[i]).BroadcastAdd(conv.Bias[i]).M
		}
	}

	return output, nil
}

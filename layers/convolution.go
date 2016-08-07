package layers

import (
	"fmt"
	"github.com/Rompei/mat"
	"os"
)

// ConvolutionLayer is layer of Convolution.
type ConvolutionLayer struct {
	*BaseLayer
	NInput     uint32
	NOutput    uint32
	KernelSize uint32
	Stride     uint32
	Padding    uint32
	Weights    [][][][]float32
	Bias       []float32
}

// NewConvolutionLayer is constructor.
func NewConvolutionLayer(name, t string, nInput, nOutput, kernelSize, stride, padding uint32) *ConvolutionLayer {
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
		for i := range input {
			input[i] = mat.NewMatrix(input[i]).Pad(uint(conv.Padding), mat.Max).M
		}
	}
	in := mat.NewMatrix(Im2Col(input, int(conv.KernelSize), int(conv.Stride)))
	kernels := make([][]float32, conv.NOutput)
	for i := 0; i < int(conv.NOutput); i++ {
		kernels[i] = Im2Col(conv.Weights[i], int(conv.KernelSize), int(conv.Stride))[0]
	}
	kernelMatrix := mat.NewMatrix(kernels).T()
	out, err := mat.Mul(in, kernelMatrix)
	if err != nil {
		return nil, err
	}
	output := make([][][]float32, conv.NOutput)
	rows := (len(input[0])-int(conv.KernelSize))/int(conv.Stride) + 1
	cols := (len(input[0][0])-int(conv.KernelSize))/int(conv.Stride) + 1
	for i := range output {
		output[i] = make([][]float32, rows)
		for j := range output[i] {
			output[i][j] = make([]float32, cols)
		}
	}
	out, err = out.Reshape(out.Cols, out.Rows)
	if err != nil {
		return nil, err
	}
	for i := range out.M {
		part := make([][]float32, 1)
		part[0] = out.M[i]
		res, err := mat.NewMatrix(part).Reshape(uint(rows), uint(cols))
		if err != nil {
			return nil, err
		}
		output[i] = res.M
	}

	fmt.Println(output)
	os.Exit(0)

	return output, nil
}

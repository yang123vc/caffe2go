package layers

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

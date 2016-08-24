package c2g

import (
	"errors"
	"fmt"

	"github.com/Rompei/caffe2go/layers"
)

// SetupConvolution setups ConvolutionLayer from caffe model.
func SetupConvolution(layer LayerParameter) *layers.ConvolutionLayer {
	blobs := layer.GetBlobs()
	param := layer.GetConvolutionParam()
	kernelSize := getKernelSize(param)
	fmt.Println("KernelSize: ", kernelSize)
	stride := getStride(param)
	fmt.Println("Stride: ", stride)
	pad := getPad(param)
	fmt.Println("Padding: ", pad)
	num := getNnum(blobs[0])
	channels := getChannels(blobs[0])
	nIn := uint32(channels) * param.GetGroup()
	fmt.Println("nInput: ", nIn)
	nOut := uint32(num)
	fmt.Println("nOutput: ", nOut)
	fmt.Println("Group: ", param.GetGroup())
	biasTerm := param.GetBiasTerm()
	fmt.Println("BiasTerm: ", biasTerm)
	convLayer := layers.NewConvolutionLayer(layer.GetName(), layers.Convolution, nIn, nOut, kernelSize, stride, pad, biasTerm)
	idx := 0
	// Calculate kernelsize
	for i := 0; i < int(nOut); i++ {
		for j := 0; j < int(nIn); j++ {
			for k := 0; k < int(kernelSize); k++ {
				for l := 0; l < int(kernelSize); l++ {
					convLayer.Weights[i][j][k][l] = blobs[0].GetData()[idx]
					idx++
				}
			}
		}
	}
	if biasTerm {
		convLayer.Bias = blobs[1].GetData()
	}
	return convLayer
}

// SetupFullconnect setups FullConnectLayer.
func SetupFullconnect(layer LayerParameter) (*layers.FullconnectLayer, error) {
	param := layer.GetInnerProductParam()
	blobs := layer.GetBlobs()
	width, err := getWidth(blobs[0])
	if err != nil {
		return nil, err
	}
	fmt.Println("Width: ", width)
	height, err := getHeight(blobs[0])
	if err != nil {
		return nil, err
	}
	fmt.Println("Height: ", height)
	biasTerm := param.GetBiasTerm()
	fmt.Println("BiasTerm: ", biasTerm)
	fcLayer := layers.NewFullconnectLayer(layer.GetName(), layers.InnerProduct, width, height, biasTerm)
	weights := blobs[0].GetData()
	for i := 0; i < len(weights)/int(width); i++ {
		fcLayer.Weights[i] = weights[i*int(width) : i*int(width)+int(width)]
	}
	fmt.Println(len(fcLayer.Weights), len(fcLayer.Weights[0]))
	if biasTerm {
		fcLayer.Bias = blobs[1].GetData()
	}

	return fcLayer, nil
}

// SetupPooling setups PoolingLayer from caffe model.
func SetupPooling(layer LayerParameter) *layers.PoolingLayer {
	param := layer.GetPoolingParam()
	kernelSize := getKernelSize(param)
	fmt.Println("KernelSize: ", kernelSize)
	stride := getStride(param)
	fmt.Println("Stride: ", stride)
	pad := getPad(param)
	fmt.Println("Padding: ", pad)
	return layers.NewPoolingLayer(layer.GetName(), layers.Pooling, kernelSize, stride, pad)
}

// SetupDropout setups DropoutLayer from caffe model.
func SetupDropout(layer LayerParameter) *layers.DropoutLayer {
	param := layer.GetDropoutParam()
	ratio := param.GetDropoutRatio()
	fmt.Println(ratio)
	return layers.NewDropoutLayer(layer.GetName(), layers.Dropout, ratio)
}

// SetupReLU setups ReLULayer from caffe model.
func SetupReLU(layer LayerParameter) *layers.ReLULayer {
	param := layer.GetReluParam()
	slope := param.GetNegativeSlope()
	fmt.Println("Slope: ", slope)
	reluLayer := layers.NewReLULayer(layer.GetName(), layers.ReLU, slope)
	return reluLayer
}

// SetupSoftmaxLoss setups SoftmaxLossLayer from caffe model.
func SetupSoftmaxLoss(layer LayerParameter) *layers.SoftmaxLossLayer {
	return layers.NewSoftmaxLossLayer(layer.GetName(), layers.SoftmaxWithLoss)
}

// SetupLRN setups LRNLayer from caffe model.
func SetupLRN(layer LayerParameter) *layers.LRN {
	param := layer.GetLrnParam()
	k := param.GetK()
	fmt.Println("k: ", k)
	localSize := param.GetLocalSize()
	fmt.Println("LocalSize: ", localSize)
	alpha := param.GetAlpha()
	fmt.Println("alpha: ", alpha)
	beta := param.GetBeta()
	fmt.Println("beta: ", beta)
	return layers.NewLRNLayer(layer.GetName(), layers.Lrn, int(localSize), float64(k), float64(alpha), float64(beta))
}

func getKernelSize(param Parameter) int {
	if kernelH := param.GetKernelH(); kernelH > 0 {
		return int(kernelH)
	}
	return int(param.GetKernelSize())
}

func getStride(param Parameter) int {
	if strideH := param.GetStrideH(); strideH > 0 {
		return int(strideH)
	}
	return int(param.GetStride())
}

func getPad(param Parameter) int {
	if padH := param.GetPadH(); padH > 0 {
		return int(padH)
	}
	return int(param.GetPad())
}

func getNnum(blob Blob) int {
	if num := blob.GetNum(); num > 0 {
		return int(num)
	}
	return int(blob.GetShape().GetDim()[0])
}

func getChannels(blob Blob) int {
	if channels := blob.GetChannels(); channels > 0 {
		return int(channels)
	}
	return int(blob.GetShape().GetDim()[1])
}

func getWidth(blob Blob) (int, error) {
	if width := blob.GetWidth(); width > 0 {
		return int(width), nil
	}
	if len(blob.GetShape().GetDim()) == 2 {
		return int(blob.GetShape().GetDim()[1]), nil
	}
	if len(blob.GetShape().GetDim()) == 4 {
		return int(blob.GetShape().GetDim()[3]), nil
	}
	return 0, errors.New("Width is not defined.")
}

func getHeight(blob Blob) (int, error) {
	if height := blob.GetHeight(); height > 0 {
		return int(height), nil
	}
	if len(blob.GetShape().GetDim()) == 2 {
		return int(blob.GetShape().GetDim()[0]), nil
	}
	if len(blob.GetShape().GetDim()) == 4 {
		return int(blob.GetShape().GetDim()[2]), nil
	}
	return 0, errors.New("Height is not defined.")
}

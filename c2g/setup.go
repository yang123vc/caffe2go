package c2g

import (
	"fmt"
	"github.com/Rompei/caffe2go/layers"
)

// SetupConvolution setups ConvolutionLayer from caffe model.
func SetupConvolution(layer LayerParameter) *layers.ConvolutionLayer {
	blobs := layer.GetBlobs()
	param := layer.GetConvolutionParam()
	kernelSize := param.GetKernelSize()
	fmt.Println("KernelSize: ", kernelSize)
	stride := param.GetStride()
	fmt.Println("Stride: ", stride)
	pad := param.GetPad()
	fmt.Println("Padding: ", pad)
	num := blobs[0].GetNum()
	channels := blobs[0].GetChannels()
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
func SetupFullconnect(layer LayerParameter) *layers.FullconnectLayer {
	param := layer.GetInnerProductParam()
	blobs := layer.GetBlobs()
	width := blobs[0].GetWidth()
	fmt.Println("Width: ", width)
	height := blobs[0].GetHeight()
	fmt.Println("Height: ", height)
	biasTerm := param.GetBiasTerm()
	fmt.Println("BiasTerm: ", biasTerm)
	fcLayer := layers.NewFullconnectLayer(layer.GetName(), layers.Fullconnect, width, height, biasTerm)
	weights := blobs[0].GetData()
	for i := 0; i < len(weights)/int(width); i++ {
		fcLayer.Weights[i] = weights[i*int(width) : i*int(width)+int(width)]
	}
	fmt.Println(len(fcLayer.Weights))
	if biasTerm {
		fcLayer.Bias = blobs[1].GetData()
	}

	return fcLayer
}

// SetupPooling setups PoolingLayer from caffe model.
func SetupPooling(layer LayerParameter) *layers.PoolingLayer {
	param := layer.GetPoolingParam()
	kernelSize := param.GetKernelSize()
	fmt.Println("KernelSize: ", kernelSize)
	stride := param.GetStride()
	fmt.Println("Stride: ", stride)
	pad := param.GetPad()
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
	return layers.NewSoftmaxLossLayer(layer.GetName(), layers.SoftmaxLoss)
}

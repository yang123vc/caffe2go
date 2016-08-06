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
	convLayer := layers.NewConvolutionLayer(layer.GetName(), layers.Convolution, nIn, nOut, kernelSize, stride, pad)
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
	convLayer.Bias = blobs[1].GetData()
	return convLayer
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

// SetupSoftmaxLoss setups SoftmaxLossLayer from caffe model.
func SetupSoftmaxLoss(layer LayerParameter) *layers.SoftmaxLossLayer {
	return layers.NewSoftmaxLossLayer(layer.GetName(), layers.SoftmaxLoss)
}

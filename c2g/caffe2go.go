package c2g

import (
	"fmt"
	"io/ioutil"
	"log"

	"github.com/Rompei/caffe2go/caffe"
	"github.com/Rompei/caffe2go/network"
	"github.com/golang/protobuf/proto"
)

// Caffe2Go is interface of caffe2go.
type Caffe2Go struct {
	Network *network.Network
}

// NewCaffe2Go is constructor.
func NewCaffe2Go(modelPath string) *Caffe2Go {
	data, err := ioutil.ReadFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}
	var netParameter caffe.NetParameter
	if err = proto.Unmarshal(data, &netParameter); err != nil {
		log.Fatalln(err)
	}
	fmt.Println(netParameter.GetName())
	var net network.Network
	if len(netParameter.Layer) != 0 {
		showLayers(netParameter.Layer)
		for i := range netParameter.GetLayer() {
			fmt.Println(netParameter.Layer[i].GetName())
		}
	} else {
		showV1Layers(netParameter.Layers)
		for i := range netParameter.GetLayers() {
			switch netParameter.Layers[i].GetType() {
			case caffe.V1LayerParameter_CONVOLUTION:
				fmt.Println(caffe.V1LayerParameter_CONVOLUTION)
				convLayer := SetupConvolution(netParameter.Layers[i])
				net.Add(convLayer)
				fmt.Println()
			case caffe.V1LayerParameter_POOLING:
				fmt.Println(caffe.V1LayerParameter_POOLING)
				poolLayer := SetupPooling(netParameter.Layers[i])
				net.Add(poolLayer)
				fmt.Println()
			case caffe.V1LayerParameter_DROPOUT:
				fmt.Println(caffe.V1LayerParameter_DROPOUT)
				dropoutLayer := SetupDropout(netParameter.Layers[i])
				net.Add(dropoutLayer)
				fmt.Println()
			case caffe.V1LayerParameter_SOFTMAX_LOSS:
				fmt.Println(caffe.V1LayerParameter_SOFTMAX_LOSS)
				softmaxLossLayer := SetupSoftmaxLoss(netParameter.Layers[i])
				net.Add(softmaxLossLayer)
				fmt.Println()
			}
		}
	}
	return &Caffe2Go{
		Network: &net,
	}
}

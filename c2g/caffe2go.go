package c2g

import (
	"errors"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"os"

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
	var net network.Network
	if len(netParameter.GetLayer()) != 0 {
		// TODO: implement
	} else {
		showV1Layers(netParameter.GetLayers())
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

// Predict start network.
func (c2g *Caffe2Go) Predict(imagePath string) ([][][]float32, error) {
	reader, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, err
	}
	input := im2vec(img)
	input, err = crop(input, 227)
	if err != nil {
		return nil, err
	}
	return c2g.Network.Predict(input)
}

func im2vec(img image.Image) [][][]float32 {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y
	res := make([][][]float32, 3)
	for i := 0; i < 3; i++ {
		res[i] = make([][]float32, height)
	}
	for y := 0; y < height; y++ {
		for i := 0; i < 3; i++ {
			res[i][y] = make([]float32, width)
		}
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			res[0][y][x] = float32(r) / 65025
			res[1][y][x] = float32(g) / 65025
			res[2][y][x] = float32(b) / 65025
		}
	}
	return res
}

func crop(tensor [][][]float32, l int) ([][][]float32, error) {
	if len(tensor[0]) < l || len(tensor[0][0]) < l {
		return nil, errors.New("Length is mismatched")
	}
	sy := (len(tensor[0]) - l) / 2
	sx := (len(tensor[0][0]) - l) / 2
	res := make([][][]float32, len(tensor))
	for i := range tensor {
		res[i] = make([][]float32, l)
		y := 0
		for _, s1 := range tensor[i][sy : sy+l] {
			res[i][y] = make([]float32, l)
			x := 0
			for _, s2 := range s1[sx : sx+l] {
				res[i][y][x] = s2
				x++
			}
			y++
		}
	}
	return res, nil
}

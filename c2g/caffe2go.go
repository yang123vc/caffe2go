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
	"github.com/nfnt/resize"
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
			case caffe.V1LayerParameter_INNER_PRODUCT:
				fmt.Println(caffe.V1LayerParameter_INNER_PRODUCT)
				fcLayer := SetupFullconnect(netParameter.Layers[i])
				net.Add(fcLayer)
				fmt.Println()
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
			case caffe.V1LayerParameter_SOFTMAX:
				fmt.Println(caffe.V1LayerParameter_SOFTMAX)
				softMaxLayer := SetupSoftmaxLoss(netParameter.Layers[i])
				net.Add(softMaxLayer)
				fmt.Println()
			case caffe.V1LayerParameter_RELU:
				fmt.Println(caffe.V1LayerParameter_RELU)
				reluLayer := SetupReLU(netParameter.Layers[i])
				net.Add(reluLayer)
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
	img = resize.Resize(224, 224, img, resize.Lanczos3)
	input := im2vec(img)
	//input, err = crop(input, 224)
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
			res[0][y][x] = (float32(r)/255 - 103.939)
			res[1][y][x] = (float32(g)/255 - 116.779)
			res[2][y][x] = (float32(b)/255 - 123.68)
		}
	}
	return res
}

func crop(tensor [][][]float32, l int) ([][][]float32, error) {
	w := len(tensor[0][0])
	h := len(tensor[0])
	if h < l || w < l {
		return nil, errors.New("Length is mismatched")
	}
	var w1, h1 int
	if w > h {
		w1 = l * w / h
		h1 = l
	} else {
		w1 = l
		h1 = l * h / w
	}
	sx := (w1 - l) / 2
	sy := (h1 - l) / 2
	res := make([][][]float32, len(tensor))
	for i := range tensor {
		res[i] = make([][]float32, l)
		for j, s1 := range tensor[i][sy : sy+l] {
			res[i][j] = s1[sx : sx+l]
		}
	}
	return res, nil
}

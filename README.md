# Caffe2Go

Caffe2Go evaluate caffemodel with Golang

## Usage

Command line interface

'''
./caffe2go -i images/plane.jpg -m models/nin\_imagenet.caffemodel -l labels/synset\_words.txt -s 224 -mf means.txt
'''

Options

'''
Usage of ./caffe2go:
-i string
Path for image.
-l string
Path for labels.
-m string
Path for caffemodel.
-mf string
Meanfile path
-s uint
Input Shape

'''

Use the library on your own software

'''go
package main

import (
	"fmt"
	_ "image/jpeg"
	_ "image/png"

	"github.com/Rompei/caffe2go/c2g"
)

func main() {
	caffe2go, err := c2g.NewCaffe2Go("lenet.caffemodel")
	if err != nil {
		panic(err)
	}
	output, err := caffe2go.Predict("mnist_zero.png", 28, nil)
	if err != nil {
		panic(err)
	}

	for i := range output {
		fmt.Printf("%d: %f\n", i, output[i][0][0])
	}
}
'''

## Supported layers

Now supports the layers below

'''
Convolution
Pooling
ReLU
FullyConnected
Dropout
Softmax
'''

## License

[BSD-2](https://opensource.org/licenses/BSD-2-Clause)

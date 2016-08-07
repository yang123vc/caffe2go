package main

import (
	"flag"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"runtime"

	"github.com/Rompei/caffe2go/c2g"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		modelPath string
		imagePath string
	)

	flag.StringVar(&modelPath, "m", "", "Path for caffemodel.")
	flag.StringVar(&imagePath, "i", "", "Path for image.")
	flag.Parse()

	if modelPath == "" || imagePath == "" {
		log.Fatalln("Option is not enough.")
	}

	caffe2go := c2g.NewCaffe2Go(modelPath)
	if output, err := caffe2go.Predict(imagePath); err != nil {
		panic(err)
		log.Fatalln(err)
	} else {
		fmt.Println(output)
		for i := range output {
			fmt.Printf("%d: %f\n", i, output[i][0][0])
		}
	}
}

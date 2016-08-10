package main

import (
	"flag"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"runtime"
	"strings"

	"github.com/Rompei/caffe2go/c2g"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		modelPath string
		imagePath string
		labelPath string
		shape     uint
	)

	flag.StringVar(&modelPath, "m", "", "Path for caffemodel.")
	flag.StringVar(&imagePath, "i", "", "Path for image.")
	flag.StringVar(&labelPath, "l", "", "Path for labels.")
	flag.UintVar(&shape, "s", 0, "Input Shape")
	flag.Parse()

	if modelPath == "" || imagePath == "" || shape == 0 {
		log.Fatalln("Option is not enough.")
	}

	caffe2go, err := c2g.NewCaffe2Go(modelPath)
	if err != nil {
		log.Fatalln(err)
	}
	output, err := caffe2go.Predict(imagePath, shape)
	if err != nil {
		log.Fatalln(err)
	}

	if labelPath != "" {
		result := make([]float32, len(output))
		for i := range output {
			result[i] = output[i][0][0]
		}
		indice := make([]int, len(result))
		labelData, err := ioutil.ReadFile(labelPath)
		if err != nil {
			log.Fatalln(err)
		}
		labels := strings.Split(string(labelData), "\n")

		Argsort(result, indice)
		for i := range indice[:10] {
			fmt.Printf("%f:%s\n", result[i], labels[indice[i]])
		}
	}
}

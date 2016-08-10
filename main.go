package main

import (
	"flag"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/Rompei/caffe2go/c2g"
)

func loadMeans(meanFile string) ([]float32, error) {
	b, err := ioutil.ReadFile(meanFile)
	if err != nil {
		return nil, err
	}
	means := strings.Split(string(b), ",")
	res := make([]float32, len(means))
	for i := range means {
		out, err := strconv.ParseFloat(means[i], 32)
		if err != nil {
			return nil, err
		}
		res[i] = float32(out)
	}
	return res, nil
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		modelPath string
		imagePath string
		labelPath string
		shape     uint
		meanFile  string
	)

	flag.StringVar(&modelPath, "m", "", "Path for caffemodel.")
	flag.StringVar(&imagePath, "i", "", "Path for image.")
	flag.StringVar(&labelPath, "l", "", "Path for labels.")
	flag.StringVar(&meanFile, "mf", "", "Meanfile path")
	flag.UintVar(&shape, "s", 0, "Input Shape")
	flag.Parse()

	if modelPath == "" || imagePath == "" || shape == 0 {
		log.Fatalln("Option is not enough.")
	}

	var means []float32
	var err error
	if meanFile != "" {
		means, err = loadMeans(meanFile)
		if err != nil {
			log.Fatalln(err)
		}
	}

	caffe2go, err := c2g.NewCaffe2Go(modelPath)
	if err != nil {
		log.Fatalln(err)
	}
	start := time.Now()
	output, err := caffe2go.Predict(imagePath, shape, means)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Done in %fs\n", time.Now().Sub(start).Seconds())

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

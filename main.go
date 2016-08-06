package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"

	"github.com/Rompei/caffe2go/c2g"
)

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
			res[0][y][x] = float32(r) / 255.0
			res[1][y][x] = float32(g) / 255.0
			res[2][y][x] = float32(b) / 255.0
		}
	}
	return res
}

func main() {

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

	reader, err := os.Open(imagePath)
	if err != nil {
		log.Fatalln(err)
	}
	defer reader.Close()

	img, _, err := image.Decode(reader)
	if err != nil {
		log.Fatalln(err)
	}
	input := im2vec(img)
	fmt.Println(input[0][0])

	c2g.NewCaffe2Go(modelPath)
}

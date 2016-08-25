package layers

import "github.com/gonum/matrix/mat64"

// Im2Col converts image 3D tensor to matrix.
func Im2Col(img [][][]float32, kernelSize, stride int) [][]float32 {
	colSize := kernelSize * kernelSize * len(img)
	rows := (len(img[0])-kernelSize)/stride + 1
	cols := (len(img[0][0])-kernelSize)/stride + 1
	res := make([][]float32, rows*cols)
	idx1 := 0
	doneCh := make(chan bool, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			go func(x, y, idx1 int) {
				col := make([]float32, colSize)
				idx2 := 0
				sy := y * stride
				sx := x * stride
				for c := range img {
					for i := sy; i < sy+kernelSize; i++ {
						for j := sx; j < sx+kernelSize; j++ {
							col[idx2] = img[c][i][j]
							idx2++
						}
					}
				}
				res[idx1] = col
				doneCh <- true
			}(x, y, idx1)
			idx1++
		}
	}
	for i := 0; i < rows*cols; i++ {
		<-doneCh
	}
	close(doneCh)
	return res
}

// ConvertMatrix converts slice of vector to mat64.Matrix
func ConvertMatrix(m [][]float32) *mat64.Dense {
	cols := len(m[0])
	flatten := make([]float64, len(m)*cols)
	for i := range m {
		for j := range m[i] {
			flatten[cols*i+j] = float64(m[i][j])
		}
	}
	return mat64.NewDense(len(m), cols, flatten)
}

// ConvertMat64 converts mat64 matrix to [][]float32
func ConvertMat64(m mat64.Matrix) [][]float32 {
	r, c := m.Dims()
	res := make([][]float32, r)
	for i := 0; i < r; i++ {
		res[i] = make([]float32, c)
		for j := 0; j < c; j++ {
			res[i][j] = float32(m.At(i, j))
		}
	}
	return res
}

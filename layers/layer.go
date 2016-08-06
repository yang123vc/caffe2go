package layers

const (
	Convolution = "CONVOLUTION"
	Pooling     = "POOLING"
	ReLU        = "RELU"
	Dropout     = "DROPOUT"
	SoftmaxLoss = "SOFTMAX_LOSS"
)

// Layer is object of network layer.
type Layer interface {
	GetName() string
	GetType() string
	Forward([][][]float32) [][][]float32
}

package layers

const (
	Data            = "Data"
	Convolution     = "Convolution"
	InnerProduct    = "InnerProduct"
	Pooling         = "Pooling"
	ReLU            = "ReLU"
	Dropout         = "Dropout"
	SoftmaxWithLoss = "SoftmaxWithLoss"
	Softmax         = "Softmax"
)

// Layer is object of network layer.
type Layer interface {
	GetName() string
	GetType() string
	Forward([][][]float32) ([][][]float32, error)
}
